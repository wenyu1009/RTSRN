import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
sys.path.append('../')
sys.path.append('./')
from utils import util, ssim_psnr
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from interfaces import base
from utils.meters import AverageMeter
from utils.metrics import get_string_aster, get_string_crnn, Accuracy
from utils.util import str_filt
from utils import utils_moran

from model import gumbel_softmax
from loss.semantic_loss import SemanticLoss
from copy import deepcopy
from tensorboardX import SummaryWriter

from ptflops import get_model_complexity_info
import string


TEST_MODEL = "MORAN"
sem_loss = SemanticLoss()
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
lossfn_ctc_lm = torch.nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)
ssim = ssim_psnr.SSIM()

ABLATION_SET = ["tsrn_tl_cascade", "srcnn_tl", "srresnet_tl", "rdn_tl", "vdsr_tl",'rtsrn']

from torch.optim.lr_scheduler import _LRScheduler
 
 
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def to_numpy(tensor):
  if torch.is_tensor(tensor):
    return tensor.cpu().numpy()
  elif type(tensor).__module__ != 'numpy':
    raise ValueError("Cannot convert {} to numpy array"
                     .format(type(tensor)))
  return tensor

def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()


def get_str_list(output, target, dataset=None):
  # label_seq
  assert output.dim() == 2 and target.dim() == 2

  end_label = dataset.char2id[dataset.EOS]
  unknown_label = dataset.char2id[dataset.UNKNOWN]
  num_samples, max_len_labels = output.size()
  num_classes = len(dataset.char2id.keys())
  assert num_samples == target.size(0) and max_len_labels == target.size(1)
  output = to_numpy(output)
  target = to_numpy(target)

  # list of char list
  pred_list, targ_list = [], []
  for i in range(num_samples):
    pred_list_i = []
    for j in range(max_len_labels):
      if output[i, j] != end_label:
        if output[i, j] != unknown_label:
          pred_list_i.append(dataset.id2char[output[i, j]])
      else:
        break
    pred_list.append(pred_list_i)

  for i in range(num_samples):
    targ_list_i = []
    for j in range(max_len_labels):
      if target[i, j] != end_label:
        if target[i, j] != unknown_label:
          targ_list_i.append(dataset.id2char[target[i, j]])
      else:
        break
    targ_list.append(targ_list_i)

  # char list to string
  # if dataset.lowercase:
  if True:
    # pred_list = [''.join(pred).lower() for pred in pred_list]
    # targ_list = [''.join(targ).lower() for targ in targ_list]
    pred_list = [_normalize_text(pred) for pred in pred_list]
    targ_list = [_normalize_text(targ) for targ in targ_list]
  else:
    pred_list = [''.join(pred) for pred in pred_list]
    targ_list = [''.join(targ) for targ in targ_list]

  return pred_list, targ_list

class TextSR(base.TextBase):

    def cal_conf(self, images_lr, rec_model):
        SR_confidence = []
        for image_lr in images_lr:
            img_np = image_lr[:, :3, :, :].data.cpu().numpy()[0] * 255
            # print("img_np:", img_np.shape)
            img_np = np.transpose((img_np * 255).astype(np.uint8), (1, 2, 0))
            img_np_L = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            l_var = cv2.Laplacian(img_np_L, cv2.CV_64F).var()
            # print("img_up:", np.unique(img_np), img_np.shape)

            aster_dict_hr = self.parse_crnn_data(image_lr[:, :3, :, :])
            label_vecs_lr = rec_model(aster_dict_hr)
            label_vecs_lr = torch.nn.functional.softmax(label_vecs_lr, -1)
            # [26, 1, 37] - > [1, 26, 37]
            SR_conf = label_vecs_lr.permute(1, 0, 2).data.cpu().numpy()

            conf_idx = np.argmax(SR_conf[0, :, :], axis=-1)
            picked_score = SR_conf[0, np.arange(SR_conf.shape[1]), conf_idx]

            SR_conf = np.sum(picked_score * (conf_idx > 0)) / float(np.sum(conf_idx > 0) + 1e-10)
            SR_confidence.append(SR_conf)

        return SR_confidence

    def train(self):
        self.global_img_val_cnt = 0
        TP_Generator_dict = {
            "CRNN": self.CRNN_init,
            "OPT": self.TPG_init
        }

        tpg_opt = self.opt_TPG

        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']

        model_list = [model]
        if not self.args.sr_share:
            for i in range(self.args.stu_iter - 1):
                model_sep = self.generator_init(i+1)['model']
                model_list.append(model_sep)

        tensorboard_dir = os.path.join("tensorboard", self.vis_dir)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            print("Directory exist, remove events...")
            os.popen("rm " + tensorboard_dir + "/*")

        self.results_recorder = SummaryWriter(tensorboard_dir)

        aster, aster_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt) #self.args.tpg default=CRNN ，it is same as CRNN_init()

        test_bible = {}

        if self.args.test_model == "CRNN":
            crnn, aster_info = self.TPG_init(recognizer_path=None, opt=tpg_opt) if self.args.CHNSR else self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }

        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "SEED":
            aster_real, aster_real_info = self.SEED_init()
            aster_info = aster_real_info
            test_bible["SEED"] = {
                'model': aster_real,
                'data_in_fn': self.parse_SEED_data,
                'string_process': get_string_aster
            } 

        # print("self.args.arch:", self.args.arch)

        if self.args.arch =='rtsrn':
            aster_student = []
            stu_iter = self.args.stu_iter

            for i in range(stu_iter):
                recognizer_path = os.path.join(self.resume, "recognizer_best_acc_" + str(i) + ".pth")
                print("recognizer_path:", recognizer_path)
                if os.path.isfile(recognizer_path):
                    aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=recognizer_path, opt=tpg_opt) #
                else:
                    aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

                if type(aster_student_) == list:
                    aster_student_ = aster_student_[i]

                aster_student_.train()
                aster_student.append(aster_student_)

        aster.eval()
        # Recognizer needs to be fixed:
        # aster
        if self.args.arch == 'rtsrn':
            optimizer_G = self.optimizer_init(model_list, recognizer=aster_student)
        elif self.args.arch not in ['bicubic']:
            optimizer_G = self.optimizer_init(model_list)
        warmup_epoch = 5
        scheduler = CosineAnnealingLR(optimizer_G, 100 - warmup_epoch)

        iter_per_epoch = len(train_dataset)
        print('iter_per_epoch:',iter_per_epoch)
        warmup_scheduler = WarmUpLR(optimizer_G, iter_per_epoch * warmup_epoch)   
        # for p in aster.parameters():
        #     p.requires_grad = False

        #print("cfg:", cfg.ckpt_dir)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_result={}
        best_acc = 0
        converge_list = []
        lr = cfg.lr
        if self.args.arch not in ['bicubic']:
            for model in model_list:
                model.train()

        for epoch in range(cfg.epochs):
            if epoch >= warmup_epoch:
                scheduler.step()
                learn_rate = scheduler.get_lr()[0]
                print("Learn_rate:%s" % learn_rate)
            for j, data in (enumerate(train_loader)):
                if epoch < 5:
                    warmup_scheduler.step()
                    warm_lr = warmup_scheduler.get_lr()
                    # print("warm_lr:%s" % warm_lr)
                iters = len(train_loader) * epoch + j + 1
                if not self.args.go_test:
                    for model in model_list:
                        for p in model.parameters():
                            p.requires_grad = True

                    images_hr, images_lr, label_strs, label_vecs, weighted_mask, weighted_tics = data
                    text_label = label_vecs

              
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)
                    loss_ssim = 0.



                    if self.args.arch == 'rtsrn':
                    
                        aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                        label_vecs_logits_hr = aster(aster_dict_hr).detach()
                        label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1)

                        cascade_images = images_lr

                        loss_img = 0.
                        loss_recog_distill = 0.

                        for i in range(self.args.stu_iter):
                            recognizer_path = os.path.join(self.resume, "recognizer_best_acc_" + str(i) + ".pth") #
                            if self.args.tpg_share:
                                tpg_pick = 0
                            else:
                                tpg_pick = i
                            stu_model = aster_student[tpg_pick]

                            # Detach from last iteration
                            # cascade_images = cascade_images.detach()

                            aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :])
                            stu_model.eval()
                            with torch.no_grad():
                                lm_input = stu_model(aster_dict_lr).transpose(0,1)
                                lm_input = torch.nn.functional.softmax(lm_input, -1)
                            stu_model.train()
                            label_vecs_logits = stu_model(aster_dict_lr)
                            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)

                            label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                            '''
                            #####################################################
                            # Sample shift
                            # [N, C, H, W]

                            N, C, H, W = label_vecs_final.shape
                            half_N = int(N / 4)
                            all_samples = torch.ones(N)
                            picked_samples = torch.zeros(half_N)
                            # First half to be the LR samples
                            all_samples[:half_N] = picked_samples

                            all_samples = Variable(all_samples).to(label_vecs_final.device).reshape(-1, 1, 1, 1)
                            # print("all_samples:", all_samples.shape, label_vecs_final.shape, label_vecs_final_hr.shape)
                            label_fusion = all_samples * label_vecs_final + (1 - all_samples) * label_vecs_final_hr

                            #####################################################
                            '''
                            # image for cascading
                            if self.args.sr_share:
                                pick = 0
                            else:
                                pick = i
                            
                            if self.args.use_label:
                                # [B, L]
                                text_sum = text_label.sum(1).squeeze(1)
                                # print("text_sum:", text_sum.shape)
                                text_pos = (text_sum > 0).float().sum(1)
                                text_len = text_pos.reshape(-1)
                                predicted_length = torch.ones(label_vecs_logits.shape[1]) * label_vecs_logits.shape[0]

                                fsup_sem_loss = ctc_loss(
                                    label_vecs_logits.log_softmax(2),
                                    weighted_mask.long().to(label_vecs_logits.device),
                                    predicted_length.long().to(label_vecs_logits.device),
                                    text_len.long()
                                )

                                # fsup_sem_loss = Variable(weighted_tics.float()).to(fsup_sem_loss.device)
                                loss_recog_distill_each = (fsup_sem_loss * Variable(weighted_tics.float()).to(fsup_sem_loss.device))# .mean()
                                # print('loss_recog_distill_each:', loss_recog_distill_each)
                                loss_recog_distill_each = loss_recog_distill_each.mean()
                                loss_recog_distill += loss_recog_distill_each

                            # [N, C, H, W] -> [N, T, C]
                            # text_label = text_label.squeeze(2).permute(2, 0, 1)
                            if self.args.use_distill:
                                # print("label_vecs_hr:", label_vecs_hr.shape)
                                loss_recog_distill_each = sem_loss(label_vecs, label_vecs_hr) * 100
                                loss_recog_distill += loss_recog_distill_each  # * (1 + 0.5 * i)
                            
                            # prior dropout
                            device = label_vecs_final.device
                            # drop_vec = (torch.rand(images_lr.shape[0]) > 0.33).float().to(device)
                            drop_vec = torch.ones(images_lr.shape[0]).float()
                            drop_vec[:int(images_lr.shape[0] // 4)] = 0.
                            drop_vec = drop_vec.to(device)
                            # TODO ADD another clues
                            if self.args.arch in ['rtsrn']:
                                # generate linguistical clue
                                # lm_input = label_vecs_final.clone().squeeze(2).transpose(1,2)
                                # lm_input = label_vecs_final.squeeze(2).transpose(1,2)
                                # B
                                lengths_input = torch.zeros(lm_input.size(0)).fill_(lm_input.size(1)).to(images_hr.device).long()
                                # print("?",lm_input.shape,'vs',lengths_input.shape)
                                if self.config.TRAIN.ngpu>1:

                                    lm_out = model_list[pick].module.lm(lm_input, lengths_input)['logits'] #for multi gpu 
                                else:
                                    lm_out = model_list[pick].lm(lm_input, lengths_input)['logits']

                                # B L 1 26
                                # generate visual clue
                                readed_txts = self.converter_crnn.decode(torch.argmax(lm_input.log_softmax(2),2).view(-1),torch.IntTensor([lm_input.size(1)]*lm_input.size(0)))
                                lm_txts = self.converter_crnn.decode(torch.argmax(lm_out.log_softmax(2),2).view(-1),torch.IntTensor([lm_input.size(1)]*lm_input.size(0)))
                                visual_clues = None

                                lm_labels, lm_label_lengths = [], []
                                for label_str in label_strs:
                                    label_str = str_filt(label_str,'lower')
                                    encoded_label, length_tensor = self.converter_crnn.encode(label_str)
                                    lm_labels.append(encoded_label)
                                    lm_label_lengths.append(length_tensor)
                                lm_labels = torch.cat(lm_labels,0)
                                lm_label_lengths = torch.cat(lm_label_lengths)
                                
                                # loss_ctc_lm = lossfn_ctc_lm(log_probs=lm_out.transpose(0,1), targets=lm_labels, input_lengths=lengths_input, target_lengths=lm_label_lengths)
                                # print(lm_out.shape,label_vecs_hr.shape)
                                # loss_lm_distill = sem_loss(lm_out.transpose(0,1), label_vecs_hr) * 10
                                # if True:
                                    # [B, L]
                                text_sum = text_label.sum(1).squeeze(1)
                                # print("text_sum:", text_sum.shape)
                                text_pos = (text_sum > 0).float().sum(1)
                                text_len = text_pos.reshape(-1)
                                predicted_length = torch.ones(label_vecs_logits.shape[1]) * label_vecs_logits.shape[0]
                                # print(lm_out.shape,weighted_mask.shape,predicted_length.shape,text_len.shape,sum(text_len))
                                # TODO
                                fsup_sem_loss = ctc_loss(
                                    lm_out.log_softmax(2).transpose(0,1),
                                    weighted_mask.long().to(label_vecs_logits.device),
                                    predicted_length.long().to(label_vecs_logits.device),
                                    text_len.long()
                                )

                                # fsup_sem_loss = Variable(weighted_tics.float()).to(fsup_sem_loss.device)
                                loss_recog_distill_each = (fsup_sem_loss * Variable(weighted_tics.float()).to(fsup_sem_loss.device))# .mean()
                                # print('loss_recog_distill_each:', loss_recog_distill_each)
                                loss_recog_distill_each = loss_recog_distill_each.mean()
                                loss_recog_distill += loss_recog_distill_each
                                lm_out = lm_out.softmax(-1).unsqueeze(2).transpose(1,3) * drop_vec.view(-1, 1, 1, 1)
                            # Add noise
                            label_vecs_final = label_vecs_final * drop_vec.view(-1, 1, 1, 1)
                            cascade_images = model_list[pick](images_lr, label_vecs_final, lm_out, visual_clues)

                            # image_crit.to(cascade_images.device)
                            loss_img_each = image_crit(cascade_images, images_hr, label = label_strs)[0]* 100
                            loss_img += loss_img_each
                            
                            if self.args.ssim_loss:
                                loss_ssim = (1 - ssim(cascade_images, images_hr).mean()) * 10.
                                loss_img += loss_ssim

                            if iters % 5 == 0 and i == self.args.stu_iter - 1:

                                self.results_recorder.add_scalar('loss/distill', float(loss_recog_distill_each.data) * 100,
                                                                 global_step=iters)

                                self.results_recorder.add_scalar('loss/SR', float(loss_img_each.data) * 100,
                                                                 global_step=iters)

                                self.results_recorder.add_scalar('loss/SSIM', float(loss_ssim) * 100,
                                                                 global_step=iters)

                        loss_im = loss_img + loss_recog_distill

                    
                    optimizer_G.zero_grad()
                    loss_im.backward()

                    for model in model_list:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer_G.step()
                    if iters % 5 == 0:
                        self.results_recorder.add_scalar('loss/total', float(loss_im.data) * 100,
                                                    global_step=iters)


                    # torch.cuda.empty_cache()
                    if iters % cfg.displayInterval == 0:
                        print('[{}]\t'
                              'Epoch: [{}][{}/{}]\t'
                              'vis_dir={:s}\t'
                              'loss_total: {:.3f} \t'
                              'loss_im: {:.3f} \t'
                              'loss_ssim: {:.3f} \t'
                              'loss_teaching: {:.3f} \t'
                              '{:.3f} \t'
                              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      epoch, j + 1, len(train_loader),
                                      self.vis_dir,
                                      float(loss_im.data),
                                      float(loss_img.data),
                                      float(loss_ssim),
                                      float(loss_recog_distill.data),
                                      lr),flush=True)

                if iters % cfg.VAL.valInterval == 0 or self.args.go_test:
                    print('======================================================')
                    current_acc_dict = {}
                    current_fps_dict = {}
                    current_psnr_dict = {}
                    current_ssim_dict = {}

                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        if self.args.arch not in ['bicubic']:
                            for model in model_list:
                                model.eval()
                                for p in model.parameters():
                                    p.requires_grad = False

                       
                        if self.args.arch == 'rtsrn':
                            for stu in aster_student:
                                stu.eval()
                                for p in stu.parameters():
                                    p.requires_grad = False

                        else:
                            aster_student = aster

                        # Tuned TPG for recognition:
                        # test_bible[self.args.test_model]['model'] = aster#aster_student[-1]

                        metrics_dict = self.eval(
                            model_list,
                            val_loader,
                            image_crit,
                            iters,
                            [test_bible[self.args.test_model], aster_student, aster], #
                            aster_info
                        )
                        # print('metrics_dict:',metrics_dict)
                        for key in metrics_dict:
                            # print(metrics_dict)
                            if key in ["cnt_psnr_avg", "cnt_ssim_avg", "psnr_avg", "ssim_avg", "accuracy"]:
                                self.results_recorder.add_scalar('eval/' + key + "_" + data_name, float(metrics_dict[key]),
                                                    global_step=iters)

                        if self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:
                            for p in aster_student.parameters():
                                p.requires_grad = True
                            aster_student.train()
                        elif self.args.arch in ABLATION_SET:
                            for stu in aster_student:
                                for p in stu.parameters():
                                    p.requires_grad = True
                                stu.train()
                        if self.args.arch not in ['bicubic']:
                            for model in model_list:
                                for p in model.parameters():
                                    p.requires_grad = True
                                model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        current_fps_dict[data_name] = float(metrics_dict['fps'])
                        current_psnr_dict[data_name] = float(metrics_dict['psnr_avg'])
                        current_ssim_dict[data_name] = float(metrics_dict['ssim_avg'])


                        if acc > best_history_acc[data_name]:  
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))

                        # if self.args.go_test:
                        #     break
                    print('fps {:,.4f} | acc_avg {:.4f} | psnr_avg {:.4f} | ssim_avg {:.4f}\t'
                        .format(sum(current_fps_dict.values())/3,sum(current_acc_dict.values())/3,
                                sum(current_psnr_dict.values())/3, sum(current_ssim_dict.values())/3, ))

                    if self.args.go_test:
                        break
                    # print('current_acc_dict.values:',current_acc_dict.values())
                    # print('current_acc_dict_sun:',sum(current_acc_dict.values()))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        # best_model_psnr = current_psnr_dict
                        # best_model_ssim = current_ssim_dict
                        best_model_info = {'accuracy': best_model_acc, 'psnr': current_psnr_dict, 'ssim': current_ssim_dict}
                        best_result = {'accuracy_avg': best_acc/3,'acc_list': current_acc_dict ,'psnr_avg': sum(current_psnr_dict.values())/3, 'ssim_avg': sum(current_ssim_dict.values())/3,'epoch':epoch}
                        print('saving best model')
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer=aster_student)
                    print(best_result)
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': current_psnr_dict, 'ssim': current_ssim_dict}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=aster_student)
            if epoch in [600,800]:
                print("Reduce LR")
                for p in optimizer_G.param_groups:
                    p['lr'] *= 0.1
            if self.args.go_test:
                break
    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info):
        n_correct = 0
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {'psnr_lr': [], 'ssim_lr': [], 'cnt_psnr_lr': [], 'cnt_ssim_lr': [], 'psnr': [], 'ssim': [], 'cnt_psnr': [], 'cnt_ssim': [],'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,'fps':0.0}
        counters = {i: 0 for i in range(self.args.stu_iter)}
        wrong_cnt = 0

        go_SR = 0
        go_LR = 0

        LRR_SRW = 0
        LRW_SRR = 0
        LRW_SRW = 0
        LRR_SRR = 0
        #SR_stat = []
        sr_infer_time=0.0

        for i, data in (enumerate(val_loader)):
            SR_stat = []
            SR_tick = False
            time_begin = time.time()


            images_hr, images_lr, label_strs, label_vecs_gt = data

            
            if self.args.random_reso:
                val_batch_size = len(images_lr)
                images_lr = [image_lr.to(self.device) for image_lr in images_lr]
                images_hr = [image_hr.to(self.device) for image_hr in images_hr]
            else:
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

            ##############################################################
            # Evaluating SR confidence
            SR_confidence = []
            iter_i = i
            if self.args.random_reso:
                print("iter:", i * self.args.batch_size)
                SR_confidence = self.cal_conf(images_lr, aster[-1])
            ##############################################################

            elif self.args.arch in ABLATION_SET:

                cascade_images = images_lr

                images_sr = []

                if self.args.random_reso:

                    for i in range(self.args.stu_iter):
                        new_cascade_images = []
                        # if i > 0:
                        #     SR_confidence = self.cal_conf(cascade_images, aster[1][i])

                        for j in range(len(cascade_images)):
                            # print("cascade_images:", len(cascade_images), j)
                            cascade_image = cascade_images[j]
                            if len(cascade_image.shape) < 4:
                                cascade_image = cascade_image.unsqueeze(0)

                            if len(images_lr[j].shape) < 4:
                                image_lr = images_lr[j].unsqueeze(0)
                            else:
                                image_lr = images_lr[j]

                            if SR_confidence[j] > 0.85 and image_lr.shape[-2] > 16:  # SR_confidence > 0.9 or
                                if i == self.args.stu_iter - 1:
                                    go_LR += 1
                                    SR_stat.append("LR")
                                new_cascade_images.append(image_lr)
                            else:
                                if i == self.args.stu_iter - 1:
                                    go_SR += 1
                                    SR_stat.append("SR")
                                if self.args.tpg_share:
                                    tpg_pick = 0
                                else:
                                    tpg_pick = i

                                stu_model = aster[1][tpg_pick]
                                aster_dict_lr = self.parse_crnn_data(cascade_image[:, :3, :, :])
                                label_vecs_logits = stu_model(aster_dict_lr)

                                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                                if self.args.sr_share:
                                    pick = 0
                                else:
                                    pick = i

                                cascade_image = model_list[pick](image_lr, label_vecs_final)
                                new_cascade_images.append(cascade_image)
                        cascade_images = new_cascade_images
                        images_sr.append(cascade_images)
                else:
                    # Get char mask
                    with torch.no_grad():
                        aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                        label_vecs_hr = aster[1][0](aster_dict_hr)
                        label_vecs_hr_pos = torch.nn.functional.softmax(label_vecs_hr, -1)[..., 1:]
                    
                        char_prob = label_vecs_hr_pos.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                        prob_val, prob_ind = char_prob.max(1)
                        # prob_val = (prob_val >= 0.1).float()
                        prob_val = torch.nn.functional.interpolate(prob_val.unsqueeze(1), images_hr.shape[-2:], mode='bilinear') 
                        # prob_val = (prob_val >= 0.01).float()
                    # print("prob_val:", prob_val.shape)


                    for i in range(self.args.stu_iter):

                        if self.args.tpg_share:
                            tpg_pick = 0
                        else:
                            tpg_pick = i

                        stu_model = aster[1][tpg_pick]
                        aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :]) #
                        label_vecs_logits = stu_model(aster_dict_lr)#.transpose(0,1)

                        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                        label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                        # generating another two clues
                        if self.args.arch in ['rtsrn']:
                            # generate linguistical clue
                            lm_input = label_vecs_final.clone().squeeze(2).detach().transpose(1,2)
                            # B
                            lengths_input = torch.zeros(lm_input.size(0)).fill_(lm_input.size(1)).to(images_hr.device).long()
                            # print("?",lm_input.shape,'vs',lengths_input.shape)
                            if self.config.TRAIN.ngpu>1:
                                lm_out = model_list[tpg_pick].module.lm(lm_input, lengths_input)['logits'].softmax(-1)
                            else:
                                lm_out = model_list[tpg_pick].lm(lm_input, lengths_input)['logits'].softmax(-1)
                            # B L 1 26
                            # generate visual clue
                            # readed_txts = self.converter_crnn.decode(torch.argmax(lm_input,2).view(-1),torch.IntTensor([lm_input.size(1)]*lm_input.size(0)))
                            # visual_clues = []
                            # for readed_txt in readed_txts:
                            #     readed_txt_lower = readed_txt.lower()
                            #     readed_txt_upper= readed_txt.upper()
                            #     readed_txt_lower = render_text_img(readed_txt_lower)
                            #     readed_txt_upper = render_text_img(readed_txt_upper)
                            #     # 6 H W
                            #     visual_clues.append(torch.cat((readed_txt_lower,readed_txt_upper),0))
                            # visual_clues = torch.stack(visual_clues, 0).to(images_hr.device)
                            visual_clues = None
                            lm_out = lm_out.unsqueeze(2).transpose(1,3)
                        if self.args.sr_share:
                            pick = 0
                        else:
                            pick = i

                        if self.args.arch in ['rtsrn']:
                                # print(label_vecs_final.shape,'vs',lm_out.shape,visual_clues.shape)
                            cascade_images = model_list[tpg_pick](images_lr, label_vecs_final, lm_out, visual_clues)
                        else:
                            cascade_images = model_list[tpg_pick](images_lr, label_vecs_final)
                        images_sr.append(cascade_images)
            
            time_end = time.time()
            tmp = time_end-time_begin
            sr_infer_time += tmp
            # aster_dict_lr = aster[0]["data_in_fn"](images_lr) # [:, :3, ...]
            # aster_dict_hr = aster[0]["data_in_fn"](images_hr) # [:, :3, ...]

            # if not self.args.random_reso:
            if self.args.test_model== 'CRNN':
                aster_dict_lr = aster[0]["data_in_fn"](images_lr)#[:, :3, ...]
                aster_dict_hr = aster[0]["data_in_fn"](images_hr)#[:, :3, ...]
            else:
                aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, ...])#[:, :3, ...]
                aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, ...])#[:, :3, ...]

            if self.args.test_model == "MORAN":
                # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)

            # print("输出",aster_output_lr.shape)
            # outputs_lr = aster_output_lr.permute(1, 0, 2).contiguous()
            # outputs_hr = aster_output_hr.permute(1, 0, 2).contiguous()
            # outputs = aster_output_sr
            #             print("取最大前", outputs)
            # outputs = torch.max(outputs,1)[1]
            # print("取最大后",outputs)
            '-----------------------get pred of sr------------------------------------'

            if type(images_sr) == list:
                predict_result_sr = []

                if self.args.arch in ABLATION_SET:

                    for i in range(self.args.stu_iter):
                        image = images_sr[i]
                        if self.args.test_model == "CRNN":
                            aster_dict_sr = aster[0]["data_in_fn"](image)
                        else:
                            aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
                        # aster_dict_sr = aster[0]["data_in_fn"](image)
                        if self.args.test_model == "MORAN":
                            # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                            aster_output_sr = aster[0]["model"](
                                aster_dict_sr[0],
                                aster_dict_sr[1],
                                aster_dict_sr[2],
                                aster_dict_sr[3],
                                test=True,
                                debug=True
                            )
                        else:
                            aster_output_sr = aster[0]["model"](aster_dict_sr)
                        # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                        if self.args.test_model == "CRNN":
                            predict_result_sr_ = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                        elif self.args.test_model == "ASTER":
                            predict_result_sr_, _ = aster[0]["string_process"](
                                aster_output_sr['output']['pred_rec'],
                                aster_dict_sr['rec_targets'],
                                dataset=aster_info
                            )
                        elif self.args.test_model == "MORAN":
                            preds, preds_reverse = aster_output_sr[0]
                            _, preds = preds.max(1)
                            sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                            if type(sim_preds) == list:
                                predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]
                            else:
                                predict_result_sr_ = [sim_preds.split('$')[0]] # [pred.split('$')[0] for pred in sim_preds]

                        elif self.args.test_model == "SEED":
                            predict_result_sr_ = []
                            output_logits = aster_output_sr["output"]["pred_rec"]

                            # print("predict_result_sr_:", predict_result_sr_.shape, predict_result_sr_)

                            alphabet = string.printable[:-6]
                            for i in range(output_logits.shape[0]):
                                out_str = ''
                                for j in range(output_logits.shape[1]):
                                    if output_logits[i][j] < 94:
                                        out_str += alphabet[output_logits[i][j]]
                                    else:
                                        break
                                # print("out_str:", out_str)
                                out_str = str_filt(out_str,'lower')
                                predict_result_sr_.append(out_str)
                        predict_result_sr.append(predict_result_sr_)

                        #if self.args.random_reso:
                        #    img_sr = self.parse_crnn_data(images_sr[-1])
                        #    img_hr = self.parse_crnn_data(images_hr)
                        #else:
                        #    img_sr = images_sr[-1]
                        #    img_hr = images_hr
                        if self.args.random_reso:
                            img_sr = self.parse_crnn_data(images_sr[-1])
                            img_hr = self.parse_crnn_data(images_hr)
                            img_lr = self.parse_crnn_data(images_lr)
                        else:
                            img_sr = images_sr[-1]
                            img_hr = images_hr
                            img_lr = images_lr
                else:
                    aster_dict_sr = aster[0]["data_in_fn"](images_sr)
                    # print("aster_dict_sr:", aster_dict_sr.shape)
                    if self.args.test_model == "MORAN":
                        # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                        aster_output_sr = aster[0]["model"](
                            aster_dict_sr[0],
                            aster_dict_sr[1],
                            aster_dict_sr[2],
                            aster_dict_sr[3],
                            test=True,
                            debug=True
                        )
                    else:
                        aster_output_sr = aster[0]["model"](aster_dict_sr)
                    # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                    if self.args.test_model == "CRNN":
                        predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                    elif self.args.test_model in ["ASTER","SEED"]:
                        predict_result_sr, _ = aster[0]["string_process"](
                            aster_output_sr['output']['pred_rec'],
                            aster_dict_sr['rec_targets'],
                            dataset=aster_info
                        )
                    elif self.args.test_model == "MORAN":
                        preds, preds_reverse = aster_output_sr[0]
                        _, preds = preds.max(1)
                        sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                        if type(sim_preds) == list:
                            predict_result_sr = [pred.split('$')[0] for pred in sim_preds]
                        else:
                            predict_result_sr = [sim_preds.split('$')[0]]  # [pred.split('$')[0] for pred in sim_preds]
                    # img_sr = self.parse_crnn_data(images_sr)
                    # img_hr = self.parse_crnn_data(images_hr)
                    elif self.args.test_model == "SEED":
                        predict_result_sr_ = []
                        output_logits = aster_output_sr["output"]["pred_rec"]

                        # print("predict_result_sr_:", predict_result_sr_.shape, predict_result_sr_)

                        alphabet = string.printable[:-6]
                        for i in range(output_logits.shape[0]):
                            out_str = ''
                            for j in range(output_logits.shape[1]):
                                if output_logits[i][j] < 94:
                                    out_str += alphabet[output_logits[i][j]]
                                else:
                                    break
                            print("out_str:", out_str)
                            out_str = str_filt(out_str,'lower')
                            predict_result_sr_.append(out_str)

                    if self.args.random_reso:
                        img_sr = self.parse_crnn_data(images_sr)
                        img_hr = self.parse_crnn_data(images_hr)
                        img_lr = self.parse_crnn_data(images_lr)
                    else:
                        img_sr = images_sr[-1]
                        img_hr = images_hr
                        img_lr = images_lr
                    # print("predict_result_sr_:", predict_result_sr_)
                # print("images_sr[-1]:", images_sr[-1].shape, images_hr.shape)

                #if images_sr[-1].shape != images_hr.shape:
                #    images_sr = [
                #        nn.functional.interpolate(images_sr[k], (images_hr.shape[2], images_hr.shape[3]))
                #                for k in range(self.args.stu_iter)
                #                 ]
                # img_lr = torch.nn.functional.interpolate(img_lr, img_sr.shape[-2:], mode="bicubic")

                prob_val = img_hr[:, :1, ...]

                metric_dict['psnr'].append(self.cal_psnr(img_sr, img_hr))
                metric_dict['ssim'].append(self.cal_ssim(img_sr, img_hr))

                # del prob_val
            else:

                aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                elif self.args.test_model == "ASTER":
                    # print("??",aster_output_sr['output'])
                    predict_result_sr, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "SEED":
                    predict_result_sr = []
                    alphabet = string.printable[:-6]
                    output_logits = aster_output_sr['output']['pred_rec']
                    for i in range(output_logits.shape[0]):
                        out_str = ''
                        for j in range(output_logits.shape[1]):
                            if output_logits[i][j] < 94:
                                out_str += alphabet[output_logits[i][j]]
                            else:
                                break
                        # print("out_str:", out_str)
                        out_str = str_filt(out_str,'lower')
                        predict_result_sr.append(out_str)
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]

                if images_sr.shape != images_hr.shape:
                    images_sr = nn.functional.interpolate(images_sr, (images_hr.shape[2], images_hr.shape[3]))

                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))
                # aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                # aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                # predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)

            # predict_result_lr = aster[0]["string_process"](outputs_lr)
            # predict_result_hr = aster[0]["string_process"](outputs_hr)

            if self.args.test_model == "CRNN":
                predict_result_lr = aster[0]["string_process"](aster_output_lr, self.args.CHNSR)
                predict_result_hr = aster[0]["string_process"](aster_output_hr, self.args.CHNSR)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                # predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                if type(sim_preds) == list:
                    predict_result_lr = [pred.split('$')[0] for pred in sim_preds]
                else:
                    predict_result_lr = [sim_preds.split('$')[0]]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                # predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

                if type(sim_preds) == list:
                    predict_result_hr = [pred.split('$')[0] for pred in sim_preds]
                else:
                    predict_result_hr = [sim_preds.split('$')[0]]

            elif self.args.test_model == "SEED":
                predict_result_lr = []
                output_logits = aster_output_lr["output"]["pred_rec"]

                # print("predict_result_sr_:", predict_result_sr_.shape, predict_result_sr_)

                alphabet = string.printable[:-6]
                for i in range(output_logits.shape[0]):
                    out_str = ''
                    for j in range(output_logits.shape[1]):
                        if output_logits[i][j] < 94:
                            out_str += alphabet[output_logits[i][j]]
                        else:
                            break
                    # print("out_str:", out_str)
                    out_str = str_filt(out_str,'lower')
                    predict_result_lr.append(out_str)

                predict_result_hr = []
                output_logits = aster_output_hr["output"]["pred_rec"]

                # print("predict_result_sr_:", predict_result_sr_.shape, predict_result_sr_)

                alphabet = string.printable[:-6]
                for i in range(output_logits.shape[0]):
                    out_str = ''
                    for j in range(output_logits.shape[1]):
                        if output_logits[i][j] < 94:
                            out_str += alphabet[output_logits[i][j]]
                        else:
                            break
                    # print("out_str:", out_str)
                    out_str = str_filt(out_str,'lower')
                    predict_result_hr.append(out_str)


            cnt = 0



            for batch_i in range(len(images_lr)):

                lr_wrong = True
                sr_wrong = True
                hr_wrong = True

                label = label_strs[batch_i]
                if self.args.arch in ABLATION_SET:
                    for k in range(self.args.stu_iter):
                        if predict_result_sr[k][batch_i] == str_filt(label, 'lower'):
                            counters[k] += 1

                    # print("predict_result_sr:", predict_result_sr)

                    if predict_result_sr[-1][batch_i] == str_filt(label, 'lower'):
                        sr_wrong = False

                else:
                    if predict_result_sr[batch_i] == str_filt(label, 'lower'):
                        sr_wrong = False
                        n_correct += 1
                    else:
                        iswrong = True
                if predict_result_lr[batch_i] == str_filt(label, 'lower'):
                    lr_wrong = False
                    n_correct_lr += 1
                else:
                    iswrong = True
                if predict_result_hr[batch_i] == str_filt(label, 'lower'):
                    hr_wrong = False
                    n_correct_hr += 1
                else:
                    iswrong = True

                if self.args.random_reso:
                    if SR_stat[i] == "SR":
                        if lr_wrong and not sr_wrong:
                            LRW_SRR += 1
                        if not lr_wrong and sr_wrong:
                            LRR_SRW += 1
                        if lr_wrong and sr_wrong:
                            LRW_SRW += 1
                        if not lr_wrong and not sr_wrong:
                            LRR_SRR += 1
                # if hr_wrong and sr_wrong and len(str_filt(label,'lower'))>=5:
                #     print(self.global_img_val_cnt+1)
                if self.vis:
                
                    # if (lr_wrong and not sr_wrong) or (hr_wrong and not sr_wrong):
                    #     #print("identity:", identity)
                    #     #i_f.write(identity[i] + "," + predict_result_lr[i] + "," + predict_result_sr[-1][i] + "," + predict_result_hr[i] + "\n")
                    #     print(predict_result_lr[i] + "," + predict_result_sr[-1][i] + "," + predict_result_hr[i] )



                    sr, lr, hr = images_sr[-1][batch_i] if type(images_sr) is list else images_sr[batch_i], images_lr[batch_i], images_hr[batch_i]


                    if len(sr.shape) > 3:
                        sr = sr.squeeze(0)
                        lr = lr.squeeze(0)
                        hr = hr.squeeze(0)

                    sr = sr.data.cpu().numpy() * 255
                    # sr = (sr >= 0) * sr
                    sr[sr < 0] = 0.
                    sr[sr > 255] = 255.
                    # print("sr:", np.unique(sr))
                    sr = np.transpose(sr[:3], (1, 2, 0)).astype(np.uint8)
                    lr = np.transpose(lr.data.cpu().numpy()[:3] * 255, (1, 2, 0)).astype(np.uint8)
                    hr = np.transpose(hr.data.cpu().numpy()[:3] * 255, (1, 2, 0)).astype(np.uint8)

                    
                    lr = cv2.resize(lr, (128, 32), interpolation=cv2.INTER_CUBIC)
                    sr = cv2.resize(sr, (128, 32), interpolation=cv2.INTER_CUBIC)
                    hr = cv2.resize(hr, (128, 32), interpolation=cv2.INTER_CUBIC)

                    # prob_val_i = cv2.resize(prob_val_i, (128, 32), interpolation=cv2.INTER_CUBIC)
                    # char_mask = cv2.resize(char_mask, (128, 32), interpolation=cv2.INTER_CUBIC)
                    #pad——image
                    # paddimg_im = np.zeros((sr.shape[0] + lr.shape[0] + hr.shape[0] + hr.shape[0] + hr.shape[0] + 25, sr.shape[1], 3))
                    # paddimg_im[:lr.shape[0], :lr.shape[1], :] = lr
                    # paddimg_im[lr.shape[0] + 5:lr.shape[0] + sr.shape[0] + 5, :sr.shape[1], :] = sr
                    # paddimg_im[lr.shape[0] + sr.shape[0] + 10:lr.shape[0] + sr.shape[0] + 10 + hr.shape[0], :hr.shape[1], :] = hr
                    
                    # paddimg_im[prob_val_i.shape[0] + lr.shape[0] + sr.shape[0] + 15:prob_val_i.shape[0] + lr.shape[0] + sr.shape[0] + 15 + hr.shape[0],
                    # :prob_val_i.shape[1], :] = cv2.cvtColor(prob_val_i, cv2.COLOR_GRAY2BGR)
                    # paddimg_im[
                    # prob_val_i.shape[0] + char_mask.shape[0] + lr.shape[0] + sr.shape[0] + 20:prob_val_i.shape[0] + char_mask.shape[0] + lr.shape[0] + sr.shape[
                    #     0] + 20 + hr.shape[0], :char_mask.shape[1], :] = cv2.cvtColor(char_mask, cv2.COLOR_GRAY2BGR)
                    file_name = str(self.global_img_val_cnt)+'_'+predict_result_lr[batch_i] + '_' + \
                                (predict_result_sr[-1][batch_i] if type(images_sr) is list else predict_result_sr[batch_i]) + '_' + \
                                predict_result_hr[batch_i] + "_" + label.lower() + '_.jpg'
                    
                    no_padding = np.concatenate((lr,sr,hr), axis=0, out=None)
                    cv2.imwrite(os.path.join(self.vis_dir, file_name), cv2.cvtColor(no_padding.astype(np.uint16), cv2.COLOR_RGB2BGR))

                    # cv2.imwrite(os.path.join(self.vis_dir, file_name), cv2.cvtColor(paddimg_im.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    # save only one
                    file_name = str(self.global_img_val_cnt)+"_"+ label.lower() + '_sr.jpg'
                    cv2.imwrite(os.path.join(self.vis_dir, file_name), cv2.cvtColor(sr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    file_name2 = str(self.global_img_val_cnt)+"_"+ label.lower() + '_lr.jpg'
                    cv2.imwrite(os.path.join(self.vis_dir, file_name2), cv2.cvtColor(lr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    file_name3 = str(self.global_img_val_cnt)+"_"+ label.lower() + '_hr.jpg'
                    cv2.imwrite(os.path.join(self.vis_dir, file_name3), cv2.cvtColor(hr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    wrong_cnt += 1

                    
                self.global_img_val_cnt += 1
            '''
            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
            '''
            #loss_im = image_crit["image_loss"](images_sr, images_hr).mean()
            # loss_rec = aster_output_sr['losses']['loss_rec'].mean()
            sum_images += val_batch_size

            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['ssim']) + 1e-10)
        fps = sum_images/sr_infer_time

        print('[{}]\t'

              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))
        metric_dict['fps'] = fps

        # print('save display images')
        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)

        if self.args.arch in ABLATION_SET:
            acc = {i: 0 for i in range(self.args.stu_iter)}
            for i in range(self.args.stu_iter):
                acc[i] = round(counters[i] / sum_images, 4)
        else:
            accuracy = round(n_correct / sum_images, 4)
        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        
        if self.args.arch in ABLATION_SET:
            for i in range(self.args.stu_iter):
                print('sr_accuray_iter' + str(i) + ': %.2f%%' % (acc[i] * 100))
            accuracy = acc[self.args.stu_iter-1]

        else:
            print('sr_accuray: %.2f%%' % (accuracy * 100))
        print('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        print('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        if self.args.random_reso:
            print('LR rate: %.2f%%' % (go_LR / sum_images * 100))
            print('SR rate: %.2f%%' % (go_SR / sum_images * 100))
            print('LRW_SRR rate: %.2f%%' % (LRW_SRR / sum_images * 100))
            print('LRR_SRW rate: %.2f%%' % (LRR_SRW / sum_images * 100))
            print('LRR_SRR rate: %.2f%%' % (LRR_SRR / sum_images * 100))
            print('LRW_SRW rate: %.2f%%' % (LRW_SRW / sum_images * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        # if self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:
        #     aster[1].train()
        #sr_accuray_iter accuracy psnr_avg ssim_avg

        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_hr)

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr["images"])
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input["images"])
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        val_batch_size = images_lr.shape[0]

        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)

def render_text_img(text, textsize=32,font_dir="utils/fonts/Ubuntu-Bold.ttf"):
    textsize = 32
    ft = ImageFont.truetype(font_dir, textsize)
    sz, offsets = ft.font.getsize(text)
    x, y = sz
    bg = Image.new('RGB', (x+4, y+4), (255, 255, 255))
    drawer = ImageDraw.Draw(bg)
    drawer.text((2,2),text,(0,0,0),font=ft)
    bg = bg.resize((64, 16),Image.ANTIALIAS)
    to_tensor = transforms.ToTensor()
    return to_tensor(bg)
if __name__ == '__main__':
    embed()