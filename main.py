import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR
os.chdir(sys.path[0])
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


def main(config, args, opt_TPG):
    Mission = TextSR(config, args, opt_TPG)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='rtsrn', choices=['tsrn_tl_cascade_sft', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'scgan', 'rdn', 'tbsrn',
                                                           'edsr', 'lapsrn', 'tsrn_tl_wmask', 'tsrn_tl_cascade', 'srcnn_tl', 'srresnet_tl', 'rdn_tl', 'vdsr_tl', 'tranSR_v4',
                                                                    "esrgan_tl", "scgan_tl", 'tbsrn_tl', 'tatt', "han", 'pcan', 'pcan_tl','TSRN_TL','C3-STISR','rtsrn'])
    parser.add_argument('--go_test', action='store_true', default=False)
    parser.add_argument('--y_domain', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../hard_space1/mjq/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--ic15sr', action='store_true', default=False, help='use IC15SR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--stu_iter', type=int, default=1, help='Default is set to 1, must be used with --arch=tsrn_tl_cascade')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN"])
    parser.add_argument('--sr_share', action='store_true', default=False)
    parser.add_argument('--tpg_share', action='store_true', default=False)
    parser.add_argument('--use_label', action='store_true', default=False)
    parser.add_argument('--use_distill', action='store_true', default=False)
    parser.add_argument('--ssim_loss', action='store_true', default=False)
    parser.add_argument('--tpg', type=str, default="CRNN", choices=['CRNN', 'OPT'])
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--CHNSR', action='store_true', default=False)
    parser.add_argument('--text_focus', action='store_true', default=False)
    parser.add_argument('--prob_insert', type=float, default=1., help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--rotate_test', type=float, default=0., help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--badset', action='store_true', default=False)
    parser.add_argument('--training_stablize', action='store_true', default=False)
    parser.add_argument('--test_distorted_fusing', type=int, default=0)
    parser.add_argument('--results_rotate', action='store_true', default=False)
    parser.add_argument('--results_rotate_angle', type=float, default=5., help='')
    parser.add_argument('--learning_STN', action='store_true', default=False)
    parser.add_argument('--tssim_loss', action='store_true', default=False)
    parser.add_argument('--mse_fuse', action='store_true', default=False)
    parser.add_argument('--for_cascading', action='store_true', default=False)
    parser.add_argument('--color_loss', action='store_true', default=False)
    parser.add_argument('--BiSR', action='store_true', default=False)
    parser.add_argument('--triple_clues', action='store_true', default=False)
    parser.add_argument('--random_reso', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--lca', action='store_true', default=False)

    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    config.TRAIN.lr = args.learning_rate


    opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "best_accuracy.pth",#"best_accuracy.pth", #"None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

    if args.CHNSR:
        opt['character'] = open("al_chinese.txt", 'r').readlines()[0].replace("\n", "")
    opt["num_class"] = len(opt['character'])
    if args.vis: #check save path exist ?

        if not os.path.exists(args.vis_dir):
            os.mkdir(args.vis_dir)


    opt = EasyDict(opt)
    print(opt)
    print(args)
    print(config)
    main(config, args, opt_TPG=opt)
