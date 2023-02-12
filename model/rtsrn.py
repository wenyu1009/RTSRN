import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import numpy as np

sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .model_transformer import FeatureEnhancer, ReasoningTransformer, FeatureEnhancerW2V
from .transformer_v2 import Transformer as Transformer_V2
from .transformer_v2 import InfoTransformer
from .transformer_v2 import PositionalEncoding
from . import torch_distortion
from .dcn import DeformFuser, DSTA
from .language_correction import BCNLanguage
from .gatedfusion import GatedFusion
SHUT_BN = False

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or  in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode=mode
        
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    


    def forward(self, x):
     
        B, C, H, W = x.shape
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)

        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        # x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        # x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=-2).reshape(B,2*C,H,W)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=-2).reshape(B,2*C,H,W)

#         x_1=self.fc_h(x)
#         x_2=self.fc_w(x)
        # x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        # x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
        
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x


class MLPbasedRecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(MLPbasedRecurrentResidualBlockTL, self).__init__()

        self.gru1 = GruBlock(channels + text_channels, channels)

        self.gru2 = GruBlock(channels, channels) # + text_channels

        self.wave_mlp = WaveBlock(channels, mlp_ratio=2, qkv_bias=False, qk_scale=None,
                      attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d,mode='fc')


    def forward(self, x, text_emb):

        residual = self.wave_mlp(x)

        residual = torch.cat([residual, text_emb], 1)

        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(x + residual)

class SGAT(nn.Module):
    def __init__(self,in_features,out_features,head=1,stride=1):
        super(SGAT,self).__init__()
        self.in_features=in_features
        self.hid_features=out_features*3
        self.head=head
        self.stride=stride
        self.trans=nn.Conv2d(in_channels=self.in_features,out_channels=self.hid_features,kernel_size=1,stride=1,padding=0,bias=False)
        self.pad=nn.Unfold(kernel_size=3,padding=1,stride=self.stride)
        self.unfo=nn.Unfold(kernel_size=1,padding=0,stride=stride)
        self.drop=nn.Dropout(0.4)
    def forward(self, x):
        # result=[]

        x=self.trans(x)
        q,k,v=torch.chunk(x,3, dim=1)

        b,c,h,w=q.shape

        q=q.reshape(b,self.head,-1,h,w)
        k=k.reshape(b,self.head,-1,h,w)
        v=v.reshape(b,self.head,-1,h,w)

        q=q.view(b*self.head,-1,h,w)
        q=self.unfo(q)
        q=q.view(b,self.head,-1,1,h//self.stride,w//self.stride).permute(0,1,4,5,3,2).contiguous()#b,head,h,w,1,c//head

        k=k.view(b*self.head,-1,h,w)
        k=self.pad(k)
        k=k.view(b,self.head,-1,9,h//self.stride,w//self.stride).permute(0,1,4,5,2,3).contiguous()#b,head,h,w,c//head,9
        
        v=v.view(b*self.head,-1,h,w)
        v=self.pad(v)
        v=v.view(b,self.head,-1,9,h//self.stride,w//self.stride).permute(0,1,4,5,3,2).contiguous()#b,head,h,w,9,c//head

        


        
        att=q@k
        att=att/math.sqrt(c//self.head)
        att=F.softmax(att,dim=-1)
        att=self.drop(att)
        
        result=att@v

        result=result.squeeze(-2).permute(0,1,4,2,3).reshape(b,-1,h//self.stride,w//self.stride)#stride 
        
        return result
        
class PAM(nn.Module):
    def __init__(self, in_features,out_features,head=1,stride=1):
        super(PAM, self).__init__()
        self.gama = nn.Parameter(torch.zeros(1))
        self.output = nn.Sequential( 
        SGAT(in_features,out_features,head), 
        nn.BatchNorm2d(in_features),
		mish()
		)
    def forward(self, x):#resnet b 2048 16 8 
        return (1-self.gama)*x+self.gama*self.output(x)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pam = PAM(in_channels,in_channels,head=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.pam(x)

        return x

class RTSRN(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #37, #26+26+1 3965
                 out_text_channels=32,
                 triple_clues=False): #256 32
        super(RTSRN, self).__init__()
        
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), MLPbasedRecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.feature_enhancer = None
        # From [1, 1] -> [16, 16]

        self.infoGen = InfoGen(text_emb, out_text_channels)

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums+2)]
        self.triple_clues = triple_clues

        if self.triple_clues:
            self.lm = BCNLanguage()
            # self.lm.load_state_dict(torch.load('ckpt/BCN_correct_model.pt'))
            self.dsta_rec = DSTA(hidden_units)
            self.dsta_vis = DSTA(hidden_units)
            self.dsta_ling = DSTA(hidden_units)
            # self.vis_rec_fuser = DeformFuser(16,hidden_units,hidden_units,4)
            # self.gated = gated(hidden_units)
            self.gated = GatedFusion(hidden_units)
            self.down_conv = nn.Conv2d(hidden_units*2,hidden_units,1,padding=0)
            self.infoGen_ling = InfoGen(text_emb, out_text_channels)
            self.infoGen_visual = InfoGen(text_emb, 10)
            self.correction_model = BCNLanguage()
            self.vis_rec_fuser = DeformFuser(16,hidden_units,hidden_units,4)
        # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, hint_ling=None, hint_vis=None):

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device) # 37 or 3965

        spatial_t_emb_gt, pr_weights_gt = None, None
        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # ,block['1'], block['1'],

        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        B,C,H,W = x.shape
        if self.triple_clues:
            if hint_ling is None:
                hint_ling = torch.zeros_like(text_emb)
            if hint_vis is None:
                hint_vis = torch.zeros((B,6,H,W)).to(x.device)
            hint_rec = self.dsta_rec(spatial_t_emb)
            # hint = spatial_t_emb
            hint_ling, _ = self.infoGen_ling(hint_ling)
            hint_ling = F.interpolate(hint_ling, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            hint_ling = self.dsta_ling(hint_ling)
            # hint = hint_ling

            # # The Trident
            # hint_vis_rec, _ = self.infoGen_visual(text_emb)
            # hint_vis_rec = F.interpolate(hint_vis_rec, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
            # hint_vis = self.vis_rec_fuser(torch.cat((hint_vis,hint_vis_rec),1))
            # hint_vis = self.dsta_vis(hint_vis)
            hint = self.gated(self.down_conv(block['1']),hint_rec,hint_ling)

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], hint)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #

        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block
        return output


class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)




class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class ImFeat2WordVec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImFeat2WordVec, self).__init__()
        self.vec_d = out_channels
        self.vec_proj = nn.Linear(in_channels, self.vec_d)

    def forward(self, x):

        b, c, h, w = x.size()
        result = x.view(b, c, h * w)
        result = torch.mean(result, 2)
        pred_vec = self.vec_proj(result)

        return pred_vec


if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
