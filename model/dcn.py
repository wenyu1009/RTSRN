import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
def default_conv(in_channels, out_channels, kernel_size,padding=None,stride=1, bias=True):
    if padding is None:
        padding = kernel_size//2
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=padding, stride=stride,bias=bias)
class DSTA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(DSTA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        # DCN is better
        self.dcn = DeformConv2d(f,f,3,padding=1,groups=f)
        self.mask = conv(f,f*3*3*3,3,padding=1)
        # two mask, multilevel fusion
        self.f = f
        self.down_conv2 = nn.Sequential(
                nn.Conv2d(f, f, 3, stride=2, padding=3//2),
                nn.ReLU(inplace=True))
        self.mask2 = conv(f,f*3*3*3,3,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(f, 2*f, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*f, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        f = x.clone()
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = self.relu(c3)
        dc3 = self.down_conv2(c3)
        off_mask2 = self.mask2(dc3)
        off_msk = self.mask(c3)
        off_mask2 = F.interpolate(off_mask2, (off_msk.size(2), off_msk.size(3)), mode='bilinear', align_corners=False)
        off_msk = off_msk + off_mask2
        off = off_msk[:, :self.f*2*3*3, ...]
        msk = torch.sigmoid(
            off_msk[:, self.f*2*3*3:, ...]
            )
        c3 = self.dcn(v_max,off,msk)
        c3 = F.relu(c3,inplace = True)
        y = self.avg_pool(c3)
        y = self.conv_du(y)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m * y

class DeformFuser(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(DeformFuser, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        self.deform_conv = DeformConv2d(in_nc,out_nc,base_ks,padding=base_ks//2,groups=in_nc)

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            # print(out.size(),'pk',out_lst[i].size())
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat