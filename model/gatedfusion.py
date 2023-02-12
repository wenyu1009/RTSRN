import torch
import torch.nn as nn
wn=lambda x: torch.nn.utils.weight_norm(x)
class GatedFusion(nn.Module):
    def __init__(self,n_feat=32):
        super(GatedFusion,self).__init__()
        self.gated1=nn.Sequential(
            wn(nn.Conv2d(2*n_feat, 2*n_feat, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.ReLU(True),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=1)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=2, dilation=2)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=4,dilation=4)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=8,dilation=8)),
            )
        self.gated2=nn.Sequential(
            wn(nn.Conv2d(2*n_feat, 2*n_feat, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.ReLU(True),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=1)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=2, dilation=2)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=4,dilation=4)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=8,dilation=8)),
            )
        self.gated3=nn.Sequential(
            wn(nn.Conv2d(2*n_feat, 2*n_feat, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.ReLU(True),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=1)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=2, dilation=2)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=4,dilation=4)),
            wn(nn.Conv2d(2*n_feat, 2*n_feat, 3, groups=2*n_feat, padding=8,dilation=8)),
            )
        self.compress=nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(4*n_feat, 2*n_feat, kernel_size=1, stride=1, padding=0, bias=False)),
        )

    def forward(self,center,out1,out2,out3=None):
        # print(center.size(),'vs',out1.size(),'vs',out2.size())
        b,c,h,w=center.shape
        inp1=torch.cat([center,out1],1)
        inp2=torch.cat([center,out2],1)
        # inp3=torch.cat([center,out3],1)
        map1=self.gated1(inp1)
        map2=self.gated2(inp2)
        # map3=self.gated3(inp3)
        # out6=self.compress(torch.cat([map1,map2,map3],1)).view(b,3,c,h,w)
        out6=self.compress(torch.cat([map1,map2],1)).view(b,2,c,h,w)
        out6=torch.softmax(out6,1)
        map1=out6[:,0]
        map2=out6[:,1]
        # map3=out6[:,2]
        return out1*map1+out2*map2#+out3*map3