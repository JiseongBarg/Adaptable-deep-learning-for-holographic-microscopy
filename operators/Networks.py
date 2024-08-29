import torch
from torch import nn
from functions.InitializationModel import weights_initialize_xavier_normal as weights_initialize
import torch.nn.functional as F


class Field_Generator_Resblk(nn.Module):
    """
    Complex-valued field generator with Residual block
    """

    def __init__(self, args, input_channel=1, out_channel = 2):
        
        super(Field_Generator_Resblk, self).__init__()

        self.use_norm = args.norm_use
        self.input_channel = input_channel
        self.output_channel = out_channel
        self.lrelu_use = args.lrelu_use

        c1 = args.initial_channel_G
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2
        
        self.l10 = ResBlk(dim_in =self.input_channel, dim_out=c1, normalize=False)
        self.SE1 = SELayer(channel=c1)
        
        self.l20 = ResBlk(dim_in=c1, dim_out=c2, normalize = True)
        self.SE2 = SELayer(channel=c2)

        self.l30 = ResBlk(dim_in=c2, dim_out=c3, normalize = True)
        self.SE3 = SELayer(channel=c3)

        self.l40 = ResBlk(dim_in=c3, dim_out=c4, normalize = True)
        self.SE4 = SELayer(channel=c4)

        self.l50 = ResBlk(dim_in=c4, dim_out=c5, normalize = True)
        self.l51 = ResBlk(dim_in=c5, dim_out=c4, normalize = True)
        self.conv_T5 = nn.ConvTranspose2d(in_channels=c4, out_channels=c4, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l61 = ResBlk(dim_in=c5, dim_out=c4, normalize = True)
        self.l60 = ResBlk(dim_in=c4, dim_out=c3, normalize = True)
        self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l71 = ResBlk(dim_in=c4, dim_out=c3, normalize = True)
        self.l70 = ResBlk(dim_in=c3, dim_out=c2, normalize = True)
        self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l81 = ResBlk(dim_in=c3, dim_out=c2, normalize = True)
        self.l80 = ResBlk(dim_in=c2, dim_out=c1, normalize = True)
        self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l91 = ResBlk(dim_in=c2, dim_out=c1, normalize = True)
        self.l90 = ResBlk(dim_in=c1, dim_out=c1, normalize = True)
        
        self.conv_out_amplitdue = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)
        self.conv_out_phase = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)
        
        self.SE_out_amplitude = SELayer(channel=c1)
        self.SE_out_phase = SELayer(channel=c1)
            
        self.apply(weights_initialize)
        self.mpool0 = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
    
        l1 = self.l10(x)
        l1_pool = self.mpool0(l1)

        l2 = self.l20(l1_pool)
        l2_pool = self.mpool0(l2)

        l3 = self.l30(l2_pool)
        l3_pool = self.mpool0(l3)

        l4 = self.l40(l3_pool)
        l4_pool = self.mpool0(l4)

        l5 = self.conv_T5(self.l51(self.l50(l4_pool)))

        l6 = torch.cat([l5, self.SE4(l4)], dim=1)
        l6 = self.conv_T6(self.l60(self.l61(l6)))

        l7 = torch.cat([l6, self.SE3(l3)], dim=1)
        l7 = self.conv_T7(self.l70(self.l71(l7)))

        l8 = torch.cat([l7, self.SE2(l2)], dim=1)
        l8 = self.conv_T8(self.l80(self.l81(l8)))

        l9 = torch.cat([l8, self.SE1(l1)], dim=1)
        out = self.l90(self.l91(l9))

        out_amplitude = self.conv_out_amplitdue(self.SE_out_amplitude(out))
        out_phase = self.conv_out_phase(self.SE_out_phase(out))

        return out_amplitude, out_phase
        
    
class SELayer(nn.Module):
    '''
    Squeeze-and-excitation network used in Generator.
    '''
    def __init__(self, channel, reduction=16):
        
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

class ResBlk(nn.Module):
    '''
    Residual Block with leakyReLU and instance normalization    
    '''
    
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x # / math.sqrt(2)  # unit variance

