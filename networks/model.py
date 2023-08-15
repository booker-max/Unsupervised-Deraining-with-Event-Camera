from locale import DAY_3
import sys
sys.path.append("./networks/basic")
sys.path.append("dclgan_util")
from base_model import BaseModel
import networks
from patchnce import PatchNCELoss_Layer as PatchNCELoss
from image_pool import ImagePool
from loss_FA import FALoss_max, FALoss_min, FALoss_max_NA
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.init as init
from numpy import *
import torch.nn.functional as F
import commentjson as json
import itertools

###DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
if True:
    def To3D(E1, group):
        [b, c, h, w] = E1.shape
        nf = int(c/group)

        E_list = []
        for i in range(0, group):
            tmp = E1[:, nf*i:nf*(i+1), :, :]
            tmp = tmp.view(b, nf, 1, h, w)
            E_list.append(tmp)
            
        E1_3d = torch.cat(E_list, 2)
        return E1_3d

    def To2D(E1_3d):
        [b, c, g, h, w] = E1_3d.shape

        E_list = []
        for i in range(0, g):
            tmp = E1_3d[:, :, i, :, :]
            tmp = tmp.view(b, c, h, w)
            E_list.append(tmp)

        E1 = torch.cat(E_list, 1)
        return E1
    
    def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias, stride = stride)

    def logsumexp_2d(tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs

### modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py

    class ConvLSTM(nn.Module):
        
        def __init__(self, input_size, hidden_size, kernel_size):
            super(ConvLSTM, self).__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            pad = kernel_size // 2

            self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

        def forward(self, input_, prev_state=None):

            # get batch and spatial sizes
            batch_size = input_.data.size()[0]
            spatial_size = input_.data.size()[2:]

            # generate empty prev_state, if None is provided
            if prev_state is None:
                state_size = [batch_size, self.hidden_size] + list(spatial_size)
                prev_state = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_hidden, prev_cell = prev_state

            # data size is [batch, channel, height, width]
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
            gates = self.Gates(stacked_inputs)

            # chunk across channel dimension
            in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

            # apply sigmoid non linearity
            in_gate = F.sigmoid(in_gate)
            remember_gate = F.sigmoid(remember_gate)
            out_gate = F.sigmoid(out_gate)

            # apply tanh non linearity
            cell_gate = F.tanh(cell_gate)

            # compute current cell and hidden state
            cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
            hidden = out_gate * F.tanh(cell)

            return hidden, cell

    class ConvLayer(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm=None, bias=True, last_bias=0):
            super(ConvLayer, self).__init__()
            padding = kernel_size // 2
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

            if last_bias!=0:
                init.constant(self.conv2d.weight, 0)
                init.constant(self.conv2d.bias, last_bias)

        def forward(self, x):
            out = self.conv2d(x)

            return out

    class ResidualBlock(nn.Module):
    
        def __init__(self, channels, groups=1, norm=None, bias=True):
            super(ResidualBlock, self).__init__()
            self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
            self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
            self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        def forward(self, x):
            
            input = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)

            out = out + input

            return out

    class Coor_M(nn.Module):
        def __init__(self, n_feat):
            super(Coor_M, self).__init__()
            
            self.conv_k7 = nn.Sequential(
                ConvLayer(n_feat, n_feat, kernel_size=7, stride=1),
                ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

            self.conv_k5 = nn.Sequential(
                ConvLayer(n_feat, n_feat, kernel_size=5, stride=1),
                ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

            self.conv_k3 = nn.Sequential(
                ConvLayer(n_feat, n_feat, kernel_size=3, stride=1),
                ConvLayer(n_feat, n_feat, kernel_size=1, stride=1))

            self.fus = ConvLayer(3*n_feat, n_feat, kernel_size=3, stride=1)
        
        def forward(self, x):
            x1 = self.conv_k7(x)
            x2 = self.conv_k5(x)
            x3 = self.conv_k3(x)

            x_fus = self.fus(torch.cat((x1, x2, x3),1))
            corr_prob = torch.sigmoid(x_fus)
            return corr_prob
    
    class CALayer(nn.Module):
        def __init__(self, channel, reduction=16, bias=False):
            super(CALayer, self).__init__()
            # global average pooling: feature --> point
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                    nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                    nn.Sigmoid()
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv_du(y)
            return x * y

    class CAB(nn.Module):
        def __init__(self, n_feat, kernel_size, reduction, bias, act):
            super(CAB, self).__init__()
            modules_body = []
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            modules_body.append(act)
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

            self.CA = CALayer(n_feat, reduction, bias=bias)
            self.body = nn.Sequential(*modules_body)

        def forward(self, x):
            res = self.body(x)
            res = self.CA(res)
            res += x
            return res
    
    class SkipUpSample(nn.Module):
        def __init__(self, in_channels,s_factor):
            super(SkipUpSample, self).__init__()
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

        def forward(self, x, y):
            x = self.up(x)
            x = x + y
            return x
    
    class DownSample(nn.Module):
        def __init__(self, in_channels,s_factor):
            super(DownSample, self).__init__()
            self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='nearest'),
                                    nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

        def forward(self, x):
            x = self.down(x)
            return x
    
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)
    
    class ChannelPool(nn.Module):
        def forward(self, x):
            return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

    class ChannelGate(nn.Module):
        def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
            super(ChannelGate, self).__init__()
            self.gate_channels = gate_channels
            self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
                )
            self.pool_types = pool_types
        def forward(self, x):
            channel_att_sum = None
            for pool_type in self.pool_types:
                if pool_type=='avg':
                    avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( avg_pool )
                elif pool_type=='max':
                    max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( max_pool )
                elif pool_type=='lp':
                    lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( lp_pool )
                elif pool_type=='lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(x)
                    channel_att_raw = self.mlp( lse_pool )

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

            scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
            return x * scale
    
    class SpatialGate(nn.Module):
        def __init__(self):
            super(SpatialGate, self).__init__()
            kernel_size = 7
            self.compress = ChannelPool()
            self.spatial = ConvLayer(2, 1, kernel_size, stride=1)
        def forward(self, x):
            x_compress = self.compress(x)
            x_out = self.spatial(x_compress)
            scale = F.sigmoid(x_out) # broadcasting
            return x * scale

    class Sym_CBAM(nn.Module):
        def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
            super(Sym_CBAM, self).__init__()
            self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
            self.SpatialGate1 = SpatialGate()
            self.SpatialGate2 = SpatialGate()
            self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        def forward(self, x):
            x1 = self.ChannelGate1(x)
            x2 = self.SpatialGate1(x1)
            x3 = self.SpatialGate2(x2)
            x4 = self.ChannelGate2(x3)
            return x4

    class multi_Sym_CBAM(nn.Module):
        def __init__(self, gate_channels, n_blocks, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
            super(multi_Sym_CBAM, self).__init__()
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            modules_body = []
            for _ in range(n_blocks//2):
                modules_body.append(Sym_CBAM(gate_channels))
                modules_body.append(self.act)
                modules_body.append(Sym_CBAM(gate_channels))
            self.body = nn.Sequential(*modules_body)

        def forward(self, x):
            out = self.body(x)
            return out
    
    class CMFB(nn.Module):
        def __init__(self, n_feat):
            super(CMFB, self).__init__()

            self.conv1 = ConvLayer(n_feat, n_feat, kernel_size=3, stride=1, groups=1, bias=True, norm=None)
            self.res1 =  ResidualBlock(n_feat, groups=1, bias=True, norm=None)
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        def forward(self, rgb, event, neg, state=None):
            if state is not None:
                x = rgb + state
            else:
                x = rgb
            
            # fus = torch.cat((x, event, -neg), 1)
            fus = x + event - neg
            out = self.res1(self.relu(self.conv1(fus)))

            return out

    class Encoder(nn.Module):
        def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
            super(Encoder, self).__init__()
            
            self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

            self.encoder_level1 = nn.Sequential(*self.encoder_level1)
            self.encoder_level2 = nn.Sequential(*self.encoder_level2)
            self.encoder_level3 = nn.Sequential(*self.encoder_level3)

            self.down12  = DownSample(n_feat, scale_unetfeats)
            self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)     

        def forward(self, x):

            enc1 = self.encoder_level1(x)
            x = self.down12(enc1)

            enc2 = self.encoder_level2(x)
            x = self.down23(enc2)

            enc3 = self.encoder_level3(x)
            
            return [enc1, enc2, enc3] #[64,128,128] [96, 64, 64] [128, 32, 32]
    
    class Decoder(nn.Module):

        def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
            super(Decoder, self).__init__()
            
            self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            
            self.decoder_level1 = nn.Sequential(*self.decoder_level1)
            self.decoder_level2 = nn.Sequential(*self.decoder_level2)
            self.decoder_level3 = nn.Sequential(*self.decoder_level3)

            self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

            self.up21  = SkipUpSample(n_feat, scale_unetfeats)
            self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

            self.tail = nn.Sequential(
                    ConvLayer(n_feat, n_feat, kernel_size=3, stride=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ConvLayer(n_feat, 3, kernel_size=1, stride=1)
            )

        def forward(self, fus, state=None):

            fus1, fus2, fus3 = fus
      
            dec3 = self.decoder_level3(fus3)

            x = self.up32(dec3, self.skip_attn2(fus2))
            dec2 = self.decoder_level2(x)

            x = self.up21(dec2, self.skip_attn1(fus1))
            dec1 = self.decoder_level1(x)

            out = self.tail(dec1)

            return out
    
    ## Supervised Attention Module
    class SAM(nn.Module):
        def __init__(self, n_feat, scale_factor, kernel_size, bias):
            super(SAM, self).__init__()
            
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                                    nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1, bias=False),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(n_feat, 3, 1, stride=1, padding=0, bias=False))
            
            self.down = nn.Sequential(nn.Upsample(scale_factor=(1/scale_factor), mode='nearest'),
                                    nn.Conv2d(n_feat, n_feat, 1, stride=1, padding=0, bias=False))

            self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
            self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

        def forward(self, x, x_img):
            x1 = self.conv1(x)
            img = self.up(x) + x_img
            x2 = torch.sigmoid(self.down(self.conv3(img)))
            x1 = x1*x2
            x1 = x1+x
            return x1, img

    class Decoder_pure(nn.Module):

        def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
            super(Decoder_pure, self).__init__()

            self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
            
            self.decoder_level1 = nn.Sequential(*self.decoder_level1)
            self.decoder_level2 = nn.Sequential(*self.decoder_level2)
            self.decoder_level3 = nn.Sequential(*self.decoder_level3)

            self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

            self.up21  = SkipUpSample(n_feat, scale_unetfeats)
            self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

            self.tail = nn.Sequential(
                    ConvLayer(n_feat, n_feat, kernel_size=3, stride=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ConvLayer(n_feat, 3, kernel_size=1, stride=1)
            )

        def forward(self, outs):

            enc1, enc2, enc3 = outs
            dec3 = self.decoder_level3(enc3)

            x = self.up32(dec3, self.skip_attn2(enc2))
            dec2 = self.decoder_level2(x)

            x = self.up21(dec2, self.skip_attn1(enc1))
            dec1 = self.decoder_level1(x)

            out = self.tail(dec1)

            return out

    class Generator(nn.Module):

        def __init__(self, opts):

            super(Generator, self).__init__()

            scale_unetfeats = opts["scale_unetfeats"]
            nf = opts["nf"]

            self.encoder = Encoder(nf, scale_unetfeats)
            self.encoder_e = Encoder(nf, scale_unetfeats)

            self.refine_e = Refine_E(nf, scale_unetfeats)
            self.fus_mm_f = MMFus(nf, scale_unetfeats)
            self.decoder = Decoder(nf, scale_unetfeats)

        def forward(self, x, x_e, layers=[]):

            feat_enc = self.encoder(x)
            feat_enc_e = self.encoder_e(x_e)
            refine_enc_e = self.refine_e(feat_enc_e)

            fus_f = self.fus_mm_f(feat_enc, refine_enc_e)

            out = self.decoder(fus_f)

            return out, feat_enc, refine_enc_e
    
    class Generator_T(nn.Module):

        def __init__(self, opts):

            super(Generator_T, self).__init__()

            scale_unetfeats = opts["scale_unetfeats"]
            nf = opts["nf"]

            self.encoder_rain = Encoder(nf, scale_unetfeats)
            self.encoder_bg = Encoder(nf, scale_unetfeats)
            self.encoder_e = Encoder(nf, scale_unetfeats)

            self.refine_e = Refine_E(nf, scale_unetfeats)

            self.fus_mm_bg = MMFus(nf, scale_unetfeats)
            self.fus_mm_rain = MMFus(nf, scale_unetfeats)

            self.decoder_rain = Decoder(nf, scale_unetfeats)
            self.decoder_bg = Decoder(nf, scale_unetfeats)

        def forward(self, x, x_e, layers=[]):

            rain_feat_f = self.encoder_rain(x)
            bg_feat_f = self.encoder_bg(x)

            feat_e = self.encoder_e(x_e)
            rain_feat_e = self.refine_e(feat_e)

            bg_feat_e = [(feat_e[i] - rain_feat_e[i]) for i in range(len(feat_e))]
            
            fus_rain = self.fus_mm_rain(rain_feat_f, rain_feat_e, bg_feat_e)
            fus_bg = self.fus_mm_bg(bg_feat_f, bg_feat_e, rain_feat_e)

            out_rain = self.decoder_rain(fus_rain)
            out_bg = self.decoder_bg(fus_bg)

            return out_rain, out_bg, rain_feat_f, bg_feat_f, rain_feat_e, bg_feat_e

    class Head_E(nn.Module):

        def __init__(self, args):
            super(Head_E, self).__init__()

            num_bins = args["num_bins"]
            nf = args["nf"]
            use_bias = True
            args["norm"] = None

            self.conv1_e = ConvLayer(num_bins*2, nf*2, kernel_size=3, stride=1, groups=2, bias=use_bias, norm=args["norm"])
            self.res1_e = ResidualBlock(nf*2, groups=2, bias=use_bias, norm=args["norm"])
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        def forward(self, rainy_event):

            X1_e = self.res1_e(self.relu(self.conv1_e(rainy_event)))

            return X1_e

    class Head_F(nn.Module):

        def __init__(self, args):
            super(Head_F, self).__init__()

            num_bins = args["num_bins"]
            nf = args["nf"]
            use_bias = True
            args["norm"] = None

            self.conv1 = ConvLayer(9, nf*3, kernel_size=3, stride=1, groups=3, bias=use_bias, norm=args["norm"])
            self.res1 = ResidualBlock(nf*3, groups=3, bias=use_bias, norm=args["norm"])

            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        def forward(self, rainy_frame):

            X1 = self.res1(self.relu(self.conv1(rainy_frame)))

            return X1
    
    class Refine_E(nn.Module):

        def __init__(self, n_feat, scale_unetfeats):
            super(Refine_E, self).__init__()

            self.CSA1 = multi_Sym_CBAM(n_feat, n_blocks = 4)
            self.CSA2 = multi_Sym_CBAM((n_feat+(scale_unetfeats*1)), n_blocks = 4)
            self.CSA3 = multi_Sym_CBAM((n_feat+(scale_unetfeats*2)), n_blocks = 4)

        def forward(self, enc_e):

            enc1_e, enc2_e, enc3_e = enc_e

            ref_enc1_e = self.CSA1(enc1_e)
            ref_enc2_e = self.CSA2(enc2_e)
            ref_enc3_e = self.CSA3(enc3_e)
            
            return [ref_enc1_e, ref_enc2_e, ref_enc3_e]

    class MMFus(nn.Module):

        def __init__(self, n_feat, scale_unetfeats):
            super(MMFus, self).__init__()

            self.CMF1 = CMFB(n_feat)
            self.CMF2 = CMFB(n_feat+(scale_unetfeats*1))
            self.CMF3 = CMFB(n_feat+(scale_unetfeats*2))

        def forward(self, enc, ref_enc_e, neg_f):

            enc1, enc2, enc3 = enc
            ref_enc1_e, ref_enc2_e, ref_enc3_e = ref_enc_e
            neg_e1, neg_e2, neg_e3 = neg_f

            fus1 = self.CMF1(enc1, ref_enc1_e, neg_e1)
            fus2 = self.CMF2(enc2, ref_enc2_e, neg_e2)
            fus3 = self.CMF3(enc3, ref_enc3_e, neg_e3)

            return [fus1, fus2, fus3]
    
    class Tem_Mof(nn.Module):

        def __init__(self, args):

            super(Tem_Mof, self).__init__()

            nf = args["nf"]
            use_bias = True
            args["norm"] = None

            self.corr = Coor_M(nf*2)
            self.corr_conv = ConvLayer(nf*2, nf*2, kernel_size=1, stride=1)

            self.conv2 = nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
            self.res2 = ResidualBlock(nf, groups=1, bias=use_bias, norm=args["norm"])
            
            self.conv2_e = nn.Conv3d(nf, nf, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
            self.res2_e = ResidualBlock(nf, groups=1, bias=use_bias, norm=args["norm"])

            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        def forward(self, X1, X1_e):

            X1_neigh = torch.cat((X1[:,:32,:,:], X1[:,64:96,:,:]),1) #now nf =32
            X1_tar = X1[:,32:64,:,:]

            coor_E = self.corr(X1_e)
            ref_X_neigh = self.corr_conv(coor_E * X1_neigh)
            ref_X1 = torch.cat((ref_X_neigh[:,:32,:,:], X1_tar, ref_X_neigh[:,32:64,:,:]), 1)

            X1_3d = To3D(ref_X1, 3)
            X2_3d = self.conv2(X1_3d)
            X2 = To2D(X2_3d)
            X2 = self.res2(self.relu(X2)) #[B,32, 128, 128]

            X1_e_3d = To3D(X1_e, 2)
            X2_e_3d = self.conv2_e(X1_e_3d)
            X2_e = To2D(X2_e_3d) 
            X2_e = self.res2_e(self.relu(X2_e)) #[B,32, 128, 128]

            return X2, X2_e

    class Pre_OP(nn.Module):

        def __init__(self, args):

            super(Pre_OP, self).__init__()

            num_bins = args["num_bins"]
            nf = args["nf"]
            use_bias = True
            args["norm"] = None

            self.conv1 = ConvLayer(9, nf*3, kernel_size=3, stride=1, groups=3, bias=use_bias, norm=args["norm"])
            self.res1 = ResidualBlock(nf*3, groups=3, bias=use_bias, norm=args["norm"])

            self.conv1_e = ConvLayer(num_bins*2, nf*2, kernel_size=3, stride=1, groups=2, bias=use_bias, norm=args["norm"])
            self.res1_e = ResidualBlock(nf*2, groups=2, bias=use_bias, norm=args["norm"])
            
            self.corr = Coor_M(nf*2)
            self.corr_conv = ConvLayer(nf*2, nf*2, kernel_size=1, stride=1)

            self.conv2 = nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
            self.res2 = ResidualBlock(nf, groups=1, bias=use_bias, norm=args["norm"])
            
            self.conv2_e = nn.Conv3d(nf, nf, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
            self.res2_e = ResidualBlock(nf, groups=1, bias=use_bias, norm=args["norm"])

            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, rainy_frame, rainy_event):

            X1 = self.res1(self.relu(self.conv1(rainy_frame)))
            X1_neigh = torch.cat((X1[:,:32,:,:], X1[:,64:96,:,:]),1) #now nf =32
            X1_tar = X1[:,32:64,:,:]

            X1_e = self.res1_e(self.relu(self.conv1_e(rainy_event)))

            coor_E = self.corr(X1_e)
            ref_X_neigh = self.corr_conv(coor_E * X1_neigh)
            ref_X1 = torch.cat((ref_X_neigh[:,:32,:,:], X1_tar, ref_X_neigh[:,32:64,:,:]), 1)

            X1_3d = To3D(ref_X1, 3)
            X2_3d = self.conv2(X1_3d)
            X2 = To2D(X2_3d)
            X2 = self.res2(self.relu(X2)) #[B,32, 128, 128]

            X1_e_3d = To3D(X1_e, 2)
            X2_e_3d = self.conv2_e(X1_e_3d)
            X2_e = To2D(X2_e_3d) 
            X2_e = self.res2_e(self.relu(X2_e)) #[B,32, 128, 128]

            return X2, X2_e
    
class RMFD(BaseModel):

    def __init__(self, args, opt, local_rank):

        BaseModel.__init__(self, opt)

        self.args = args
        self.opt = opt
        self.local_rank = local_rank
        
        self.nce_layers = [0,1,2]

        self.head_event = Head_E(self.args).to(local_rank)

        self.head_frame = Head_F(self.args).to(local_rank)

        self.tem_mof = Tem_Mof(self.args).to(local_rank)

        self.G_twostream = Generator_T(self.args).to(local_rank)

        self.netD_Bg = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            [], opt).to(local_rank)

        self.train_model_names = ["head_event", "head_frame", "tem_mof", "G_twostream", "netD_Bg"]
        self.ddp_model_names = ["head_event", "head_frame", "tem_mof", "G_twostream", "netD_Bg"]
        self.eval_model_names = ["head_event", "head_frame", "tem_mof", "G_twostream", "netD_Bg"]
        self.optimizer_names = ["optimizer_G", "optimizer_D"]

        self.load_ddp()

        self.fake_Bg_pool = ImagePool(opt.pool_size)
        
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.head_event.parameters(), self.head_frame.parameters(), self.tem_mof.parameters(),\
                                                            self.G_twostream.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))

        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Bg.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        
        self.criterionL1 = torch.nn.L1Loss().to(local_rank)
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(local_rank)
        self.fa_min = FALoss_min().to(local_rank)
        self.fa_max = FALoss_max().to(local_rank)
    
    def load_ddp(self):

        for name in self.ddp_model_names:

            if isinstance(name, str):
                setattr(self, name, torch.nn.SyncBatchNorm.convert_sync_batchnorm(getattr(self, name)).to(self.local_rank))
                setattr(self, name, DDP(getattr(self, name), device_ids = [self.local_rank], output_device = self.local_rank, broadcast_buffers = False))

    def set_input(self, data):

        self.rainy_frame = data["Rain_frame"]
        self.rainy_event = data["Rain_event"]
        self.clean = data["clean"]
        self.gan_clean = data["gan_clean"]

        b,c,h,w = self.rainy_frame.shape

        self.target_index = (c//3)%2

        self.target_frame = self.rainy_frame[:, (self.target_index)*3:(self.target_index+1)*3, :, :]

    def set_input_test(self, data):

        self.rainy_frame = data["Rain_frame"]
        self.rainy_event = data["Rain_event"]
        self.clean = data["clean"]

        b,c,h,w = self.rainy_frame.shape

        self.target_index = (c//3)%2

        self.target_frame = self.rainy_frame[:, (self.target_index)*3:(self.target_index+1)*3, :, :]

    def forward(self):

        x1, x1_e = self.head_frame(self.rainy_frame), self.head_event(self.rainy_event)

        x2, x2_e = self.tem_mof(x1, x1_e)
        
        self.Pred_rl, self.Pred_bg, self.rain_fea, self.bg_fea, self.rain_fea_e, self.bg_fea_e = self.G_twostream(x2, x2_e) #list

    def backward_D_Bg(self):

        fake_Bg = self.fake_Bg_pool.query(self.Pred_bg)
        self.loss_D_Bg = self.backward_D_basic(self.netD_Bg, self.gan_clean, fake_Bg)
        
    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):

        self.forward()
        self.set_requires_grad([self.netD_Bg], True)
        self.optimizer_D.zero_grad()
        self.backward_D_Bg()
        self.optimizer_D.step()

        self.set_requires_grad([self.netD_Bg], False)
        self.optimizer_G.zero_grad()
        self.loss = self.compute_G_loss()
        self.loss.backward()
        self.optimizer_G.step()

    def compute_G_loss(self):

        self.loss_consist = self.criterionL1((self.Pred_rl + self.Pred_bg), self.target_frame)
        
        pred_fake_Bg = self.netD_Bg(self.Pred_bg)

        self.loss_adv_g = self.criterionGAN(pred_fake_Bg, True).mean()
        
        self.loss_pos = self.fa_max(self.rain_fea[0], self.rain_fea_e[0]) + \
                        self.fa_max(self.rain_fea[1], self.rain_fea_e[1]) + \
                        self.fa_max(self.rain_fea[2], self.rain_fea_e[2])

        self.loss_neg_MM = self.fa_min(self.rain_fea[0], self.bg_fea_e[0]) + \
                        self.fa_min(self.rain_fea[1], self.bg_fea_e[1]) + \
                        self.fa_min(self.rain_fea[2], self.bg_fea_e[2])

        self.loss_neg_F = self.fa_min(self.rain_fea[0], self.bg_fea[0]) + \
                        self.fa_min(self.rain_fea[1], self.bg_fea[1]) + \
                        self.fa_min(self.rain_fea[2], self.bg_fea[2])

        self.loss_neg_E = self.fa_min(self.rain_fea_e[0], self.bg_fea_e[0]) + \
                        self.fa_min(self.rain_fea_e[1], self.bg_fea_e[1]) + \
                        self.fa_min(self.rain_fea_e[2], self.bg_fea_e[2])

        self.loss_neg = self.loss_neg_MM + self.loss_neg_F + self.loss_neg_E

        self.loss_NCE_layer = self.loss_pos + self.loss_neg
        
        self.loss_G = self.loss_adv_g + self.loss_consist + self.loss_NCE_layer

        return self.loss_G

    if True:
        def calculate_neg_loss(self, x1, x2):

            n_layers = len(self.nce_layers)

            feat_q, _ = self.netF1(x1, self.opt.num_patches, None)
            feat_k, _ = self.netF2(x2, self.opt.num_patches, None)

            total_nce_loss = 0.0

            for f_q, f_k, crit, nce_layer in zip(feat_q, feat_k, self.criterionNeg, self.nce_layers):
                loss = crit(f_q, f_k)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers

        def calculate_pos_loss(self, x1):

            n_layers = len(self.nce_layers)

            feat_q, _ = self.netF1(x1, self.opt.num_patches, None)
            feat_k, _ = self.netF2(x1, self.opt.num_patches, None)

            total_nce_loss = 0.0

            for f_q, f_k, crit, nce_layer in zip(feat_q, feat_k, self.criterionPos, self.nce_layers):
                loss = crit(f_q, f_k)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers

        def calculate_MMPos_loss(self, x1, x2):
            
            n_layers = len(self.nce_layers)

            feat_q, _ = self.netF1(x1, self.opt.num_patches, None)
            feat_k, _ = self.netF2(x2, self.opt.num_patches, None)

            total_nce_loss = 0.0

            for f_q, f_k, crit, nce_layer in zip(feat_q, feat_k, self.criterionPos, self.nce_layers):
                loss = crit(f_q, f_k)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers

        def calculate_MMNeg_loss(self, x1, x2):
            
            n_layers = len(self.nce_layers)

            feat_q, _ = self.netF1(x1, self.opt.num_patches, None)
            feat_k, _ = self.netF2(x2, self.opt.num_patches, None)

            total_nce_loss = 0.0

            for f_q, f_k, crit, nce_layer in zip(feat_q, feat_k, self.criterionNeg, self.nce_layers):
                loss = crit(f_q, f_k)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers

        def calculate_NCE_loss_Bg(self, src, tgt):
            n_layers = len(self.nce_layers)

            feat_head_q = self.pre_op(torch.cat((tgt,tgt,tgt),1))
            feat_head_k = self.pre_op(torch.cat((src,src,src),1))

            feat_q = self.G_Bg(feat_head_q, self.nce_layers, encode_only=True)
            feat_k = self.G_Bg(feat_head_k, self.nce_layers, encode_only=True)

            feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
            feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
            total_nce_loss = 0.0
            for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
                loss = crit(f_q, f_k)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers
        
        def calculate_NCE_loss_layer(self, feat_kn, feat_q):
            n_layers = len(self.nce_layers)
            
            feat_kn_pool, _ = self.netF1(feat_kn, self.opt.num_patches, None)
            feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, None)
            feat_kp_pool, _ = self.netF2(feat_q, self.opt.num_patches, None)

            # if self.local_rank == 0:

            #     print(feat_kn[0].shape) # [8, 32, 128, 128]
            #     print(feat_kn_pool[0].shape) #[2048, 256], 这里的2048=8*256, 8是batch_size，256是num_patches
            #     print(feat_kn[1].shape) #[8, 64, 64, 64]
            #     print(feat_kn_pool[1].shape) #[2048, 256], netF中利用几个线性层将特征通道维度扩展到256
            #     print(feat_kn[2].shape) #[8, 96, 32, 32]
            #     print(feat_kn_pool[2].shape) #[2048, 256]

            total_nce_loss = 0.0
            for f_q, f_kp, f_kn, crit, nce_layer in zip(feat_q_pool, feat_kp_pool, feat_kn_pool, self.criterionNCE, self.nce_layers):
                loss = crit(f_q, f_kp, f_kn)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers
        
        def calculate_NCE_loss_layer_1(self, src, tgt):
            n_layers = len(self.nce_layers)

            feat_head_q = self.pre_op(torch.cat((tgt,tgt,tgt),1))
            feat_head_kn = self.pre_op(torch.cat((src,src,src),1))

            feat_q = self.G_Rain(feat_head_q, self.nce_layers, encode_only=True)
            feat_kn = self.G_Bg(feat_head_kn, self.nce_layers, encode_only=True)

            feat_kn_pool, _ = self.netF1(feat_kn, self.opt.num_patches, None)
            feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, None)
            feat_kp_pool, _ = self.netF2(feat_q, self.opt.num_patches, None)

            total_nce_loss = 0.0
            for f_q, f_kp, f_kn, crit, nce_layer in zip(feat_q_pool, feat_kp_pool, feat_kn_pool, self.criterionNCE, self.nce_layers):
                loss = crit(f_q, f_kp, f_kn)
                total_nce_loss += loss.mean()
            return total_nce_loss / n_layers

    def train(self):

        for name in self.train_model_names:

            if isinstance(name, str):

                net = getattr(self, name)

                net.train()

    def eval(self):

        for name in self.eval_model_names:

            if isinstance(name, str):

                net = getattr(self, name)

                net.eval()

    def get_losses(self):

        loss_consist = self.loss_consist.item()
        loss_adv_gen = self.loss_adv_g.item()
        loss_adv_dis = self.loss_D_Bg.item()
        loss_contrast_layer = self.loss_NCE_layer.item()
        loss_pos = self.loss_pos.item()
        loss_neg = self.loss_neg.item()

        loss_neg_MM = self.loss_neg_MM.item()
        loss_neg_E = self.loss_neg_E.item()
        loss_neg_F = self.loss_neg_F.item()

        loss_G_sum = self.loss_G.item()

        loss_record = { "G_sum": loss_G_sum,
                        "consist": loss_consist,
                        "adv_gen": loss_adv_gen,
                        "adv_dis": loss_adv_dis,
                        "contrstive_layer":loss_contrast_layer,
                        "pos":loss_pos,
                        "neg":loss_neg,
                        "neg_MM": loss_neg_MM,
                        "neg_E": loss_neg_E,
                        "neg_F": loss_neg_F
                        }
        return loss_record
