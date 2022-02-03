import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# import pac
# import pac_simplified as pac
from icecream import ic

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

def LSregress(pred, gt, origin, if_clamp_coeff=True):
    # print(pred.shape)
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    # coef = (torch.sum(pred * gt, dim = 1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)).detach()
    coef = (torch.sum(pred * gt, dim = 1) / (torch.sum(pred * pred, dim=1) + 1e-6)).detach()
    # print(coef)
    if if_clamp_coeff:
        coef = torch.clamp(coef, 0.001, 1000)
    # print(coef, pred.shape)
    for n in range(0, len(origSize) -1):
        coef = coef.unsqueeze(-1)
    pred = pred.view(origSize)

    predNew = origin * coef.expand(origSize)

    return predNew

def LSregressDiffSpec(diff, spec, imOrig, diffOrig, specOrig):
    nb, nc, nh, nw = diff.size()
    
    # Mask out too bright regions
    mask = (imOrig < 0.9).float() 
    diff = diff * mask 
    spec = spec * mask 
    im = imOrig * mask

    diff = diff.view(nb, -1)
    spec = spec.view(nb, -1)
    im = im.view(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim = 1)
    b2 = torch.sum(spec * im, dim = 1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22  - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    # Compute the coefficients assuming diffuse albedo only
    coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
    coef4 = coef3.clone() * 0

    frac = (frac / (nc * nh * nw)).detach()
    fracInd = (frac > 1e-2).float()

    coefDiffuse = fracInd * coef1 + (1 - fracInd) * coef3
    coefSpecular = fracInd * coef2 + (1 - fracInd) * coef4

    for n in range(0, 3):
        coefDiffuse = coefDiffuse.unsqueeze(-1)
        coefSpecular = coefSpecular.unsqueeze(-1)

    coefDiffuse = torch.clamp(coefDiffuse, min=0, max=1000)
    coefSpecular = torch.clamp(coefSpecular, min=0, max=1000)

    diffScaled = coefDiffuse.expand_as(diffOrig) * diffOrig
    specScaled = coefSpecular.expand_as(specOrig) * specOrig 

    # Do the regression twice to avoid clamping
    renderedImg = torch.clamp(diffScaled + specScaled, 0, 1)
    renderedImg = renderedImg.view(nb, -1)
    imOrig = imOrig.view(nb, -1)

    coefIm = (torch.sum(renderedImg * imOrig, dim = 1) \
            / torch.clamp(torch.sum(renderedImg * renderedImg, dim=1), min=1e-5)).detach()
    coefIm = torch.clamp(coefIm, 0.001, 1000)
    
    coefIm = coefIm.view(nb, 1, 1, 1)

    diffScaled = coefIm * diffScaled 
    specScaled = coefIm * specScaled

    return diffScaled, specScaled, coefIm


class encoder0(nn.Module):
    def __init__(self, opt, cascadeLevel = 0, isSeg = False, in_channels = 3, encoder_exclude=[]):

        super(encoder0, self).__init__()
        self.isSeg = isSeg
        self.opt = opt
        self.cascadeLevel = cascadeLevel

        self.encoder_exclude = encoder_exclude + self.opt.cfg.MODEL_BRDF.encoder_exclude

        self.channel_multi = opt.cfg.MODEL_BRDF.channel_multi

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64*self.channel_multi, kernel_size=4, stride=2, bias =True)
        else:
            self.conv1 = nn.Conv2d(in_channels = 17, out_channels = 64*self.channel_multi, kernel_size =4, stride =2, bias = True)

        self.gn1 = nn.GroupNorm(num_groups = 4*self.channel_multi, num_channels = 64*self.channel_multi)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels = 64*self.channel_multi, out_channels = 128*self.channel_multi, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups = 8*self.channel_multi, num_channels = 128*self.channel_multi)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 128*self.channel_multi, out_channels = 256*self.channel_multi, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups = 16*self.channel_multi, num_channels = 256*self.channel_multi)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels = 256*self.channel_multi, out_channels = 256*self.channel_multi, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups = 16*self.channel_multi, num_channels = 256*self.channel_multi)

        if 'x5' not in self.encoder_exclude:
            self.pad5 = nn.ZeroPad2d(1)
            self.conv5 = nn.Conv2d(in_channels = 256*self.channel_multi, out_channels = 512*self.channel_multi, kernel_size=4, stride=2, bias=True)
            self.gn5 = nn.GroupNorm(num_groups = 32*self.channel_multi, num_channels = 512*self.channel_multi)

        if 'x6' not in self.encoder_exclude:
            self.pad6 = nn.ZeroPad2d(1)
            self.conv6 = nn.Conv2d(in_channels = 512*self.channel_multi, out_channels = 1024*self.channel_multi, kernel_size=3, stride=1, bias=True)
            self.gn6 = nn.GroupNorm(num_groups = 64*self.channel_multi, num_channels = 1024*self.channel_multi)

    def forward(self, x, input_dict_extra=None):
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))), True)
        
        if 'x5' not in self.encoder_exclude:
            x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))), True)
        else:
            x5 = x1

        if 'x6' not in self.encoder_exclude:
            x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))), True)
        else:
            x6 = x1

        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape) # [16, 3, 192, 256, ]) [16, 64, 96, 128, ]) [16, 128, 48, 64,) [16, 256, 24, 32,) [16, 256, 12, 16,) [16, 512, 6, 8],  [16, 1024, 6, 8]
        return x1, x2, x3, x4, x5, x6, {}

class decoder0(nn.Module):
    def __init__(self, opt, mode=-1, modality='', out_channel=3, input_dict_guide=None,  if_PPM=False, if_not_final_fc=False):
        super(decoder0, self).__init__()
        self.opt = opt
        self.mode = mode
        self.modality = modality

        self.channel_multi = opt.cfg.MODEL_BRDF.channel_multi
        
        self.if_PPM = if_PPM

        self.dconv1 = nn.Conv2d(in_channels=1024*self.channel_multi, out_channels=512*self.channel_multi, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32*self.channel_multi, num_channels=512*self.channel_multi )

        self.dconv2 = nn.Conv2d(in_channels=1024*self.channel_multi, out_channels=256*self.channel_multi, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16*self.channel_multi, num_channels=256*self.channel_multi )

        self.dconv3 = nn.Conv2d(in_channels=512*self.channel_multi, out_channels=256*self.channel_multi, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16*self.channel_multi, num_channels=256*self.channel_multi )

        self.dconv4 = nn.Conv2d(in_channels=512*self.channel_multi, out_channels=128*self.channel_multi, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8*self.channel_multi, num_channels=128*self.channel_multi )

        self.dconv5 = nn.Conv2d(in_channels=256*self.channel_multi, out_channels=64*self.channel_multi, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4*self.channel_multi, num_channels=64*self.channel_multi )

        self.dconv6 = nn.Conv2d(in_channels=128*self.channel_multi, out_channels=64*self.channel_multi, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=4*self.channel_multi, num_channels=64*self.channel_multi )

        self.relu = nn.ReLU(inplace = True )

        fea_dim = 64*self.channel_multi
        if self.if_PPM:
            bins=(1, 2, 3, 6)
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2

        self.if_not_final_fc = if_not_final_fc
        if not self.if_not_final_fc:
            self.dpadFinal = nn.ReplicationPad2d(1)

        dconv_final_in_channels = 64
        if self.if_PPM:
            dconv_final_in_channels = 128

        self.dconvFinal = nn.Conv2d(in_channels=dconv_final_in_channels*self.channel_multi, out_channels=out_channel, kernel_size = 3, stride=1, bias=True)

        self.flag = True

    def forward(self, im, x1, x2, x3, x4, x5, x6, input_dict_extra=None):

        extra_output_dict = {}

        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )
        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1 )
        # if  :
        #     xin5 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin5, scale_feat_map=2)
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')

        if self.if_PPM:
            dx6 = self.ppm(dx6)

        return_dict = {'extra_output_dict': extra_output_dict, 'dx1': dx1, 'dx2': dx2, 'dx3': dx3, 'dx4': dx4, 'dx5': dx5, 'dx6': dx6}

        if self.if_not_final_fc:
            return return_dict


        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )

        if self.mode == 0: # modality='al'
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
        elif self.mode == 1: # modality='no'
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2: # modality='ro'
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4: # modality='de'
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1) # -> [-1, 1]
        # elif self.mode == 5: # clip to 0., inf
        #     x_out = self.relu(torch.mean(x_orig, dim=1).unsqueeze(1)) # -> [0, inf]
        #     # x_out[x_out < 1e-8] = 1e-8
        # elif self.mode == 6: # sigmoid to 0., 1. -> inverse to 0., inf
        #     x_out = torch.sigmoid(torch.mean(x_orig, dim=1).unsqueeze(1))
        #     x_out = 1. / (x_out + 1e-6) # -> [0, inf]
        else:
            x_out = x_orig

        return_dict.update({'x_out': x_out})

        return return_dict
