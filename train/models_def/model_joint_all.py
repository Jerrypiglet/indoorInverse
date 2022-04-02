import torch
import torch.nn as nn

from utils.utils_misc import *
from utils.utils_training import freeze_bn_in_module, unfreeze_bn_in_module
import torch.nn.functional as F


import models_def.models_brdf as models_brdf # basic model
import models_def.models_light as models_light 

from icecream import ic

import models_def.BilateralLayer as bs

class Model_Joint(nn.Module):
    def __init__(self, opt, logger):
        super(Model_Joint, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger
        self.non_learnable_layers = {}

        self.load_brdf_gt = self.opt.cfg.DATA.load_brdf_gt

        if self.cfg.MODEL_BRDF.enable:
            in_channels = 3

            self.encoder_to_use = models_brdf.encoder0
            self.decoder_to_use = models_brdf.decoder0

            self.BRDF_Net = nn.ModuleDict({
                    'encoder': self.encoder_to_use(opt, cascadeLevel = self.opt.cascadeLevel, in_channels = in_channels)
                    })

            if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if 'al' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'albedoDecoder': self.decoder_to_use(opt, mode=0, modality='al')})
                    if self.cfg.MODEL_BRDF.if_bilateral:
                        self.BRDF_Net.update({'albedoBs': bs.BilateralLayer(mode = 0)})

                if 'no' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'normalDecoder': self.decoder_to_use(opt, mode=1, modality='no')})
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        self.BRDF_Net.update({'normalBs': bs.BilateralLayer(mode = 1)})

                if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'roughDecoder': self.decoder_to_use(opt, mode=2, modality='ro')})
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        self.BRDF_Net.update({'roughBs': bs.BilateralLayer(mode = 2)})

                if 'de' in self.cfg.MODEL_BRDF.enable_list:
                    assert self.cfg.MODEL_BRDF.depth_activation in ['relu', 'sigmoid', 'tanh']
                    if self.cfg.MODEL_BRDF.depth_activation == 'relu':
                        self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=5, modality='de')}) # default # -> [0, inf]
                    elif self.cfg.MODEL_BRDF.depth_activation == 'sigmoid':
                        self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=6, modality='de')}) # -> [0, inf]
                    elif self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                        self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=4, modality='de')}) # -> [-1, 1]
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        self.BRDF_Net.update({'depthBs': bs.BilateralLayer(mode = 4)})
                    

            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()


        if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF:
            self.load_pretrained_MODEL_BRDF(if_load_encoder=self.cfg.MODEL_BRDF.pretrained_if_load_encoder, if_load_decoder=self.cfg.MODEL_BRDF.pretrained_if_load_decoder, if_load_Bs=self.cfg.MODEL_BRDF.pretrained_if_load_Bs)

        if self.cfg.MODEL_LIGHT.enable:
            self.LIGHT_Net = nn.ModuleDict({})
            self.LIGHT_Net.update({'lightEncoder':  models_light.encoderLight(cascadeLevel = opt.cascadeLevel, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            if 'axis' in opt.cfg.MODEL_LIGHT.enable_list:
                self.LIGHT_Net.update({'axisDecoder':  models_light.decoderLight(mode=0, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            if 'lamb' in opt.cfg.MODEL_LIGHT.enable_list:
                self.LIGHT_Net.update({'lambDecoder':  models_light.decoderLight(mode = 1, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            if 'weight' in opt.cfg.MODEL_LIGHT.enable_list:
                self.LIGHT_Net.update({'weightDecoder':  models_light.decoderLight(mode = 2, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})

            self.non_learnable_layers['renderLayer'] = models_light.renderingLayer(isCuda = opt.if_cuda, 
                imWidth=opt.cfg.MODEL_LIGHT.envCol, imHeight=opt.cfg.MODEL_LIGHT.envRow, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight)
            self.non_learnable_layers['output2env'] = models_light.output2env(isCuda = opt.if_cuda, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight, SGNum = opt.cfg.MODEL_LIGHT.SGNum )

            if self.cfg.MODEL_LIGHT.freeze_BRDF_Net:
                self.turn_off_names(['BRDF_Net'])
                freeze_bn_in_module(self.BRDF_Net)

            if self.cfg.MODEL_LIGHT.if_freeze:
                self.turn_off_names(['LIGHT_Net'])
                freeze_bn_in_module(self.LIGHT_Net)


            if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_LIGHT:
                self.load_pretrained_MODEL_LIGHT()

    def freeze_BN(self):
        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            self.turn_off_names(['BRDF_Net'])
            freeze_bn_in_module(self.BRDF_Net)

        if self.cfg.MODEL_LIGHT.if_freeze:
            self.turn_off_names(['LIGHT_Net'])
            freeze_bn_in_module(self.LIGHT_Net)


    def forward(self, input_dict, if_has_gt_BRDF=True):
        return_dict = {}
        input_dict_extra = {}

        if self.cfg.MODEL_BRDF.enable:
            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()

            return_dict_brdf = self.forward_brdf(input_dict, input_dict_extra=input_dict_extra, if_has_gt_BRDF=if_has_gt_BRDF)
        else:
            return_dict_brdf = {}
        return_dict.update(return_dict_brdf)

        if self.cfg.MODEL_LIGHT.enable:
            if self.cfg.MODEL_LIGHT.if_freeze:
                self.LIGHT_Net.eval()
            # if self.opt.cfg.DATASET.if_no_gt_BRDF:
            #     return_dict_light = self.forward_light_real(input_dict, return_dict_brdf=return_dict_brdf)
            # else:
            return_dict_light = self.forward_light(input_dict, return_dict_brdf=return_dict_brdf)
        else:
            return_dict_light = {}
        return_dict.update(return_dict_light)

        return return_dict


    def forward_brdf(self, input_dict, input_dict_extra={}, if_has_gt_BRDF=True):
        if_has_gt_BRDF = if_has_gt_BRDF and (not self.opt.cfg.DATASET.if_no_gt_BRDF) and self.load_brdf_gt and not self.opt.cfg.DEBUG.if_test_real
        if_has_gt_segBRDF = if_has_gt_BRDF and not self.opt.cfg.DEBUG.if_nyud and not self.opt.cfg.DEBUG.if_iiw and not self.opt.cfg.DEBUG.if_test_real

        input_list = [input_dict['input_batch_brdf']]

        input_tensor = torch.cat(input_list, 1)
        x1, x2, x3, x4, x5, x6, extra_output_dict = self.BRDF_Net['encoder'](input_tensor, input_dict_extra=input_dict_extra)

        return_dict = {'encoder_outputs': {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'brdf_extra_output_dict': extra_output_dict}}
        albedo_output = {}

        if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
            if 'al' in self.cfg.MODEL_BRDF.enable_list:
                albedo_output = self.BRDF_Net['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                albedoPred = 0.5 * (albedo_output['x_out'] + 1)

                if if_has_gt_segBRDF:
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
                # if not self.cfg.MODEL_BRDF.use_scale_aware_albedo:
                if if_has_gt_BRDF and if_has_gt_segBRDF:
                    albedoPred_aligned = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                    albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                    return_dict.update({'albedoPred_aligned': albedoPred_aligned, 'albedo_extra_output_dict': albedo_output['extra_output_dict']})

                if self.cfg.MODEL_BRDF.if_bilateral:
                    albedoBsPred, albedoConf = self.BRDF_Net['albedoBs'](input_dict['imBatch'], albedoPred.detach(), albedoPred )
                    if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                        albedoBsPred = models_brdf.LSregress(albedoBsPred * input_dict['segBRDFBatch'].expand_as(albedoBsPred ),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoBsPred )
                    albedoBsPred = torch.clamp(albedoBsPred, 0, 1 )
                    if if_has_gt_segBRDF:
                        albedoBsPred = input_dict['segBRDFBatch'] * albedoBsPred
                    return_dict.update({'albedoBsPred': albedoBsPred, 'albedoConf': albedoConf})

                    if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                        albedoBsPred_aligned, albedoConf_aligned = self.BRDF_Net['albedoBs'](input_dict['imBatch'], albedoPred_aligned.detach(), albedoPred_aligned )
                        return_dict.update({'albedoBsPred_aligned': albedoBsPred_aligned})

            if 'no' in self.cfg.MODEL_BRDF.enable_list:
                normal_output = self.BRDF_Net['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                normalPred = normal_output['x_out']
                return_dict.update({'normalPred': normalPred, 'normal_extra_output_dict': normal_output['extra_output_dict']})
                if if_has_gt_BRDF:
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        normalBsPred = normalPred.clone().detach()
                        normalConf = albedoConf.clone().detach()
                        return_dict.update({'normalBsPred': normalBsPred, 'normalConf': normalConf})

            if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                rough_output = self.BRDF_Net['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                roughPred = rough_output['x_out']
                return_dict.update({'roughPred': roughPred, 'rough_extra_output_dict': rough_output['extra_output_dict']})
                # if if_has_gt_BRDF:
                if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    roughBsPred, roughConf = self.BRDF_Net['roughBs'](input_dict['imBatch'], albedoPred.detach(), 0.5*(roughPred+1.) )
                    roughBsPred = torch.clamp(2 * roughBsPred - 1, -1, 1)
                    return_dict.update({'roughBsPred': roughBsPred, 'roughConf': roughConf})

            if 'de' in self.cfg.MODEL_BRDF.enable_list:
                depth_output = self.BRDF_Net['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                depthPred = depth_output['x_out']
                return_dict.update({'depthPred': depthPred})

                if if_has_gt_BRDF:
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], albedoPred.detach(), 0.5*(depthPred+1.) if self.cfg.MODEL_BRDF.depth_activation=='tanh' else depthPred )
                        if if_has_gt_segBRDF:
                            depthBsPred = models_brdf.LSregress(depthBsPred *  input_dict['segAllBatch'].expand_as(depthBsPred),
                                    input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthBsPred)
                        else:
                            depthBsPred = models_brdf.LSregress(depthBsPred, input_dict['depthBatch'], depthBsPred)
                        return_dict.update({'depthBsPred': depthBsPred, 'depthConf': depthConf})
                    
                    if self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                        depthPred_aligned = 0.5 * (depthPred + 1) # [-1, 1] -> [0, 1]
                    else:
                        depthPred_aligned = depthPred # [0, inf]
                    if if_has_gt_segBRDF:
                        depthPred_aligned = models_brdf.LSregress(depthPred_aligned *  input_dict['segAllBatch'].expand_as(depthPred_aligned),
                                input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred_aligned)
                    else:
                        depthPred_aligned = models_brdf.LSregress(depthPred_aligned, input_dict['depthBatch'], depthPred_aligned)
                    return_dict.update({'depthPred_aligned': depthPred_aligned, 'depth_extra_output_dict': depth_output['extra_output_dict']})

                if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    if 'albedoPred' in return_dict:
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), depthPred )
                    else:
                        assert self.opt.cfg.DEBUG.if_load_dump_BRDF_offline
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], input_dict['albedoBatch'].detach(), depthPred )
                    return_dict.update({'depthBsPred': depthBsPred})
                    if if_has_gt_BRDF and 'de' in self.opt.cfg.DATA.data_read_list:
                        assert 'depthPred_aligned' in return_dict
                        depthBsPred_aligned, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), depthPred_aligned )
                        return_dict.update({'depthBsPred_aligned': depthBsPred_aligned})


        return return_dict

    def forward_LIGHT_Net(self, input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput, ):
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width
        assert self.cfg.DATA.pad_option == 'const'

        # bn, ch, nrow, ncol = albedoInput.size()
        # if not self.cfg.MODEL_LIGHT.use_scale_aware_loss:
        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        # bn, ch, nrow, ncol = depthInput.size()
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)

        imBatchLarge = F.interpolate(imBatch, scale_factor=2, mode='bilinear')
        albedoInputLarge = F.interpolate(albedoInput, scale_factor=2, mode='bilinear')
        depthInputLarge = F.interpolate(depthInput, scale_factor=2, mode='bilinear')
        normalInputLarge = F.interpolate(normalInput, scale_factor=2, mode='bilinear')
        roughInputLarge = F.interpolate(roughInput, scale_factor=2, mode='bilinear')

        input_batch = torch.cat([imBatchLarge, albedoInputLarge, normalInputLarge, roughInputLarge, depthInputLarge ], dim=1 )

        if self.opt.cascadeLevel == 0:
            # print(input_batch.shape)
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach() )
        else:
            assert self.opt.cascadeLevel > 0
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach(), input_dict['envmapsPreBatch'].detach() )

        # print(input_batch.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape) # torch.Size([4, 11, 480, 640]) torch.Size([4, 128, 60, 80]) torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 15, 20]) torch.Size([4, 512, 7, 10]) torch.Size([4, 512, 3, 5]) torch.Size([4, 1024, 3, 5])

        # Prediction
        if 'axis' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            axisPred_ori = self.LIGHT_Net['axisDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 3, 120, 160])
        else:
            axisPred_ori = input_dict['sg_axis_Batch'] # (4, 120, 160, 12, 3)
            axisPred_ori = axisPred_ori.permute(0, 3, 4, 1, 2)
        if 'lamb' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            lambPred_ori = self.LIGHT_Net['lambDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 120, 160])
        else:
            lambPred_ori = input_dict['sg_lamb_Batch'] # (4, 120, 160, 12, 1)
            lambPred_ori = lambPred_ori.squeeze(4).permute(0, 3, 1, 2)

        if 'weight' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            weightPred_ori = self.LIGHT_Net['weightDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 36, 120, 160])
        else:
            weightPred_ori = input_dict['sg_weight_Batch'] # (4, 120, 160, 12, 3)
            weightPred_ori = weightPred_ori.flatten(3).permute(0, 3, 1, 2)
            # weightPred_ori = torch.ones_like(weightPred_ori).cuda() * 0.1
            # weightPred_ori[weightPred_ori>500] = 500.
            # weightPred_ori = weightPred_ori / 500.
        # print(torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori))

        # print(axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape)
        return axisPred_ori, lambPred_ori, weightPred_ori


    def forward_light(self, input_dict, return_dict_brdf):
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width

        # Normalize Albedo and depth
        if 'al' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            albedoInput = return_dict_brdf['albedoPred'].detach().clone()
        else:
            albedoInput = input_dict['albedoBatch'].detach().clone()

        if 'de' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            depthInput = return_dict_brdf['depthPred'].detach().clone()
            # print('-', depthInput.shape, torch.max(depthInput), torch.min(depthInput), torch.median(depthInput))
            if self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                depthInput = 0.5 * (depthInput + 1) # [-1, 1] -> [0, 1]
            # print('->', depthInput.shape, torch.max(depthInput), torch.min(depthInput), torch.median(depthInput))
        else:
            depthInput = input_dict['depthBatch'].detach().clone()

        if 'no' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            normalInput = return_dict_brdf['normalPred'].detach().clone()
        else:
            normalInput = input_dict['normalBatch'].detach().clone()

        if 'ro' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            roughInput = return_dict_brdf['roughPred'].detach().clone()
        else:
            roughInput = input_dict['roughBatch'].detach().clone()

        imBatch = input_dict['imBatch']
        segBRDFBatch = input_dict['segBRDFBatch']

        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            assert self.BRDF_Net.training == False

        axisPred_ori, lambPred_ori, weightPred_ori = self.forward_LIGHT_Net(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)

        if_print = False

        if self.opt.is_master and if_print:
            print('--(unet) weight', torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori), weightPred_ori.shape)
            print('--(unet) lamb', torch.max(lambPred_ori), torch.min(lambPred_ori), torch.median(lambPred_ori), lambPred_ori.shape)
            print('--(unet) axis', torch.max(axisPred_ori), torch.min(axisPred_ori), torch.median(axisPred_ori), axisPred_ori.shape)

        if self.opt.is_master and if_print:
            print('--weight', torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori), weightPred_ori.shape)
            print('--axis', torch.max(axisPred_ori), torch.min(axisPred_ori), torch.median(axisPred_ori), axisPred_ori.shape)
            print('--lamb', torch.max(lambPred_ori), torch.min(lambPred_ori), torch.median(lambPred_ori), lambPred_ori.shape)

        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        segBRDFBatchSmall = F.interpolate(segBRDFBatch, scale_factor=0.5, mode="nearest")
        notDarkEnv = (torch.mean(torch.mean(torch.mean(input_dict['envmapsBatch'], 4), 4), 1, True ) > 0.001 ).float()
        segEnvBatch = (segBRDFBatchSmall * input_dict['envmapsIndBatch'].expand_as(segBRDFBatchSmall) ).unsqueeze(-1).unsqueeze(-1)

        segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        return_dict = {}

        # Compute the recontructed error
        if self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsPredImage = input_dict['envmapsBatch'].detach().clone()
        else:
            envmapsPredImage, axisPred, lambPred, weightPred = self.non_learnable_layers['output2env'].output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg)

        # print(axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape, envmapsPredImage.shape, '=====')

        pixelNum_recon = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)
        if self.cfg.MODEL_LIGHT.use_GT_light_sg:
            envmapsPredScaledImage = envmapsPredImage * (input_dict['hdr_scaleBatch'].flatten().view(-1, 1, 1, 1, 1, 1))
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsPredScaledImage = envmapsPredImage # gt envmap already scaled in dataloader
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            envmapsPredScaledImage = envmapsPredImage # not aligning envmap
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        else: # scale-invariant
            if self.cfg.MODEL_LIGHT.if_align_log_envmap:
                # assert False, 'disabled'
                # envmapsPredScaledImage = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                #     torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                #     if_clamp_coeff=False)
                envmapsPredScaledImage_offset_log_ = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset), 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                envmapsPredScaledImage = torch.exp(envmapsPredScaledImage_offset_log_) - self.cfg.MODEL_LIGHT.offset
            else:
                envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                print('-envmapsPredScaledImage-', torch.max(envmapsPredScaledImage), torch.min(envmapsPredScaledImage), torch.mean(envmapsPredScaledImage), torch.median(envmapsPredScaledImage))
                # envmapsPredScaledImage_offset_log_ = torch.log(torch.clamp(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset, min=self.cfg.MODEL_LIGHT.offset))
                envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)

        if self.opt.is_master and if_print:
            ic(torch.max(input_dict['envmapsBatch']), torch.min(input_dict['envmapsBatch']),torch.median(input_dict['envmapsBatch']))
            ic(torch.max(envmapsPredImage), torch.min(envmapsPredImage),torch.median(envmapsPredImage))
            ic(torch.max(envmapsPredScaledImage), torch.min(envmapsPredScaledImage),torch.median(envmapsPredScaledImage))
            ic(torch.max(envmapsPredScaledImage_offset_log_), torch.min(envmapsPredScaledImage_offset_log_),torch.median(envmapsPredScaledImage_offset_log_))

        return_dict.update({'envmapsPredImage': envmapsPredImage, 'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_offset_log_': envmapsPredScaledImage_offset_log_, \
            'segEnvBatch': segEnvBatch, \
            'imBatchSmall': imBatchSmall, 'segBRDFBatchSmall': segBRDFBatchSmall, 'pixelNum_recon': pixelNum_recon}) 

        # Compute the rendered error
        pixelNum_render = max( (torch.sum(segBRDFBatchSmall ).cpu().data).item(), 1e-5 )
        
        normal_input, rough_input = normalInput, roughInput

        if self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsImage_input = input_dict['envmapsBatch']
        else:
            envmapsImage_input = envmapsPredImage

        diffusePred, specularPred = self.non_learnable_layers['renderLayer'].forwardEnv(normalPred=normal_input.detach(), envmap=envmapsImage_input, diffusePred=albedoInput.detach(), roughPred=rough_input.detach())

        if self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            diffusePredScaled, specularPredScaled = diffusePred, specularPred
        else:
            diffusePredScaled, specularPredScaled, _ = models_brdf.LSregressDiffSpec(
                diffusePred.detach(),
                specularPred.detach(),
                imBatchSmall,
                diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = renderedImPred_hdr

        if self.opt.is_master and if_print:
            print('--renderedImPred', torch.max(renderedImPred), torch.min(renderedImPred), torch.median(renderedImPred), renderedImPred.shape)

        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        return_dict.update({'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr, 'pixelNum_render': pixelNum_render}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori, 'segEnvBatch': segEnvBatch, 'notDarkEnv': notDarkEnv}})

        return return_dict

    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            if_trainable_str = white_blue('True') if param.requires_grad else green('False')
            self.logger.info(name + str(param.shape) + ' ' + if_trainable_str)
            if param.requires_grad:
                count_grads += 1
        self.logger.info(magenta('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads

    def load_pretrained_MODEL_BRDF(self, if_load_encoder=True, if_load_decoder=True, if_load_Bs=True):
        # if self.opt.if_cluster:
        #     pretrained_path_root = Path('/viscompfs/users/ruizhu/models_ckpt/')
        # else:
        #     pretrained_path_root = Path('/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt/')
        pretrained_path_root = Path(self.opt.cfg.PATH.models_ckpt_path)
        # loaded_strings = []
        module_names = []
        if if_load_encoder:
            module_names.append('encoder')    
        if if_load_decoder:
            if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['albedoDecoder']
            if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['normalDecoder']
            if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['roughDecoder']
            if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['depthDecoder']
        if if_load_Bs:
            if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['albedoBs']
            if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['normalBs']
            if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['roughBs']
            if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['depthBs']
            # module_names += ['albedoBs']
            # assert self.cfg.MODEL_BRDF.if_bilateral_albedo_only

        saved_names_dict = {
            'encoder': 'encoder', 
            'albedoDecoder': 'albedo', 
            'normalDecoder': 'normal', 
            'roughDecoder': 'rough', 
            'depthDecoder': 'depth', 
            'albedoBs': 'albedoBs', 
            'normalBs': 'normalBs', 
            'roughBs': 'roughBs', 
            'depthBs': 'depthBs'
        }
        pretrained_pth_name_dict = {
            'encoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'albedoDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'normalDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'roughDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'depthDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'albedoBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'normalBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'roughBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'depthBs':self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0
        }
        for module_name in module_names:
            saved_name = saved_names_dict[module_name]
            pickle_path = str(pretrained_path_root / pretrained_pth_name_dict[module_name]) % saved_name
            print('Loading %s into module [%s]'%(pickle_path, module_name))
            self.BRDF_Net[module_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            # loaded_strings.append(saved_name)

            self.logger.info(magenta('Loaded pretrained BRDFNet-%s from %s'%(module_name, pickle_path)))
    
    def load_pretrained_MODEL_LIGHT(self):
        pretrained_path_root = Path(self.opt.cfg.PATH.models_ckpt_path)
        loaded_strings = []
        for saved_name in ['lightEncoder', 'axisDecoder', 'lambDecoder', 'weightDecoder', ]:
            # pickle_path = '{0}/{1}{2}_{3}.pth'.format(pretrained_path, saved_name, cascadeLevel, epochIdFineTune) 
            pickle_path = str(pretrained_path_root / self.opt.cfg.MODEL_LIGHT.pretrained_pth_name_cascade0) % saved_name
            print('Loading ' + pickle_path)
            self.LIGHT_Net[saved_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            loaded_strings.append(saved_name)

            self.logger.info(magenta('Loaded pretrained LightNet-%s from %s'%(saved_name, pickle_path)))


    def turn_off_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info(colored('only_enable_camH_bboxPredictor', 'white', 'on_red'))

    def turn_on_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        self.logger.info(colored('turned on all params', 'white', 'on_red'))

    def turn_on_names(self, in_names, if_print=True):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = True
                    if if_print:
                        self.logger.info(colored('turn_ON_names: ' + in_name, 'white', 'on_red'))

    def turn_off_names(self, in_names, exclude_names=[], if_print=True):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if_not_in_exclude = all([exclude_name not in name for exclude_name in exclude_names]) # any item in exclude_names must not be in the paramater name
                if in_name in name and if_not_in_exclude:
                    param.requires_grad = False
                    if if_print:
                        self.logger.info(colored('turn_OFF_names: ' + in_name, 'white', 'on_red'))

    def freeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.normalDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.normalDecoder, if_print=if_print)
        if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.depthDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.depthDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def unfreeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.normalDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.normalDecoder, if_print=if_print)
        if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.depthDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.depthDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def freeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.albedoDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.albedoDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def unfreeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.albedoDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.albedoDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)


    def freeze_bn_semantics(self):
        freeze_bn_in_module(self.SEMSEG_Net)

    def freeze_bn_matseg(self):
        freeze_bn_in_module(self.MATSEG_Net)