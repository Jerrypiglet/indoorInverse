import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils
from icecream import ic
from models_def.models_brdf import LSregressDiffSpec


# def get_im_input(data_batch, opt, return_input_batch_as_list=False)

def get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=False):
    input_dict = {}
    
    input_dict['im_paths'] = data_batch['image_path']
    # Load the image from cpu to gpu
    # im_cpu = (data_batch['im_trainval'].permute(0, 3, 1, 2) )
    im_cpu = data_batch['im_trainval']
    input_dict['imBatch'] = im_cpu.cuda(non_blocking=True).contiguous()
    # print(torch.max(input_dict['imBatch']), torch.min(input_dict['imBatch']), '+++')

    input_dict['brdf_loss_mask'] = data_batch['brdf_loss_mask'].cuda(non_blocking=True)
    input_dict['pad_mask'] = data_batch['pad_mask'].cuda(non_blocking=True)
    if 'frame_info' in data_batch:
        input_dict['frame_info'] = data_batch['frame_info']
    if opt.cfg.DEBUG.if_test_real:
        input_dict['im_h_resized_to'] = data_batch['im_h_resized_to']
        input_dict['im_w_resized_to'] = data_batch['im_w_resized_to']

    if_load_mask = (opt.cfg.DATA.load_brdf_gt or opt.cfg.DEBUG.if_load_dump_BRDF_offline) and (not opt.cfg.DATASET.if_no_gt_BRDF) and not (opt.cfg.DEBUG.if_test_real)
    # if_load_mask = opt.cfg.DATASET.if_no_gt_BRDF

    # print(opt.cfg.DATA.load_brdf_gt, opt.cfg.DEBUG.if_load_dump_BRDF_offline, opt.cfg.DATA.data_read_list)
    if opt.cfg.DATA.load_brdf_gt or opt.cfg.DEBUG.if_load_dump_BRDF_offline:
        # Load data from cpu to gpu
        if 'al' in opt.cfg.DATA.data_read_list:
            albedo_cpu = data_batch['albedo']
            input_dict['albedoBatch'] = albedo_cpu.cuda(non_blocking=True)

        if 'no' in opt.cfg.DATA.data_read_list:
            normal_cpu = data_batch['normal']
            input_dict['normalBatch'] = normal_cpu.cuda(non_blocking=True)

        if 'ro' in opt.cfg.DATA.data_read_list:
            rough_cpu = data_batch['rough']
            input_dict['roughBatch'] = rough_cpu.cuda(non_blocking=True)

        if 'de' in opt.cfg.DATA.data_read_list:
            depth_cpu = data_batch['depth']
            input_dict['depthBatch'] = depth_cpu.cuda(non_blocking=True)

    if if_load_mask:
        if 'mask' in data_batch:
            mask_cpu = data_batch['mask'].permute(0, 3, 1, 2) # [b, 3, h, w]
            input_dict['maskBatch'] = mask_cpu.cuda(non_blocking=True)

        segArea_cpu = data_batch['segArea']
        segEnv_cpu = data_batch['segEnv']
        segObj_cpu = data_batch['segObj']

        seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
        segBatch = seg_cpu.cuda(non_blocking=True)

        input_dict['segBRDFBatch'] = segBatch[:, 2:3, :, :]
        input_dict['segAllBatch'] = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]
    else:
        input_dict['segBRDFBatch'] = torch.ones((im_cpu.shape[0], 1, im_cpu.shape[2], im_cpu.shape[3]), dtype=torch.float32).cuda(non_blocking=True)
        input_dict['segAllBatch'] = input_dict['segBRDFBatch']

    # print(input_dict['segBRDFBatch'].shape, input_dict['brdf_loss_mask'].shape)
    input_dict['segBRDFBatch'] = input_dict['segBRDFBatch'] * input_dict['brdf_loss_mask'].unsqueeze(1)
    input_dict['segAllBatch'] = input_dict['segAllBatch'] * input_dict['brdf_loss_mask'].unsqueeze(1)

    preBatchDict = {}
    if opt.cfg.DATA.load_brdf_gt:
        if opt.cascadeLevel > 0:
            albedoPre_cpu = data_batch['albedoPre']
            albedoPreBatch = albedoPre_cpu.cuda(non_blocking=True)

            normalPre_cpu = data_batch['normalPre']
            normalPreBatch = normalPre_cpu.cuda(non_blocking=True)

            roughPre_cpu = data_batch['roughPre']
            roughPreBatch = roughPre_cpu.cuda(non_blocking=True)

            depthPre_cpu = data_batch['depthPre']
            depthPreBatch = depthPre_cpu.cuda(non_blocking=True)

            diffusePre_cpu = data_batch['diffusePre']
            diffusePreBatch = diffusePre_cpu.cuda(non_blocking=True)

            specularPre_cpu = data_batch['specularPre']
            specularPreBatch = specularPre_cpu.cuda(non_blocking=True)

            if albedoPreBatch.size(2) < opt.imHeight or albedoPreBatch.size(3) < opt.imWidth:
                albedoPreBatch = F.interpolate(albedoPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if normalPreBatch.size(2) < opt.imHeight or normalPreBatch.size(3) < opt.imWidth:
                normalPreBatch = F.interpolate(normalPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if roughPreBatch.size(2) < opt.imHeight or roughPreBatch.size(3) < opt.imWidth:
                roughPreBatch = F.interpolate(roughPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if depthPreBatch.size(2) < opt.imHeight or depthPreBatch.size(3) < opt.imWidth:
                depthPreBatch = F.interpolate(depthPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

            # Regress the diffusePred and specular Pred
            envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
            imBatchSmall = F.adaptive_avg_pool2d(input_dict['imBatch'], (envRow, envCol) )
            diffusePreBatch, specularPreBatch = LSregressDiffSpec(
                    diffusePreBatch, specularPreBatch, imBatchSmall,
                    diffusePreBatch, specularPreBatch )

            if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
                diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
                specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

            renderedImBatch = diffusePreBatch + specularPreBatch

    assert opt.cascadeLevel == 0
    input_batch = [input_dict['imBatch']]
    if not return_input_batch_as_list:
        input_batch = torch.cat(input_batch, 1)

    return input_batch, input_dict, preBatchDict

def postprocess_brdf(input_dict, output_dict, loss_dict, opt, time_meters, eval_module_list=[], tid=-1, if_loss=True):
    if opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        opt.albeW, opt.normW, opt.rougW, opt.deptW = opt.cfg.MODEL_BRDF.albedoWeight, opt.cfg.MODEL_BRDF.normalWeight, opt.cfg.MODEL_BRDF.roughWeight, opt.cfg.MODEL_BRDF.depthWeight

        pixelObjNum = (torch.sum(input_dict['segBRDFBatch'] ).cpu().data).item()
        pixelAllNum = (torch.sum(input_dict['segAllBatch'] ).cpu().data).item()

        if opt.cfg.MODEL_LIGHT.enable:
            extra_dict = []

        if opt.cfg.MODEL_BRDF.enable_BRDF_decoders and if_loss:
            loss_dict['loss_brdf-ALL'] = 0.

        if 'al' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            # albedoPreds = []
            if opt.cfg.MODEL_BRDF.use_scale_aware_albedo or opt.cfg.DEBUG.if_test_real:
                albedoPred = output_dict['albedoPred']

            if 'al' in opt.cfg.DATA.data_read_list:
                if not opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                    albedoPred = output_dict['albedoPred_aligned']
                if if_loss and 'al':
                    loss_dict['loss_brdf-albedo'] = []
                    loss = torch.sum( (albedoPred - input_dict['albedoBatch'])
                        * (albedoPred - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0
                    loss_dict['loss_brdf-albedo'].append(loss)

                    loss_dict['loss_brdf-ALL'] += 4 * opt.albeW * loss_dict['loss_brdf-albedo'][-1]
                    loss_dict['loss_brdf-albedo'] = loss_dict['loss_brdf-albedo'][-1]

                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        albedoBsPred = output_dict['albedoBsPred']
                        if not opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                            albedoBsPred = output_dict['albedoBsPred_aligned']
                        loss_bs = torch.sum( (albedoBsPred - input_dict['albedoBatch'])
                            * (albedoBsPred - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0
                        loss_dict['loss_brdf-ALL'] += 4 * opt.albeW * loss_bs
                        loss_dict['loss_brdf-albedo-bs'] = loss_bs


        if 'no' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            normalPred = output_dict['normalPred']

            if if_loss and 'no' in opt.cfg.DATA.data_read_list:
                loss_dict['loss_brdf-normal'] = []
                loss_dict['loss_brdf-normal'].append( torch.sum( (normalPred - input_dict['normalBatch'])
                    * (normalPred - input_dict['normalBatch']) * input_dict['segAllBatch'].expand_as(input_dict['normalBatch']) ) / pixelAllNum / 3.0)
                loss_dict['loss_brdf-ALL'] += opt.normW * loss_dict['loss_brdf-normal'][-1]
                loss_dict['loss_brdf-normal'] = loss_dict['loss_brdf-normal'][-1]

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            roughPred = output_dict['roughPred']
            if if_loss and 'ro' in opt.cfg.DATA.data_read_list:
                loss_dict['loss_brdf-rough'] = []
                loss_dict['loss_brdf-rough'].append( torch.sum( (roughPred - input_dict['roughBatch'])
                    * (roughPred - input_dict['roughBatch']) * input_dict['segBRDFBatch'] ) / pixelObjNum )
                loss_dict['loss_brdf-ALL'] += opt.rougW * loss_dict['loss_brdf-rough'][-1]
                loss_dict['loss_brdf-rough'] = loss_dict['loss_brdf-rough'][-1]
                loss_dict['loss_brdf-rough-paper'] = loss_dict['loss_brdf-rough'] / 4.

                if opt.cfg.MODEL_BRDF.if_bilateral:
                    loss_bs = torch.sum( (output_dict['roughBsPred'] - input_dict['roughBatch'])
                        * (output_dict['roughBsPred'] - input_dict['roughBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['roughBatch'] ) ) / pixelObjNum
                    loss_dict['loss_brdf-ALL'] += 4 * opt.rougW * loss_bs
                    loss_dict['loss_brdf-rough-bs'] = loss_bs
                    loss_dict['loss_brdf-rough-bs-paper'] = loss_bs / 4.


        if 'de' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            if opt.cfg.MODEL_BRDF.use_scale_aware_depth or opt.cfg.DEBUG.if_test_real:
                depthPred = output_dict['depthPred']
            else:
                depthPred = output_dict['depthPred_aligned']

            if 'de' in opt.cfg.DATA.data_read_list:
                if if_loss:
                    loss_dict['loss_brdf-depth'] = []

                    if opt.cfg.MODEL_BRDF.loss.depth.if_use_paper_loss:
                        loss =  torch.sum( 
                                (torch.log(depthPred+1) - torch.log(input_dict['depthBatch']+0.001) )
                                * ( torch.log(depthPred+0.001) - torch.log(input_dict['depthBatch']+0.001) )
                                * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) 
                            ) / pixelAllNum
                    else:
                        assert opt.cfg.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss
                        loss =  torch.sum(
                                (torch.log(depthPred+1) - torch.log(input_dict['depthBatch']+1) )
                                * ( torch.log(depthPred+1) - torch.log(input_dict['depthBatch']+1) )
                                * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) 
                            ) / pixelAllNum 

                    loss_dict['loss_brdf-depth'].append(loss)
                    loss_dict['loss_brdf-ALL'] += opt.deptW * loss_dict['loss_brdf-depth'][-1]
                    loss_dict['loss_brdf-depth'] = loss_dict['loss_brdf-depth'][-1]
                    loss_dict['loss_brdf-depth-paper'] = torch.sum( (torch.log(depthPred+0.001) - torch.log(input_dict['depthBatch']+0.001) )
                        * ( torch.log(depthPred+0.001) - torch.log(input_dict['depthBatch']+0.001) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum

                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        depthBsPred = output_dict['depthBsPred']
                        if not opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                            depthBsPred = output_dict['depthBsPred_aligned']
                        loss_bs =  torch.sum(
                                    (torch.log(depthBsPred+1) - torch.log(input_dict['depthBatch']+1) )
                                    * ( torch.log(depthBsPred+1) - torch.log(input_dict['depthBatch']+1) )
                                    * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) 
                                ) / pixelAllNum 
                        loss_dict['loss_brdf-ALL'] += 4 * opt.deptW * loss_bs
                        loss_dict['loss_brdf-depth-bs-paper'] = torch.sum( (torch.log(depthBsPred+0.001) - torch.log(input_dict['depthBatch']+0.001) )
                        * ( torch.log(depthBsPred+0.001) - torch.log(input_dict['depthBatch']+0.001) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum



    return output_dict, loss_dict