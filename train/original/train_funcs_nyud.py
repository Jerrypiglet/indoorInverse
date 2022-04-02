import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def get_labels_dict_nyud(data_batch, opt, return_input_batch_as_list=False):
    input_dict = {}
    
    input_dict['im_paths'] = data_batch['image_path']
    im_cpu = data_batch['im_trainval']
    input_dict['imBatch'] = im_cpu.cuda(non_blocking=True).contiguous()

    # if opt.cfg.DEBUG.if_test_real:
    input_dict['im_h_resized_to'] = data_batch['im_h_resized_to']
    input_dict['im_w_resized_to'] = data_batch['im_w_resized_to']

    assert opt.cascadeLevel == 0
    input_batch = [input_dict['imBatch']]
    if not return_input_batch_as_list:
        input_batch = torch.cat(input_batch, 1)

    normal_cpu = data_batch['normal']
    normalBatch = Variable(normal_cpu ).cuda()

    depth_cpu = data_batch['depth']
    depthBatch = Variable(depth_cpu ).cuda()

    seg_cpu = data_batch['segNormal']
    segNormalBatch = Variable( seg_cpu ).cuda()

    seg_cpu = data_batch['segDepth']
    segDepthBatch = Variable(seg_cpu ).cuda()
    
    input_dict.update({'normalBatch': normalBatch, 'depthBatch': depthBatch, 'segNormalBatch': segNormalBatch, 'segDepthBatch': segDepthBatch})

    if 'segMask_ori' in data_batch:
        segMask_ori_cpu = data_batch['segMask_ori']
        normal_ori_cpu = data_batch['normal_ori']
        depth_ori_cpu = data_batch['depth_ori']

        input_dict.update({'segMask_ori_cpu': segMask_ori_cpu, 'normal_ori_cpu': normal_ori_cpu, 'depth_ori_cpu': depth_ori_cpu})

    return input_batch, input_dict

def postprocess_nyud(input_dict, output_dict, loss_dict, opt, time_meters, eval_module_list=[], tid=-1, if_loss=True):

    # if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
    #     output_dict['roughPreds'] = [output_dict['roughPred']]

    # if 'al' in opt.cfg.MODEL_BRDF.enable_list:
    #     if opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
    #         output_dict['albedoPreds'] = [output_dict['albedoPred']]
    #     else:
    #         output_dict['albedoPreds'] = [output_dict['albedoPred_aligned']]

    # if 'de' in opt.cfg.MODEL_BRDF.enable_list:
    #     if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
    #         depthPred = output_dict['depthPred']
    #     else:
    #         if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt:
    #             depthPred = output_dict['depthPred_aligned']
    #     output_dict['depthPreds'] = [output_dict['depthPred']]

    # if 'no' in opt.cfg.MODEL_BRDF.enable_list:
    #     output_dict['normalPreds'] = [output_dict['normalPred']]

    if if_loss:
        loss_dict['loss_nyud-ALL'] = 0.

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalPred = output_dict['normalPred']
            normalBatch = input_dict['normalBatch']
            segNormalBatch = input_dict['segNormalBatch']

            pixelAllNumNormal = (torch.sum(segNormalBatch ).cpu().data).item()

            normalErr = torch.sum( (normalPred - normalBatch)
                * (normalPred - normalBatch) * segNormalBatch.expand_as(normalBatch) ) / pixelAllNumNormal / 3.0
            
            # both in [-1, 1]
            # print(normalPred.shape, torch.max(normalPred), torch.min(normalPred), torch.median(normalPred))
            # print(normalBatch.shape, torch.max(normalBatch), torch.min(normalBatch), torch.median(normalBatch))

            angleMean = torch.sum(torch.acos( torch.clamp(torch.sum(normalPred * normalBatch, dim=1).unsqueeze(1), -1, 1) ) / np.pi * 180 * segNormalBatch) / pixelAllNumNormal

            # normalPred_np = normalPred.data.cpu().numpy()
            # normalBatch_np = normalBatch.data.cpu().numpy()
            # segNormalBatch_np = segNormalBatch.cpu().numpy()
            # theta = np.arccos( np.clip(np.sum(normalPred_np * normalBatch_np, axis=1)[:, np.newaxis, :, :], -1, 1) ) / np.pi * 180
        
            loss_dict['loss_nyud-normal'] = normalErr
            output_dict['angleErr'] = angleMean
    
            normNYUW = opt.normalNYUWeight
            loss_dict['loss_nyud-ALL'] += normNYUW * loss_dict['loss_nyud-normal']

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPred = output_dict['depthPred']
            depthBatch = input_dict['depthBatch']
            segDepthBatch = input_dict['segDepthBatch']

            pixelAllNumDepth = (torch.sum(segDepthBatch ).cpu().data).item()
            depthErr = torch.sum( (torch.log(depthPred + 0.1) - torch.log(depthBatch + 0.1 ) )
                * ( torch.log(depthPred + 0.1) - torch.log(depthBatch + 0.1) ) * segDepthBatch.expand_as(depthBatch ) ) / pixelAllNumDepth

            loss_dict['loss_nyud-depth'] = depthErr

            depthNYUW = opt.depthNYUWeight
            loss_dict['loss_nyud-ALL'] += depthNYUW * loss_dict['loss_nyud-depth']

    return output_dict, loss_dict