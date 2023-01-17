import numpy as np
import torch
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import time
import torchvision.utils as vutils
from utils.loss import hinge_embedding_loss, surface_normal_loss, parameter_loss, \
    class_balanced_cross_entropy_loss
from utils.match_segmentation import MatchSegmentation
from utils.utils_vis import vis_index_map, reindex_output_map
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *

def get_labels_dict_matseg(data_batch, opt):
    input_dict = {}

    # input_dict['im_paths'] = data_batch['image_path']
    input_dict['im_batch_matseg'] = data_batch['im_matseg_transformed_trainval'].cuda(non_blocking=True)

    # if opt.if_hdr_input_matseg:
    #     im_cpu = data_batch['im']
    # else:
    #     im_cpu = data_batch['im_matseg_transformed_trainval']
    # input_dict['im_batch'] = im_cpu.to(opt.device)
    # print('-', im_cpu.dtype, im_cpu.shape, torch.max(im_cpu), torch.min(im_cpu), torch.median(im_cpu))

    input_dict['num_mat_masks_batch'] = data_batch['num_mat_masks'].int()

    input_dict['mat_aggre_map_cpu'] = data_batch['mat_aggre_map'].permute(0, 3, 1, 2) # [b, 1, h, w]
    input_dict['mat_notlight_mask_cpu'] = input_dict['mat_aggre_map_cpu']!=0
    input_dict['mat_notlight_mask_gpu_float'] = input_dict['mat_notlight_mask_cpu'].to(opt.device).float()

    # input_dict['mat_aggre_map_reindex_cpu'] = data_batch['mat_aggre_map_reindex'].permute(0, 3, 1, 2) # [b, 1, h, w]
    # input_dict['mat_aggre_map_reindex_batch'] = Variable(input_dict['mat_aggre_map_reindex_cpu'] ).to(opt.device)
    
    input_dict['instance'] = data_batch['instance'].to(opt.device)
    input_dict['instance_valid'] = data_batch['instance_valid'].to(opt.device)
    input_dict['semantic'] = data_batch['semantic'].to(opt.device)

    return input_dict

def postprocess_matseg(input_dict, output_dict, loss_dict, opt, time_meters):
    logit, embedding = output_dict['logit'], output_dict['embedding']

    # ======= Calculate loss
    loss_all_list, loss_pull_list, loss_push_list, loss_binary_list = [], [], [], []
    # batch_size = input_dict['im_batch_matseg'].size(0)
    batch_size = input_dict['im_trainval_SDR'].size(0)
    for i in range(batch_size):
        # if i == 0 and opt.is_master:
        #     print(embedding[i:i+1][0, :, :2, 0])
        _loss_all, _loss_pull, _loss_push = hinge_embedding_loss(embedding[i:i+1], input_dict['num_mat_masks_batch'][i:i+1],
                                                                input_dict['instance'][i:i+1], opt.device)

        _loss_binary = class_balanced_cross_entropy_loss(logit[i], input_dict['semantic'][i]) * 0.
        _loss_all += _loss_binary
        loss_all_list.append(_loss_all)
        loss_pull_list.append(_loss_pull)
        loss_push_list.append(_loss_push)
        loss_binary_list.append(_loss_binary)

        # print('------', i, _loss_all, _loss_pull, _loss_push, _loss_binary)


    loss_dict['loss_matseg-ALL'] = torch.mean(torch.stack(loss_all_list))
    loss_dict['loss_matseg-pull'] = torch.mean(torch.stack(loss_pull_list))
    loss_dict['loss_matseg-push'] = torch.mean(torch.stack(loss_push_list))
    loss_dict['loss_matseg-binary'] = torch.mean(torch.stack(loss_binary_list))

    #  # meah shift
    # gt_seg = input_dict['mat_aggre_map_reindex_batch']
    # # print(gt_seg.dtype)
    # segmentations, sample_segmentations, centers, sample_probs, sample_gt_segs = \
    #     bin_mean_shift(logit, embedding, gt_seg)
    # # print(segmentations[0].shape, sample_segmentations[0].shape, centers[0].shape, sample_probs[0].shape, sample_gt_segs[0].shape)

    # output_dict.update({'logit_matseg': logit, 'matseg_matseg': embedding})


    return output_dict, loss_dict

def val_epoch_matseg(brdfLoaderVal, model, bin_mean_shift, params_mis):

    writer, logger, opt, tid, loss_meters = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['loss_meters']
    logger.info(red('===Evaluating matseg for %d batches'%len(brdfLoaderVal)))

    model.eval()

    # losses = AverageMeter()
    # losses_pull = AverageMeter()
    # losses_push = AverageMeter()
    # losses_binary = AverageMeter()

    time_meters = {}
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    

    match_segmentation = MatchSegmentation()

    with torch.no_grad():
        for i, data_batch in tqdm(enumerate(brdfLoaderVal)):
            ts_iter_start = time.time()

            input_dict = get_labels_dict_matseg(data_batch, opt)
            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = postprocess_matseg(input_dict, model, opt, time_meters)
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # loss = loss_dict['loss_all']
            
            # ======= update loss
            loss_meters['loss_matseg-ALL'].update(loss_dict_reduced['loss_all'].item())
            loss_meters['loss_matseg-pull'].update(loss_dict_reduced['loss_pull'].item())
            loss_meters['loss_matseg-push'].update(loss_dict_reduced['loss_push'].item())
            loss_meters['loss_matseg-binary'].update(loss_dict_reduced['loss_binary'].item())

            # # ======= visualize clusters
            # if i == 0 and opt.is_master:

            #     b, c, h, w = output_dict['logit'].size()

            #     for j, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):
            #         sample_idx = j

            #         # prob_single = torch.sigmoid(logit_single)
            #         prob_single = input_dict['mat_notlight_mask_cpu'][j].to(opt.device).float()
            #         # fast mean shift
            #         segmentation, sampled_segmentation = bin_mean_shift.test_forward(
            #             prob_single, embedding_single, mask_threshold=0.1)
                    
            #         # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
            #         # we thus use avg_pool_2d to smooth the segmentation results
            #         b = segmentation.t().view(1, -1, h, w)
            #         pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            #         b = pooling_b.view(-1, h*w).t()
            #         segmentation = b

            #         # greedy match of predict segmentation and ground truth segmentation using cross entropy
            #         # to better visualization
            #         gt_plane_num = input_dict['num_mat_masks_batch'][j]
            #         matching = match_segmentation(segmentation, prob_single.view(-1, 1), input_dict['instance'][j], gt_plane_num)

            #         # return cluster results
            #         predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            #         # reindexing to matching gt segmentation for better visualization
            #         matching = matching.cpu().numpy().reshape(-1)
            #         used = set([])
            #         max_index = max(matching) + 1
            #         for i, a in zip(range(len(matching)), matching):
            #             if a in used:
            #                 matching[i] = max_index
            #                 max_index += 1
            #             else:
            #                 used.add(a)
            #         predict_segmentation = matching[predict_segmentation]

            #         # mask out non planar region
            #         predict_segmentation[prob_single.cpu().numpy().reshape(-1) <= 0.1] = opt.invalid_index
            #         predict_segmentation = predict_segmentation.reshape(h, w)

            #         # ===== vis
            #         im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

            #         writer.add_image('VAL_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

            #         mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
            #         matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
            #         writer.add_image('VAL_mat_aggre_map_GT/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

            #         mat_aggre_map_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
            #         matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
            #         writer.add_image('VAL_mat_aggre_map_PRED/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')


            #         mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
            #         writer.add_image('VAL_mat_notlight_mask_GT/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

            #         writer.add_text('VAL_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)

            #         if j > 12:
            #             break

    # if opt.is_master:
    #     writer.add_scalar('loss_eval/loss_all', loss_meters['loss_matseg-ALL'].avg, tid)
    #     writer.add_scalar('loss_eval/loss_pull', loss_meters['loss_matseg-pull'].avg, tid)
    #     writer.add_scalar('loss_eval/loss_push', loss_meters['loss_matseg-push'].avg, tid)
    #     writer.add_scalar('loss_eval/loss_binary', loss_meters['loss_matseg-binary'].avg, tid)
    logger.info('Evaluation timings: ' + time_meters_to_string(time_meters))


def get_labels_dict_matseg_combine(data_batch, opt):
    labels_dict_matseg = get_labels_dict_matseg(data_batch, opt)
    input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf = get_labels_dict_brdf(data_batch, opt)
    input_dict = {**labels_dict_matseg, **labels_dict_brdf}
    input_dict.update({'input_batch_brdf': input_batch_brdf, 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    return input_dict


def val_epoch_combine(brdfLoaderVal, model, bin_mean_shift, params_mis):

    writer, logger, opt, tid, loss_meters = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['loss_meters']
    logger.info(red('===Visualizing for %d batches'%len(brdfLoaderVal)))

    model.eval()

    loss_keys = [
        'loss_matseg-ALL', 
        'loss_matseg-pull', 
        'loss_matseg-push', 
        'loss_matseg-binary', 
        'loss_brdf-albedo', 
        'loss_brdf-normal', 
        'loss_brdf-rough', 
        'loss_brdf-depth', 
        'loss_brdf-ALL', 
    ]
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}

    time_meters = {}
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    



    match_segmentation = MatchSegmentation()

    with torch.no_grad():
        for i, data_batch in tqdm(enumerate(brdfLoaderVal)):
            ts_iter_start = time.time()

            input_dict = get_labels_dict_matseg(data_batch, opt)
            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = postprocess_matseg(input_dict, model, opt, time_meters)
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # loss = loss_dict['loss_all']
            
            # ======= update loss
            for loss_key in loss_dict_reduced:
                loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= visualize clusters
            if i == 0 and opt.is_master:

                b, c, h, w = output_dict['logit'].size()

                for j, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):
                    sample_idx = j

                    # prob_single = torch.sigmoid(logit_single)
                    prob_single = input_dict['mat_notlight_mask_cpu'][j].to(opt.device).float()
                    # fast mean shift
                    segmentation, sampled_segmentation = bin_mean_shift.test_forward(
                        prob_single, embedding_single, mask_threshold=0.1)
                    
                    # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
                    # we thus use avg_pool_2d to smooth the segmentation results
                    b = segmentation.t().view(1, -1, h, w)
                    pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
                    b = pooling_b.view(-1, h*w).t()
                    segmentation = b

                    # greedy match of predict segmentation and ground truth segmentation using cross entropy
                    # to better visualization
                    gt_plane_num = input_dict['num_mat_masks_batch'][j]
                    matching = match_segmentation(segmentation, prob_single.view(-1, 1), input_dict['instance'][j], gt_plane_num)

                    # return cluster results
                    predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

                    # reindexing to matching gt segmentation for better visualization
                    matching = matching.cpu().numpy().reshape(-1)
                    used = set([])
                    max_index = max(matching) + 1
                    for i, a in zip(range(len(matching)), matching):
                        if a in used:
                            matching[i] = max_index
                            max_index += 1
                        else:
                            used.add(a)
                    predict_segmentation = matching[predict_segmentation]

                    # mask out non planar region
                    predict_segmentation[prob_single.cpu().numpy().reshape(-1) <= 0.1] = opt.invalid_index
                    predict_segmentation = predict_segmentation.reshape(h, w)

                    # ===== vis
                    im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

                    writer.add_image('VAL_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

                    mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                    matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                    writer.add_image('VAL_mat_aggre_map_GT/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

                    # mat_aggre_map_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
                    mat_aggre_map_single = predict_segmentation.squeeze()
                    matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                    writer.add_image('VAL_mat_aggre_map_PRED/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')


                    mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
                    writer.add_image('VAL_mat_notlight_mask_GT/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

                    writer.add_text('VAL_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)


    if opt.is_master:
        writer.add_scalar('loss_eval/loss_all', losses.avg, tid)
        writer.add_scalar('loss_eval/loss_pull', losses_pull.avg, tid)
        writer.add_scalar('loss_eval/loss_push', losses_push.avg, tid)
        writer.add_scalar('loss_eval/loss_binary', losses_binary.avg, tid)
        logger.info('Evaluation timings: ' + time_meters_to_string(time_meters))


