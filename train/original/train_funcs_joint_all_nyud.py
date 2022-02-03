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
from utils.utils_vis import vis_index_map, reindex_output_map, vis_disp_colormap, colorize
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *
from utils.utils_semseg import intersectionAndUnionGPU
import torchvision.utils as vutils
import torch.distributed as dist
import cv2
from PIL import Image

from train_funcs_matseg import get_labels_dict_matseg, postprocess_matseg, val_epoch_matseg
from train_funcs_semseg import get_labels_dict_semseg, postprocess_semseg
from train_funcs_brdf import get_labels_dict_brdf, postprocess_brdf
from train_funcs_light import get_labels_dict_light, postprocess_light
from train_funcs_layout_object_emitter import get_labels_dict_layout_emitter, postprocess_layout_object_emitter
from train_funcs_matcls import get_labels_dict_matcls, postprocess_matcls
from train_funcs_detectron import postprocess_detectron, gather_lists
# from utils.comm import synchronize
from train_funcs_nyud import get_labels_dict_nyud, postprocess_nyud

from utils.utils_metrics import compute_errors_depth_nyu
from train_funcs_matcls import getG1IdDict, getRescaledMatFromID
# from pytorch_lightning.metrics import Precision, Recall, F1, Accuracy
from pytorch_lightning.metrics import Accuracy

from icecream import ic
import pickle
import matplotlib.pyplot as plt

from train_funcs_layout_object_emitter import vis_layout_emitter

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog
from utils.utils_dettectron import py_cpu_nms
# from detectron2.utils.visualizer import Visualizer, ColorMode

from contextlib import ExitStack, contextmanager
from skimage.segmentation import mark_boundaries
from skimage.transform import resize as scikit_resize

def get_time_meters_joint_nyud():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_nyud'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_nyud_meters(opt):
    nyud_meters = {}
    normal_mean_meter = AverageMeter('normal_mean_meter')
    normal_median_meter = AverageMeter('normal_median_meter')
    depth_mean_meter = AverageMeter('depth_mean_meter')
    nyud_meters.update({'normal_mean_meter': normal_mean_meter, 'normal_median_meter': normal_median_meter, 'depth_mean_meter': depth_mean_meter})
    return nyud_meters

def get_labels_dict_joint_nyud(data_batch, opt):
    # prepare input_dict from data_batch (from dataloader)
    labels_dict = {'im_fixedscale_SDR': data_batch['im_fixedscale_SDR'].cuda(non_blocking=True), 'batch_idx': data_batch['image_index']}
    if 'im_fixedscale_SDR_next' in data_batch:
        labels_dict['im_fixedscale_SDR_next'] = data_batch['im_fixedscale_SDR_next'].cuda(non_blocking=True)

    input_batch_nyud, labels_dict_nyud = get_labels_dict_nyud(data_batch, opt, return_input_batch_as_list=True)
    labels_dict.update({'input_batch_brdf': torch.cat(input_batch_nyud, dim=1)})
    labels_dict.update(labels_dict_nyud)

    return labels_dict

def forward_joint_nyud(is_train, labels_dict, model, opt, time_meters, if_vis=False, if_loss=True, tid=-1, loss_dict=None):
    # forward model + compute losses

    # Forward model
    output_dict = model(labels_dict, if_has_gt_BRDF=True)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    # Post-processing and computing losses
    if loss_dict is None:
        loss_dict = {}

    output_dict, loss_dict = postprocess_nyud(labels_dict, output_dict, loss_dict, opt, time_meters, tid=tid, if_loss=True)
    time_meters['loss_nyud'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    return output_dict, loss_dict

def val_epoch_joint_nyud(nyud_loader_val, model, params_mis):
    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']

    logger.info(red('=== [nyud] Evaluating for %d batches'%len(nyud_loader_val)))

    model.eval()
    
    loss_keys = [
        'loss_nyud-normal', 
        'loss_nyud-depth', 
        'loss_nyud-ALL', 
    ]
        
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}
    time_meters = get_time_meters_joint_nyud()

    nyud_meters = get_nyud_meters(opt)

    with torch.no_grad():

        nyud_dataset_val = params_mis['nyud_dataset_val']
        count_samples_this_rank = 0

        for batch_id, data_batch in tqdm(enumerate(nyud_loader_val)):

            ts_iter_start = time.time()

            input_dict = get_labels_dict_joint_nyud(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint_nyud(False, input_dict, model, opt, time_meters)

            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            
            # ======= update loss
            if len(loss_dict_reduced.keys()) != 0:
                for loss_key in loss_dict_reduced:
                    loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering normal
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                if opt.cfg.MODEL_BRDF.if_bilateral:
                    # albedoPred_numpy = output_dict['albedoBsPred'].detach().cpu().numpy()
                    assert False
                else:
                    normalPred_numpy = output_dict['normalPred'].detach().cpu().numpy()
                    normalOri_numpy = input_dict['normal_ori_cpu'].numpy()
                    segMaskOri_numpy = input_dict['segMask_ori_cpu'].numpy()
                # print(np.amax(albedoPred_numpy), np.amin(albedoPred_numpy), np.median(albedoPred_numpy), np.mean(albedoPred_numpy))
                assert len(normalPred_numpy) == len(normalOri_numpy) == len(segMaskOri_numpy)

                thetaMean_list = []
                thetaMedian_list = []
                for normalPred_numpy_single, normalOri_numpy_single, segMaskOri_numpy_single in zip(normalPred_numpy, normalOri_numpy, segMaskOri_numpy):
                    if opt.cfg.DATA.if_pad_to_32x:
                        normalPred_numpy_single = normalPred_numpy_single[:, :opt.cfg.DATA.im_height, :]
                    normalPred_numpy_single = np.transpose(normalPred_numpy_single, (1, 2, 0))
                    assert opt.cfg.DATA.nyud.im_height / normalPred_numpy_single.shape[0] == opt.cfg.DATA.nyud.im_width / normalPred_numpy_single.shape[1] 
                    normal = cv2.resize(normalPred_numpy_single, (opt.cfg.DATA.nyud.im_width, opt.cfg.DATA.nyud.im_height), cv2.INTER_LINEAR)
                    normalGt = np.transpose(normalOri_numpy_single, (1, 2, 0))
                    mask = np.transpose(segMaskOri_numpy_single, (1, 2, 0))
                    # ic(normal.shape, normalGt.shape, mask.shape, mask.dtype)

                    mask = np.min(mask[:, :, :], axis=2)
                    assert mask.dtype == np.uint8
                    mask = (mask == 255 )
                    mask = mask.astype(np.float32)[:, :, np.newaxis]
                    normalGt = normalGt.astype(np.float32 )

                    normalGt = normalGt.astype(np.float32 )
                    normalGt = (normalGt - 127.5) / 127.5
                    # normalNorm = np.sqrt(np.sum(normalGt * normalGt, axis=2 ) )[:, :, np.newaxis]

                    normalGt = normalGt / (np.sqrt(np.sum( (normalGt * normalGt ), axis=2 )[:, :, np.newaxis] ) + 1e-6)
                    normal = normal / (np.sqrt(np.sum( (normal * normal ), axis=2 )[:, :, np.newaxis] ) + 1e-6)

                    cosTheta =  np.clip(np.sum(normal * normalGt, axis=2), -1, 1)
                    theta = np.arccos(cosTheta ) / np.pi * 180

                    thetaMean = np.sum(theta * mask[:, :, 0] ) / (np.sum(mask[:, :, 0] ) + 1e-6)
                    thetaMedian = np.median(theta[mask[:, :, 0] != 0]  )
                    # print(thetaMean, thetaMedian, np.sum(mask[:, :, 0] ), mask[:, :, 0].shape)

                    thetaMean_list.append(thetaMean)
                    thetaMedian_list.append(thetaMedian)

                if opt.distributed:
                    thetaMean_list_gathered = gather_lists([thetaMean_list], opt.num_gpus)
                    thetaMean_list_all = [item for sublist in thetaMean_list_gathered for item in sublist]
                    thetaMedian_list_gathered = gather_lists([thetaMedian_list], opt.num_gpus)
                    thetaMedian_list_all = [item for sublist in thetaMedian_list_gathered for item in sublist]
                else:
                    thetaMean_list_all = thetaMean_list
                    thetaMedian_list_all = thetaMedian_list

                if opt.is_master:
                    for thetaMean, thetaMedian in zip(thetaMean_list_all, thetaMedian_list_all):
                        nyud_meters['normal_mean_meter'].update(thetaMean)
                        nyud_meters['normal_median_meter'].update(thetaMedian)

            # ======= Metering depth
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                if opt.cfg.MODEL_BRDF.if_bilateral:
                    # albedoPred_numpy = output_dict['albedoBsPred'].detach().cpu().numpy()
                    assert False
                else:
                    depthPred_numpy = output_dict['depthPred'].detach().cpu().numpy()
                    # print(np.amax(depthPred_numpy), np.amin(depthPred_numpy), np.median(depthPred_numpy), np.mean(depthPred_numpy))
                    depthOri_numpy = input_dict['depth_ori_cpu'].numpy()
                    # print(np.amax(depthOri_numpy), np.amin(depthOri_numpy), np.median(depthOri_numpy), np.mean(depthOri_numpy))
                assert depthPred_numpy.shape[0] == depthOri_numpy.shape[0]

                depthErrAccu_list = []
                for depthPred_numpy_single, depthOri_numpy_single in zip(depthPred_numpy, depthOri_numpy):
                    depthPred_numpy_single = np.transpose(depthPred_numpy_single, (1, 2, 0))
                    assert opt.cfg.DATA.nyud.im_height / depthPred_numpy_single.shape[0] == opt.cfg.DATA.nyud.im_width / depthPred_numpy_single.shape[1] 
                    depthPred_numpy_single = depthPred_numpy_single.squeeze()

                    depth = cv2.resize(depthPred_numpy_single, (opt.cfg.DATA.nyud.im_width, opt.cfg.DATA.nyud.im_height), cv2.INTER_LINEAR)
                    # ic(depth.shape, depthGt.shape)

                    depthGt = depthOri_numpy_single.squeeze()
                    depthMask = np.logical_and(depthGt > 1, depthGt < 10).astype(np.float32)

                    depth = np.log(depth + 1e-20)
                    depthGt = np.log(depthGt + 1e-20)

                    depth = depth - np.mean(depth);
                    depthGt = depthGt - np.mean(depthGt )

                    error = np.sum(np.power(depth - depthGt, 2) * depthMask ) / np.sum(depthMask )
                    depthErrAccu_list.append(np.sqrt(error))

                if opt.distributed:
                    depthErrAccu_list_gathered = gather_lists([depthErrAccu_list], opt.num_gpus)
                    depthErrAccu_list_all = [item for sublist in depthErrAccu_list_gathered for item in sublist]
                else:
                    depthErrAccu_list_all = depthErrAccu_list

                if opt.is_master:
                    for depthErrAccu in depthErrAccu_list_all:
                        nyud_meters['depth_mean_meter'].update(depthErrAccu)

    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('nyud_loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
            logger.info('[nyud] Logged val loss for %s'%loss_key)

        logger.info(red('[nyud] Evaluation timings: ' + time_meters_to_string(time_meters)))
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            writer.add_scalar('VAL/nyud-normal-mean', nyud_meters['normal_mean_meter'].avg, tid)
            writer.add_scalar('VAL/nyud-normal-median', nyud_meters['normal_median_meter'].avg, tid)
            print('[nyud-normal(mean, median)', nyud_meters['normal_mean_meter'].avg, nyud_meters['normal_median_meter'].avg)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            writer.add_scalar('VAL/nyud-depth-mean', nyud_meters['depth_mean_meter'].avg, tid)
            print('[nyud-depth(mean)', nyud_meters['depth_mean_meter'].avg)



def vis_val_epoch_joint_nyud(nyud_loader_val, model, params_mis):

    writer, logger, opt, tid, batch_size = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis']
    logger.info(red('=== [nyud] [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(nyud_loader_val), opt.rank)))

    model.eval()
    opt.if_vis_debug_pac = True

    time_meters = get_time_meters_joint_nyud()

    im_paths_list = []
    imBatch_list = []

    normalBatch_list = []
    depthBatch_list = []
    segNormalBatch_list = []
    segDepthBatch_list = []

    normalPreds_list = []
    depthPreds_list = []
    depthPreds_aligned_list = []

    depthBsPreds_list = []
    depthBsPreds_aligned_list = []

    im_h_resized_to_list, im_w_resized_to_list = [], []
    depth_min_and_scale_list = []

    # ===== Gather vis of N batches
    with torch.no_grad():
        im_single_list = []
        for batch_id, data_batch in enumerate(nyud_loader_val):
            if batch_size*batch_id >= opt.cfg.TEST.vis_max_samples:
                break

            input_dict = get_labels_dict_joint_nyud(data_batch, opt)

            # ======= Forward
            output_dict, _ = forward_joint_nyud(False, input_dict, model, opt, time_meters, if_vis=True)
            
            # ======= Vis imagges

            for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_fixedscale_SDR'], data_batch['image_path'])):
                sample_idx = sample_idx_batch+batch_size*batch_id
                print('[Image] Visualizing %d sample...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break

                im_single = im_single.numpy().squeeze()
                im_single = im_single[:opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width, :]
                im_single_list.append(im_single)

                im_h_resized_to, im_w_resized_to = data_batch['im_h_resized_to'][sample_idx_batch], data_batch['im_w_resized_to'][sample_idx_batch]

                if opt.is_master:
                    writer.add_image('nyud_VAL_im/%d'%(sample_idx), im_single[:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((im_single*255.).astype(np.uint8)).save('{0}/nyud_{1}_im_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx) )

                    writer.add_text('nyud_VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]

            
            # ===== Vis BRDF 1/2
            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                im_paths_list.append(input_dict['im_paths'])
                im_h_resized_to_list.append(data_batch['im_h_resized_to'])
                im_w_resized_to_list.append(data_batch['im_w_resized_to'])

                imBatch_list.append(input_dict['imBatch'])

                n = 0

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalPreds_list.append(output_dict['normalPred'])
                    normalBatch_list.append(input_dict['normalBatch'])
                    segNormalBatch_list.append(input_dict['segNormalBatch'])


                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthPreds_list.append(output_dict['depthPred'])
                    if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                        depthPreds_aligned_list.append(output_dict['depthPred_aligned'])
                    depthBatch_list.append(input_dict['depthBatch'])
                    segDepthBatch_list.append(input_dict['segDepthBatch'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        depthBsPreds_list.append(output_dict['depthBsPred'])
                        if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                            depthBsPreds_aligned_list.append(output_dict['depthBsPred_aligned'])

    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        im_paths_list = flatten_list(im_paths_list)
        im_h_resized_to_list = flatten_list(im_h_resized_to_list)
        im_w_resized_to_list = flatten_list(im_w_resized_to_list)

        # ==== GTs
        if 'no' in opt.cfg.DATA.data_read_list:
            normalBatch_vis = torch.cat(normalBatch_list)
            normal_gt_batch_vis_sdr = (0.5*(normalBatch_vis + 1) ).data
            normal_gt_batch_vis_sdr = normal_gt_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
            seg_normal_gt_batch_vis = torch.cat(segNormalBatch_list)
            # print(seg_normal_gt_batch_vis.shape, seg_normal_gt_batch_vis.dtype, torch.max(seg_normal_gt_batch_vis), torch.min(seg_normal_gt_batch_vis))

        if 'de' in opt.cfg.DATA.data_read_list:
            depthBatch_vis = torch.cat(depthBatch_list)
            depthOut = 1 / torch.clamp(depthBatch_vis + 1, 1e-6, 10) # invert the gt depth just for visualization purposes!
            depth_gt_batch_vis_sdr = depthOut.data
            depth_gt_batch_vis_sdr = depth_gt_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
            seg_depth_gt_batch_vis = torch.cat(segDepthBatch_list)

        print('Saving vis to ', '{0}'.format(opt.summary_vis_path_task, tid))

        imBatch_vis = torch.cat(imBatch_list)
        im_batch_vis_sdr = ( (imBatch_vis)**(1.0/2.2) ).data

        if opt.is_master:
            vutils.save_image(im_batch_vis_sdr,
                '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
            if 'no' in opt.cfg.DATA.data_read_list:
                vutils.save_image(normal_gt_batch_vis_sdr,
                    '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
                normal_gt_batch_vis_sdr_numpy = normal_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
                seg_normal_gt_batch_vis_numpy = seg_normal_gt_batch_vis.cpu().numpy().transpose(0, 2, 3, 1)
            if 'de' in opt.cfg.DATA.data_read_list:
                vutils.save_image(depth_gt_batch_vis_sdr,
                    '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )
                depth_gt_batch_vis_sdr_numpy = depth_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
                seg_depth_gt_batch_vis_numpy = seg_depth_gt_batch_vis.cpu().numpy().transpose(0, 2, 3, 1)

        if not opt.cfg.DATASET.if_no_gt_BRDF and opt.is_master:
            for sample_idx in range(im_batch_vis_sdr.shape[0]):
                if 'no' in opt.cfg.DATA.data_read_list:
                    writer.add_image('nyud_VAL_brdf-normal_GT/%d'%sample_idx, normal_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    writer.add_image('nyud_VAL_brdf-seg_normal_GT/%d'%sample_idx, seg_normal_gt_batch_vis_numpy[sample_idx].squeeze(), tid, dataformats='HW')
                if 'de' in opt.cfg.DATA.data_read_list:
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)
                    depth_min_and_scale_list.append(depth_min_and_scale)
                    writer.add_image('nyud_VAL_brdf-depth_GT/%d'%sample_idx, depth_normalized, tid, dataformats='HWC')
                    writer.add_image('nyud_VAL_brdf-seg_depth_GT/%d'%sample_idx, seg_depth_gt_batch_vis_numpy[sample_idx].squeeze(), tid, dataformats='HW')

        # ==== Preds
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalPreds_vis = torch.cat(normalPreds_list)
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            normal_pred_batch_vis_sdr = normal_pred_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
            if opt.is_master:
                vutils.save_image(normal_pred_batch_vis_sdr,
                    '{0}/nyud_{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPreds_vis = torch.cat(depthPreds_list)
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)
            depth_pred_batch_vis_sdr = depthOut.data
            depth_pred_batch_vis_sdr = depth_pred_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
            if opt.is_master:
                vutils.save_image(depth_pred_batch_vis_sdr,
                    '{0}/nyud_{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

            if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                depthPreds_aligned_vis = torch.cat(depthPreds_aligned_list)
                depthOut = 1 / torch.clamp(depthPreds_aligned_vis + 1, 1e-6, 10)
                depth_pred_aligned_batch_vis_sdr = depthOut.data
                depth_pred_aligned_batch_vis_sdr = depth_pred_aligned_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
                if opt.is_master:
                    vutils.save_image(depth_pred_aligned_batch_vis_sdr,
                        '{0}/nyud_{1}_depthPred_aligned_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                depth_pred_aligned_batch_vis_sdr_numpy = depth_pred_aligned_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

            if opt.cfg.MODEL_BRDF.if_bilateral:
                depthBsPreds_vis = torch.cat(depthBsPreds_list)
                depthOut = 1 / torch.clamp(depthBsPreds_vis + 1, 1e-6, 10)
                depth_bs_pred_batch_vis_sdr = depthOut.data
                depth_bs_pred_batch_vis_sdr = depth_bs_pred_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
                if opt.is_master:
                    vutils.save_image(depth_bs_pred_batch_vis_sdr,
                        '{0}/nyud_{1}_depthBsPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                depth_bs_pred_batch_vis_sdr_numpy = depth_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

                if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                    depthBsPreds_aligned_vis = torch.cat(depthBsPreds_aligned_list)
                    depthOut = 1 / torch.clamp(depthBsPreds_aligned_vis + 1, 1e-6, 10)
                    depth_bs_pred_aligned_batch_vis_sdr = depthOut.data
                    depth_bs_pred_aligned_batch_vis_sdr = depth_bs_pred_aligned_batch_vis_sdr[:, :, :opt.cfg.DATA.im_height, :opt.cfg.DATA.im_width]
                    if opt.is_master:
                        vutils.save_image(depth_bs_pred_aligned_batch_vis_sdr,
                            '{0}/nyud_{1}_depthBsPred_aligned_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                    depth_bs_pred_aligned_batch_vis_sdr_numpy = depth_bs_pred_aligned_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

        if opt.is_master:
            for sample_idx in tqdm(range(im_batch_vis_sdr.shape[0])):
                im_h_resized_to = im_h_resized_to_list[sample_idx]
                im_w_resized_to = im_w_resized_to_list[sample_idx]

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('nyud_VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/nyud_{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_not_normalized_pred = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()[:im_h_resized_to, :im_w_resized_to], normalize=True)[0]
                    writer.add_image('nyud_VAL_brdf-depth_PRED/%d'%sample_idx, depth_not_normalized_pred, tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((depth_not_normalized_pred).astype(np.uint8)).save('{0}/nyud_{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                        depth_normalized_pred, _ = vis_disp_colormap(depth_pred_aligned_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])
                        writer.add_image('nyud_brdf-depth_syncScale_PRED/%d'%sample_idx, depth_normalized_pred, tid, dataformats='HWC')
                        if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                            Image.fromarray((depth_normalized_pred).astype(np.uint8)).save('{0}/nyud_{1}_depthPred_aligned_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        depth_not_normalized_bs_pred = vis_disp_colormap(depth_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze()[:im_h_resized_to, :im_w_resized_to], normalize=True)[0]
                        writer.add_image('nyud_VAL_brdf-depth_PRED-BS/%d'%sample_idx, depth_not_normalized_bs_pred, tid, dataformats='HWC')
                        if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                            Image.fromarray((depth_not_normalized_bs_pred).astype(np.uint8)).save('{0}/nyud_{1}_depthBsPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                            depth_normalized_bs_pred, _ = vis_disp_colormap(depth_bs_pred_aligned_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])
                            writer.add_image('nyud_VAL_brdf-depth_syncScale_PRED-BS/%d'%sample_idx, depth_normalized_bs_pred, tid, dataformats='HWC')
                            if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                                Image.fromarray((depth_normalized_bs_pred).astype(np.uint8)).save('{0}/nyud_{1}_depthBsPred_aligned_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))