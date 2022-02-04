import numpy as np
import torch
from tqdm import tqdm
import time
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *
from utils.utils_vis import vis_disp_colormap
import torchvision.utils as vutils
import cv2
from PIL import Image, ImageOps
import scipy.ndimage as ndimage

from train_funcs_brdf import get_labels_dict_brdf, postprocess_brdf
from train_funcs_light import get_labels_dict_light, postprocess_light


from utils.utils_metrics import compute_errors_depth_nyu

from icecream import ic
import pickle
import matplotlib.pyplot as plt

def get_time_meters_joint():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_brdf'] = AverageMeter()
    time_meters['loss_light'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_brdf_meters(opt):
    brdf_meters = {}
    if 'no' in opt.cfg.MODEL_BRDF.enable_list:
        normal_mean_error_meter = AverageMeter('normal_mean_error_meter')
        normal_median_error_meter = AverageMeter('normal_median_error_meter')
        # inv_depth_mean_error_meter = AverageMeter('inv_depth_mean_error_meter')
        # inv_depth_median_error_meter = AverageMeter('inv_depth_median_error_meter')
        brdf_meters.update({'normal_mean_error_meter': normal_mean_error_meter, 'normal_median_error_meter': normal_median_error_meter})
    if 'de' in opt.cfg.MODEL_BRDF.enable_list:
        brdf_meters.update(get_depth_meters(opt))
    return brdf_meters

def get_depth_meters(opt):
    return {metric: AverageMeter(metric) for metric in opt.depth_metrics}

def get_light_meters(opt):
    light_meters = {}
    return light_meters

def get_labels_dict_joint(data_batch, opt):

    # prepare input_dict from data_batch (from dataloader)
    labels_dict = {'im_trainval_SDR': data_batch['im_trainval_SDR'].cuda(non_blocking=True), 'im_fixedscale_SDR': data_batch['im_fixedscale_SDR'].cuda(non_blocking=True), 'batch_idx': data_batch['image_index']}
    if 'im_fixedscale_SDR_next' in data_batch:
        labels_dict['im_fixedscale_SDR_next'] = data_batch['im_fixedscale_SDR_next'].cuda(non_blocking=True)

    input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf = get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=True)
    list_from_brdf = [input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf]
    labels_dict.update({'input_batch_brdf': torch.cat(input_batch_brdf, dim=1), 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    labels_dict.update(labels_dict_brdf)

    if opt.cfg.DATA.load_light_gt:
        input_batch_light, labels_dict_light, pre_batch_dict_light, extra_dict_light = get_labels_dict_light(data_batch, opt, list_from_brdf=list_from_brdf, return_input_batch_as_list=True)
        labels_dict.update({'input_batch_light': torch.cat(input_batch_light, dim=1), 'pre_batch_dict_light': pre_batch_dict_light})
    else:
        labels_dict_light = {}
        extra_dict_light = {}
    labels_dict.update(labels_dict_light)
    labels_dict.update(extra_dict_light)

    return labels_dict

def forward_joint(is_train, labels_dict, model, opt, time_meters, if_vis=False, if_loss=True, tid=-1, loss_dict=None):
    # forward model + compute losses

    # Forward model
    # c = time.time()
    output_dict = model(labels_dict)
    # print(time.time() - c)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    # Post-processing and computing losses
    if loss_dict is None:
        loss_dict = {}

    if opt.cfg.MODEL_BRDF.enable:
        if_loss_brdf = if_loss and opt.cfg.DATA.load_brdf_gt and (not opt.cfg.DATASET.if_no_gt_BRDF)
        output_dict, loss_dict = postprocess_brdf(labels_dict, output_dict, loss_dict, opt, time_meters, tid=tid, if_loss=if_loss_brdf)
        time_meters['loss_brdf'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_LIGHT.enable:
        output_dict, loss_dict = postprocess_light(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_light'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()


    return output_dict, loss_dict

def val_epoch_joint(brdf_loader_val, model, params_mis):
    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    ENABLE_BRDF = opt.cfg.MODEL_BRDF.enable and opt.cfg.DATA.load_brdf_gt
    ENABLE_LIGHT = opt.cfg.MODEL_LIGHT.enable

    logger.info(red('===Evaluating for %d batches'%len(brdf_loader_val)))

    model.eval()
    
    loss_keys = []

    if opt.cfg.MODEL_BRDF.enable:
        if opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
            loss_keys += ['loss_brdf-ALL', ]
            if 'al' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-albedo', ]
                if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_albedo:
                    loss_keys += ['loss_brdf-albedo-reg']
            if 'no' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-normal', ]
            if 'ro' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-rough', 'loss_brdf-rough-paper', ]
            if 'de' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-depth', 'loss_brdf-depth-paper']
                if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_depth:
                    loss_keys += ['loss_brdf-depth-reg']

            if opt.cfg.MODEL_BRDF.if_bilateral:
                loss_keys += [
                    'loss_brdf-albedo-bs', ]
                if not opt.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    loss_keys += [
                        # 'loss_brdf-normal-bs', 
                        'loss_brdf-rough-bs', 
                        'loss_brdf-depth-bs', 
                        'loss_brdf-rough-bs-paper', 
                        'loss_brdf-depth-bs-paper', 
                    ]


    if opt.cfg.MODEL_LIGHT.enable:
        loss_keys += [
            'loss_light-ALL', 
            'loss_light-reconstErr', 
            'loss_light-renderErr', 
        ]
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}
    time_meters = get_time_meters_joint()
    if ENABLE_BRDF:
        brdf_meters = get_brdf_meters(opt)
    if ENABLE_LIGHT:
        light_meters = get_light_meters(opt)

    with torch.no_grad():
        brdf_dataset_val = params_mis['brdf_dataset_val']
        count_samples_this_rank = 0

        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):

            ts_iter_start = time.time()

            input_dict = get_labels_dict_joint(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint(False, input_dict, model, opt, time_meters)

            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))

            # ======= update loss
            if len(loss_dict_reduced.keys()) != 0:
                for loss_key in loss_dict_reduced:
                    loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_LIGHT:
                pass

            if ENABLE_BRDF:
                frame_info_list = input_dict['frame_info']

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_input = input_dict['depthBatch'].detach().cpu().numpy()
                    depth_output = output_dict['depthPred'].detach().cpu().numpy()
                    seg_obj = data_batch['segObj'].cpu().numpy()
                    min_depth, max_depth = 0.1, 8.
                    depth_output = depth_output * seg_obj
                    depth_input = depth_input * seg_obj

                    depth_input[depth_input < min_depth] = min_depth
                    depth_output[depth_output < min_depth] = min_depth
                    depth_input[depth_input > max_depth] = max_depth
                    depth_output[depth_output > max_depth] = max_depth

                    for depth_input_single, depth_output_single in zip(depth_input.squeeze(), depth_output.squeeze()):
                        metrics_results = compute_errors_depth_nyu(depth_input_single, depth_output_single)
                        for metric in metrics_results:
                            brdf_meters[metric].update(metrics_results[metric])

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normal_input = input_dict['normalBatch'].detach().cpu().numpy()
                    normal_output = output_dict['normalPred'].detach().cpu().numpy()
                    normal_input_Nx3 = np.transpose(normal_input, (0, 2, 3, 1)).reshape(-1, 3)
                    normal_output_Nx3 = np.transpose(normal_output, (0, 2, 3, 1)).reshape(-1, 3)
                    normal_in_n_out_dot = np.sum(np.multiply(normal_input_Nx3, normal_output_Nx3), 1)
                    normal_error = normal_in_n_out_dot / (np.linalg.norm(normal_input_Nx3, axis=1) * np.linalg.norm(normal_output_Nx3, axis=1) + 1e-6)
                    normal_error = np.arccos(normal_error) / np.pi * 180.
                    brdf_meters['normal_mean_error_meter'].update(np.mean(normal_error))
                    brdf_meters['normal_median_error_meter'].update(np.median(normal_error))

    # ======= Metering
        
    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
            logger.info('Logged val loss for %s:%.6f'%(loss_key, loss_meters[loss_key].avg))


        if ENABLE_BRDF:
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                for metric in opt.depth_metrics:
                    writer.add_scalar('VAL/BRDF-depth_%s'%metric, brdf_meters[metric].avg, tid)
                logger.info('Val result - depth: ' + ', '.join(['%s: %.4f'%(metric, brdf_meters[metric].avg) for metric in opt.depth_metrics]))
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                writer.add_scalar('VAL/BRDF-normal_mean_val', brdf_meters['normal_mean_error_meter'].avg, tid)
                writer.add_scalar('VAL/BRDF-normal_median_val', brdf_meters['normal_median_error_meter'].get_median(), tid)
                logger.info('Val result - normal: mean: %.4f, median: %.4f'%(brdf_meters['normal_mean_error_meter'].avg, brdf_meters['normal_median_error_meter'].get_median()))

    # synchronize()
    logger.info(red('Evaluation timings: ' + time_meters_to_string(time_meters)))


def vis_val_epoch_joint(brdf_loader_val, model, params_mis):

    writer, logger, opt, tid, batch_size = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis']
    logger.info(red('=== [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()

    time_meters = get_time_meters_joint()

    im_paths_list = []
    albedoBatch_list = []
    normalBatch_list = []
    roughBatch_list = []
    depthBatch_list = []
    imBatch_list = []
    imBatch_vis_list = []
    segAllBatch_list = []
    segBRDFBatch_list = []
    
    im_w_resized_to_list = []
    im_h_resized_to_list = []

    if opt.cfg.MODEL_BRDF.enable:

        diffusePreBatch_list = []
        specularPreBatch_list = []
        renderedImBatch_list = []
        
        albedoPreds_list = []
        albedoPreds_aligned_list = []
        normalPreds_list = []
        roughPreds_list = []
        depthPreds_list = []
        depthPreds_aligned_list = []

        albedoBsPreds_list = []
        albedoBsPreds_aligned_list = []
        roughBsPreds_list = []
        depthBsPreds_list = []
        depthBsPreds_aligned_list = []


    # ===== Gather vis of N batches

    with torch.no_grad():
        im_single_list = []
        real_sample_results_path_list = []
        im_h_w_list = []
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            if batch_size*batch_id >= opt.cfg.TEST.vis_max_samples:
                break


            input_dict = get_labels_dict_joint(data_batch, opt)

            # ======= Forward
            output_dict, _ = forward_joint(False, input_dict, model, opt, time_meters, if_vis=True, if_loss=False)
            # print(output_dict.keys(), output_dict['extra_output_dict'].keys()) # dict_keys(['al-attn_matrix_encoder_0', 'al-attn_matrix_encoder_1', 'al-attn_matrix_encoder_2', 'al-attn_matrix_encoder_3', 'al-attn_matrix_encoder_4', 'al-attn_matrix_encoder_5', 'al-attn_matrix_decoder_0', 'al-attn_matrix_decoder_1', 'al-attn_matrix_decoder_2', 'al-attn_matrix_decoder_3', 'al-attn_matrix_decoder_4', 'al-attn_matrix_decoder_5'])

            # ======= Vis imagges
            for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_fixedscale_SDR'], data_batch['image_path'])):
                sample_idx = sample_idx_batch+batch_size*batch_id
                print('[Image] Visualizing %d sample...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break
                
                im_h_resized_to_batch, im_w_resized_to_batch = data_batch['im_h_resized_to'], data_batch['im_w_resized_to']
                im_h_resized_to_list.append(im_h_resized_to_batch)
                im_w_resized_to_list.append(im_w_resized_to_batch)
                if opt.cfg.DEBUG.if_test_real:
                    real_sample_name = im_path.split('/')[-2]
                    real_sample_results_path = Path(opt.summary_vis_path_task) / real_sample_name
                    real_sample_results_path.mkdir(parents=True, exist_ok=False)
                    real_sample_results_path_list.append([real_sample_results_path, (im_h_resized_to_batch, im_w_resized_to_batch)])

                im_single = im_single.numpy().squeeze()
                im_single_list.append(im_single)
                if opt.is_master:
                    writer.add_image('VAL_im/%d'%(sample_idx), im_single, tid, dataformats='HWC')
                    writer.add_image('VAL_im_cropped/%d'%(sample_idx), im_single[:im_h_resized_to_batch[sample_idx_batch], :im_w_resized_to_batch[sample_idx_batch]], tid, dataformats='HWC')
                    writer.add_image('VAL_pad_mask/%d'%(sample_idx), data_batch['pad_mask'][sample_idx_batch]*255, tid, dataformats='HW')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((im_single*255.).astype(np.uint8)).save('{0}/{1}_im_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx) )

                    writer.add_text('VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]
                    # print(sample_idx, im_path)

                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_im_path = real_sample_results_path / 'im_.png'
                        assert len(im_h_resized_to_batch) == 1
                        im_h_resized_to, im_w_resized_to = im_h_resized_to_batch[0], im_w_resized_to_batch[0]
                        im_ = Image.fromarray((im_single*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        im_ = im_.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        im_.save(str(real_sample_im_path))

            # ===== Vis BRDF 1/2
            # print(((input_dict['albedoBatch'] ) ** (1.0/2.2) ).data.shape) # [b, 3, h, w]
            im_paths_list.append(input_dict['im_paths'])
            imBatch_list.append(input_dict['imBatch'])
            imBatch_vis_list.append(data_batch['im_fixedscale_SDR'])
            segAllBatch_list.append(input_dict['segAllBatch'])
            segBRDFBatch_list.append(input_dict['segBRDFBatch'])
            if 'al' in opt.cfg.DATA.data_read_list:
                albedoBatch_list.append(input_dict['albedoBatch'])
            if 'no' in opt.cfg.DATA.data_read_list:
                normalBatch_list.append(input_dict['normalBatch'])
            if 'ro' in opt.cfg.DATA.data_read_list:
                roughBatch_list.append(input_dict['roughBatch'])
            if 'de' in opt.cfg.DATA.data_read_list:
                depthBatch_list.append(input_dict['depthBatch'])

            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if opt.cascadeLevel > 0:
                    diffusePreBatch_list.append(input_dict['pre_batch_dict_brdf']['diffusePreBatch'])
                    specularPreBatch_list.append(input_dict['pre_batch_dict_brdf']['specularPreBatch'])
                    renderedImBatch_list.append(input_dict['pre_batch_dict_brdf']['renderedImBatch'])
                n = 0

                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    # if (not opt.cfg.DATASET.if_no_gt_semantics):
                    # albedoPreds_list.append(output_dict['albedoPreds'][n])
                    albedoPreds_list.append(output_dict['albedoPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        albedoBsPreds_list.append(output_dict['albedoBsPred'])
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                        albedoPreds_aligned_list.append(output_dict['albedoPred_aligned'])

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalPreds_list.append(output_dict['normalPred'])

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    roughPreds_list.append(output_dict['roughPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        roughBsPreds_list.append(output_dict['roughBsPred'])

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthPreds_list.append(output_dict['depthPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        depthBsPreds_list.append(output_dict['depthBsPred'])
                    if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                        depthPreds_aligned_list.append(output_dict['depthPred_aligned'])
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            depthBsPreds_aligned_list.append(output_dict['depthBsPred_aligned'])

            # ===== LIGHT
            if opt.cfg.MODEL_LIGHT.enable:
                envmapsPredImage = output_dict['envmapsPredImage'].detach().cpu().numpy()
                if not opt.cfg.DATASET.if_no_gt_light:
                    envmapsPredScaledImage = output_dict['envmapsPredScaledImage'].detach().cpu().numpy()
                    envmapsBatch = input_dict['envmapsBatch'].detach().cpu().numpy()
                else:
                    envmapsPredScaledImage = [None] * envmapsPredImage.shape[0]
                    envmapsBatch = [None] * envmapsPredImage.shape[0]
                renderedImPred = output_dict['renderedImPred'].detach().cpu().numpy()
                renderedImPred_sdr = output_dict['renderedImPred_sdr'].detach().cpu().numpy()
                imBatchSmall = output_dict['imBatchSmall'].detach().cpu().numpy()
                if not opt.cfg.DATASET.if_no_gt_light:
                    segEnvBatch = output_dict['segEnvBatch'].detach().cpu().numpy() # (4, 1, 120, 160, 1, 1)
                    reconstErr_loss_map_batch = output_dict['reconstErr_loss_map'].detach().cpu().numpy() # [4, 3, 120, 160, 8, 16]
                    reconstErr_loss_map_2D_batch = reconstErr_loss_map_batch.mean(-1).mean(-1).mean(1)

                # print(reconstErr_loss_map_2D_batch.shape, np.amax(reconstErr_loss_map_2D_batch), np.amin(reconstErr_loss_map_2D_batch),np.median(reconstErr_loss_map_2D_batch)) # (4, 120, 160) 3.9108467 0.0 0.22725311

                for sample_idx_batch in range(envmapsPredImage.shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    # assert envmapsPredScaledImage.shape[0] == batch_size
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_env_path = real_sample_results_path / 'env.npz'
                        real_sample_env_path_hdr = real_sample_results_path / 'env.hdr'
                        env_save_half = envmapsPredImage[sample_idx_batch].transpose(1, 2, 3, 4, 0) # -> (120, 160, 8, 16, 3); >>> s = 'third_parties_outside/VirtualObjectInsertion/data_/Example1/env.npz' ['env']: (106, 160, 8, 16, 3)
                        env_save_full = np.zeros((env_save_half.shape[0]*2, env_save_half.shape[1]*2, env_save_half.shape[2], env_save_half.shape[3], env_save_half.shape[4]), dtype=env_save_half.dtype) # (106x2, 160x2, 8, 16, 3) 

                        # Flip to be conincide with our dataset [ Rui: important...... to fix the blue-ish hue of inserted objects]
                        np.savez_compressed(real_sample_env_path,
                            env = np.ascontiguousarray(env_save_half[:, :, :, :, ::-1] ) )
                        # writeEnvToFile(output_dict['envmapsPredImage'][sample_idx_batch], 0, real_sample_env_path_hdr, nrows=24, ncols=16 )

                        I_hdr =envmapsPredImage[sample_idx_batch] * 1000.
                        H_grid, W_grid, h, w = I_hdr.shape[1:]
                        downsize_ratio = 4
                        H_grid_after = H_grid // 4 * 4
                        W_grid_after = W_grid // 4 * 4
                        I_hdr_after = I_hdr[:, :H_grid_after, :W_grid_after, :, :]
                        xx, yy = np.meshgrid(np.arange(0, H_grid_after, downsize_ratio), np.arange(0, W_grid_after, downsize_ratio))
                        I_hdr_downsampled = I_hdr_after[:, xx.T, yy.T, :, :]
                        I_hdr_downsampled = I_hdr_downsampled.transpose(1, 3, 2, 4, 0).reshape(H_grid_after*h//downsize_ratio, W_grid_after*w//downsize_ratio, 3)
                        if opt.is_master:
                            cv2.imwrite('{0}/{1}-{2}_{3}.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred') , I_hdr_downsampled[:, :, [2, 1, 0]])
                            if opt.cfg.DEBUG.if_dump_full_envmap:
                                # cv2.imwrite('{0}/{1}-{2}_{3}_ori.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred') , )
                                with open('{0}/{1}-{2}_{3}_ori.pickle'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred'),"wb") as f:
                                    pickle.dump({'env': I_hdr[[2, 1, 0], :, :, :, :]}, f)
                    else:
                        for I_hdr, name_tag in zip([envmapsPredImage[sample_idx_batch], envmapsPredScaledImage[sample_idx_batch], envmapsBatch[sample_idx_batch]], ['light_Pred', 'light_Pred_Scaled', 'light_GT']):
                            if I_hdr is None:
                                continue
                            H_grid, W_grid, h, w = I_hdr.shape[1:]
                            downsize_ratio = 4
                            assert H_grid % downsize_ratio == 0
                            assert W_grid % downsize_ratio == 0
                            xx, yy = np.meshgrid(np.arange(0, H_grid, downsize_ratio), np.arange(0, W_grid, downsize_ratio))
                            I_hdr_downsampled = I_hdr[:, xx.T, yy.T, :, :]
                            I_hdr_downsampled = I_hdr_downsampled.transpose(1, 3, 2, 4, 0).reshape(H_grid*h//downsize_ratio, W_grid*w//downsize_ratio, 3)
                            if opt.is_master:
                                cv2.imwrite('{0}/{1}-{2}_{3}.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag) , I_hdr_downsampled[:, :, [2, 1, 0]])
                                with open('{0}/{1}-{2}_{3}_ori.pickle'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag),"wb") as f:
                                    pickle.dump({'env': I_hdr[[2, 1, 0], :, :, :, :]}, f)

                    for I_png, name_tag in zip([renderedImPred[sample_idx_batch], renderedImPred_sdr[sample_idx_batch], imBatchSmall[sample_idx_batch], imBatchSmall[sample_idx_batch]**(1./2.2)], ['renderedImPred', 'renderedImPred_sdr', 'imBatchSmall_GT', 'imBatchSmall_GT_sdr']):
                        I_png = np.clip(I_png, 0., 1.)
                        I_png = (I_png.transpose(1, 2, 0) * 255.).astype(np.uint8)
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%(name_tag, sample_idx), I_png, tid, dataformats='HWC')
                        Image.fromarray(I_png).save('{0}/{1}-{2}_light-{3}.png'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag))

                    if not opt.cfg.DATASET.if_no_gt_light:
                        segEnv = segEnvBatch[sample_idx_batch].squeeze()
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%('segEnv_mask', sample_idx), segEnv, tid, dataformats='HW')

                        reconstErr_loss_map_2D = reconstErr_loss_map_2D_batch[sample_idx_batch].squeeze()
                        reconstErr_loss_map_2D = reconstErr_loss_map_2D / np.amax(reconstErr_loss_map_2D)
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%('reconstErr_loss_map_2D', sample_idx), reconstErr_loss_map_2D, tid, dataformats='HW')
        

    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    im_paths_list = flatten_list(im_paths_list)
    im_h_resized_to_list = flatten_list(im_h_resized_to_list)
    im_w_resized_to_list = flatten_list(im_w_resized_to_list)
    

    if 'al' in opt.cfg.DATA.data_read_list:
        albedoBatch_vis = torch.cat(albedoBatch_list)
    if 'no' in opt.cfg.DATA.data_read_list:
        normalBatch_vis = torch.cat(normalBatch_list)
    if 'ro' in opt.cfg.DATA.data_read_list:
        roughBatch_vis = torch.cat(roughBatch_list)
    if 'de' in opt.cfg.DATA.data_read_list:
        depthBatch_vis = torch.cat(depthBatch_list)

    imBatch_vis = torch.cat(imBatch_vis_list)
    segAllBatch_vis = torch.cat(segAllBatch_list)
    segBRDFBatch_vis = torch.cat(segBRDFBatch_list)

    print('Saving vis to ', '{0}'.format(opt.summary_vis_path_task, tid))
    # Save the ground truth and the input
    im_batch_vis_sdr = (imBatch_vis ).data.permute(0, 3, 1, 2)

    ## ---- GTs
    # if (not opt.cfg.DATASET.if_no_gt_semantics):
    if 'al' in opt.cfg.DATA.data_read_list:
        albedo_gt_batch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data
    if 'no' in opt.cfg.DATA.data_read_list:
        normal_gt_batch_vis_sdr = (0.5*(normalBatch_vis + 1) ).data
    if 'ro' in opt.cfg.DATA.data_read_list:
        rough_gt_batch_vis_sdr = (0.5*(roughBatch_vis + 1) ).data
    if 'de' in opt.cfg.DATA.data_read_list:
        depthOut = 1 / torch.clamp(depthBatch_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBatch_vis) # invert the gt depth just for visualization purposes!
        depth_gt_batch_vis_sdr = ( depthOut*segAllBatch_vis.expand_as(depthBatch_vis) ).data
    
    if opt.is_master:
        vutils.save_image(im_batch_vis_sdr,
                '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
        if not opt.cfg.DATASET.if_no_gt_BRDF:
            if 'al' in opt.cfg.DATA.data_read_list:
                vutils.save_image(albedo_gt_batch_vis_sdr,
                    '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'no' in opt.cfg.DATA.data_read_list:
                vutils.save_image(normal_gt_batch_vis_sdr,
                    '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'ro' in opt.cfg.DATA.data_read_list:
                vutils.save_image(rough_gt_batch_vis_sdr,
                    '{0}/{1}_roughGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'de' in opt.cfg.DATA.data_read_list:
                vutils.save_image(depth_gt_batch_vis_sdr,
                    '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )

    if 'al' in opt.cfg.DATA.data_read_list:
        albedo_gt_batch_vis_sdr_numpy = albedo_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'no' in opt.cfg.DATA.data_read_list:
        normal_gt_batch_vis_sdr_numpy = normal_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'ro' in opt.cfg.DATA.data_read_list:
        rough_gt_batch_vis_sdr_numpy = rough_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'de' in opt.cfg.DATA.data_read_list:
        depth_gt_batch_vis_sdr_numpy = depth_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    depth_min_and_scale_list = []
    segAll_list = []
    if not opt.cfg.DATASET.if_no_gt_BRDF and opt.is_master:
        for sample_idx in range(im_batch_vis_sdr.shape[0]):
            writer.add_image('VAL_brdf-segBRDF_GT/%d'%sample_idx, segBRDFBatch_vis[sample_idx].cpu().detach().numpy().squeeze(), tid, dataformats='HW')
            segAll = segAllBatch_vis[sample_idx].cpu().detach().numpy().squeeze()
            segAll = segAll.squeeze()
            segAll = ndimage.binary_erosion(segAll, structure=np.ones((7, 7) ),
                    border_value=1)
            segAll_list.append(segAll)

            writer.add_image('VAL_brdf-segAll_GT/%d'%sample_idx, segAll, tid, dataformats='HW')
            if 'al' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-albedo_GT/%d'%sample_idx, albedo_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'no' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-normal_GT/%d'%sample_idx, normal_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'ro' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-rough_GT/%d'%sample_idx, rough_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'de' in opt.cfg.DATA.data_read_list:
                depth_normalized, depth_min_and_scale = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, valid_mask=segAll==1)
                depth_min_and_scale_list.append(depth_min_and_scale)
                writer.add_image('VAL_brdf-depth_GT/%d'%sample_idx, depth_normalized, tid, dataformats='HWC')


    ## ---- ESTs
    # if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
    if opt.cfg.MODEL_BRDF.enable:
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedoPreds_vis = torch.cat(albedoPreds_list)
            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedoPreds_aligned_vis = torch.cat(albedoPreds_aligned_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedoBsPreds_vis = torch.cat(albedoBsPreds_list)

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalPreds_vis = torch.cat(normalPreds_list)

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            roughPreds_vis = torch.cat(roughPreds_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                roughBsPreds_vis = torch.cat(roughBsPreds_list)

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPreds_vis = torch.cat(depthPreds_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depthBsPreds_vis = torch.cat(depthBsPreds_list)
            if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                # use aligned
                depthPreds_vis = torch.cat(depthPreds_aligned_list)
                if opt.cfg.MODEL_BRDF.if_bilateral:
                    depthBsPreds_vis = torch.cat(depthBsPreds_aligned_list)

        if opt.cascadeLevel > 0:
            diffusePreBatch_vis = torch.cat(diffusePreBatch_list)
            specularPreBatch_vis = torch.cat(specularPreBatch_list)
            renderedImBatch_vis = torch.cat(renderedImBatch_list)

        if opt.cascadeLevel > 0 and opt.is_master:
            vutils.save_image( ( (diffusePreBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_diffusePre.png'.format(opt.summary_vis_path_task, tid) )
            vutils.save_image( ( (specularPreBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_specularPre.png'.format(opt.summary_vis_path_task, tid) )
            vutils.save_image( ( (renderedImBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_renderedImage.png'.format(opt.summary_vis_path_task, tid) )

        # ==== Preds
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr = ( (albedoPreds_vis ) ** (1.0/2.2) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedo_bs_pred_batch_vis_sdr = ( (albedoBsPreds_vis ) ** (1.0/2.2) ).data

            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedo_pred_aligned_batch_vis_sdr = ( (albedoPreds_aligned_vis ) ** (1.0/2.2) ).data
            if opt.is_master:
                vutils.save_image(albedo_pred_batch_vis_sdr,
                        '{0}/{1}_albedoPred.png'.format(opt.summary_vis_path_task, tid) )

                if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                    vutils.save_image(albedo_pred_aligned_batch_vis_sdr,
                            '{0}/{1}_albedoPred_aligned.png'.format(opt.summary_vis_path_task, tid) )

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(normal_pred_batch_vis_sdr,
                        '{0}/{1}_normalPred.png'.format(opt.summary_vis_path_task, tid) )

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                rough_bs_pred_batch_vis_sdr = ( 0.5*(roughBsPreds_vis + 1)).data
            if opt.is_master:
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/{1}_roughPred.png'.format(opt.summary_vis_path_task, tid) )

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
            depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depthBsOut = 1 / torch.clamp(depthBsPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBsPreds_vis)
                depth_bs_pred_batch_vis_sdr = ( depthBsOut * segAllBatch_vis.expand_as(depthBsPreds_vis) ).data

            depthOut_colored_single_numpy_list = []
            for idxx, depthPreds_vis_single_numpy in enumerate(depthOut.cpu().detach().numpy()):
                # print(idxx, len(segAll_list))
                if opt.cfg.DATASET.if_no_gt_BRDF:
                    depthOut_colored_single_numpy_list.append(vis_disp_colormap(depthPreds_vis_single_numpy.squeeze(), normalize=True)[0])
                else:
                    depthOut_colored_single_numpy_list.append(vis_disp_colormap(depthPreds_vis_single_numpy.squeeze(), normalize=True, valid_mask=segAll_list[idxx]==1)[0])
            depthOut_colored_batch = np.stack(depthOut_colored_single_numpy_list).transpose(0, 3, 1, 2).astype(np.float32) / 255.
            depth_pred_batch_vis_sdr_colored = ( torch.from_numpy(depthOut_colored_batch).cuda() * segAllBatch_vis.expand_as(depthPreds_vis) ).data

            if opt.is_master:
                vutils.save_image(depth_pred_batch_vis_sdr,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                vutils.save_image(depth_pred_batch_vis_sdr_colored,
                    '{0}/{1}_depthPred_colored_{2}.png'.format(opt.summary_vis_path_task, tid, n) )


        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr_numpy = albedo_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedo_pred_aligned_batch_vis_sdr_numpy = albedo_pred_aligned_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedo_bs_pred_batch_vis_sdr_numpy = albedo_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr_numpy = rough_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                rough_bs_pred_batch_vis_sdr_numpy = rough_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depth_bs_pred_batch_vis_sdr_numpy = depth_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

        if opt.is_master:
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                writer.add_histogram('VAL_brdf-depth_PRED', np.clip(depthPreds_vis.cpu().numpy().flatten(), 0., 200.), tid)

            for sample_idx in tqdm(range(im_batch_vis_sdr.shape[0])):

                im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]

                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-albedo_PRED/%d'%sample_idx, albedo_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        writer.add_image('VAL_brdf-albedo_PRED-BS/%d'%sample_idx, albedo_bs_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                        writer.add_image('VAL_brdf-albedo_scaleAligned_PRED/%d'%sample_idx, albedo_pred_aligned_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((albedo_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_albedoGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, _ = real_sample_results_path_list[sample_idx]
                        real_sample_albedo_path = real_sample_results_path / 'albedo.png'
                        albedo = Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        albedo = albedo.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        albedo.save(str(real_sample_albedo_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_albedo_bs_path = real_sample_results_path / 'albedo_bs.png'
                            albedo_bs = Image.fromarray((albedo_bs_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                            albedo_bs = albedo_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            albedo_bs.save(str(real_sample_albedo_bs_path))


                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'no' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((normal_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_normal_path = real_sample_results_path / 'normal.png'
                        normal = Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        normal = normal.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        normal.save(str(real_sample_normal_path))

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_PRED/%d'%sample_idx, rough_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        writer.add_image('VAL_brdf-rough_PRED-BS/%d'%sample_idx, rough_bs_pred_batch_vis_sdr_numpy[sample_idx][:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')

                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'ro' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((rough_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_rough_path = real_sample_results_path / 'rough.png'
                        rough = Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        rough = rough.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        rough.save(str(real_sample_rough_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_rough_bs_path = real_sample_results_path / 'rough_bs.png'
                            rough_bs = Image.fromarray((rough_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                            rough_bs = rough_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            rough_bs.save(str(real_sample_rough_bs_path))

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                        depth_normalized_pred, _ = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx], )
                        writer.add_image('VAL_brdf-depth_syncScale_PRED/%d'%sample_idx, depth_normalized_pred[:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                        # print(segAllBatch_vis.shape, segAllBatch_vis.dtype, torch.max(segAllBatch_vis))
                    _ = depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()
                    im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]
                    _ = _[:im_h_resized_to, :im_w_resized_to]
                    depth_not_normalized_pred, depth_not_normalized_pred_scaling = vis_disp_colormap(_, normalize=True)
                    writer.add_image('VAL_brdf-depth_PRED/%d'%sample_idx, depth_not_normalized_pred[:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-depth_PRED_thres%.2f/%d'%(opt.cfg.MODEL_LIGHT.depth_thres, sample_idx), depthPreds_vis[sample_idx].cpu().numpy().squeeze()>opt.cfg.MODEL_LIGHT.depth_thres, tid, dataformats='HW')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                            depth_bs_normalized_pred = vis_disp_colormap(depth_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])[0]
                            writer.add_image('VAL_brdf-depth_syncScale_PRED-BS/%d'%sample_idx, depth_bs_normalized_pred[:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')
                        _ = depth_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze()
                        im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]
                        _ = _[:im_h_resized_to, :im_w_resized_to]
                        depth_bs_not_normalized_pred = vis_disp_colormap(_, normalize=True, min_and_scale=depth_not_normalized_pred_scaling)[0]
                        writer.add_image('VAL_brdf-depth_PRED-BS/%d'%sample_idx, depth_bs_not_normalized_pred[:im_h_resized_to, :im_w_resized_to], tid, dataformats='HWC')

                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((depth_not_normalized_pred).astype(np.uint8)).save('{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((depth_normalized_pred).astype(np.uint8)).save('{0}/{1}_depthPred_syncScale_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                            depth_normalized_gt, _ = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)
                            Image.fromarray((depth_normalized_gt).astype(np.uint8)).save('{0}/{1}_depthGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

                    pickle_save_path = Path(opt.summary_vis_path_task) / ('results_depth_%d.pickle'%sample_idx)
                    save_dict = {'depthPreds_vis': depthPreds_vis[sample_idx].detach().cpu().squeeze().numpy()}
                    if opt.if_save_pickles:
                        with open(str(pickle_save_path),"wb") as f:
                            pickle.dump(save_dict, f)

                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_depth_path = real_sample_results_path / 'depth.png'
                        depth = Image.fromarray(depth_not_normalized_pred)
                        depth = depth.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        depth.save(str(real_sample_depth_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_depth_bs_path = real_sample_results_path / 'depth_bs.png'
                            depth_bs = Image.fromarray(depth_bs_not_normalized_pred)
                            depth_bs = depth_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            depth_bs.save(str(real_sample_depth_bs_path))


    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))


def writeEnvToFile(envmaps, envId, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0] )
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0/2.2) ) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )