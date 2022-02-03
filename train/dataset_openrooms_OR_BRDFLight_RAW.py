# import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils import data
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce 
import h5py
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm
import torchvision.transforms as T
# import PIL
from utils.utils_misc import *
from pathlib import Path
# import pickle
import pickle5 as pickle
from icecream import ic
# from utils.utils_total3D.utils_OR_imageops import loadHdr_simple, to_nonhdr
import math
from scipy.spatial import cKDTree
import copy


import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")

from utils import transform

from utils_dataset_openrooms_OR_BRDFLight_RAW import make_dataset

class openrooms(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_BRDF=None, 
            split='train', task=None, if_for_training=True, load_first = -1, rseed = 1, 
            cascadeLevel = 0,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12):

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.rseed = rseed
        self.dataset_name = self.cfg.DATASET.dataset_name
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training
        self.data_root = self.opt.cfg.DATASET.dataset_path
        split_to_list = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
        data_list = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.dataset_list)
        data_list = os.path.join(data_list, split_to_list[split])
        self.data_list, self.meta_split_scene_name_frame_id_list, self.all_scenes_list = make_dataset(opt, split, self.task, self.data_root, data_list, logger=self.logger)
        assert len(self.data_list) == len(self.meta_split_scene_name_frame_id_list)
        if load_first != -1:
            self.data_list = self.data_list[:load_first] # [('/data/ruizhu/openrooms_mini-val/mainDiffLight_xml1/scene0509_00/im_1.hdr', '/data/ruizhu/openrooms_mini-val/main_xml1/scene0509_00/imsemLabel_1.npy'), ...
            self.meta_split_scene_name_frame_id_list = self.meta_split_scene_name_frame_id_list[:load_first] # [('mainDiffLight_xml1', 'scene0509_00', 1)

        logger.info(white_blue('%s-%s: total frames: %d; total scenes %d'%(self.dataset_name, self.split, len(self.data_list),len(self.all_scenes_list))))

        self.cascadeLevel = cascadeLevel

        self.transforms_BRDF = transforms_BRDF

        self.logger = logger
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        # ====== per-pixel lighting =====
        if self.opt.cfg.DATA.load_light_gt:
            self.envWidth = envWidth
            self.envHeight = envHeight
            self.envRow = envRow
            self.envCol = envCol
            self.SGNum = SGNum

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        hdr_image_path, _ = self.data_list[index]
        meta_split, scene_name, frame_id = self.meta_split_scene_name_frame_id_list[index]
        assert frame_id > 0

        if self.opt.cfg.DATASET.tmp:
            png_image_path = Path(hdr_image_path.replace('.hdr', '.png').replace('.rgbe', '.png'))
        else:
            png_image_path = Path(self.opt.cfg.DATASET.png_path) / meta_split / scene_name / ('im_%d.png'%frame_id)
        frame_info = {'index': index, 'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id, 'frame_key': '%s-%s-%d'%(meta_split, scene_name, frame_id), \
            'png_image_path': png_image_path}
        batch_dict = {'image_index': index, 'frame_info': frame_info}

        if_load_immask = True
        if_load_immask = if_load_immask or self.opt.cfg.DATA.load_masks
        self.opt.if_load_immask = if_load_immask

        if if_load_immask:
            seg_path = hdr_image_path.replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
            # Read segmentation
            seg = 0.5 * (self.loadImage(seg_path ) + 1)[0:1, :, :]
            semantics_path = hdr_image_path.replace('DiffMat', '').replace('DiffLight', '')
            mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
            mask = self.loadBinary(mask_path, channels = 3, dtype=np.int32, if_resize=True, modality='mask').squeeze() # [h, w, 3]
        else:
            seg = np.ones((1, self.im_height, self.im_width), dtype=np.float32)
            mask_path = ''
            mask = np.ones((self.im_height, self.im_width, 3), dtype=np.uint8)

        seg_ori = np.copy(seg)
        brdf_loss_mask = np.ones((self.im_height, self.im_width), dtype=np.uint8)
        pad_mask = np.ones((self.im_height, self.im_width), dtype=np.uint8)
        # if self.if_extra_op:
        #     if mask.dtype not in [np.int32, np.float32]:
        #         mask = self.extra_op(mask, name='mask') # if resize, willl not work because mask is of dtype int32
        #     seg = self.extra_op(seg, if_channel_first=True, name='seg')
        #     brdf_loss_mask = self.extra_op(brdf_loss_mask, if_channel_2_input=True, name='brdf_loss_mask', if_padding_constant=True)
        #     pad_mask = self.extra_op(pad_mask, if_channel_2_input=True, name='pad_mask', if_padding_constant=True)

        batch_dict.update({'brdf_loss_mask': torch.from_numpy(brdf_loss_mask), 'pad_mask': torch.from_numpy(pad_mask)})
        batch_dict.update({'im_w_resized_to': self.im_width, 'im_h_resized_to': self.im_height})

        if self.opt.cfg.DATA.if_load_png_not_hdr:
            hdr_scale = 1.
            # Read PNG image
            image = Image.open(str(png_image_path))
            im_fixedscale_SDR_uint8 = np.array(image)
            im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )

            im_trainval_SDR = self.transforms_BRDF(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]; already padded
            # print(im_trainval_SDR.shape, type(im_trainval_SDR), torch.max(im_trainval_SDR), torch.min(im_trainval_SDR), torch.mean(im_trainval_SDR))
            im_trainval = im_trainval_SDR # channel first for training

            im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
            # if self.if_extra_op:
                # im_fixedscale_SDR = self.extra_op(im_fixedscale_SDR, name='im_fixedscale_SDR')

            batch_dict.update({'image_path': str(png_image_path)})
        else:
            # Read HDR image
            im_ori = self.loadHdr(hdr_image_path)

            # Random scale the image
            im_trainval, hdr_scale = self.scaleHdr(im_ori, seg_ori, forced_fixed_scale=False, if_print=True) # channel first for training
            im_trainval_SDR = np.clip(im_trainval**(1.0/2.2), 0., 1.)
            # if self.if_extra_op:
            #     im_trainval = self.extra_op(im_trainval, name='im_trainval', if_channel_first=True)
            #     im_trainval_SDR = self.extra_op(im_trainval_SDR, name='im_trainval_SDR', if_channel_first=True)

            # == no random scaling:
            im_fixedscale, _ = self.scaleHdr(im_ori, seg_ori, forced_fixed_scale=True)
            im_fixedscale_SDR = np.clip(im_fixedscale**(1.0/2.2), 0., 1.)
            # if self.if_extra_op:
            #     im_fixedscale = self.extra_op(im_fixedscale, name='im_fixedscale', if_channel_first=True)
            #     im_fixedscale_SDR = self.extra_op(im_fixedscale_SDR, name='im_fixedscale_SDR', if_channel_first=True)
            im_fixedscale_SDR_uint8 = (255. * im_fixedscale_SDR).transpose(1, 2, 0).astype(np.uint8)

            im_fixedscale_SDR = np.transpose(im_fixedscale_SDR, (1, 2, 0)) # [240, 320, 3], np.ndarray

            batch_dict.update({'image_path': str(hdr_image_path)})

        # image_transformed_fixed: normalized, not augmented [only needed in semseg]
        # im_trainval: normalized, augmented; HDR (same as im_trainval_SDR in png case) -> for input to network
        # im_trainval_SDR: normalized, augmented; LDR (SRGB space)
        # im_fixedscale_SDR: normalized, NOT augmented; LDR
        # im_fixedscale_SDR_uint8: im_fixedscale_SDR -> 255

        # print('------', image_transformed_fixed.shape, im_trainval.shape, im_trainval_SDR.shape, im_fixedscale_SDR.shape, im_fixedscale_SDR_uint8.shape, )
        # png: ------ torch.Size([3, 240, 320]) (240, 320, 3) torch.Size([3, 240, 320]) (240, 320, 3) (240, 320, 3)
        # hdr: ------ torch.Size([3, 240, 320]) (3, 240, 320) (3, 240, 320) (3, 240, 320) (240, 320, 3)

        batch_dict.update({'hdr_scale': hdr_scale, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR, 'im_fixedscale_SDR_uint8': im_fixedscale_SDR_uint8})

        # ====== BRDF =====
        # image_path = batch_dict['image_path']
        # if self.opt.cfg.DATA.load_brdf_gt and (not self.opt.cfg.DATASET.if_no_gt_semantics):
        batch_dict_brdf, frame_info = self.load_brdf_lighting(hdr_image_path, if_load_immask, mask_path, mask, seg, seg_ori, hdr_scale, frame_info)
        batch_dict.update(batch_dict_brdf)


        return batch_dict

    def load_brdf_lighting(self, hdr_image_path, if_load_immask, mask_path, mask, seg, seg_ori, hdr_scale, frame_info):
        batch_dict_brdf = {}
        # Get paths for BRDF params
        # print(self.cfg.DATA.load_brdf_gt, self.cfg.DATA.data_read_list)
        if self.cfg.DATA.load_brdf_gt:
            if 'al' in self.cfg.DATA.data_read_list:
                albedo_path = hdr_image_path.replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
                if self.opt.cfg.DATASET.dataset_if_save_space:
                    albedo_path = albedo_path.replace('DiffLight', '')
                # Read albedo
                frame_info['albedo_path'] = albedo_path
                albedo = self.loadImage(albedo_path, isGama = False)
                albedo = (0.5 * (albedo + 1) ) ** 2.2
                # if self.if_extra_op:
                #     albedo = self.extra_op(albedo, if_channel_first=True, name='albedo')
                batch_dict_brdf.update({'albedo': torch.from_numpy(albedo)})

            if 'no' in self.cfg.DATA.data_read_list:
                normal_path = hdr_image_path.replace('im_', 'imnormal_').replace('rgbe', 'png').replace('hdr', 'png')
                if self.opt.cfg.DATASET.dataset_if_save_space:
                    normal_path = normal_path.replace('DiffLight', '').replace('DiffMat', '')
                # normalize the normal vector so that it will be unit length
                frame_info['normal_path'] = normal_path
                normal = self.loadImage(normal_path )
                normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
                # if self.if_extra_op:
                #     normal = self.extra_op(normal, if_channel_first=True, name='normal')

                batch_dict_brdf.update({'normal': torch.from_numpy(normal),})

            if 'ro' in self.cfg.DATA.data_read_list:
                rough_path = hdr_image_path.replace('im_', 'imroughness_').replace('rgbe', 'png').replace('hdr', 'png')
                if self.opt.cfg.DATASET.dataset_if_save_space:
                    rough_path = rough_path.replace('DiffLight', '')
                frame_info['rough_path'] = rough_path
                # Read roughness
                rough = self.loadImage(rough_path )[0:1, :, :]
                # if self.if_extra_op:
                #     rough = self.extra_op(rough, if_channel_first=True, name='rough')

                batch_dict_brdf.update({'rough': torch.from_numpy(rough),})

            if 'de' in self.cfg.DATA.data_read_list or 'de' in self.cfg.DATA.data_read_list:
                depth_path = hdr_image_path.replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
                if self.opt.cfg.DATASET.dataset_if_save_space:
                    depth_path = depth_path.replace('DiffLight', '').replace('DiffMat', '')
                frame_info['depth_path'] = depth_path
                # Read depth
                depth = self.loadBinary(depth_path)
                # if self.if_extra_op:
                #     depth = self.extra_op(depth, if_channel_first=True, name='depth')
                batch_dict_brdf.update({'depth': torch.from_numpy(depth),})

            # print('al', albedo.shape, np.amax(albedo), np.amin(albedo), np.median(albedo), np.mean(albedo))
            # print('no', normal.shape, np.amax(normal), np.amin(normal), np.median(normal), np.mean(normal))
            # print('ro', rough.shape, np.amax(rough), np.amin(rough), np.median(rough), np.mean(rough))
            # print('de', depth.shape, np.amax(depth), np.amin(depth), np.median(depth), np.mean(depth))
            # al (3, 256, 320) 1.0 0.0 0.42050794 0.38241568
            # no (3, 256, 320) 0.99998415 -0.99431545 0.2006149 0.2737651
            # ro (1, 256, 320) 1.0 -1.0 0.2 0.067728244
            # de (1, 256, 320) 4.679752 0.0 2.4866104 2.4604745

        if if_load_immask:
            segArea = np.logical_and(seg_ori > 0.49, seg_ori < 0.51 ).astype(np.float32 )
            segEnv = (seg_ori < 0.1).astype(np.float32 )
            segObj = (seg_ori > 0.9) 

            if self.opt.cfg.MODEL_LIGHT.enable:
                segObj = segObj.squeeze()
                segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                        border_value=1)
                segObj = segObj[np.newaxis, :, :]

            segObj = segObj.astype(np.float32 )
        else:
            segObj = np.ones_like(seg_ori, dtype=np.float32)
            segEnv = np.zeros_like(seg_ori, dtype=np.float32)
            segArea = np.zeros_like(seg_ori, dtype=np.float32)

        # if self.if_extra_op:
        #     segObj = self.extra_op(segObj, if_channel_first=True, name='segObj')
        #     segEnv = self.extra_op(segEnv, if_channel_first=True, name='segEnv')
        #     segArea = self.extra_op(segArea, if_channel_first=True, name='segArea')

        
        if self.opt.cfg.DATA.load_light_gt:
            if self.cascadeLevel == 0:
                env_path = hdr_image_path.replace('im_', 'imenv_')
            else:
                env_path = hdr_image_path.replace('im_', 'imenv_')
                envPre_path = hdr_image_path.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) )
                
                albedoPre_path = hdr_image_path.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
                normalPre_path = hdr_image_path.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
                roughPre_path = hdr_image_path.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
                depthPre_path = hdr_image_path.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )

                diffusePre_path = hdr_image_path.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
                specularPre_path = hdr_image_path.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )

            if self.opt.cfg.DEBUG.if_fast_light_labels:
                if frame_info['meta_split']=='main_xml1':
                    root_path_scene = '/ruidata/openrooms_raw_light_main_xml1'
                else:
                    root_path_scene = '/newdata/ruizhu/openrooms_raw_light'
                env_path = env_path.replace(self.opt.cfg.DATASET.dataset_path_local, root_path_scene)
            print(env_path)

            envmaps, envmapsInd = self.loadEnvmap(env_path )
            envmaps = envmaps * hdr_scale 
            # print(frame_info, self.split, hdr_scale, np.amax(envmaps),np.amin(envmaps), np.median(envmaps))
            if self.cascadeLevel > 0: 
                envmapsPre = self.loadH5(envPre_path ) 
                if envmapsPre is None:
                    print("Wrong envmap pred")
                    envmapsInd = envmapsInd * 0 
                    envmapsPre = np.zeros((84, 120, 160), dtype=np.float32 ) 

            if self.opt.cfg.MODEL_LIGHT.load_GT_light_sg:
                sgEnv_path = hdr_image_path.replace('im_', 'imsgEnv_').replace('.hdr', '.h5')
                sgEnv = self.loadH5(sgEnv_path) # (120, 160, 12, 6)
                sgEnv_torch = torch.from_numpy(sgEnv)
                sg_theta_torch, sg_phi_torch, sg_lamb_torch, sg_weight_torch = torch.split(sgEnv_torch, [1, 1, 1, 3], dim=3)
                sg_axisX = torch.sin(sg_theta_torch ) * torch.cos(sg_phi_torch )
                sg_axisY = torch.sin(sg_theta_torch ) * torch.sin(sg_phi_torch )
                sg_axisZ = torch.cos(sg_theta_torch )
                sg_axis_torch = torch.cat([sg_axisX, sg_axisY, sg_axisZ], dim=3)


        if self.cascadeLevel > 0:
            # Read albedo
            albedoPre = self.loadH5(albedoPre_path )
            albedoPre = albedoPre / np.maximum(np.mean(albedoPre ), 1e-10) / 3

            # normalize the normal vector so that it will be unit length
            normalPre = self.loadH5(normalPre_path )
            normalPre = normalPre / np.sqrt(np.maximum(np.sum(normalPre * normalPre, axis=0), 1e-5) )[np.newaxis, :]
            normalPre = 0.5 * (normalPre + 1)

            # Read roughness
            roughPre = self.loadH5(roughPre_path )[0:1, :, :]
            roughPre = 0.5 * (roughPre + 1)

            # Read depth
            depthPre = self.loadH5(depthPre_path )
            depthPre = depthPre / np.maximum(np.mean(depthPre), 1e-10) / 3

            diffusePre = self.loadH5(diffusePre_path )
            diffusePre = diffusePre / max(diffusePre.max(), 1e-10)

            specularPre = self.loadH5(specularPre_path )
            specularPre = specularPre / max(specularPre.max(), 1e-10)

        # if if_load_immask:
        batch_dict_brdf.update({
                'mask': torch.from_numpy(mask), 
                'maskPath': mask_path, 
                'segArea': torch.from_numpy(segArea),
                'segEnv': torch.from_numpy(segEnv),
                'segObj': torch.from_numpy(segObj),
                'object_type_seg': torch.from_numpy(seg), 
                })
        # if self.transform is not None and not self.opt.if_hdr:

        if self.opt.cfg.DATA.load_light_gt:
            batch_dict_brdf['envmaps'] = envmaps
            batch_dict_brdf['envmapsInd'] = envmapsInd
            # print(envmaps.shape, envmapsInd.shape)

            if self.cascadeLevel > 0:
                batch_dict_brdf['envmapsPre'] = envmapsPre

            if self.opt.cfg.MODEL_LIGHT.load_GT_light_sg:
                batch_dict_brdf['sg_theta'] = sg_theta_torch
                batch_dict_brdf['sg_phi'] = sg_phi_torch
                batch_dict_brdf['sg_lamb'] = sg_lamb_torch
                batch_dict_brdf['sg_axis'] = sg_axis_torch
                batch_dict_brdf['sg_weight'] = sg_weight_torch

        if self.cascadeLevel > 0:
            batch_dict_brdf['albedoPre'] = albedoPre
            batch_dict_brdf['normalPre'] = normalPre
            batch_dict_brdf['roughPre'] = roughPre
            batch_dict_brdf['depthPre'] = depthPre

            batch_dict_brdf['diffusePre'] = diffusePre
            batch_dict_brdf['specularPre'] = specularPre

        return batch_dict_brdf, frame_info


    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            self.logger.warning('File does not exist: ' + imName )
            assert(False), 'File does not exist: ' + imName 

        im = Image.open(imName)
        im = im.resize([self.im_width, self.im_height], Image.ANTIALIAS )

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadHdr(self, imName):
        if not(osp.isfile(imName ) ):
            if osp.isfile(imName.replace('.hdr', '.rgbe')):
                imName = imName.replace('.hdr', '.rgbe')
            else:
                print(imName )
                assert(False )
        im = cv2.imread(imName, -1)
        # print(imName, im.shape, im.dtype)

        if im is None:
            print(imName )
            assert(False )
        im = cv2.resize(im, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg, forced_fixed_scale=False, if_print=False):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split == 'train' and not forced_fixed_scale:
            # print('randommmm', np.random.random(), random.random())
            # scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
            scale = (0.95 - 0.1 * random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
            # if if_print:
            #     print(self.split, not forced_fixed_scale, scale)

        # print('-', hdr.shape, np.max(hdr), np.min(hdr), np.median(hdr), np.mean(hdr))
        # print('----', seg.shape, np.max(seg), np.min(seg), np.median(seg), np.mean(seg))
        # print('-------', scale)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName, channels = 1, dtype=np.float32, if_resize=True, modality=''):
        assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
        if not(osp.isfile(imName ) ):
            assert(False ), '%s doesnt exist!'%imName
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * channels * width * height )
            if dtype == np.float32:
                decode_char = 'f'
            elif dtype == np.int32:
                decode_char = 'i'
            depth = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
            depth = depth.reshape([height, width, channels] )
            if if_resize:
                # print(self.im_width, self.im_height, width, height)
                if dtype == np.float32:
                    depth = cv2.resize(depth, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA )
                elif dtype == np.int32:
                    depth = cv2.resize(depth.astype(np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                    depth = depth.astype(np.int32)

            depth = np.squeeze(depth)

        # if modality=='mask':
        #     print(depth.shape, depth[np.newaxis, :, :].shape)

        return depth[np.newaxis, :, :]

    def loadH5(self, imName ): 
        try:
            hf = h5py.File(imName, 'r')
            im = np.array(hf.get('data' ) )
            return im 
        except:
            return None

    def loadEnvmap(self, envName ):
        # print('>>>>loadEnvmap', envName)
        if not osp.isfile(envName ):
            env = np.zeros( [3, self.envRow, self.envCol,
                self.envHeight, self.envWidth], dtype = np.float32 )
            envInd = np.zeros([1, 1, 1], dtype=np.float32 )
            print('Warning: the envmap %s does not exist.' % envName )
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert( (envHeightOrig / self.envHeight) == (envWidthOrig / self.envWidth) )
            assert( envHeightOrig % self.envHeight == 0)
            
            env = cv2.imread(envName, -1 ) 

            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                    envWidthOrig, 3) # (1920, 5120, 3) -> (120, 16, 160, 32, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) ) # -> (3, 120, 160, 16, 32)

                scale = envHeightOrig / self.envHeight
                if scale > 1:
                    env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

                envInd = np.ones([1, 1, 1], dtype=np.float32 )
                return env, envInd
            else:
                env = np.zeros( [3, self.envRow, self.envCol,
                    self.envHeight, self.envWidth], dtype = np.float32 )
                envInd = np.zeros([1, 1, 1], dtype=np.float32 )
                print('Warning: the envmap %s does not exist.' % envName )
                return env, envInd

    def loadNPY(self, imName, dtype=np.int32, if_resize=True):
        depth = np.load(imName)
        if if_resize:
            #t0 = timeit.default_timer()
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)
                #print('Resize float npy: %.4f' % (timeit.default_timer() - t0) )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(
                    np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)
                #print('Resize int32 npy: %.4f' % (timeit.default_timer() - t0) )

        depth = np.squeeze(depth)

        return depth

default_collate = torch.utils.data.dataloader.default_collate
def collate_fn_OR(batch):
    """
    Data collater.

    Assumes each instance is a dict.    
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    # print(batch[0].keys())
    for key in batch[0]:
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey in ['bdb2D_full', 'bdb3D_full']: # lists of original & more information (e.g. color)
                    continue
                if subkey in ['mask', 'random_id', 'cat_name']: # list of lists
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    try:
                        tensor_batch = torch.cat(list_of_tensor)
                        # print(subkey, [x['boxes_batch'][subkey].shape for x in batch], tensor_batch.shape)
                    except RuntimeError:
                        print(subkey, [x.shape for x in list_of_tensor])
                collated_batch[key][subkey] = tensor_batch
        elif key in ['frame_info', 'boxes_valid_list', 'emitter2wall_assign_info_list', 'emitters_obj_list', 'gt_layout_RAW', 'cell_info_grid', 'image_index', \
                'gt_obj_path_alignedNew_normalized_list', 'gt_obj_path_alignedNew_original_list', \
                'detectron_sample_dict', 'detectron_sample_dict']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except:
                print('[!!!!] Type error in collate_fn_OR: ', key)

    if 'boxes_batch' in batch[0]:
        interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
        collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    return collated_batch

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem
