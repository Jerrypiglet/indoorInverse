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
import json
import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")

def open_exr(file,img_wh):
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)[...,[2,1,0]]
    img = cv2.resize(img,img_wh,cv2.INTER_LANCZOS4)
    img = torch.from_numpy(img.astype(np.float32))
    return img

class Indoor(data.Dataset):
    def __init__(
        self, opt, 
        # data_list=None, 
        logger=basic_logger(), 
        transforms_BRDF=None, 
        split='train', task=None, if_for_training=True, 
        load_first = -1, rseed = 1, # useless
        cascadeLevel=None, # useless
        ):
        '''
        dataloader for Indoor scenes (e.g. kitchen)
        
        currently image only
        '''

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        # self.rseed = rseed
        self.dataset_name = self.cfg.DATASET.dataset_name.split('-')[1] # e.g. kitchen
        self.split = split
        assert self.split in ['train', 'val']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training
        assert not self.opt.if_cluster

        self.data_root = Path(self.opt.cfg.DATASET.dataset_path) / self.dataset_name / split
        assert self.data_root.exists(), 'data_root does not exist: %s'%str(self.data_root)
        
        # self.hdr_root = Path(self.data_root) / 'Image'
        #  if not self.opt.if_cluster else Path(self.data_root)#/'imhdr'
        # self.png_root = Path(self.data_root) / 'Image'
        # Path(self.opt.cfg.DATASET.png_path) if not self.opt.if_cluster else Path(self.data_root)#/'impng'
        # self.mask_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'immask'
        # self.cadmatobj_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'imcadmatobj'
        # self.baseColor_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'imbaseColor'
        # self.normal_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'imnormal'
        # self.roughness_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'imroughness'
        # self.depth_root = Path(self.data_root) if not self.opt.if_cluster else Path(self.data_root)#/'imdepth'

        with open(str(self.data_root / 'transforms.json'), 'r') as f:
            self.meta = json.load(f)

        self.total_frame_num = len(self.meta['frames'])
        self.frame_idx_list = list(range(self.total_frame_num))

        logger.info(white_blue('%s-%s-%s: total frames: %d'%(self.dataset_name, self.task, self.split, len(self.frame_idx_list))))

        self.transforms_BRDF = transforms_BRDF

        self.logger = logger
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

    def __len__(self):
        return self.total_frame_num

    def __getitem__(self, index):

        hdr_image_path = os.path.join(self.data_root, 'Image', '{:03d}_0001.exr'.format(index))
        png_image_path = os.path.join(self.data_root, 'Image', '{:03d}_0001.png'.format(index))

        # hdr_image_path, _ = self.data_list[index]
        # meta_split, scene_name, frame_id = self.meta_split_scene_name_frame_id_list[index]
        # assert frame_id > 0

        frame_info = {'index': index, 'frame_key': '%s-%d'%(self.dataset_name, index), 'png_image_path': png_image_path}
        batch_dict = {'image_index': index, 'frame_info': frame_info, 'im_w_resized_to': self.im_width, 'im_h_resized_to': self.im_height}

        if self.opt.cfg.DATA.if_load_png_not_hdr:
            hdr_scale = 1.
            # Read PNG image
            image = Image.open(str(png_image_path))
            im_fixedscale_SDR_uint8 = np.array(image)
            im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )
            # print(type(im_fixedscale_SDR_uint8), im_fixedscale_SDR_uint8.shape)

            im_trainval_SDR = self.transforms_BRDF(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]; already padded
            # print('-->', type(im_trainval_SDR), im_trainval_SDR.shape, torch.amax(im_trainval_SDR), torch.amin(im_trainval_SDR))
            im_trainval = im_trainval_SDR # channel first for training

            im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.

            batch_dict.update({'image_path': str(png_image_path)})
        else:
            # Read HDR image
            im_ori = open_exr(hdr_image_path, self.img_wh).reshape(self.img_wh[1], self.img_wh[0], 3)

            # Random scale the image
            # im_trainval, hdr_scale = self.scaleHdr(im_ori, seg_ori, forced_fixed_scale=False, if_print=True) # channel first for training
            im_trainval_SDR = np.clip(im_ori**(1.0/2.2), 0., 1.)

            # == no random scaling:
            # im_fixedscale, _ = self.scaleHdr(im_ori, seg_ori, forced_fixed_scale=True)
            hdr_scale = 1.
            im_fixedscale_SDR = np.clip(im_ori**(1.0/2.2), 0., 1.)
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

        return batch_dict