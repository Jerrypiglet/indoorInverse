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
from utils.utils_total3D.utils_OR_imageops import loadHdr_simple, to_nonhdr
import math
from utils.utils_total3D.data_config import RECON_3D_CLS_OR_dict
from scipy.spatial import cKDTree
import copy
# import math
# from detectron2.structures import BoxMode
# from detectron2.data.dataset_mapper import DatasetMapper

from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB
# from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog

from utils.utils_scannet import read_ExtM_from_txt, read_img
import utils.utils_nvidia.mdataloader.m_preprocess as m_preprocess
import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")

from utils import transform

from semanticInverse.train.utils_dataset_openrooms_OR_BRDFLight_RAW import *
import json
class nyud(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', task=None, if_for_training=True, load_first = -1, rseed = 1, 
            imWidthMax = 600, imWidthMin = 560,
            cascadeLevel = 0,
            maxNum = 800 ):

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.rseed = rseed
        self.dataset_name = 'nyud'

        self.NYU_root = Path(opt.cfg.DATASET.nyud_path)
        self.im_root = self.NYU_root / 'images'
        self.normal_root = self.NYU_root / 'normals'
        self.depth_root = self.NYU_root / 'depths'
        self.seg_root = self.NYU_root / 'masks'

        self.split = split
        assert self.split in ['train', 'val']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training

        self.maxNum = maxNum

        self.data_root = self.opt.cfg.DATASET.nyud_path
        data_list_path = Path(self.cfg.PATH.root) / self.cfg.DATASET.nyud_list_path
        # self.data_list = make_dataset_real(opt, self.data_root, data_list_path, logger=self.logger)

        if split == 'train':
            with open(str(data_list_path / 'NYUTrain.txt'), 'r') as fIn:
                self.data_list = [Path(x.strip()) for x in fIn.readlines()]
        elif split == 'val':
            with open(str(data_list_path / 'NYUTest.txt'), 'r') as fIn:
                self.data_list = [Path(x.strip()) for x in fIn.readlines()]
        else:
            raise RuntimeError("Invalid split %s for nyud!"%split)

        logger.info(white_blue('%s-%s: total frames: %d'%(self.dataset_name, self.split, len(self.data_list))))

        self.im_list = [str(self.im_root / x ) for x in self.data_list ]
        self.normal_list = [str(self.normal_root / x ) for x in self.data_list ]
        self.seg_list = [str(self.seg_root / x ) for x in self.data_list ]
        self.depth_list = [str(self.depth_root / x ).replace('.png', '.tiff') for x in self.data_list ]

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        self.im_width, self.im_height = self.cfg.DATA.nyud.im_width, self.cfg.DATA.nyud.im_height # original dimension (480x640)
        self.im_height_padded_to, self.im_width_padded_to = self.cfg.DATA.nyud.im_height_padded_to, self.cfg.DATA.nyud.im_width_padded_to # (256x320)
        self.im_height_resized_to, self.im_width_resized_to = self.cfg.DATA.im_height, self.cfg.DATA.im_width # (240x320)
        self.imWidthMax = imWidthMax
        self.imWidthMin = imWidthMin

        self.if_extra_op = False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        png_image_path = self.data_list[index]
        # frame_info = {'index': index, 'png_image_path': png_image_path}
        batch_dict = {'image_index': index}

        pad_mask = np.zeros((self.im_height_padded_to, self.im_width_padded_to), dtype=np.uint8)
        pad_mask[:self.im_height_resized_to, :] = 1 # assert pad to bottom

        hdr_scale = 1.
        # Read PNG image

        if self.split.lower() == 'train':
            scale = np.random.random();
            imCropWidth = int( np.round( (self.imWidthMax - self.imWidthMin ) * scale + self.imWidthMin ) )
            imCropHeight = int( float(self.im_height_resized_to) / float(self.im_width_resized_to ) * imCropWidth )
            rs = int(np.round( (480 - imCropHeight) * np.random.random() ) )
            re = rs + imCropHeight
            cs = int(np.round( (640 - imCropWidth) * np.random.random() ) )
            ce = cs + imCropWidth
        elif self.split.lower() == 'val':
            imCropWidth = self.im_width
            imCropHeight = self.im_height
            rs, re, cs, ce = 0, self.im_height, 0, self.im_width

        segNormal = 0.5 * ( self.loadImage(self.seg_list[index], rs, re, cs, ce) + 1)[0:1, :, :] # [0, 1]

        # Read Image
        im = 0.5 * (self.loadImage(self.im_list[index], rs, re, cs, ce, isGama = True ) + 1).astype(np.float32) # already did 2.2 with isGama = True
        
        # print(self.seg_list[index], segNormal.shape, rs, re, cs, ce, imCropWidth, imCropHeight, im.shape)

        # ic(np.amax(segMask_ori), np.amin(segMask_ori), np.median(segMask_ori))

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage( self.normal_list[index], rs, re, cs, ce, tag='normal')
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

        # Read depth
        depth = self.loadDepth(self.depth_list[index], rs, re, cs, ce )

        if imCropHeight != self.im_height_resized_to or imCropWidth != self.im_width_resized_to:
            depth = np.squeeze(depth, axis=0)
            depth = cv2.resize(depth, (self.im_width_resized_to, self.im_height_resized_to), interpolation = cv2.INTER_LINEAR)
            depth = depth[np.newaxis, :, :]
        segDepth = np.logical_and(depth > 1, depth < 10).astype(np.float32 )

        if imCropHeight != self.im_height_resized_to or imCropWidth != self.im_width_resized_to:
            normal = normal.transpose([1, 2, 0] )
            normal = cv2.resize(normal, (self.im_width_resized_to, self.im_height_resized_to), interpolation = cv2.INTER_LINEAR)
            normal = normal.transpose([2, 0, 1] )
        normal = normal / np.maximum(np.sqrt(np.sum(normal * normal, axis=0 )[np.newaxis, :, :] ), 1e-5)


        if imCropHeight != self.im_height_resized_to or imCropWidth != self.im_width_resized_to:
            segNormal = np.squeeze(segNormal, axis=0)
            segNormal = cv2.resize(segNormal, (self.im_width_resized_to, self.im_height_resized_to), interpolation = cv2.INTER_LINEAR)
            segNormal = segNormal[np.newaxis, :, :]

            im = im.transpose([1, 2, 0] )
            im = cv2.resize(im, (self.im_width_resized_to, self.im_height_resized_to), interpolation = cv2.INTER_LINEAR )
            im = im.transpose([2, 0, 1] )

        if self.split.lower() == 'train':
            if np.random.random() > 0.5:
                normal = np.ascontiguousarray(normal[:, :, ::-1] )
                normal[0, :, :] = -normal[0, :, :]
                depth = np.ascontiguousarray(depth[:, :, ::-1] )
                segNormal = np.ascontiguousarray(segNormal[:, :, ::-1] )
                segDepth = np.ascontiguousarray(segDepth[:, :, ::-1] )
                im = np.ascontiguousarray(im[:, :, ::-1] )
            scale = 1 + ( np.random.random(3) * 0.4 - 0.2 )
            scale = scale.reshape([3, 1, 1] )
            im = im * scale

        # ic(self.if_extra_op, normal)
        if self.if_extra_op:
            im = self.extra_op(im, if_channel_first=True, name='im')
            normal = self.extra_op(normal, if_channel_first=True, name='normal')
            depth = self.extra_op(depth, if_channel_first=True, name='depth')
            segNormal = self.extra_op(segNormal, if_channel_first=True, name='segNormal')
            segDepth = self.extra_op(segDepth, if_channel_first=True, name='segDepth')
        # ic(normal.shape)

        # ic(normal.shape, depth.shape, segNormal.shape, segDepth.shape, im.shape)
        # ic(normal.dtype, depth.dtype, segNormal.dtype, segDepth.dtype, im.dtype)
        batch_dict.update({'normal': normal, 'depth': depth, 'segNormal': segNormal, 'segDepth': segDepth})
        if self.split != 'train':
            segMask_ori = self.loadImage(self.seg_list[index], rs, re, cs, ce, if_crop=False, if_normalize=False)
            depth_ori = self.loadDepth(self.depth_list[index], rs, re, cs, ce, if_crop=False )
            normal_ori = self.loadImage( self.normal_list[index], rs, re, cs, ce, if_crop=False, if_normalize=False)
            # normal_ori = normal_ori / np.sqrt(np.maximum(np.sum(normal_ori * normal_ori, axis=0), 1e-5) )[np.newaxis, :]

            batch_dict.update({'segMask_ori': segMask_ori, 'normal_ori': normal_ori, 'depth_ori': depth_ori})
            # print(segMask_ori.shape, normal_ori.shape, depth_ori.shape) # all 480x640

        im_fixedscale_HDR = im
        im_fixedscale_SDR = im_fixedscale_HDR ** (1./2.2)

        im_trainval = torch.from_numpy(im_fixedscale_HDR).float() # channel first for training
        im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR).float()
        im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR).float()

        batch_dict.update({'image_path': str(png_image_path), 'pad_mask': pad_mask, 'brdf_loss_mask': pad_mask})
        batch_dict.update({'im_w_resized_to': self.im_width_resized_to, 'im_h_resized_to': self.im_height_resized_to})
        batch_dict.update({'hdr_scale': hdr_scale, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR.permute(1, 2, 0)}) # im_fixedscale_SDR for Tensorboard logging

        return batch_dict

        
    def loadImage(self, imName, rs, re, cs, ce, isGama = False, if_normalize=True, if_crop=True, tag=''):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = cv2.imread(imName)
        if len(im.shape ) == 3:
            im = im[:, :, ::-1]

        assert im.shape[:2] == (self.im_height, self.im_width)

        if if_crop:
            im = im[rs:re, cs:ce, :]

        if if_normalize:
            im = im.astype(np.float32 )
            if isGama:
                im = (im / 255.0) ** 2.2
                im = 2 * im - 1
            else:
                im = (im - 127.5) / 127.5

        im = np.ascontiguousarray(im)

        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadDepth(self, imName, rs, re, cs, ce, if_crop=True ):
        if not osp.isfile(imName):
            print(imName )
            assert(False )

        im = cv2.imread(imName, -1)

        assert im.shape[:2] == (self.im_height, self.im_width)
        
        if if_crop:
            im = im[rs:re, cs:ce]

        im = im[np.newaxis, :, :]
        return im


default_collate = torch.utils.data.dataloader.default_collate
def collate_fn_nyud(batch):
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
        if key == '':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey in []: # lists of original & more information (e.g. color)
                    continue
                if subkey in []: # list of lists
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    try:
                        tensor_batch = torch.cat(list_of_tensor)
                        # print(subkey, [x['boxes_batch'][subkey].shape for x in batch], tensor_batch.shape)
                    except RuntimeError:
                        print(subkey, [x.shape for x in list_of_tensor])
                collated_batch[key][subkey] = tensor_batch
        elif key in ['eq', 'darker', 'judgements']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except RuntimeError as e:
                print('[!!!!] Type error in collate_fn_OR: ', key, e)

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
