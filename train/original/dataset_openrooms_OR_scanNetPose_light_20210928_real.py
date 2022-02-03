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
class openrooms(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', task=None, if_for_training=True, load_first = -1, rseed = 1, 
            cascadeLevel = 0,
            # is_light = False, is_all_light = False,
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
        # assert self.split in ['train', 'val', 'test']
        assert self.split in ['val']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training

        self.data_root = self.opt.cfg.DATASET.real_images_root_path
        data_list_path = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.real_images_list_path)
        self.data_list = make_dataset_real(opt, self.data_root, data_list_path, logger=self.logger)

        logger.info(white_blue('%s: total frames: %d'%(self.dataset_name, len(self.data_list))))

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height
        self.im_height_padded, self.im_width_padded = self.cfg.DATA.im_height_padded_to, self.cfg.DATA.im_width_padded_to

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        png_image_path = self.data_list[index]
        frame_info = {'index': index, 'png_image_path': png_image_path}
        batch_dict = {'image_index': index}

        im_height_padded, im_width_padded = self.im_height_padded, self.im_width_padded

        hdr_scale = 1.
        # Read PNG image
        image = Image.open(str(png_image_path))
        im_fixedscale_SDR_uint8 = np.array(image)
        im_h, im_w = im_fixedscale_SDR_uint8.shape[0], im_fixedscale_SDR_uint8.shape[1]

        pad_mask = np.zeros((im_height_padded, im_width_padded), dtype=np.uint8)
        if float(im_h) / float(im_w) < float(im_height_padded) / float(im_width_padded): # flatter
            im_w_resized_to = im_width_padded
            im_h_resized_to = int(float(im_h) / float(im_w) * im_w_resized_to)
            assert im_h_resized_to <= im_height_padded
            pad_mask[:im_h_resized_to, :] = 1
        else: # taller
            im_h_resized_to = im_height_padded
            im_w_resized_to = int(float(im_w) / float(im_h) * im_h_resized_to)
            assert im_w_resized_to <= im_width_padded
            pad_mask[:, :im_w_resized_to] = 1

        im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (im_w_resized_to, im_h_resized_to), interpolation = cv2.INTER_AREA )
        # print(im_w_resized_to, im_h_resized_to, im_w, im_h)
        assert self.opt.cfg.DATA.pad_option == 'const'
        im_fixedscale_SDR_uint8 = cv2.copyMakeBorder(im_fixedscale_SDR_uint8, 0, im_height_padded-im_h_resized_to, 0, im_width_padded-im_w_resized_to, cv2.BORDER_CONSTANT, value=0)
        # print(im_fixedscale_SDR_uint8.shape, pad_mask.shape)
        im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        im_fixedscale_SDR = im_fixedscale_SDR.transpose(2, 0, 1)


        if self.opt.cfg.DATA.if_load_png_not_hdr:
            # [PNG]
            assert False, 'all models are trained with HDR input for now; should convert real images to HDR images by ** 2.2'
            im_fixedscale_HDR = (im_fixedscale_SDR - 0.5) / 0.5
            im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
            im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
            im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)
        else:
            # [HDR]
            im_fixedscale_HDR = im_fixedscale_SDR ** 2.2
            im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
            im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
            im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)

        # image_transformed_fixed = self.transforms_fixed(im_fixedscale_SDR_uint8)
        # im_trainval_SDR = self.transforms_resize(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]; already padded
        # # print(im_trainval_SDR.shape, type(im_trainval_SDR), torch.max(im_trainval_SDR), torch.min(im_trainval_SDR), torch.mean(im_trainval_SDR))
        # im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training

        batch_dict.update({'image_path': str(png_image_path), 'pad_mask': pad_mask, 'brdf_loss_mask': pad_mask})
        batch_dict['frame_info'] = {'image_path': str(png_image_path)}
        batch_dict.update({'im_w_resized_to': im_w_resized_to, 'im_h_resized_to': im_h_resized_to})
        # batch_dict.update({'hdr_scale': hdr_scale, 'image_transformed_fixed': image_transformed_fixed, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR, 'im_fixedscale_SDR_uint8': im_fixedscale_SDR_uint8})
        batch_dict.update({'hdr_scale': hdr_scale, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR.permute(1, 2, 0)}) # im_fixedscale_SDR for Tensorboard logging

        return batch_dict

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
            except RuntimeError as e:
                print('[!!!!] Type error in collate_fn_OR: ', key, e)
                # print(type(batch[0][key]))
                # print(batch[0][key].dtype)

    if 'boxes_batch' in batch[0]:
        interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
        collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    # boxes_valid_list = [item for sublist in collated_batch['boxes_valid_list'] for item in sublist]
    # boxes_valid_nums = [sum(x) for x in collated_batch['boxes_valid_list']]
    # boxes_total_nums = [len(x) for x in collated_batch['boxes_valid_list']]
    # if sum(boxes_valid_list)==0:
    #     print(boxes_valid_nums, '/', boxes_total_nums, red(sum(boxes_valid_list)), '/', len(boxes_valid_list), boxes_valid_list)
    # else:
    #     print(boxes_valid_nums, '/', boxes_total_nums, sum(boxes_valid_list), '/', len(boxes_valid_list), boxes_valid_list)

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
