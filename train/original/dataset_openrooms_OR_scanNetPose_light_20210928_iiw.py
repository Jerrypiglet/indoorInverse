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
class iiw(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', task=None, if_for_training=True, load_first = -1, rseed = 1, 
            cascadeLevel = 0,
            maxNum = 800 ):

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.rseed = rseed
        self.dataset_name = 'IIW'
        self.split = split
        assert self.split in ['train', 'val']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training

        self.maxNum = maxNum

        self.data_root = self.opt.cfg.DATASET.iiw_path
        data_list_path = Path(self.cfg.PATH.root) / self.cfg.DATASET.iiw_list_path
        # self.data_list = make_dataset_real(opt, self.data_root, data_list_path, logger=self.logger)

        if split == 'train':
            with open(str(data_list_path / 'IIWTrain.txt'), 'r') as fIn:
                im_list = fIn.readlines()
            self.data_list = [osp.join(self.data_root, x.strip()) for x in im_list ]
        elif split == 'val':
            with open(str(data_list_path / 'IIWTest.txt'), 'r') as fIn:
                im_list = fIn.readlines()
            self.data_list = [osp.join(self.data_root, x.strip()) for x in im_list ]
        else:
            raise RuntimeError("Invalid split %s for iiw!"%split)

        self.json_list = [x.replace('.png', '.json') for x in self.data_list]



        logger.info(white_blue('%s: total frames: %d'%(self.dataset_name, len(self.data_list))))

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.iiw.im_width, self.cfg.DATA.iiw.im_height
        self.im_height_padded, self.im_width_padded = self.cfg.DATA.iiw.im_height_padded_to, self.cfg.DATA.iiw.im_width_padded_to

        self.if_extra_op = False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        png_image_path = self.data_list[index]
        # frame_info = {'index': index, 'png_image_path': png_image_path}
        batch_dict = {'image_index': index}

        pad_mask = np.zeros((self.im_height_padded, self.im_width_padded), dtype=np.uint8)

        hdr_scale = 1.
        # Read PNG image
        image = Image.open(str(png_image_path))
        hdr_scale = 1.
        # Read PNG image
        image = Image.open(str(png_image_path))
        # im_fixedscale_SDR_uint8 = np.array(image)
        # im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )

        # image_transformed_fixed = self.transforms_fixed(im_fixedscale_SDR_uint8)
        # im_trainval_SDR = self.transforms_resize(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]; already padded
        # # print(im_trainval_SDR.shape, type(im_trainval_SDR), torch.max(im_trainval_SDR), torch.min(im_trainval_SDR), torch.mean(im_trainval_SDR))
        # im_trainval = im_trainval_SDR # channel first for training

        # im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        # if self.if_extra_op:
        #     im_fixedscale_SDR = self.extra_op(im_fixedscale_SDR, name='im_fixedscale_SDR')

        # batch_dict.update({'image_path': str(png_image_path)})

        im_fixedscale_SDR_uint8 = np.array(image)
        im_h, im_w = im_fixedscale_SDR_uint8.shape[0], im_fixedscale_SDR_uint8.shape[1]
        if float(im_h) / float(im_w) < float(self.im_height_padded) / float(self.im_width_padded): # flatter
            im_w_resized_to = self.im_width_padded
            im_h_resized_to = int(float(im_h) / float(im_w) * im_w_resized_to)
            assert im_h_resized_to <= self.im_height_padded
            pad_mask[:im_h_resized_to, :] = 1

            # rs, re = 0, im_h_resized_to
            # cs, ce = 0, im_w_resized_to
            # gap = im_h_resized_to - self.im_height_padded
            # rs = np.random.randint(gap + 1)
            # re = rs + self.im_height_padded

        else: # taller
            im_h_resized_to = self.im_height_padded
            im_w_resized_to = int(float(im_w) / float(im_h) * im_h_resized_to)
            assert im_w_resized_to <= self.im_width_padded
            pad_mask[:, :im_w_resized_to] = 1

            # rs, re = 0, self.im_height_padded
            # gap = im_w_resized_to - self.im_width_padded
            # cs = np.random.randint(gap + 1)
            # ce = cs + self.im_width_padded


        im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (im_w_resized_to, im_h_resized_to), interpolation = cv2.INTER_AREA )
        # print(im_w_resized_to, im_h_resized_to, im_w, im_h)
        assert self.opt.cfg.DATA.pad_option == 'const'
        im_fixedscale_SDR_uint8 = cv2.copyMakeBorder(im_fixedscale_SDR_uint8, 0, self.im_height_padded-im_h_resized_to, 0, self.im_width_padded-im_w_resized_to, cv2.BORDER_CONSTANT, value=0)
        im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        im_fixedscale_SDR = im_fixedscale_SDR.transpose(2, 0, 1)


        # if self.opt.cfg.DATA.if_load_png_not_hdr:
        #     # [PNG]
        #     im_fixedscale_HDR = (im_fixedscale_SDR - 0.5) / 0.5
        #     im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
        #     im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
        #     im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)
        # else:
        # [HDR]
        im_fixedscale_HDR = im_fixedscale_SDR ** 2.2
        im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
        im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
        im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)

        batch_dict.update({'image_path': str(png_image_path), 'pad_mask': pad_mask, 'brdf_loss_mask': pad_mask})
        batch_dict.update({'im_w_resized_to': im_w_resized_to, 'im_h_resized_to': im_h_resized_to})
        batch_dict.update({'hdr_scale': hdr_scale, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR.permute(1, 2, 0)}) # im_fixedscale_SDR for Tensorboard logging

        # load judgements labels
        rs, re = 0, im_h_resized_to
        cs, ce = 0, im_w_resized_to

        judgements = json.load(open(self.json_list[index]))
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}

        eqPoint, eqWeight = [0, 0, 0, 0], [0]
        darkerPoint, darkerWeight = [0, 0, 0, 0], [0]
        for c in comparisons:
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            r1, c1 = int(point1['y'] * im_h_resized_to ), int(point1['x'] * im_w_resized_to )
            r2, c2 = int(point2['y'] * im_h_resized_to ), int(point2['x'] * im_w_resized_to )

            pr1 = float(r1 - rs) / float(self.im_height_padded -1 )
            pc1 = float(c1 - cs ) / float(self.im_width_padded - 1 )
            pr2 = float(r2 - rs ) / float(self.im_height_padded - 1 )
            pc2 = float(c2 - cs ) / float(self.im_width_padded - 1 )

            if not pr1 >= 0.0 or not pr1 <= 1.0:
                continue
            assert(pr1 >= 0.0 and pr1 <= 1.0)
            if pc1 < 0.0 or pc1 > 1.0:
                continue
            assert(pc1 >= 0.0 and pc1 <= 1.0)
            if not pr2 >= 0.0 or not pr2 <= 1.0:
                continue
            assert(pr2 >= 0.0 and pr2 <= 1.0)
            if pc2 < 0.0 or pc2 > 1.0:
                continue
            assert(pc2 >= 0.0 and pc2 <= 1.0)

            prId1 = int(pr1 * (self.im_height_padded - 1) )
            pcId1 = int(pc1 * (self.im_width_padded - 1) )
            prId2 = int(pr2 * (self.im_height_padded - 1) )
            pcId2 = int(pc2 * (self.im_width_padded - 1) )


            # the second point should be darker than the first point
            if darker == 'E':
                eqPoint = eqPoint + [prId1, pcId1, prId2, pcId2 ]
                eqWeight.append(weight )
            elif darker == '1':
                darkerPoint = darkerPoint + [prId2, pcId2, prId1, pcId1 ]
                darkerWeight.append(weight )
            elif darker == '2':
                darkerPoint = darkerPoint + [prId1, pcId1, prId2, pcId2 ]
                darkerWeight.append(weight )

        eqWeight = np.asarray(eqWeight, dtype=np.float32 )
        eqPoint = np.asarray(eqPoint, dtype=np.long )
        eqPoint = eqPoint.reshape([-1, 4] )
        darkerWeight = np.asarray(darkerWeight, dtype=np.float32 )
        darkerPoint = np.asarray(darkerPoint, dtype=np.float32 )
        darkerPoint = darkerPoint.reshape([-1, 4] )

        assert(eqPoint.shape[0] == eqWeight.shape[0] )
        assert(darkerPoint.shape[0] == darkerWeight.shape[0] )

        eqNum = eqPoint.shape[0]
        if eqNum < self.maxNum:
            gap = self.maxNum - eqNum
            eqPoint = np.concatenate([eqPoint, np.zeros( (gap, 4), dtype=np.long) ], axis=0 )
            eqWeight = np.concatenate([eqWeight, np.zeros(gap, dtype=np.float32)], axis=0 )
        elif eqNum > self.maxNum:
            index = np.random.permutation(np.arange(eqNum ) )
            eqPoint = eqPoint[index, :]
            eqWeight = eqWeight[index ]

            eqPoint = eqPoint[0:self.maxNum, :]
            eqWeight = eqWeight[0:self.maxNum ]
            eqNum = self.maxNum

        darkerNum = darkerPoint.shape[0]
        if darkerNum < self.maxNum:
            gap = self.maxNum - darkerNum
            darkerPoint = np.concatenate([darkerPoint, np.zeros( (gap, 4), dtype=np.long) ], axis=0 )
            darkerWeight = np.concatenate([darkerWeight, np.zeros(gap, dtype=np.float32)], axis=0 )
        elif darkerNum > self.maxNum:
            index = np.random.permutation(np.arange(darkerNum ) )
            darkerPoint = darkerPoint[index, :]
            darkerWeight = darkerWeight[index ]

            darkerPoint = darkerPoint[0:self.maxNum, :]
            darkerWeight = darkerWeight[0:self.maxNum]
            darkerNum = self.maxNum

        batch_dict.update({
                'eq': {'point' : eqPoint, 'weight' : eqWeight, 'num': eqNum },
                'darker': {'point' : darkerPoint, 'weight' : darkerWeight, 'num' : darkerNum },
                'judgements': judgements
                })

        return batch_dict


default_collate = torch.utils.data.dataloader.default_collate
def collate_fn_iiw(batch):
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
