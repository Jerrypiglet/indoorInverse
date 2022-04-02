# import glob
import re
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
import math
from scipy.spatial import cKDTree
import copy

import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")

import tables



def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

def get_valid_scenes(opt, frames_list_path, split, logger=None):
    scenes_list_path = str(frames_list_path).replace('.txt', '_scenes.txt')
    if not os.path.isfile(scenes_list_path):
        # raise (RuntimeError("Scene list file do not exist: " + scenes_list_path + "\n"))
        return []
    if logger is None:
        logger = basic_logger()

    meta_split_scene_name_list = []
    list_read = open(scenes_list_path).readlines()
    logger.info("Totally {} scenes in {} set.".format(len(list_read), split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            assert False, 'No support for test split for now.'
        else:
            if len(line_split) not in [2]:
                raise (RuntimeError("Scene list file read line error : " + line + "\n"))

        meta_split, scene_name = line_split
        meta_split_scene_name_list.append([meta_split, scene_name])

    return meta_split_scene_name_list

def make_dataset(opt, split, task, data_root=None, data_lists=None, logger=None):
    assert split in ['train', 'val', 'test', 'trainval']

    if logger is None:
        logger = basic_logger()
    image_label_list = []
    meta_split_scene_name_frame_id_list = []

    for data_list_idx, data_list in enumerate(data_lists):
        if not os.path.isfile(data_list):
            raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
        list_read = open(data_list).readlines()
        logger.info("Totally {} samples in {} set - list #{}.".format(len(list_read), split, data_list_idx))
        logger.info("Starting Checking image&label pair from {} set - list #{}...".format(split, data_list_idx))
        for line in list_read:
            line = line.strip()
            line_split = line.split(' ')
            if split == 'test':
                image_name = os.path.join(data_root, line_split[2])
                if len(line_split) != 3:
                    label_name = os.path.join(data_root, line_split[3])
                    # raise (RuntimeError("Image list file read line error : " + line + "\n"))
                else:
                    label_name = image_name  # just set place holder for label_name, not for use
            else:
                if len(line_split) not in [3, 4]:
                    raise (RuntimeError("Image list file read line error : " + line + "\n"))
                image_name = os.path.join(data_root, line_split[2])
                # label_name = os.path.join(data_root, line_split[3])
                label_name = ''
            '''
            following check costs some time
            if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
                item = (image_name, label_name)
                image_label_list.append(item)
            else:
                raise (RuntimeError("Image list file line error : " + line + "\n"))
            '''

            meta_split = line_split[2].split('/')[0]
            # print(meta_split, opt.meta_splits_skip, meta_split in opt.meta_splits_skip)
            if opt.meta_splits_skip is not None and meta_split in opt.meta_splits_skip:
                continue
            item = (image_name, label_name)
            image_label_list.append(item)
            meta_split_scene_name_frame_id_list.append((meta_split, line_split[0], int(line_split[1])))

    logger.info("==> Checking image&label pair [%s] list done! %d frames."%(split, len(image_label_list)))
    
    if opt.cfg.DATASET.first_scenes != -1:
        # return image_label_list[:opt.cfg.DATASET.first_scenes], meta_split_scene_name_frame_id_list[:opt.cfg.DATASET.first_scenes]
        assert False
    elif opt.cfg.DATASET.if_quarter and task in ['train']:
        all_scenes = get_valid_scenes(opt, data_list, split, logger=logger)
        meta_split_scene_name_frame_id_list_quarter = return_percent(meta_split_scene_name_frame_id_list, 0.25)
        all_scenes = list(set(['/'.join([x[0], x[1]]) for x in meta_split_scene_name_frame_id_list_quarter]))
        all_scenes = [x.split('/') for x in all_scenes]
        return return_percent(image_label_list, 0.25), meta_split_scene_name_frame_id_list_quarter, all_scenes
    else:
        return image_label_list, meta_split_scene_name_frame_id_list, None

class openrooms_pickle(data.Dataset):
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
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training
        self.data_root = self.opt.cfg.DATASET.dataset_path_pickle

        split_to_list = {'train': ['train.txt'], 'val': ['val.txt'], 'trainval': ['train.txt', 'val.txt'], 'test': ['test.txt']}
        data_list = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.dataset_list)
        data_lists = [os.path.join(data_list, x) for x in split_to_list[split]]

        self.data_list, self.meta_split_scene_name_frame_id_list, self.all_scenes_list = \
            make_dataset(opt, split, self.task, self.data_root, data_lists, logger=self.logger)


        assert len(self.data_list) == len(self.meta_split_scene_name_frame_id_list)
        if load_first != -1:
            self.data_list = self.data_list[:load_first] # [('/data/ruizhu/openrooms_mini-val/mainDiffLight_xml1/scene0509_00/im_1.hdr', '/data/ruizhu/openrooms_mini-val/main_xml1/scene0509_00/imsemLabel_1.npy'), ...
            self.meta_split_scene_name_frame_id_list = self.meta_split_scene_name_frame_id_list[:load_first] # [('mainDiffLight_xml1', 'scene0509_00', 1)

        logger.info(white_blue('%s-%s: total frames: %d; total scenes %d; from path %s'%(self.dataset_name, self.split, len(self.data_list), len(self.all_scenes_list) if self.all_scenes_list is not None else -1, self.data_root   )))

        self.cascadeLevel = cascadeLevel

        self.transforms_BRDF = transforms_BRDF

        self.logger = logger
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        self.if_extra_op = False

    def __len__(self):
        return len(self.data_list)

    def load_one_pickle_tables(self, frame_info, if_load_immask=False):
        pickle_return_dict = {}
        file_path_h5 = Path(self.data_root) / frame_info['meta_split'] / frame_info['scene_name'] / ('%06d.h5'%frame_info['frame_id'])
        assert file_path_h5.exists(), '%s does not exist!'%(str(file_path_h5))
        try:
            # print(str(file_path_h5))
            h5file = tables.open_file(str(file_path_h5), driver="H5FD_CORE")
            im_uint8_array = h5file.root.im_uint8.read()
            pickle_return_dict['im_uint8_array'] = im_uint8_array 
            if if_load_immask:
                seg_uint8_array = h5file.root.seg_uint8.read()
                mask_int32_array = h5file.root.mask_int32.read().astype(np.int32)

                pickle_return_dict['seg_uint8_array'] = seg_uint8_array
                pickle_return_dict['mask_int32_array'] = mask_int32_array
                if 'al' in self.cfg.DATA.data_read_list:
                    albedo_uint8_array = h5file.root.albedo_uint8.read()
                    pickle_return_dict['albedo_uint8_array'] = albedo_uint8_array
                if 'de' in self.cfg.DATA.data_read_list:
                    depth_float32_array = h5file.root.depth_float32.read()
                    pickle_return_dict['depth_float32_array'] = depth_float32_array
                if 'no' in self.cfg.DATA.data_read_list:
                    normal_float32_array = h5file.root.normal_float32.read()
                    pickle_return_dict['normal_float32_array'] = normal_float32_array
                if 'ro' in self.cfg.DATA.data_read_list:
                    rough_float32_array = h5file.root.rough_float32.read()
                    pickle_return_dict['rough_float32_array'] = rough_float32_array
                if 'semseg' in self.cfg.DATA.data_read_list:
                    semseg_uint8_array = h5file.root.semseg_uint8.read()
                    pickle_return_dict['semseg_uint8_array'] = semseg_uint8_array
            h5file.close()                

        except OSError:
            print('[!!!!!!] Error reading '+str(file_path_h5))

        return pickle_return_dict

    def __getitem__(self, index):

        hdr_image_path, semseg_label_path = self.data_list[index]
        meta_split, scene_name, frame_id = self.meta_split_scene_name_frame_id_list[index]
        assert frame_id > 0

        if self.opt.cfg.DATASET.tmp:
            png_image_path = Path(hdr_image_path.replace('.hdr', '.png').replace('.rgbe', '.png'))
        else:
            png_image_path = Path(self.opt.cfg.DATASET.png_path) / meta_split / scene_name / ('im_%d.png'%frame_id)
        frame_info = {'index': index, 'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id, 'frame_key': '%s-%s-%d'%(meta_split, scene_name, frame_id), \
            'png_image_path': png_image_path}
        batch_dict = {'image_index': index, 'frame_info': frame_info}

        # if_load_immask = self.opt.cfg.DATA.load_brdf_gt and (not self.opt.cfg.DATASET.if_no_gt_semantics)
        if_load_immask = self.opt.cfg.DATA.load_brdf_gt or self.opt.cfg.MODEL_MATSEG.enable
        # if_load_immask = False
        self.opt.if_load_immask = if_load_immask

        pickle_return_dict = self.load_one_pickle_tables(frame_info, if_load_immask=if_load_immask)

        if if_load_immask:
            # Read segmentation
            seg = 0.5 * (self.loadImage(im=pickle_return_dict['seg_uint8_array']) + 1)[0:1, :, :] # [1, h, w]

            semantics_path = hdr_image_path.replace('DiffMat', '').replace('DiffLight', '')
            mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
            mask = self.loadBinary(im=pickle_return_dict['mask_int32_array'], dtype=np.int32).squeeze() # [h, w, 3]
        else:
            seg = np.ones((1, self.im_height, self.im_width), dtype=np.float32)
            mask_path = ''
            mask = np.ones((self.im_height, self.im_width, 3), dtype=np.uint8)

        brdf_loss_mask = np.ones((self.im_height, self.im_width), dtype=np.uint8)
        pad_mask = np.ones((self.im_height, self.im_width), dtype=np.uint8)

        hdr_scale = 1.

        assert self.opt.cfg.DATA.if_load_png_not_hdr, 'only images in PNG format ara available in this version of pickled OR dataset'

        # Read PNG image
        im_fixedscale_SDR_uint8 = pickle_return_dict['im_uint8_array']
        assert im_fixedscale_SDR_uint8.shape[:2] == (self.opt.cfg.DATA.im_height_ori, self.opt.cfg.DATA.im_width_ori)
        im_trainval_SDR = self.transforms_BRDF(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]
        im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        im_trainval = im_trainval_SDR # [3, 240, 320], tensor, not in [0., 1.]

        batch_dict.update({'image_path': str(png_image_path), 'brdf_loss_mask': torch.from_numpy(brdf_loss_mask), 'pad_mask': torch.from_numpy(pad_mask)})
        batch_dict.update({'im_w_resized_to': self.im_width, 'im_h_resized_to': self.im_height})

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
        if self.opt.cfg.DATA.load_brdf_gt:
            batch_dict_brdf = self.load_brdf_lighting(pickle_return_dict, hdr_image_path, if_load_immask, mask_path, mask, seg, hdr_scale, frame_info)
            batch_dict.update(batch_dict_brdf)

        # ====== semseg =====
        if self.opt.cfg.DATA.load_semseg_gt:
            batch_dict_semseg = self.load_semseg(pickle_return_dict)
            batch_dict.update(batch_dict_semseg)

        return batch_dict

    def convert_write_png(self, hdr_image_path, seg, png_image_path):
        # Read HDR image
        im_ori = self.loadHdr(hdr_image_path)
        # == no random scaling for inference
        im_fixedscale, _ = self.scaleHdr(im_ori, seg, forced_fixed_scale=True)
        im_fixedscale_SDR = np.clip(im_fixedscale**(1.0/2.2), 0., 1.)
        im_fixedscale_SDR_uint8 = (255. * im_fixedscale_SDR).transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(im_fixedscale_SDR_uint8).save(png_image_path)
        print(yellow('>>> Saved png file to %s'%png_image_path))


    def load_brdf_lighting(self, pickle_return_dict, hdr_image_path, if_load_immask, mask_path, mask, seg, hdr_scale, frame_info):
        batch_dict_brdf = {}
        # Get paths for BRDF params
        if 'al' in self.cfg.DATA.data_read_list:
            albedo = self.loadImage(im=pickle_return_dict['albedo_uint8_array'], isGama = False)

            if self.opt.cfg.MODEL_BRDF.albedo.if_HDR:
                albedo = (0.5 * (albedo + 1) ) ** 2.2
            else:
                albedo = 0.5 * (albedo + 1)

            batch_dict_brdf.update({'albedo': torch.from_numpy(albedo)})

        if 'no' in self.cfg.DATA.data_read_list:
            normal = pickle_return_dict['normal_float32_array'] # [-1, 1], [1, H, W]
            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

            batch_dict_brdf.update({'normal': torch.from_numpy(normal),})

        if 'ro' in self.cfg.DATA.data_read_list:
            rough = pickle_return_dict['rough_float32_array'] # [-1, 1], [1, H, W]

            batch_dict_brdf.update({'rough': torch.from_numpy(rough),})

        if 'de' in self.cfg.DATA.data_read_list or 'de' in self.cfg.DATA.data_read_list:
            depth = self.loadBinary(im=pickle_return_dict['depth_float32_array'])

            batch_dict_brdf.update({'depth': torch.from_numpy(depth),})

        if if_load_immask:
            # Example: https://i.imgur.com/G5yGAaH.png
            # for albedo and roughness: segBRDFBatch = segObj; 
            # for depth and geometry: segAllBatch = segObj + segArea
            segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 ) # lamps
            segEnv = (seg < 0.1).astype(np.float32 ) # windows
            segObj = (seg > 0.9) # other objects/stuff
            segObj = segObj.astype(np.float32 )
        else:
            segObj = np.ones_like(seg, dtype=np.float32)
            segEnv = np.zeros_like(seg, dtype=np.float32)
            segArea = np.zeros_like(seg, dtype=np.float32)

        # if if_load_immask:
        batch_dict_brdf.update({
                'mask': torch.from_numpy(mask), 
                'maskPath': mask_path, 
                'segArea': torch.from_numpy(segArea),
                'segEnv': torch.from_numpy(segEnv),
                'segObj': torch.from_numpy(segObj),
                'object_type_seg': torch.from_numpy(seg), 
                })

        return batch_dict_brdf

    def load_semseg(self, pickle_return_dict, ):
        semseg_label = pickle_return_dict['semseg_uint8_array'][np.newaxis, :]

        return {'semseg_label': torch.from_numpy(semseg_label).long().squeeze(0)}

    def loadImage(self, imName=None, im=None, isGama = False, if_flip_to_channel_first=True, if_already_float=False):

        if im is None:
            if not(osp.isfile(imName ) ):
                self.logger.warning('File does not exist: ' + imName )
                assert(False), 'File does not exist: ' + imName 
            im = Image.open(imName)
            im = im.resize([self.im_width, self.im_height], Image.ANTIALIAS )
            im = np.asarray(im, dtype=np.float32)
        
        if if_already_float:
            pass
        else:
            if isGama:
                im = (im / 255.0) ** 2.2
                im = 2 * im - 1
            else:
                im = (im - 127.5) / 127.5

        if len(im.shape) == 2:
            im = im[:, np.newaxis]

        if if_flip_to_channel_first:
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
            scale = (0.95 - 0.1 * random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
            # if if_print:
            #     print(self.split, not forced_fixed_scale, scale)

        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName=None, im=None, channels = 1, dtype=np.float32, if_resize=True):
        if im is None:
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
                im = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
                im = im.reshape([height, width, channels] )
                if if_resize:
                    # print(self.im_width, self.im_height, width, height)
                    if dtype == np.float32:
                        im = cv2.resize(im, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA )
                    elif dtype == np.int32:
                        im = cv2.resize(im.astype(np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                        im = im.astype(np.int32)

                im = np.squeeze(im)

        return im[np.newaxis, :, :]

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
                if subkey in ['mask']: # list of lists
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    assert False
                collated_batch[key][subkey] = tensor_batch
        elif key in ['frame_info', 'image_index']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except:
                print('[!!!!] Type error in collate_fn_OR: ', key)

    return collated_batch