# import glob
import numpy as np
import h5py
import torch
import torchvision.transforms as T
from utils.utils_misc import *
from pathlib import Path
# import pickle


import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")


def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

def get_valid_scenes(opt, frames_list_path, split, logger=None):
    scenes_list_path = str(frames_list_path).replace('.txt', '_scenes.txt')
    if not os.path.isfile(scenes_list_path):
        raise (RuntimeError("Scene list file do not exist: " + scenes_list_path + "\n"))
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

def make_dataset(opt, split, task, data_root=None, data_list=None, logger=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    if logger is None:
        logger = basic_logger()
    image_label_list = []
    meta_split_scene_name_frame_id_list = []
    list_read = open(data_list).readlines()
    logger.info("Totally {} samples in {} set.".format(len(list_read), split))
    logger.info("Starting Checking image&label pair {} list...".format(split))
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

    all_scenes = get_valid_scenes(opt, data_list, split, logger=logger)

    if opt.cfg.DATASET.first_scenes != -1:
        # return image_label_list[:opt.cfg.DATASET.first_scenes], meta_split_scene_name_frame_id_list[:opt.cfg.DATASET.first_scenes]
        assert False
    # elif opt.cfg.DATASET.if_quarter and task != 'vis':
    elif opt.cfg.DATASET.if_quarter and task in ['train']:
        meta_split_scene_name_frame_id_list_quarter = return_percent(meta_split_scene_name_frame_id_list, 0.25)
        all_scenes = list(set(['/'.join([x[0], x[1]]) for x in meta_split_scene_name_frame_id_list_quarter]))
        all_scenes = [x.split('/') for x in all_scenes]
        return return_percent(image_label_list, 0.25), meta_split_scene_name_frame_id_list_quarter, all_scenes

    else:
        return image_label_list, meta_split_scene_name_frame_id_list, all_scenes

def make_dataset_real(opt, data_root, data_list, logger=None):
    split = 'test_real'

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    if logger is None:
        logger = basic_logger()

    list_read = open(data_list).readlines()
    logger.info("Totally {} samples in {} set.".format(len(list_read), split))
    logger.info("Starting Checking image&label pair {} list...".format(split))

    image_list = []
    for line in list_read:
        line = line.strip()
        image_path = Path(data_root) / line
        image_list.append(image_path)

    return image_list

