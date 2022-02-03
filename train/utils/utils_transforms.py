import numpy as np
import torch
from utils import transform


def get_transform_BRDF(split, opt, pad_op_override=None):
    assert split in ['train', 'val', 'test']
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if split == 'train':
        transform_list_train = [
            # transform.RandScale([opt.semseg_configs.scale_min, opt.semseg_configs.scale_max]),
            # transform.RandRotate([opt.semseg_configs.rotate_min, opt.semseg_configs.rotate_max], padding=mean, ignore_label=opt.semseg_configs.ignore_label),
            # transform.RandomGaussianBlur(),
            # transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
        train_transform = transform.Compose(transform_list_train)
        return train_transform
    else:
        transform_list_val = [
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ]
        val_transform = transform.Compose(transform_list_val)
    return val_transform