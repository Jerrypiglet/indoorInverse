import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch


class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label=None):
        for t in self.segtransform:
            if label is not None:
                image, label = t(image, label)
            else:
                image = t(image)

        if label is not None:
            return image, label
        else:
            return image


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label=None):
        if not isinstance(image, np.ndarray):
            raise (RuntimeError("image.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if label is not None and not isinstance(label, np.ndarray):
            raise (RuntimeError("label.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if label is not None and not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        if label is not None:
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()
            return image, label
        else:
            return image


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label=None):
        # print(torch.max(image), torch.min(image), '--')
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        # print(torch.max(image), torch.min(image), '<--', self.mean, self.std)
        if label is not None:
            return image, label
        else:
            return image


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (w, h).
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2) and isinstance(size, tuple)
        self.size = size

    def __call__(self, image, label=None):
        # if min(image.shape)==0:
        #     print(image.shape, self.size)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            return image

class Resize_flexible(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (w, h).
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2) and isinstance(size, tuple)
        self.size = size

    def __call__(self, image, label=None, if_channel_first=False, if_channel_2_input=False, name=''):
        if image is None:
            return image

        if if_channel_2_input:
            image = image[:, :, np.newaxis]
        assert len(image.shape)==3

        if if_channel_first:
            image = image.transpose(1, 2, 0)
            if label is not None:
                label = label.transpose(1, 2, 0)

        h, w, c = image.shape
        assert c in [1, 3]

        # print(image.shape, image.dtype, self.size)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if label is not None:    
            label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)

        if if_channel_first:
            if c == 1:
                image = image[np.newaxis, :, :]
            else:
                image = image.transpose(2, 0, 1)
            if label is not None:
                if c == 1:
                    label = label[:, :, np.newaxis]
                else:
                    label = label.transpose(2, 0, 1)

        if if_channel_2_input:
            assert len(image.shape)==2
            if label is not None:
                assert len(label.shape)==2

        if label is not None:
            return image, label
        else:
            return image


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label=None):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            return image


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label=None):
        # h, w = label.shape
        h, w, _ = image.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            if label is not None:    
                label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        # h, w = label.shape
        h, w, _ = image.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if label is not None:
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
            return image, label
        else:
            return image

class Pad(object):
    def __init__(self, pad_to_hw, padding_with=0, pad_option='const'):
        self.pad_option = pad_option
        assert self.pad_option in ['const', 'reflect']
        if isinstance(pad_to_hw, int):
            self.pad_h = pad_to_hw
            self.pad_w = pad_to_hw
        elif isinstance(pad_to_hw, collections.Iterable) and len(pad_to_hw) == 2 \
                and isinstance(pad_to_hw[0], int) and isinstance(pad_to_hw[1], int) \
                and pad_to_hw[0] > 0 and pad_to_hw[1] > 0:
            self.pad_h = pad_to_hw[0]
            self.pad_w = pad_to_hw[1]
        else:
            print(pad_to_hw, isinstance(pad_to_hw, collections.Iterable), len(pad_to_hw))
            raise (RuntimeError("pad to size error.\n"))
        self.padding_with = padding_with
        assert isinstance(self.padding_with, numbers.Number)

    def __call__(self, image, label=None, if_channel_first=False, if_channel_2_input=False, name='', if_padding_constant=False):
        if if_padding_constant or self.pad_option=='const':
            pad_mode = cv2.BORDER_CONSTANT
        else:
            pad_mode = cv2.BORDER_REFLECT
            
        if image is None:
            return image
        # h, w = label.shape
        # print(image.shape, name, len(image.shape))
        if if_channel_2_input:
            image = image[:, :, np.newaxis]
        assert len(image.shape)==3

        if if_channel_first:
            image = image.transpose(1, 2, 0)
            if label is not None:
                label = label.transpose(1, 2, 0)

        h, w, c = image.shape
        assert c in [1, 3]
        pad_h = max(self.pad_h - h, 0)
        pad_w = max(self.pad_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.padding_with is None:
                raise (RuntimeError("segtransform.Pad() need padding while padding argument is None\n"))
            # if name=='brdf_loss_mask':
            #     print(image.shape)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, pad_mode, value=self.padding_with)
            # if name=='brdf_loss_mask':
            #     print(image.shape, '---')
            if label is not None:    
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, pad_mode, value=self.padding_with)

        if if_channel_first:
            if c == 1:
                image = image[np.newaxis, :, :]
            else:
                image = image.transpose(2, 0, 1)
            if label is not None:
                if c == 1:
                    label = label[:, :, np.newaxis]
                else:
                    label = label.transpose(2, 0, 1)

        if if_channel_2_input:
            assert len(image.shape)==2
        #     if name=='brdf_loss_mask':
        #         print(image.shape, '--------')
        #     image = image.squeeze(2)
            if label is not None:
                assert len(label.shape)==2
        #         label = label.squeeze(2)

        # print(image.shape)

        if label is not None:
            return image, label
        else:
            return image


class CropBdb(object):
    """Crops the given ndarray image (H*W*C or H*W) from a bounding box bdb: [x1, x2, y1, y2].
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, bdb):
        self.bdb = bdb

    def __call__(self, image, label=None):
        h, w, _ = image.shape
        x1, y1, x2, y2 = int(np.round(self.bdb[0])), int(np.round(self.bdb[1])), int(np.round(self.bdb[2])), int(np.round(self.bdb[3]))
        x1 = np.clip(x1, 0, w-1)
        x2 = np.clip(x2, 0, w-1)
        y1 = np.clip(y1, 0, h-1)
        y2 = np.clip(y2, 0, h-1)
        image = image[y1:y2, x1:x2]
        if label is not None:
            label = label[y1:y2, x1:x2]
            return image, label
        else:
            return image


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            # h, w = label.shape
            h, w, _ = image.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            if label is not None:
                label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        if label is not None:
            return image, label
        else:
            return image

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            if label is not None:
                label = cv2.flip(label, 1)
        if label is not None:
            return image, label
        else:
            return image


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label=None):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            if label is not None:
                label = cv2.flip(label, 0)
        if label is not None:
            return image, label
        else:
            return image


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label=None):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        if label is not None:
            return image, label
        else:
            return image


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label=None):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if label is not None:
            return image, label
        else:
            return image


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label is not None:
            return image, label
        else:
            return image
