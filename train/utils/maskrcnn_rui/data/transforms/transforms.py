[]# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import PIL
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import Lambda
from torch.nn.functional import interpolate, pad
# from torchvision.transforms import RandomCrop
from torch import Tensor
from typing import Tuple, List, Optional
import numpy as np
from utils.utils_misc import checkEqual1
import numbers

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, interpolation='BILINEAR'):
        # interpolation: 'BILINEAR', 'NEAREST', ...
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size        
        assert interpolation in ['BILINEAR', 'NEAREST'], 'interpolation option not supported!'
        if interpolation=='BILINEAR':
            self.PIL_interpolation = PIL.Image.BILINEAR # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize
            self.F_interpolation = 'bilinear'
        elif interpolation=='NEAREST':
            self.PIL_interpolation = PIL.Image.NEAREST
            self.F_interpolation = 'nearest'

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target=None):
        if isinstance(images, list):
            size = self.get_size(images[0].size)
            image_size = images[0].size
            assert checkEqual1([X.size for X in images]), 'sizes of an image list should all equal!'
            images = [F.resize(X, size, interpolation=self.PIL_interpolation) for X in images]
        else:
            size = self.get_size(images.size)
            image_size = images.size
            images = F.resize(images, size, interpolation=self.PIL_interpolation)

        if target is None:
            return images
        # target = target.resize((image.size[0], image.size[1], -1))
        target.unsqueeze_(0)
        target = interpolate(target, size=(image_size[1], image_size[0]), mode=self.F_interpolation) # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
        target.squeeze_(0)
        return image, target

# class RandomCrop(object):
#     def __init__(self, min_size, max_size):
#         if not isinstance(min_size, (list, tuple)):
#             min_size = (min_size,)
#         self.min_size = min_size
#         self.max_size = max_size        

#     # modified from torchvision to add support for max size
#     def get_size(self, image_size):
#         w, h = image_size
#         size = random.choice(self.min_size)
#         max_size = self.max_size
#         if max_size is not None:
#             min_original_size = float(min((w, h)))
#             max_original_size = float(max((w, h)))
#             if max_original_size / min_original_size * size > max_size:
#                 size = int(round(max_size * min_original_size / max_original_size))

#         if (w <= h and w == size) or (h <= w and h == size):
#             return (h, w)

#         if w < h:
#             ow = size
#             oh = int(size * h / w)
#         else:
#             oh = size
#             ow = int(size * w / h)

#         return (oh, ow)

#     @staticmethod
#     def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
#         """Get parameters for ``crop`` for a random crop.
#         Taken from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop

#         Args:
#             img (PIL Image or Tensor): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.

#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         w, h = _get_image_size(img)
#         th, tw = output_size
#         if w == tw and h == th:
#             return 0, 0, h, w

#         i = torch.randint(0, h - th + 1, size=(1, )).item()
#         j = torch.randint(0, w - tw + 1, size=(1, )).item()
#         return i, j, th, tw

#     def __call__(self, image, target=None):
#         crop_size = self.get_size(image.size) # (h, w)
#         assert image.size[0] > crop_size[1] # im_W > crop_W
#         assert image.size[1] > crop_size[0] # im_H > crop_H
#         i, j, h, w = self.get_params(image, crop_size)
#         # image = F.resize(image, size, interpolation=self.PIL_interpolation)
#         image = F.crop(image, i, j, h, w)
#         if target is None:
#             return image
#         # target = target.resize((image.size[0], image.size[1], -1))
#         # target.unsqueeze_(0)
#         # print('--0', target.shape)
#         # target = F.crop(image, i, j, h, w) # ONLY FOR PYTORCH>=1.6.0
#         # print('--1', target.shape)
#         # target.squeeze_(0)
#         return image, target[..., i:i+h, j:j+w]

class RandomCrop(object):
    def __init__(self, H_cropto, W_cropto):
        # if not isinstance(min_size, (list, tuple)):
        #     min_size = (min_size,)
        self.H_cropto = H_cropto
        self.W_cropto = W_cropto

    # # modified from torchvision to add support for max size
    # def get_size(self, image_size):
    #     w, h = image_size
    #     size = random.choice(self.min_size)
    #     max_size = self.max_size
    #     if max_size is not None:
    #         min_original_size = float(min((w, h)))
    #         max_original_size = float(max((w, h)))
    #         if max_original_size / min_original_size * size > max_size:
    #             size = int(round(max_size * min_original_size / max_original_size))

    #     if (w <= h and w == size) or (h <= w and h == size):
    #         return (h, w)

    #     if w < h:
    #         ow = size
    #         oh = int(size * h / w)
    #     else:
    #         oh = size
    #         ow = int(size * w / h)

    #     return (oh, ow)

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        Taken from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __call__(self, images, target=None):
        # crop_size = self.get_size(image.size) # (h, w)
        crop_size = (self.H_cropto, self.W_cropto)
        if isinstance(images, list):
            image_size = images[0].size
            assert checkEqual1([X.size for X in images]), 'sizes of an image list should all equal!'
            sample_image = images[0]
        else:
            image_size = images.size
            sample_image = images
        assert image_size[0] >= crop_size[1] # im_W > crop_W
        assert image_size[1] >= crop_size[0] # im_H > crop_H
        i, j, h, w = self.get_params(sample_image, crop_size)
        if isinstance(images, list):
            images = [F.crop(X, i, j, h, w) for X in images]
        else:
            images = F.crop(images, i, j, h, w)
        if target is None:
            return images
        # target = target.resize((image.size[0], image.size[1], -1))
        # target.unsqueeze_(0)
        # print('--0', target.shape)
        # target = F.crop(image, i, j, h, w) # ONLY FOR PYTORCH>=1.6.0
        # print('--1', target.shape)
        # target.squeeze_(0)
        return images, target[..., i:i+h, j:j+w]


class CenterCrop(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size        

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target=None):
        if isinstance(images, list):
            image_size = images[0].size
            assert checkEqual1([X.size for X in images]), 'sizes of an image list should all equal!'
            sample_image = images[0]
        else:
            image_size = images.size
            sample_image = images

        crop_size = self.get_size(sample_image.size) # (h, w)
        assert image_size[0] > crop_size[1] # im_W > crop_W
        assert image_size[1] > crop_size[0] # im_H > crop_H
        # image = F.resize(image, size, interpolation=self.PIL_interpolation)

        if isinstance(images, list):
            images = [F.center_crop(X, crop_size) for X in images]
        else:
            images = F.center_crop(images, crop_size)
        if target is None:
            return images

        image_height = sample_image.size[1]
        image_width = sample_image.size[0]
        crop_top = int(round((image_height - crop_size[0]) / 2.))
        crop_left = int(round((image_width - crop_size[1]) / 2.))
        return images, target[..., crop_top:crop_top+crop_size[0], crop_left:crop_left+crop_size[1]]

class Pad(object):
    def __init__(self, H_padto, W_padto):
        # if not isinstance(min_size, (list, tuple)):
        #     min_size = (min_size,)
        self.H_padto = H_padto
        self.W_padto = W_padto

    def __call__(self, images, target=None):
        if isinstance(images, list):
            image_size = images[0].size
            assert checkEqual1([X.size for X in images]), 'sizes of an image list should all equal!'
            sample_image = images[0]
        else:
            image_size = images.size
            sample_image = images

        image_height = image_size[1]
        image_width = image_size[0]
        if self.H_padto <= 0 or self.W_padto <= 0:
            H_padto = int(np.ceil(image_height/8/2)*2*8)
            W_padto = int(np.ceil(image_width/8/2)*2*8)
        else:
            H_padto = self.H_padto
            W_padto = self.W_padto
        assert image_width <= W_padto, 'Pad to W %d has to be smaller than imW %d'%(W_padto, image_width) # im_W > crop_W
        assert image_height <= H_padto, 'Pad to H %d has to be smaller than imH %d'%(H_padto, image_height) # im_H > crop_H
        pad_right = W_padto - image_width
        pad_bottom = H_padto - image_height

        if isinstance(images, list):
            images = [F.pad(X, (0, 0, pad_right, pad_bottom)) for X in images]
        else:
            images = F.pad(images, (0, 0, pad_right, pad_bottom))

        if target is None:
            return images
        target_padded = pad(target, (0, pad_right, 0, pad_bottom), 'constant', 0)
        # print('>>>target_padded', target_padded.shape)
        return images, target_padded


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target=None):
        if random.random() < self.prob:
            if isinstance(images, list):
                images = [F.hflip(X) for X in images]
            else:
                images = F.hflip(images)
            if target is not None:
                target = target.flip(2)
        return images, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            if isinstance(images, list):
                images = [F.vflip(X) for X in images]
            else:
                images = F.vflip(images)
            if target is not None:
                target = target.flip(1)
        return images, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        # self.color_jitter = torchvision.transforms.ColorJitter(
        self.color_jitter = ColorJitter_torch(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, images, target=None):
        if isinstance(images, list):
            _, jitter_params = self.color_jitter(images[0], not_adjust=True)
            images = [self.color_jitter(X, jitter_params=jitter_params)[0] for X in images]
        else:
            images, _ = self.color_jitter(images)
        if target is None:
            return images
        return images, target

class ColorJitter_torch(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img, jitter_params={}, not_adjust=False):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        # assert len(jitter_params) in [0, 2]
        if jitter_params:
            fn_idx = jitter_params['fn_idx']
        else:
            fn_idx = torch.randperm(4)
            # print('Copy jittering params')
        jitter_params_return = {'fn_idx': fn_idx}
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                if jitter_params:
                    brightness_factor = jitter_params['brightness_factor']
                else:
                    brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                jitter_params_return.update({'brightness_factor': brightness_factor})
                if not not_adjust:
                    img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                if jitter_params:
                    contrast_factor = jitter_params['contrast_factor']
                else:
                    contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                jitter_params_return.update({'contrast_factor': contrast_factor})
                if not not_adjust:
                    img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                if jitter_params:
                    saturation_factor = jitter_params['saturation_factor']
                else:
                    saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                jitter_params_return.update({'saturation_factor': saturation_factor})
                if not not_adjust:
                    img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                if jitter_params:
                    hue_factor = jitter_params['hue_factor']
                else:
                    hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                jitter_params_return.update({'hue_factor': hue_factor})
                if not not_adjust:
                    img = F.adjust_hue(img, hue_factor)

        return img, jitter_params_return

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class ToTensor(object):
    def __call__(self, images, target=None):
        if isinstance(images, list):
            images_tensors = [F.to_tensor(X) for X in images]
        else:
            images_tensors = F.to_tensor(image)
        if target is None:
            return images_tensors
        return images_tensors, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, images, target=None):
        if isinstance(images, list):
            if self.to_bgr255:
                images = [X[[2, 1, 0]] * 255 for X in images]
            images = [F.normalize(X, mean=self.mean, std=self.std) for X in images]
        else:
            if self.to_bgr255:
                images = images[[2, 1, 0]] * 255
            images = F.normalize(images, mean=self.mean, std=self.std)
        if target is None:
            return images
        return images, target
