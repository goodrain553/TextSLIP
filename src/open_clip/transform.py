# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import ImageFilter
import random


from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, RandomApply, ColorJitter, RandomGrayscale, RandomHorizontalFlip


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def get_mean_std(args=None):
    mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    return mean, std


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        inmem = False
):
    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        if inmem:
            return Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                F.pil_to_tensor
            ])
        else:
            return Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DualAugmentTransform:
    """
    返回同一张图像的三个版本用于 SLIP loss:
    - original: 标准增强（用于 CLIP loss）
    - aug1: 强增强（用于 SimCLR loss）
    - aug2: 强增强（用于 SimCLR loss）
    """
    def __init__(self, standard_transform, strong_transform):
        """
        Args:
            standard_transform: 标准增强 pipeline (resize + normalize)
            strong_transform: 强增强 pipeline (random crop, color jitter, etc.)
        """
        self.standard_transform = standard_transform
        self.strong_transform = strong_transform
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            tuple: (original, aug1, aug2) 三个版本
        """
        original = self.standard_transform(img)
        aug1 = self.strong_transform(img)
        aug2 = self.strong_transform(img)
        return (original, aug1, aug2)


def image_transform_slip(
        image_size: int = 224,
        is_train: bool = True,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        inmem: bool = False
):
    """
    SLIP 训练用的图像变换
    返回三个版本：标准增强（CLIP） + 两个强增强（SimCLR）
    """
    # 使用 ImageNet 的 mean 和 std（与 SLIP 论文一致）
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        image_size = image_size[0]
    
    normalize = Normalize(mean=mean, std=std)
    
    if is_train:
        # 标准增强：用于 CLIP loss（与标准 CLIP 相同）
        if inmem:
            standard_transform = Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                F.pil_to_tensor,
            ])
        else:
            standard_transform = Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
        
        # 强增强：用于 SimCLR loss（SimCLR 风格的强数据增强）
        if inmem:
            strong_transform = Compose([
                RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomApply([
                    ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                RandomGrayscale(p=0.2),
                RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                RandomHorizontalFlip(),
                _convert_to_rgb,
                F.pil_to_tensor,
            ])
        else:
            strong_transform = Compose([
                RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomApply([
                    ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                RandomGrayscale(p=0.2),
                RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                RandomHorizontalFlip(),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
        
        # 返回 DualAugmentTransform，产生 (original, aug1, aug2)
        return DualAugmentTransform(standard_transform, strong_transform)
    else:
        # 测试时使用标准的 center crop，不需要增强
        if inmem:
            return Compose([
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                _convert_to_rgb,
                F.pil_to_tensor,
            ])
        else:
            return Compose([
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])