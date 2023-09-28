import math
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, seg=None):
        for t in self.transforms:
            image, seg = t(image, seg)
        return image, seg

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Resize:
    def __init__(self, size, ratio, interpolation=T.InterpolationMode.BILINEAR):
        self.img_size = size
        self.seg_size = (int(size[0] / ratio), int(size[1] / ratio))
        self.interpolation = interpolation

    def __call__(self, image, seg=None):
        image = (
            F.resize(image, self.img_size, self.interpolation)
            if image is not None
            else None
        )
        seg = F.resize(seg, self.seg_size, T.InterpolationMode.NEAREST) if seg else None
        return image, seg


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, seg=None):
        if random.random() < self.prob:
            image = F.hflip(image) if image is not None else None
            seg = F.hflip(seg) if seg else None
        return image, seg


class Pad:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, seg=None):
        image = (
            F.pad(image, self.padding, self.fill, self.padding_mode)
            if image is not None
            else None
        )
        seg = F.pad(seg, self.padding, self.fill, self.padding_mode) if seg else None
        return image, seg


class RandomCrop:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, seg=None):
        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w) if image is not None else None
        seg = F.crop(seg, i, j, h, w) if seg else None
        return image, seg


class ToTensor:
    def __init__(self, n_labels=24):
        self.n_labels = n_labels

    def __call__(self, image, seg=None):
        image = F.to_tensor(image) if image is not None else None
        if seg is not None:
            seg = np.array(seg)
            if seg.ndim > 2:
                # only use I of densepose
                seg = seg[:, :, 2]
            seg = np.eye(self.n_labels)[seg]
            seg = torch.from_numpy(seg).permute(2, 0, 1)
        return image, seg


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, seg=None):
        image = (
            F.normalize(image, mean=self.mean, std=self.std)
            if image is not None
            else None
        )
        return image, seg


def build_dual_transforms(height, width, n_labels, ratio, is_train=True):
    normalize_transform = Normalize(mean=0.5, std=0.5)

    if is_train:
        # TODO: add option for random crop
        transform = Compose(
            [
                # RandomCrop((height, width)),
                Resize((height, width), ratio=ratio),
                RandomHorizontalFlip(0.5),
                ToTensor(n_labels),
                normalize_transform,
            ]
        )
    else:
        transform = Compose(
            [
                # RandomCrop((height, width)),
                Resize((height, width), ratio=ratio),
                ToTensor(n_labels),
                normalize_transform,
            ]
        )
    return transform


PART2PIXEL = {
    "background": [0, 0, 0],
    "top": [255, 250, 250],
    "skirt": [250, 235, 215],
    "leggings": [70, 130, 180],
    "dress": [16, 78, 139],
    "outer": [255, 250, 205],
    "pants": [255, 140, 0],
    "bag": [50, 205, 50],
    "neckwear": [220, 220, 220],
    "headwear": [255, 0, 0],
    "eyeglass": [127, 255, 212],
    "belt": [0, 100, 0],
    "footwear": [255, 255, 0],
    "hair": [211, 211, 211],
    "skin": [144, 238, 144],
    "face": [245, 222, 179],
}


def build_transforms(height, width, is_train=True):
    normalize_transform = T.Normalize(mean=0.5, std=0.5)

    if is_train:
        transform = T.Compose(
            [
                # T.RandomResizedCrop((height, width), scale=(0.5, 1.0)),
                T.Resize((height, width)),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((height, width)),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform


def get_mask(segm, part, height, width):
    segm = segm.convert("RGB").resize((width, height), 0)
    segm = np.array(segm)
    pixel = PART2PIXEL[part]
    mask = (
        (segm[:, :, 0] == pixel[0])
        & (segm[:, :, 1] == pixel[1])
        & (segm[:, :, 2] == pixel[2])
    )
    mask = torch.from_numpy(mask)
    return mask
