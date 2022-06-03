"""
AutoAugment with Mask
Inspired by TokenLabeling code
https://github.com/zihangJiang/TokenLabeling/blob/main/tlt/data/random_augment_label.py
"""
import math
import random
import numpy as np
from scipy import ndimage
import torch

from timm.data.auto_augment import rotate, shear_x, shear_y, translate_x_rel, translate_y_rel, _LEVEL_DENOM


def rotate_tensor(img, degrees, **kwargs):
    w, h = img.shape[-2:]
    post_trans = (0, 0)
    rotn_center = (w / 2.0, h / 2.0)
    angle = math.radians(degrees)
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(
        -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
    )
    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]
    return affine_tensor(img, matrix)


def shear_x_tensor(img, factor, **kwargs):
    return affine_tensor(img, (1, 0, 0, factor, 1, 0))


def shear_y_tensor(img, factor, **kwargs):
    return affine_tensor(img, (1, factor, 0, 0, 1, 0))


def translate_x_rel_tensor(img, pct, **kwargs):
    pixels = pct * img.shape[-1]
    return affine_tensor(img, (1, 0, 0, 0, 1, pixels))


def translate_y_rel_tensor(img, pct, **kwargs):
    pixels = pct * img.shape[-2]
    return affine_tensor(img, (1, 0, pixels, 0, 1, 0))


def affine_tensor(img, matrix):
    a, b, c, d, e, f = matrix
    affine_matrix = [[1, 0, 0, 0], [0, a, b, c], [0, d, e, f]]

    if img.dim() == 3:  # (C, H, W) format
        ret = ndimage.affine_transform(img, matrix=affine_matrix, order=0, mode="constant")
    elif img.dim() == 4:  # (2, K, H, W) format
        value = ndimage.affine_transform(img[0], matrix=affine_matrix, order=0, mode="constant")
        index = ndimage.affine_transform(img[1], matrix=affine_matrix, order=0, mode="nearest")
        ret = np.stack([value, index], axis=0)
    else:
        raise ValueError

    return torch.from_numpy(ret)


_GEOMETRIC_TRANSFORMS = {
    rotate: rotate_tensor,
    shear_x: shear_x_tensor,
    shear_y: shear_y_tensor,
    translate_x_rel: translate_x_rel_tensor,
    translate_y_rel: translate_y_rel_tensor,
}


class RandAugmentWithMask:
    def __init__(self, transform):
        self.t = transform
        # convert geometric ops to paired version
        for i, op in enumerate(self.t.ops):
            self.t.ops[i] = AugmentOpWithMask(op)

    def __call__(self, img, seg):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.t.ops, self.t.num_layers, replace=self.t.choice_weights is None, p=self.t.choice_weights)
        for op in ops:
            img, seg = op(img, seg)
        return img, seg


class AugmentOpWithMask:
    def __init__(self, op):
        self.op = op
        self.aug_fn = op.aug_fn

    def __call__(self, img, seg):
        if self.op.prob < 1.0 and random.random() > self.op.prob:
            return img, seg
        magnitude = self.op.magnitude
        if self.op.magnitude_std > 0:
            # magnitude randomization enabled
            if self.op.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.op.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.op.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.op.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.op.level_fn(magnitude, self.op.hparams) if self.op.level_fn is not None else tuple()

        aug_fn = self.op.aug_fn
        img = aug_fn(img, *level_args, **self.op.kwargs)

        if aug_fn in _GEOMETRIC_TRANSFORMS.keys():
            if isinstance(seg, torch.Tensor):  # tensor
                aug_fn_tensor = _GEOMETRIC_TRANSFORMS[aug_fn]
                seg = aug_fn_tensor(seg, *level_args, **self.op.kwargs)
            else:  # pil image
                seg = aug_fn(seg, *level_args, **self.op.kwargs)

        return img, seg
