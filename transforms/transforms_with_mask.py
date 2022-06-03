import math
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from timm.data.random_erasing import _get_pixels
from timm.data.mixup import Mixup, cutmix_bbox_and_lam, mixup_target
from .auto_augment_with_mask import RandAugmentWithMask

from torchvision.ops import roi_align


TRANSFORMS_WITH_MASK = [
    'RandomResizedCropAndInterpolation',
    'Resize',
    'RandomHorizontalFlip',
    'RandAugment',
    'RandomErasing',
    'CenterCrop',
]


class ComposeWithMask:
    def __init__(self, transform):
        assert isinstance(transform, T.Compose)
        # convert transforms to paired version
        self.transforms = []
        for t in transform.transforms:
            name = type(t).__name__
            if name in TRANSFORMS_WITH_MASK:
                t = globals()[name + 'WithMask'](t)
            elif name in ['ToTensor', 'Normalize']:
                pass
            else:
                raise NotImplementedError('Mask version of {} is not implemented'.format(name))
            self.transforms.append(t)

    def __call__(self, img, seg):
        for t in self.transforms:
            name = type(t).__name__
            if name.replace('WithMask', '') in TRANSFORMS_WITH_MASK:
                img, seg = t(img, seg)
            elif name == 'ToTensor':
                img = img if isinstance(img, torch.Tensor) else t(img)
                seg = seg if isinstance(seg, torch.Tensor) else t(seg)
            else:
                img = t(img)
        return img, seg


class RandomResizedCropAndInterpolationWithMask:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img, seg):
        i, j, h, w = self.t.get_params(img, self.t.scale, self.t.ratio)
        if isinstance(self.t.interpolation, (tuple, list)):
            interpolation = random.choice(self.t.interpolation)
        else:
            interpolation = self.t.interpolation

        img_h, img_w = img.size[-2:]  # get size before crop
        img = TF.resized_crop(img, i, j, h, w, self.t.size, interpolation)

        if isinstance(seg, torch.Tensor) and img.size[:-2] != seg.shape[:-2]:  # relabel
            seg_h, seg_w = seg.shape[-2:]  # torch Tensor

            x1 = j * seg_w / img_h
            y1 = i * seg_h / img_w
            x2 = (j + w) * seg_w / img_h
            y2 = (i + h) * seg_h / img_w

            boxes = torch.tensor([0, x1, y1, x2, y2])
            seg = roi_align(input=seg.unsqueeze(0), boxes=boxes.unsqueeze(0), output_size=(seg_h, seg_w), aligned=True)[0]
        else:
            seg = TF.resized_crop(seg, i, j, h, w, self.t.size, TF.InterpolationMode.NEAREST)

        return img, seg


class ResizeWithMask:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img, seg):
        img = TF.resize(img, self.t.size, self.t.interpolation)

        if isinstance(seg, torch.Tensor) and img.size[:-2] != seg.shape[:-2]:  # relabel
            pass  # skip augmentation for relabel
        else:
            seg = TF.resize(seg, self.t.size, TF.InterpolationMode.NEAREST)

        return img, seg


class CenterCropWithMask:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img, seg):
        img_h, img_w = img.size[-2:]  # get size before crop
        img = self.t(img)

        if isinstance(seg, torch.Tensor) and img.size[:-2] != seg.shape[:-2]:  # relabel
            seg_h, seg_w = seg.shape[-2:]  # torch Tensor
            size_h, size_w = self.t.size  # crop size

            x1 = int((img_w - size_w) / 2) * seg_w / img_w
            y1 = int((img_h - size_h) / 2) * seg_h / img_h
            x2 = int((img_w + size_w) / 2) * seg_w / img_w
            y2 = int((img_h + size_h) / 2) * seg_h / img_h

            boxes = torch.tensor([0, x1, y1, x2, y2])
            seg = roi_align(input=seg.unsqueeze(0), boxes=boxes.unsqueeze(0), output_size=(seg_h, seg_w), aligned=True)[0]
        else:
            seg = self.t(seg)

        return img, seg


class RandomHorizontalFlipWithMask:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img, seg):
        if torch.rand(1) < self.t.p:
            img = TF.hflip(img)
            seg = TF.hflip(seg)
        return img, seg


class RandomErasingWithMask:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img, seg):
        if len(img.size()) == 3:
            self._erase(img, seg, *img.size(), img.dtype)
        else:
            batch_size, chan, img_h, img_w = img.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.t.num_splits if self.t.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(img[i], seg[i], chan, img_h, img_w, img.dtype)
        return img, seg

    def _erase(self, img, seg, chan, img_h, img_w, dtype):
        if random.random() > self.t.probability:
            return
        area = img_h * img_w
        count = self.t.min_count if self.t.min_count == self.t.max_count else \
            random.randint(self.t.min_count, self.t.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.t.min_area, self.t.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.t.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)

                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.t.per_pixel, self.t.rand_color, (chan, h, w),
                        dtype=dtype, device=self.t.device)

                    if img.shape[:-2] == seg.shape[:-2]:  # skip for relabel
                        erase = _get_pixels(False, False, (seg.shape[0], h, w),  # fill with zeros
                                            dtype=dtype, device=self.t.device)
                        seg[:, top:top + h, left:left + w] = erase
                    break


class MixupWithMask(Mixup):
    """Mixup with multi-label and with-mask features"""

    def __init__(self, multi_label=False, **kwargs):
        super().__init__(**kwargs)
        self.multi_label = multi_label

    def _mix_batch_with_mask(self, img, seg):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            if img.shape[-2:] == seg.shape[-2:]:
                # image size = mask size
                (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                    img.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                img[:, :, yl:yh, xl:xh] = img.flip(0)[:, :, yl:yh, xl:xh]
                seg[:, :, yl:yh, xl:xh] = seg.flip(0)[:, :, yl:yh, xl:xh]
            else:
                # image size != mask size
                (seg_yl, seg_yh, seg_xl, seg_xh), lam = cutmix_bbox_and_lam(
                    seg.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                img_h, img_w = img.shape[-2:]
                seg_h, seg_w = seg.shape[-2:]
                img_yl = int(seg_yl * (img_h / seg_h))
                img_yh = int(seg_yh * (img_h / seg_h))
                img_xl = int(seg_xl * (img_w / seg_w))
                img_xh = int(seg_xh * (img_w / seg_w))
                img[:, :, img_yl:img_yh, img_xl:img_xh] = img.flip(0)[:, :, img_yl:img_yh, img_xl:img_xh]
                seg[:, :, seg_yl:seg_yh, seg_xl:seg_xh] = seg.flip(0)[:, :, seg_yl:seg_yh, seg_xl:seg_xh]
        else:
            img_flipped = img.flip(0).mul_(1. - lam)
            seg_flipped = seg.flip(0).mul_(1. - lam)
            img.mul_(lam).add_(img_flipped)
            seg.mul_(lam).add_(seg_flipped)
        return lam

    def __call__(self, x, y):
        # parse inputs and targets
        for_input = isinstance(x, (list, tuple))
        for_target = isinstance(y, (list, tuple))

        if for_input:
            img, seg = x
        else:
            img = x

        if for_target:
            target, seg = y
        else:
            target = y

        # apply mixup
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            if for_input or for_target:
                raise NotImplementedError
            else:
                lam = self._mix_elem(img)
        elif self.mode == 'pair':
            if for_input or for_target:
                raise NotImplementedError
            else:
                lam = self._mix_pair(img)
        else:
            if for_input or for_target:
                lam = self._mix_batch_with_mask(img, seg)
            else:
                lam = self._mix_batch(img)

        if self.multi_label:
            target = target * lam + target.flip(0) * (1. - lam)
        else:
            target = mixup_target(target, self.num_classes, lam, self.label_smoothing)

        # concat seg to input and/or target
        if for_input:
            img = (img, seg)
        if for_target:
            target = (target, seg)

        return img, target
