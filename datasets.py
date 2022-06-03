# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import OrderedDict

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from robustness.tools.imagenet_helpers import ImageNetHierarchy

from transforms import ComposeWithMask
from utils import interpolate


BG_CHALLENGE_BG_ONLY = ['only_bg_b', 'only_bg_t', 'no_fg']
BG_CHALLENGE_MIXED = ['mixed_same', 'mixed_rand', 'mixed_next']
BG_CHALLENGE = ['imagenet9', 'only_fg'] + BG_CHALLENGE_BG_ONLY + BG_CHALLENGE_MIXED

IMAGENET_SHIFTED = ['imagenet-r', 'imagenet-stylized', 'imagenet-sketch', 'imagenet-a', 'imagenet-c', 'imagenet-v2', 'objectnet']
IMAGENET9_SHIFTED = ['{}-9'.format(name) for name in IMAGENET_SHIFTED]

DATASET_INFO = {
    'cifar10': {'path': 'cifar10', 'split': ['train', 'val']},
    'cifar100': {'path': 'cifar100', 'split': ['train', 'val']},
    'cub': {'path': 'CUB_200_2011', 'split': ['train', 'test']},
    'flowers': {'path': 'flowers102', 'split': ['train', 'test']},
}


class ImageFolderWithInfo(ImageFolder):
    def __init__(self, root, return_info=False, **kwargs):
        super().__init__(root, **kwargs)
        self.return_info = return_info

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        size = sample.size  # image_size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_info:
            info = {'path': path, 'size': size}
            return (sample, info), target
        else:
            return sample, target


class DatasetWithMask(Dataset):
    """Dataset with segmentation mask"""

    def __init__(self, mask_type='image', mask_root=None, mask_size=(14, 14), for_input=False, for_target=False):
        Dataset.__init__(self)
        assert mask_type in ['image', 'mask', 'tensor']
        self.mask_type = mask_type
        self.mask_root = mask_root
        self.mask_size = mask_size
        self.for_input = for_input
        self.for_target = for_target

    @staticmethod
    def parse_kwargs(**all_kwargs):
        mask_kwargs = {
            'mask_type': all_kwargs.pop('mask_type', 'image'),
            'mask_root': all_kwargs.pop('mask_root', None),
            'mask_size': all_kwargs.pop('mask_size', (14, 14)),
            'for_input': all_kwargs.pop('for_input', False),
            'for_target': all_kwargs.pop('for_target', False),
        }
        data_kwargs = all_kwargs  # remaining ones

        return data_kwargs, mask_kwargs

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = self.loader(path)

        if self.mask_type in ['image', 'mask']:
            assert self.mask_root is not None
            mask_path = self.get_mask_path_image(path)
            seg = self.load_mask_image(mask_path)
        else:
            assert self.mask_root is not None
            mask_path = self.get_mask_path_tensor(path)
            seg = self.load_mask_relabel(mask_path)

        # apply paired transform
        if self.transform is not None:
            img, seg = self.transform(img, seg)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # post-process mask
        if isinstance(seg, torch.Tensor):  # skip for PIL image
            seg = self.postprocess(seg)

        # concat seg to input and/or target
        if self.for_input:
            img = (img, seg)
        if self.for_target:
            target = (target, seg)

        return img, target

    def postprocess(self, seg):
        if self.mask_type == 'image':
            seg = remove_gray_area(seg)  # remove gray area (due to data aug)
            seg = seg.mean(dim=0).unsqueeze(0)  # (1, H, W)
            seg = (seg > 0).float()  # get non-black area
        elif self.mask_type == 'mask':
            seg = remove_gray_area(seg)  # remove gray area (due to data aug)
            seg = seg.mean(dim=0).unsqueeze(0)  # (1, H, W)
        return seg

    def get_mask_path_image(self, path):
        return os.path.join(self.mask_root, '/'.join(path.split('/')[-2:]))

    def load_mask_image(self, mask_path):
        return self.loader(mask_path)

    def get_mask_path_tensor(self, path):
        return self.get_mask_path_image(path).split('.')[0] + '.pt'

    def load_mask_relabel(self, mask_path):
        seg = torch.load(mask_path).float()  # (2, 5, H, W)
        seg_val = interpolate(seg[0], self.mask_size, mode='bilinear')
        seg_idx = interpolate(seg[1], self.mask_size, mode='nearest')
        seg = torch.zeros((1000, self.mask_size[0], self.mask_size[1]))  # (1000, H, W)
        seg = seg.scatter_(0, seg_idx.long(), seg_val.float())
        return seg


class ImageFolderWithMask(ImageFolder, DatasetWithMask):
    """ImageFolder with segmentation mask"""
    def __init__(self, root, **kwargs):
        data_kwargs, mask_kwargs = DatasetWithMask.parse_kwargs(**kwargs)
        ImageFolder.__init__(self, root, **data_kwargs)
        DatasetWithMask.__init__(self, **mask_kwargs)

    def __getitem__(self, index):
        return DatasetWithMask.__getitem__(self, index)


class BGOnlyWithMask(ImageFolderWithMask):
    def get_mask_path_image(self, path):
        if self.mask_type == 'tensor':
            filename = path.split('/')[-1].replace('JPEG', 'pt')
            path = os.path.join(self.mask_root, path.split('/')[-2], filename)

        return path  # get image path for dummy

    def postprocess(self, seg):
        '''
        if self.mask_type in ['image', 'mask']:
            seg = seg.mean(dim=0).unsqueeze(0)  # (1, H, W)
            seg = torch.zeros_like(seg)  # no object exists
        '''

        if self.mask_type == 'image':
            seg = remove_gray_area(seg)  # remove gray area (due to data aug)
            seg = seg.mean(dim=0).unsqueeze(0)  # (1, H, W)
            seg = (seg > 0).float()  # get non-black area
        elif self.mask_type == 'mask':
            seg = remove_gray_area(seg)  # remove gray area (due to data aug)
            seg = seg.mean(dim=0).unsqueeze(0)  # (1, H, W)

        else:
            #seg = torch.ones((1000, self.mask_size[0], self.mask_size[1])) / 1000
            pass
        return seg


class BGMixedWithMask(ImageFolderWithMask):
    def get_mask_path_image(self, path):
        if self.mask_type == 'tensor':
            filename = path.split('/')[-1].replace('JPEG', 'pt')
            mask_path = os.path.join(self.mask_root, path.split('/')[-2], filename)
        else:
            #filename = '_'.join(path.split('/')[-1].split('_')[1:3]) + '.JPEG'
            #mask_path = os.path.join(self.mask_root, path.split('/')[-2], filename)
            mask_path = os.path.join(self.mask_root, '/'.join(path.split('/')[-2:]))

        return mask_path


class ImageNetNineClass(Dataset):
    def __init__(self, root, split, return_info=False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.return_info = return_info
        self.transform = transform
        self.target_transform = target_transform

        self.classes = OrderedDict({
            'dog': 'n02084071',
            'bird': 'n01503061',
            'vehicle': 'n04576211',
            'reptile': 'n01661091',
            'carnivore': 'n02075296',
            'insect': 'n02159955',
            'instrunment': 'n03800933',
            'primate': 'n02469914',
            'fish': 'n02512053',
        })
        self.samples = self.get_files_list()

    def get_files_list(self):
        in_hier = ImageNetHierarchy(self.root, './imagenet_info')

        samples = []
        for target, class_id in enumerate(self.classes.values()):
            superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(
                len(in_hier.tree[class_id].descendants_all), ancestor_wnid=class_id)
            for subclass in superclass_wnid:
                subclass_dir = os.path.join(self.root, self.split, subclass)
                for fn in os.listdir(subclass_dir):
                    samples.append((os.path.join(subclass_dir, fn), target))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        size = sample.size  # image_size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_info:
            info = {'path': path, 'size': size}
            return (sample, info), target
        else:
            return sample, target

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.samples)


'''
class ImageNetDrawing(ImageFolder):
    def __init__(self, root, split, return_info=False, transform=None, target_transform=None):
        self.root = root
        super().__init__(self.root, transform=transform, target_transform=target_transform)
        self.split = split
        self.return_info = return_info
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        size = sample.size  # image_size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_info:
            info = {'path': path, 'size': size}
            return (sample, info), target
        else:
            return sample, target

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.samples)
'''


class ImageNetDrawing(Dataset):
    def __init__(self, root, split, return_info=False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.samples = self.get_filenames()

        self.return_info = return_info
        self.transform = transform
        self.target_transform = target_transform

    def get_filenames(self):
        file_name_list = list(sorted(os.listdir(self.root)))
        new_samples = []
        for file_name in file_name_list:
            new_samples.append((os.path.join(self.root, file_name), 0))
        return new_samples

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        size = sample.size  # image_size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_info:
            info = {'path': path, 'size': size}
            return (sample, info), target
        else:
            return sample, target

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.samples)


class ImageNetNineClassWithMask(ImageNetNineClass, DatasetWithMask):
    def __init__(self, root, split, **kwargs):
        data_kwargs, mask_kwargs = DatasetWithMask.parse_kwargs(**kwargs)
        ImageNetNineClass.__init__(self, root, split, **data_kwargs)
        DatasetWithMask.__init__(self, **mask_kwargs)

    def __getitem__(self, index):
        return DatasetWithMask.__getitem__(self, index)


class ImageNetReal(Dataset):
    def __init__(self, root, split, return_info=False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.classes = OrderedDict({
            'dog': 'n02084071',
            'bird': 'n01503061',
            'vehicle': 'n04576211',
            'reptile': 'n01661091',
            'carnivore': 'n02075296',
            'insect': 'n02159955',
            'instrunment': 'n03800933',
            'primate': 'n02469914',
            'fish': 'n02512053',
        })

        self.subclasses_list, self.subclass_to_nineclass = self.get_subclasses()
        self.class_to_wordnetid = self.get_class_to_wordnetid_dict()
        self.real_classes_list = self.get_real_classes_list()
        self.filter_images()
        self.transform = transform
        self.target_transform = target_transform
        self.return_info = return_info

    def get_subclasses(self):
        in_hier = ImageNetHierarchy(self.root, './imagenet_info')

        subclasses_list = []
        subclass_to_nineclass = dict()
        samples = []
        for idx, class_id in enumerate(self.classes.values()):
            superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(
                len(in_hier.tree[class_id].descendants_all), ancestor_wnid=class_id)
            subclasses_list.extend(superclass_wnid)
            for subclass in superclass_wnid:
                subclass_to_nineclass[subclass] = idx

        return subclasses_list, subclass_to_nineclass

    def get_class_to_wordnetid_dict(self):
        with open('./imagenet_info/imagenet_class_index.json') as class_file:
            class_to_wordnetid_filename = json.load(class_file)

        class_to_wordnetid = dict()
        for key in list(class_to_wordnetid_filename.keys()):
            wordnetid_and_filename = class_to_wordnetid_filename[key]
            wordnetid = wordnetid_and_filename[0]
            class_to_wordnetid[key] = wordnetid
        return class_to_wordnetid

    def get_real_classes_list(self):
         with open('./real.json') as real_file:
            real_classes = json.load(real_file)
            return real_classes

    def filter_images(self):
        '''
        image_class_names = list(sorted(os.listdir(os.path.join(self.root, 'val_backup'))))
        image_file_names = []
        for class_ in image_class_names:
            image_file_names_for_class = list(sorted(os.listdir(os.path.join(self.root, 'val_backup', class_))))
            for file_path in image_file_names_for_class:
                image_file_names.append(os.path.join(class_, file_path))
        '''
        image_file_names = list(sorted(os.listdir(os.path.join(self.root, 'val_backup'))))
        #for path in image_class_names:
        #    image_file_names.append(os.path.join(self.root, 'val_backup', path))

        new_samples = []
        for i in range(len(image_file_names)):
            file_path = image_file_names[i]
            real_class_list = self.real_classes_list[i]

            for class_ in real_class_list:
                class_wordnetid = self.class_to_wordnetid[str(class_)]
                if class_wordnetid in self.subclasses_list:
                    target = self.subclass_to_nineclass[class_wordnetid]
                    full_path = os.path.join(self.root, 'val_backup', file_path)
                    new_samples.append((full_path, target))

        self.samples = new_samples
        self.classes = list(self.classes.keys())

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        size = sample.size  # image_size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_info:
            info = {'path': path, 'size': size}
            return (sample, info), target
        else:
            return sample, target

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.samples)


class ImageNetRealWithMask(ImageNetReal, DatasetWithMask):
    def __init__(self, root, split, **kwargs):
        data_kwargs, mask_kwargs = DatasetWithMask.parse_kwargs(**kwargs)
        ImageNetReal.__init__(self, root, split, **data_kwargs)
        DatasetWithMask.__init__(self, **mask_kwargs)

    def __getitem__(self, index):
        return DatasetWithMask.__getitem__(self, index)


class FlowersWithMask(ImageFolderWithMask):
    """Flowers dataset with mask"""
    def get_mask_path_image(self, path):
        suffix_name = path.split('/')[-1]
        suffix_idx = int(suffix_name[7:-4])
        segname = "segmim_%05d.jpg" % suffix_idx
        mask_path = os.path.join(self.mask_root, segname)
        return mask_path

    def load_mask_image(self, mask_path):
        seg = self.loader(mask_path)
        seg = np.array(seg)
        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
        seg = (seg * 255).astype('uint8').repeat(3,axis=2)
        seg = Image.fromarray(seg)
        return seg


class WaterBirds(ImageFolder):
    def __init__(self, root, is_train=True, group_no=None, **kwargs):
        # group_no
        #   None: Full train or val set
        #   1: waterbirds on water background 
        #   2: waterbirds on land background 
        #   3: landbirds on water background 
        #   4: landbirds on land background 
        #
        super().__init__(root, **kwargs)
        self.root = root
        self.is_train = is_train
        self.group_no = group_no
        self.samples = self.filter_samples()

    def get_condition(self, y, place):
        if self.group_no is None:
            return True
        elif self.group_no == 1 and y == 1 and place == 1:
            return True
        elif self.group_no == 2 and y == 1 and place == 0:
            return True
        elif self.group_no == 3 and y == 0 and place == 1:
            return True
        elif self.group_no == 4 and y == 0 and place == 0:
            return True
        else:
            return False

    def filter_samples(self):
        metadata_df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        split_array = metadata_df['split'].values.tolist()
        filename_array = metadata_df['img_filename'].values.tolist()
        y_array = metadata_df['y'].values.tolist()
        place_array = metadata_df['place'].values.tolist()

        filtered_samples = []
        for i, item in enumerate(self.samples):
            path, target = item 
            filename = '/'.join(path.split('/')[-2:])
            idx = filename_array.index(filename)
            if self.get_condition(y_array[idx], place_array[idx]):
                # print(split_array[idx], int(not self.is_train), split_array[idx] == int(not self.is_train))
                if split_array[idx] == int(not self.is_train):
                    filtered_samples.append((path, target))

        return filtered_samples 


class WaterBirdsWithMask(WaterBirds, DatasetWithMask):
    def __init__(self, root, **kwargs):
        data_kwargs, mask_kwargs = DatasetWithMask.parse_kwargs(**kwargs)
        WaterBirds.__init__(self, root, **data_kwargs)
        DatasetWithMask.__init__(self, **mask_kwargs)

    def __getitem__(self, index):
        return DatasetWithMask.__getitem__(self, index)

    def get_mask_path_image(self, path):
        mask_path = os.path.join(self.mask_root, '/'.join(path.split('/')[-2:])).replace('jpg', 'png')
        return mask_path


class Pets(ImageFolder):
    def __init__(self, root, split, **kwargs):
        super().__init__(root)
        self.root = root
        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        classes = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1])-1
            samples.append((path, label))
            if label not in classes:
                classes.append(label)

        self.samples = samples
        self.classes = classes
        self.transform = kwargs.get('transform', None)
        self.target_transform = kwargs.get('target_transform', None)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)


class PetsWithMask(Pets, DatasetWithMask):
    def __init__(self, root, split, **kwargs):
        data_kwargs, mask_kwargs = DatasetWithMask.parse_kwargs(**kwargs)
        Pets.__init__(self, root, split, **data_kwargs)
        DatasetWithMask.__init__(self, **mask_kwargs)

    def __getitem__(self, index):
        raise NotImplementedError
        # return DatasetWithMask.__getitem__(self, index)


class Food101(ImageFolder):
    def __init__(self, root, split, **kwargs):
        super().__init__(os.path.join(root, 'images'))
        self.root = root
        with open(os.path.join(root, 'meta', 'classes.txt')) as f:
            classes = [line.strip() for line in f]
        with open(os.path.join(root, 'meta', f'{split}.json')) as f:
            annotations = json.load(f)

        samples = []
        dataset_classes = []
        for i, cls in enumerate(classes):
            for path in annotations[cls]:
                samples.append((os.path.join(root, 'images', f'{path}.jpg'), i))
                if i not in dataset_classes:
                    dataset_classes.append(i)

        self.samples = samples
        self.classes = dataset_classes
        self.transform = kwargs.get('transform', None)
        self.target_transform = kwargs.get('target_transform', None)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)


class Food101WithMask(Food101):
    def __init__(self, root, split, **kwargs):
        super().__init__(root, split, **kwargs)

        self.mask_root = kwargs.get('mask_root', '')
        self.for_input = kwargs.get('for_input', False)
        self.for_target = kwargs.get('for_target', False)
        self.mask_type = kwargs.get('mask_type', 'relabel')
        self.mask_size = kwargs.get('mask_size', (14, 14))
        self.no_normalize = kwargs.get('no_normalize', False)
        assert self.mask_type in ['gt', 'relabel']

    def load_mask_relabel(self, mask_path, mask_size):
        seg = torch.load(mask_path).float()  # (2, 5, H, W)
        seg_val = interpolate(seg[0], mask_size, mode='bilinear')
        seg_idx = interpolate(seg[1], mask_size, mode='nearest')
        seg = torch.zeros((1000, mask_size[0], mask_size[1]))  # (1000, H, W)
        seg = seg.scatter_(0, seg_idx.long(), seg_val.float())
        return seg

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = self.loader(path)

        # load segmentation masks
        if self.mask_type == 'gt':
            seg = self.load_mask(path)
        else:
            detail_path_list = '/'.join(path.split('/')[-2:]).split('.')
            mask_path = os.path.join(self.mask_root,
               '.'.join(detail_path_list[:-1]) + '.pt')

            seg = load_mask_relabel(mask_path, self.mask_size)

        # apply paired transform
        if self.transform is not None:
            img, seg = self.transform(img, seg)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # convert seg to pixel-level class map
        if self.mask_type == 'gt':
            seg = (seg.mean(dim=0) > 0).float() * (target + 1)  # (H, W)
            seg = one_hot(seg, len(self.classes), self.mask_size)  # (C, H', W')
        else:
            seg_nonorm = seg
            #seg = F.softmax(seg, dim=0)  # (C, H', W')

        # concat seg to input and/or target
        if self.for_input:
            if self.no_normalize:
                img = (img, seg_nonorm)
            else:
                img = (img, seg)
        if self.for_target:
            target = (target, seg)

        return img, target


def remove_gray_area(seg):
    assert seg.shape[0] == 3  # 3-channel image
    GRAY = (0.4863, 0.4549, 0.4078)
    GRAY = torch.tensor(GRAY).view(3, 1, 1)
    is_gray = ((seg - GRAY).abs().sum(dim=0, keepdim=True) < 1e-4).bool()
    seg[is_gray.repeat(3, 1, 1)] = 0  # remove gray area (due to aug)
    return seg


def build_dataset(is_train, args, resize_only=False, normalize=True, return_info=False):
    root_dir = Path(args.root_dir)

    mask_for_input = args.mask_attention
    mask_for_target = args.token_label
    use_mask = mask_for_input or mask_for_target
    mask_path = None  # None if not specified

    transform = build_transform(is_train, args, use_mask=use_mask,
                                resize_only=resize_only, normalize=normalize)

    kwargs = {'transform': transform}
    if use_mask:
        if args.patch_label == 'gt':
            mask_type = 'image'  # rgb image
        elif args.patch_label == 'bigbigan':
            mask_type = 'mask'  # binary mask
        else:
            mask_type = 'tensor'  # (2, K, H, W) tensor

        kwargs.update({
            'mask_type': mask_type,
            'mask_size': (args.mask_size, args.mask_size),
            'for_input': mask_for_input,
            'for_target': mask_for_target,
        })

    if args.data_set == 'imagenet':
        split = 'train' if is_train else 'val'
        data_path = root_dir / 'ILSVRC/Data/CLS-LOC' / split

        if use_mask:
            dataset = ImageFolderWithMask(data_path, mask_path, **kwargs)
        else:
            dataset = ImageFolder(data_path, **kwargs)

        if split == 'train':
            random.seed(42)
            dataset.samples = random.sample(dataset.samples, int(args.dataset_ratio * len(dataset.samples)))

        num_classes = 1000
        multi_label = False

    elif args.data_set == 'imagenet_drawing':
        split = 'val_backup'
        data_path = root_dir / 'ILSVRC/Data/CLS-LOC' / split

        dataset = ImageNetDrawing(data_path, split, return_info=True, **kwargs)

        num_classes = 1000
        multi_label = False

    elif args.data_set in BG_CHALLENGE:
        split = 'train' if is_train else 'val'

        if args.data_set == 'imagenet9':
            data_path = str(root_dir / 'bg_challenge/original_changed' / split)  # val set = bg_challenge ver.
        else:
            data_path = str(root_dir / 'bg_challenge/bg_challenge' / args.data_set / 'val')  # no train set exists

        if args.patch_label == 'gt':
            mask_path = str(root_dir / 'bg_challenge/only_fg' / split)
        elif args.patch_label == 'relabel':
            #if args.data_set == 'no_fg':
            #    mask_path = str(root_dir / 'label_top5_{}_nfnet_in9'.format(split))
            #elif args.data_set == 'only_bg_b':
            #    mask_path = str(root_dir / 'label_top5_{}_nfnet_only_bg_b'.format(split))
            #else:
            mask_path = str(root_dir / 'label_top5_{}_nfnet_{}'.format(split, args.data_set))
            #mask_path = str(root_dir / 'label_top5_{}_nfnet_{}'.format(split, 'only_fg'))
        else:
            #mask_path = str(root_dir / 'bigbigan_mask_in9' / split)
            mask_path = str(root_dir / 'bigbigan_mask_{}'.format(args.data_set) / split)

        if use_mask:
            if args.data_set in BG_CHALLENGE_BG_ONLY:
                dataset = BGOnlyWithMask(data_path, mask_root=mask_path, **kwargs)
            elif args.data_set in BG_CHALLENGE_MIXED:
                dataset = BGMixedWithMask(data_path, mask_root=mask_path, **kwargs)
            else:
                dataset = ImageFolderWithMask(data_path, mask_root=mask_path, **kwargs)
        else:
            dataset = ImageFolderWithInfo(data_path, return_info=return_info, **kwargs)

        num_classes = len(dataset.classes)
        multi_label = False

    elif args.data_set == 'imagenet_real_9':
        #split = 'val'  # no train set for shfited datasets
        split = 'val_backup'
        data_path = str(root_dir / 'ILSVRC/Data/CLS-LOC')

        if args.patch_label == 'gt':
            raise ValueError
        elif args.patch_label == 'relabel':
            raise NotImplementedError
        else:
            mask_path = str(root_dir / f'bigbigan_mask_imagenet_{split}')

        if use_mask:
            #dataset = ImageNetNineClassWithMask(data_path, split, mask_root=mask_path, **kwargs)
            dataset = ImageNetRealWithMask(data_path, split, mask_root=mask_path, **kwargs)
        else:
            dataset = ImageNetReal(data_path, split, return_info=return_info, **kwargs)

        num_classes = len(dataset.classes)
        multi_label = False

    elif args.data_set in IMAGENET9_SHIFTED:
        split = 'val'  # no train set for shfited datasets

        data_set = args.data_set.replace('-9', '')
        if args.data_set == 'imagenet-9':
            print('here')
            data_path = str(root_dir / 'ILSVRC/Data/CLS-LOC')
        else:
            data_path = str(root_dir / data_set)

        if args.patch_label == 'gt':
            raise ValueError
        elif args.patch_label == 'relabel':
            #raise NotImplementedError
            mask_path = str(root_dir / 'label_top5_nfnet_{}'.format(data_set) / split)
        else:
            mask_path = str(root_dir / 'bigbigan_mask_{}'.format(data_set) / split)

        if use_mask:
            dataset = ImageNetNineClassWithMask(data_path, split, mask_root=mask_path, **kwargs)
        else:
            dataset = ImageNetNineClass(data_path, split, return_info=return_info, **kwargs)

        num_classes = len(dataset.classes)
        multi_label = False

    elif args.data_set in ['cifar10', 'cifar100', 'cub', 'flowers']:
        split = DATASET_INFO[args.data_set]['split'][int(not is_train)]
        data_path = str(root_dir / DATASET_INFO[args.data_set]['path'] / split)

        if args.patch_label == 'gt':
            raise ValueError('No GT mask for {}'.format(args.data_set))
        elif args.patch_label == 'relabel':
            mask_path = root_dir / 'label_top5_{}_nfnet_{}'.format(split, args.data_set)

        if use_mask:
            dataset = ImageFolderWithMask(data_path, mask_root=mask_path, **kwargs)
        else:
            dataset = ImageFolder(data_path, **kwargs)

        num_classes = len(dataset.classes)
        multi_label = False

    elif args.data_set == 'pets':
        split = 'trainval' if is_train else 'test'
        data_path = str(root_dir / 'Pets')

        if args.patch_label == 'relabel':
            mask_path = root_dir / 'label_top5_nfnet_Pets'

        if use_mask:
            dataset = PetsWithMask(data_path, split, mask_root=mask_path, **kwargs)
        else:
            dataset = Pets(data_path, split, **kwargs)

        num_classes = len(dataset.classes)
        multi_label = False

    elif args.data_set == 'food':
        split = 'train' if is_train else 'test'
        data_path = root_dir / 'food-101'

        mask_path = root_dir / 'label_top5_nfnet_food'

        if use_mask:
            dataset = Food101WithMask(data_path, split, mask_root=mask_path, **kwargs)
        else:
            dataset = Food101(data_path, split, **kwargs)

        nb_classes = len(dataset.classes)
        print(f'nb_classes: {nb_classes}')
        multi_label = False

    else:
        raise ValueError

    return dataset, num_classes, multi_label


def build_transform(is_train, args, use_mask=False, resize_only=False, normalize=True):
    if is_train and not resize_only:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
    else:
        t = []
        if resize_only:
            t.append(T.Resize((args.input_size, args.input_size), interpolation=3))
        else:
            size = int((256 / 224) * args.input_size)
            t.append(T.Resize(size, interpolation=3))  # to maintain same ratio w.r.t. 224 images
            t.append(T.CenterCrop(args.input_size))

        t.append(T.ToTensor())
        if normalize:
            t.append(T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        transform = T.Compose(t)

    if use_mask:
        transform = ComposeWithMask(transform)

    return transform
