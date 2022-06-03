# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import os
from tqdm import tqdm

import torch
import torchvision.transforms as T

from datasets import build_dataset
from patch_models import create_unet
import utils

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Pre-compute patch labels', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--root_dir', default='/data', type=str,
                        help='root directory')
    parser.add_argument('--data-set', default='imagenet', type=str,
                        help='dataset name')

    parser.add_argument('--output_dir', default='/data/bigbigan_mask_in',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    utils.init_distributed_mode(args)

    # set default arguments
    args.patch_label = 'bigbigan'
    args.mask_attention = False
    args.token_label = False

    device = torch.device(args.device)

    dataset_train, _, _ = build_dataset(is_train=True, args=args, resize_only=True, normalize=False, return_info=True)
    dataset_val, _, _ = build_dataset(is_train=False, args=args, resize_only=True, normalize=False, return_info=True)
    print(f'len train: {len(dataset_train)}, len val: {len(dataset_val)}')

    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_unet()
    model = model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    save_masks(data_loader_train, model, args.output_dir, device=device)
    save_masks(data_loader_val, model, args.output_dir, device=device)


def save_masks(data_loader, model, output_dir, device):
    for data, _ in tqdm(data_loader):
        batch, info = data
        batch = batch.to(device)

        with torch.no_grad():
            batch = utils.interpolate(batch, (128, 128))
            mask_batch = (1.0 - torch.softmax(model(batch), dim=1))[:, 0]

        size_info = torch.stack(info['size'], dim=1)
        for mask, path, size in zip(mask_batch, info['path'], size_info):
            path = os.path.join(output_dir, '/'.join(path.split('/')[-3:]))
            dir = '/'.join(path.split('/')[:-1])
            os.makedirs(dir, exist_ok=True)

            size = list(size.flip(0))  # (W, H) -> (H, W)
            mask = T.ToPILImage()(mask)
            mask = T.Resize(size)(mask)
            mask.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReMixer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

