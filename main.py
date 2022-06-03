# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import time
import json
import random
import numpy as np
from pathlib import Path
from functools import partial

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from transforms import MixupWithMask
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, TokenLabelLoss, MultiLabelLoss
from samplers import RASampler
from models import *
import utils

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--debug', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int, help='images patch size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--root_dir', default='/data', type=str,
                        help='root directory')
    parser.add_argument('--data-set', default='imagenet9', type=str,
                        help='dataset name')

    parser.add_argument('--output_dir', default='ckpt/debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--waterbird-group-no', type=int, default=None, metavar='N',
                        help='Group number of Waterbird dataset')
    parser.add_argument('--dataset-ratio', default=1., type=float)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # ReMixer configs
    parser.add_argument('--mask-attention', action='store_true',
                        help='Use masked attention')
    parser.add_argument('--mask-layer', default=None, type=int, nargs='+',
                        help='Layers to mask attention')
    parser.add_argument('--token-label', action='store_true',
                        help='Use token labeling for final layer')

    parser.add_argument('--patch-label', type=str, default='gt', choices=['relabel', 'bigbigan'],
                        help='Type for patch-wise labels')
    parser.add_argument('--patch-dist', type=str, default='l1', choices=['l1', 'cosine'],
                        help='Type for distance function')
    parser.add_argument('--mask-func', type=str, default='exp', choices=['exp', 'minus'],
                        help='Type for mask function')
    parser.add_argument('--mask-lr-scale', type=float, default=1.,
                        help='Use different lr for mask parameters')

    parser.add_argument('--attention-type', type=str, default='vit', choices=['vit', 'convit', 'coat'],
                        help='Use different attention for vision Transformer')
    parser.add_argument('--mixer-mask-alg', type=str, default='linearize',
                        help='Mask algorithm for mlp-mixer')
    parser.add_argument('--no-share-conv', action='store_true',
                        help='Do not share conv for conv-mixer')

    # Arguments for transfer learning
    parser.add_argument('--run-transfer', action='store_true')
    parser.add_argument('--transfer-layer', type=str, default='none')

    return parser


def set_default_arguments(args):
    # model arguments
    args.is_swin = args.model.startswith('swin')

    if args.is_swin:
        args.patch_size = 4
    elif 's32' in args.model:
        args.patch_size = 32
    else:
        args.patch_size = 16

    args.mask_size = args.input_size // args.patch_size
    # training arguments
    if args.run_transfer:
        args.epochs = 100
        args.warmup_epochs = 20
        #args.batch_size = 128
        args.patch_label = 'relabel'
        args.patch_dist = 'cosine'

    if args.patch_label == 'relabel':
        args.patch_dist = 'cosine'

    args.apply_softmax = (args.patch_label == 'relabel')

    return args


def main(args):
    utils.init_distributed_mode(args)

    # set default arguments
    args = set_default_arguments(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.num_classes, args.multi_label = build_dataset(is_train=True, args=args)
    dataset_val, _, _ = build_dataset(is_train=False, args=args)
    print(f'len train: {len(dataset_train)}, len val: {len(dataset_val)}')

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = MixupWithMask(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes,
            multi_label=args.multi_label)  # additional features

    process_input = partial(utils.process_mask, size=args.mask_size, apply_softmax=args.apply_softmax)
    process_target = partial(utils.process_mask, size=args.mask_size, apply_softmax=args.apply_softmax,
                             for_target=True, is_swin=args.is_swin)

    print(f"Creating model: {args.model}")
    model_kwargs = dict()   # model-specific arguments
    if 'mixer' in args.model and 'convmixer' not in args.model:
        model_kwargs['mask_alg'] = args.mixer_mask_alg
    if 'convmixer' in args.model:
        model_kwargs['share_conv'] = (not args.no_share_conv)

    if args.patch_label == 'relabel':
        num_patch_classes = 1000
    else:
        num_patch_classes = args.num_classes

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # custom arguments (common)
        mask_attention=args.mask_attention,
        mask_layer=args.mask_layer,
        token_label=args.token_label,
        patch_dist=args.patch_dist,
        mask_func=args.mask_func,
        attention_type=args.attention_type,
        # custom arguments (transfer)
        transfer_layer=args.transfer_layer,
        num_patch_classes=num_patch_classes,
        **model_kwargs,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    if args.mask_lr_scale != 1:  # use different lr for mask_params
        base_params, mask_params = [], []
        for name, param in model_without_ddp.named_parameters():
            if name.split('.')[-1] in BaseMaskedModule.MASKED_PARAMETERS:
                # warning: the name should not be duplicated in other modules
                mask_params.append(param)
            else:
                base_params.append(param)

        optimizer.param_groups.clear()
        optimizer.add_param_group({'params': base_params})
        optimizer.add_param_group({'params': mask_params, 'lr': args.mask_lr_scale * args.lr})

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.multi_label:
        criterion = MultiLabelLoss()
    else:
        criterion = SoftTargetCrossEntropy()

    if args.token_label:  # apply token-label
        criterion = TokenLabelLoss(criterion, cls_weight=1., tok_weight=0.5)

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.num_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    if args.multi_label:
        eval_criterion = MultiLabelLoss()
        eval_metric = utils.mAP
        eval_metric_aux = utils.patch_accuracy
    else:
        eval_criterion = torch.nn.CrossEntropyLoss()
        eval_metric = utils.class_accuracy
        eval_metric_aux = utils.patch_accuracy

    eval_criterion = utils.EvalWrapper(eval_criterion)
    eval_metric = utils.EvalWrapper(eval_metric, eval_metric_aux, name='acc', return_dict=True)
    main_metric = 'acc'  # use acc instead of acc1

    eval_kwargs = dict(eval_criterion=eval_criterion, eval_metric=eval_metric,
                       process_input=process_input, process_target=process_target,
                       debug=args.debug)

    output_dir = Path(args.output_dir)
    if utils.get_rank() == 0:
        logger = SummaryWriter(args.output_dir)
        utils.backup_code(args.output_dir)
        with open(output_dir / 'args.txt', 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        except Exception as e:
            print(e)  # print warning for non-matching parameters
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    model_without_ddp.print_model_hparams()  # print model status

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, **eval_kwargs)
        ckpt = args.resume.split('/')[-2]
        metric = test_stats[main_metric]
        print(f"Accuracy of {ckpt}, {args.data_set}: {metric:.2f}%")
        model_without_ddp.print_model_params()
        with open('./ckpt/results.txt', 'a') as f:
            f.write('{}\t{}\t{:.2f}\n'.format(ckpt, args.data_set, metric))
        return

    print(f"\nStart training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            process_input=process_input,
            process_target=process_target,
            debug=args.debug,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device, **eval_kwargs)
        print(f"Accuracy: {test_stats[main_metric]:.2f}%")

        if max_accuracy < test_stats[main_metric]:
            max_accuracy = test_stats[main_metric]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        print('\n[EXPERIMENT NAME] {}'.format(args.output_dir.split('/')[-1]))
        model_without_ddp.print_model_hparams()
        model_without_ddp.print_model_params()
        if utils.get_rank() == 0:
            update_logger(logger, train_stats, test_stats, model_without_ddp, optimizer, epoch)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def update_logger(logger, train_stats, test_stats, model, optimizer, epoch):
    for k, v in train_stats.items():
        logger.add_scalar(f'train_{k}', v, global_step=epoch)
    for k, v in test_stats.items():
        logger.add_scalar(f'test_{k}', v, global_step=epoch)

    for i, j, blk in model.get_all_blocks():
        for name, param in blk.param_dict.items():
            for h, p in enumerate(param):
                logger.add_scalar('{}/layer-{}_block-{}_head-{}'.format(name, i, j, h), p, global_step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReMixer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
