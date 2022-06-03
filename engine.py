# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, process_input=None, process_target=None, debug=False,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    it = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = to_device(samples, device, non_blocking=True)
        targets = to_device(targets, device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if process_input is not None:
            samples = process_input(samples)
        if process_target is not None:
            targets = process_target(targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs, loss = utils.process_output(outputs)  # remove dummy
            loss = loss + criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        for name, param in model.named_parameters():
            if param.grad is None:
                raise ValueError('Parameter {} is not used'.format(name))

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug and it == 1:  # to print log_every
            break
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, eval_criterion, eval_metric,
             process_input=None, process_target=None, debug=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    it = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = to_device(samples, device, non_blocking=True)
        targets = to_device(targets, device, non_blocking=True)

        if process_input is not None:
            samples = process_input(samples)
        if process_target is not None:
            targets = process_target(targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = eval_criterion(outputs, targets)

        metrics = eval_metric(outputs, targets)

        batch_size = samples[0].shape[0] if isinstance(samples, (list, tuple)) else samples.shape[0]
        metric_logger.update(loss=loss.item())
        for k, v in metrics.items():
            metric_logger.meters[k].update(v.item(), n=batch_size)

        if debug and it == 1:  # to print log_every
            break
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def to_device(tensor, device, non_blocking=True):
    if isinstance(tensor, (list, tuple)):
        return [t.to(device, non_blocking=non_blocking) for t in tensor]
    else:
        return tensor.to(device, non_blocking=non_blocking)
