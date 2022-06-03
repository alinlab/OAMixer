# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import datetime
import shutil
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from timm.utils import accuracy


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class SimpleLogger(object):
    """Log max, mean, min of tensors"""

    def __init__(self, window_size=1000):
        self._max = SmoothedValue(window_size=window_size)
        self._mean = SmoothedValue(window_size=window_size)
        self._min = SmoothedValue(window_size=window_size)

    def update(self, value):
        self._max.update(value.max().item())
        self._mean.update(value.mean().item())
        self._min.update(value.min().item())

    def stats(self):
        return [self._max.avg, self._mean.avg, self._min.avg]


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def one_hot(x, num_classes, size):
    x = F.one_hot(x.long(), num_classes=num_classes + 1).permute(2, 0, 1).float()  # (C+1, H, W)
    x = interpolate(x, size, mode='bilinear')  # (C+1, H', W')
    bg, fg = x[:1, :, :], x[1:, :, :]  # (1, H', W'), (C, H', W')
    uniform = torch.ones_like(fg) / num_classes  # (C, H', W')
    x = fg + bg * uniform  # (C, H', W')
    return x


def interpolate_and_rearrange(target, size, mode='bilinear'):
    target = interpolate(target, (size, size), mode=mode)
    target = rearrange(target, 'B C H W -> B (H W) C')  # (B, N, C)
    return target


def process_mask(x, size, for_target=False, is_swin=False, apply_softmax=False):
    """Resize mask align to patch embeddings"""

    # return single tensor
    if not isinstance(x, (list, tuple)):
        return x

    # process target
    x_cls, x_aux = x

    if not for_target:  # return: (tensor, tensor)
        x_aux = interpolate_and_rearrange(x_aux, size)
        if apply_softmax:
            x_aux = torch.nn.functional.softmax(x_aux, -1)
        x = (x_cls, x_aux)

    else:  # return: (tensor, Dict[tensor])
        if is_swin:
            x_aux_1 = interpolate_and_rearrange(x_aux, size // 8)
            x_aux_2 = interpolate_and_rearrange(x_aux, size // 4)
            x_aux_3 = interpolate_and_rearrange(x_aux, size // 2)
            x_aux_4 = interpolate_and_rearrange(x_aux, size)

            if apply_softmax:
                x_aux_1 = torch.nn.functional.softmax(x_aux_1, -1)
                x_aux_2 = torch.nn.functional.softmax(x_aux_2, -1)
                x_aux_3 = torch.nn.functional.softmax(x_aux_3, -1)
                x_aux_4 = torch.nn.functional.softmax(x_aux_4, -1)

            x_aux_dict = dict()
            x_aux_dict['tok'] = x_aux_1
            x_aux_dict['int'] = [x_aux_4] * 2 +\
                                [x_aux_3] * 2 +\
                                [x_aux_2] * 6 +\
                                [x_aux_1] * 2
        else:
            x_aux = interpolate_and_rearrange(x_aux, size)
            if apply_softmax:
                x_aux = torch.nn.functional.softmax(x_aux, -1)

            x_aux_dict = dict()
            x_aux_dict['tok'] = x_aux
            x_aux_dict['int'] = x_aux

        x = (x_cls, x_aux_dict)

    return x


def process_output(output, return_loss=True):
    """Remove dummy value from outputs"""
    loss = 0  # default value

    if isinstance(output, (list, tuple)):
        output_cls, output_aux = output
        loss = output_aux.pop('dummy', 0)  # dummy loss

        if len(output_aux) > 0:
            output = (output_cls, output_aux)
        else:
            output = output_cls

    if return_loss:
        return output, loss
    else:
        return output


class EvalWrapper(object):
    """Wrapper of criterion and metric to handle multiple outputs"""

    def __init__(self, func, func_aux=None, name=None, return_dict=False):
        self.func = func
        self.func_aux = func_aux if func_aux is not None else func
        self.name = name
        self.return_dict = return_dict

    def __call__(self, output, target):
        output = process_output(output, return_loss=False)  # remove dummy

        if not isinstance(output, (list, tuple)):
            if self.return_dict:
                ret = {self.name: self.func(output, target)}
            else:
                ret = self.func(output, target)
        else:
            output_cls, output_aux = output
            target_cls, target_aux = target

            if self.return_dict:
                ret = {self.name: self.func(output_cls, target_cls)}
                for name in output_aux.keys():
                    out, tgt = output_aux[name], target_aux[name]
                    if not isinstance(out, (list, tuple)):
                        key = 'aux/{}_{}'.format(self.name, name)
                        value = self.func_aux(out, tgt)
                        update_value(ret, key, value)
                    else:
                        tgt = repeat(tgt, len(out))
                        for i in range(len(out)):
                            key = 'aux/{}_{}_{}'.format(self.name, name, i)
                            value = self.func_aux(out[i], tgt[i])
                            update_value(ret, key, value)
            else:
                ret = self.func(output_cls, target_cls)

        return ret


def update_value(x, key, value):
    if isinstance(value, dict):
        for k, v in value.items():
            x['{}_{}'.format(key, k)] = v
    else:
        x[key] = value


def repeat(li, length):
    if isinstance(li, (list, tuple)):
        assert len(li) == length
    else:
        return [li] * length


def class_accuracy(output, target):
    return accuracy(output, target, topk=(1,))[0]


def patch_accuracy(output, target):
    C = output.shape[-1]
    output = output.reshape(-1, C)
    target = target.reshape(-1, C)
    target = target.max(dim=1)[1]  # argmax
    return class_accuracy(output, target)


# code from https://github.com/Alibaba-MIIL/ASL/blob/main/src/helper_functions/helper_functions.py
def mAP(output, target):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if np.size(output) == 0:
        return 0
    ap = np.zeros((output.shape[1]))
    # compute average precision for each class
    for k in range(output.shape[1]):
        # sort scores
        scores = output[:, k]
        targets = target[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


# code from https://github.com/Alibaba-MIIL/ASL/blob/main/src/helper_functions/helper_functions.py
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def interpolate(x, size=(14, 14), mode='bilinear'):
    """Interpolate to smaller than half produces incorrect results"""
    dim = x.dim()

    if dim == 2:
        x = x.view((1, 1, *x.size()))
    elif dim == 3:
        x = x.view((1, *x.size()))

    if not isinstance(size, (list, tuple)):
        size = (size, size)

    h1, w1 = x.shape[-2:]
    h2, w2 = size

    if h1 == h2 and w1 == w2:
        # no need of interpolation
        pass
    elif h1 > h2 or w1 > w2:
        # downsample needs iterative procedure
        while h1 > h2 or w1 > w2:
            h1 = max(h1 // 2, h2)
            w1 = max(w1 // 2, w2)
            x = F.interpolate(x, (h1, w1), mode=mode)
    else:
        # upsample is okay for single procedure
        x = F.interpolate(x, (h2, w2))

    if dim == 2:
        x = x.view((h2, w2))
    elif dim == 3:
        x = x.view((-1, h2, w2))

    return x


BACKUP_DIRS = [  # directories to backup (default: pass)
    'models', 'transforms',
]


def backup_code(log_dir):
    # save current code except excluded files
    code_dir = os.path.join(log_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    os.mkdir(code_dir)
    for fn in os.listdir(os.getcwd()):
        if os.path.isdir(fn) and fn in BACKUP_DIRS:
            shutil.copytree(fn, os.path.join(code_dir, fn))
        elif not os.path.isdir(fn) and fn[0] not in ('.', '_'):
            shutil.copy(fn, os.path.join(code_dir, fn))
