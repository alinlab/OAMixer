from abc import abstractmethod
import torch
import torch.nn as nn

import utils


class BaseMaskedModel(nn.Module):
    """Base model for masekd ViT, Mixer, etc."""

    def __init__(self, mask_attention=False, mask_layer=None, token_label=False,
                 num_patch_classes=None, transfer_layer='none', attention_type='vit'):
        nn.Module().__init__()
        assert transfer_layer in ['linear', 'fixed', 'none']
        self.mask_attention = mask_attention
        self.mask_layer = mask_layer
        self.token_label = token_label
        self.num_patch_classes = num_patch_classes
        self.transfer_layer = transfer_layer
        self.attention_type = attention_type

    @staticmethod
    def parse_kwargs(**all_kwargs):
        base_kwargs = {
            'mask_attention': all_kwargs.pop('mask_attention', False),
            'mask_layer': all_kwargs.pop('mask_layer', None),
            'token_label': all_kwargs.pop('token_label', False),
            'num_patch_classes': all_kwargs.pop('num_patch_classes', None),
            'transfer_layer': all_kwargs.pop('transfer_layer', 'none'),
            'attention_type': all_kwargs.pop('attention_type', 'vit'),
        }
        mask_kwargs = {
            'patch_dist': all_kwargs.pop('patch_dist', 'l1'),
            'mask_func': all_kwargs.pop('mask_func', 'exp'),
        }
        model_kwargs = all_kwargs  # remaining kwargs

        return base_kwargs, model_kwargs, mask_kwargs

    @property
    def hparam_dict(self):
        d = dict()
        d['mask_layer'] = '[{}]'.format(','.join(map('{}'.format, self.mask_layer)))
        d['token_label'] = self.token_label
        d['patch_class'] = self.num_patch_classes
        d['transfer'] = self.transfer_layer
        return d

    @property
    def hparam_info(self):
        info = ', '.join(['{}: {}'.format(k, v) for (k, v) in self.hparam_dict.items()])
        return info

    def print_model_hparams(self):
        print('\n[MODEL HYPER-PARAMETERS]')
        print('[all] ' + self.hparam_info)
        for i, j, blk in self.get_all_blocks():
            print('[layer {:2d} | block {:2d}] '.format(i, j) + blk.hparam_info)

    def print_model_params(self):
        print('\n[MODEL PARAMETERS]')
        for i, j, blk in self.get_all_blocks():
            if len(blk.param_info_list) > 0:
                print('[layer {:2d} | block {:2d}] '.format(i, j) + ' | '.join(blk.param_info_list))

    def _init_mask_layer(self):
        if self.mask_layer is None:
            self.mask_layer = list(range(self.depth))
        if not self.mask_attention:
            self.mask_layer = []  # empty list

    def _init_classifier(self, num_features, num_classes):
        self.head = nn.Linear(num_features, num_classes)
        if self.token_label:
            self.tok_head = nn.Linear(num_features, self.num_patch_classes)

        if self.transfer_layer == 'linear':
            self.transfer = nn.Linear(self.num_patch_classes, 10, bias=False)
        elif self.transfer_layer == 'fixed':
            self.register_buffer('transfer_layer_param', torch.randn((self.num_patch_classes, 10)))
            self.transfer = (lambda x: x @ self.transfer_layer_param)

    @property
    def depth(self):
        return 0

    @abstractmethod
    def get_all_blocks(self):
        pass

    def set_block_attributes(self, name, value):
        for i, j, blk in self.get_all_blocks():
            blk.set_attribute(name, value)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x, patch_class = x  # input: (img, patch_class)
        else:
            x, patch_class = x, None  # input: img

        # pre-process patch_class
        if self.transfer_layer is not 'none':
            patch_class = self.transfer(patch_class)

        # compute features (CORE LOGIC OF EACH MODEL)
        x = self.forward_features(x, patch_class)

        if isinstance(x, (list, tuple)):
            x_cls, x_aux = x  # output: (x_cls, x_aux)
        else:
            x_cls = x  # output: x_cls
            x_aux = dict()  # empty dict

        x_cls = self.head(x_cls)  # class output

        if self.token_label:  # token labeling
            x_aux['tok'] = self.tok_head(x_aux['tok'])

        if len(x_aux) > 0:
            return x_cls, x_aux
        else:
            return x_cls

    @abstractmethod
    def forward_features(self, x, patch_class=None):
        pass


class BaseMaskedBlock(nn.Module):
    """Base masked block"""

    def __init__(self):
        super().__init__()
        self._masked_modules = dict()

    @abstractmethod
    def forward(self, x, patch_class=None):
        pass

    def add_masked_module(self, name, module):
        self._masked_modules.update({name: module})

    def set_attribute(self, name, value):
        """Set attribute for masked modules"""
        for module in self._masked_modules.values():
            if hasattr(module, name):
                setattr(module, name, value)

    def get_attribute(self, name):
        """Get attribute from masked modules"""
        attr = []
        for module in self._masked_modules.values():
            if hasattr(module, name):
                attr.append(getattr(module, name))

        if len(attr) > 1:
            raise ValueError('Multiple attributes {} exist'.format(name))

        return attr[0] if len(attr) == 1 else None

    def __getattr__(self, name):
        if name in self._masked_modules.keys():
            return self._masked_modules[name]
        else:
            attr = self.get_attribute(name)
            if attr is None:
                attr = super().__getattr__(name)  # parent
            return attr


class BaseMaskedModule(nn.Module):
    """Base masked module"""

    MASKED_PARAMETERS = ['mask_scale']

    def __init__(self, mask_attention=False, patch_dist='l1', mask_func='exp', num_heads=1):
        super().__init__()
        assert patch_dist in ['l1', 'cosine']
        assert mask_func in ['exp', 'minus']

        self.mask_attention = mask_attention

        if self.mask_attention:
            self.num_heads = num_heads
            self.patch_dist = patch_dist
            self.mask_func = mask_func

            self.mask_scale = nn.Parameter(torch.zeros(1))
            self.dist_stats = utils.SimpleLogger()
            self.mask_stats = utils.SimpleLogger()

    @property
    def hparam_dict(self):
        d = dict()
        d['mask_attn'] = self.mask_attention
        if self.mask_attention:
            d['patch_dist'] = self.patch_dist
            d['mask_func'] = self.mask_func
        return d

    @property
    def param_dict(self):
        d = dict()
        if self.mask_attention:
            d['mask_scale'] = self.mask_scale.view(-1).tolist()
            d['dist_stats'] = self.dist_stats.stats()
            d['mask_stats'] = self.mask_stats.stats()
        return d

    @property
    def hparam_info(self):
        info = ', '.join(['{}: {}'.format(k, v) for (k, v) in self.hparam_dict.items()])
        return info

    @property
    def param_info_list(self):
        info = []  # list of params
        for name, param in self.param_dict.items():
            _info = '{}: '.format(name) + ', '.join(['{:.3f}'.format(p) for p in param])
            info.append(_info)
        return info

    @property
    def dummy(self):
        dummy = 0
        for p in self.parameters():
            dummy = dummy + p.mean() * 0
        return dummy

    @abstractmethod
    def forward(self, x, patch_class=None):
        pass

    def get_mask(self, patch_class):
        """Compute mask from patch classes"""
        assert self.mask_attention

        # compute distance
        if self.patch_dist == 'l1':
            dist = torch.cdist(patch_class, patch_class, p=1)  # (B, N, N)
        else:
            patch_class = patch_class / patch_class.norm(dim=-1, keepdim=True)
            dist = 1 - patch_class @ patch_class.transpose(1, 2)  # (B, N, N)

        # compute mask
        if self.mask_func == 'exp':
            self.mask_scale.data = self.mask_scale.data.clamp(min=0)  # range: [0, \infty)
            mask = (-self.mask_scale * dist.unsqueeze(1)).exp()  # (B, H, N, N)
        else:
            self.mask_scale.data = self.mask_scale.data.clamp(min=0, max=1)  # range: [0, 1]
            mask = 1 - self.mask_scale * dist.unsqueeze(1)  # (B, H, N, N)

        if mask.shape[1] == 1:
            mask = mask.repeat(1, self.num_heads, 1, 1)

        # log values
        self.dist_stats.update(dist.clamp(min=0, max=1))
        self.mask_stats.update(mask.clamp(min=0, max=1))

        return mask

