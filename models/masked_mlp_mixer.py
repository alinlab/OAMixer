# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn

from timm.models.mlp_mixer import MlpMixer, _cfg
from timm.models.registry import register_model

from .masked_base import BaseMaskedModel, BaseMaskedBlock, BaseMaskedModule


__all__ = [
    'mixer_s32', 'mixer_s16', 'mixer_b32', 'mixer_b16',
]


class MaskedMLPMixer(BaseMaskedModel, MlpMixer):
    """MLP-Mixer with Masked weight"""

    def __init__(self, mask_alg=None, **all_kwargs):
        base_kwargs, model_kwargs, mask_kwargs = BaseMaskedModel.parse_kwargs(**all_kwargs)

        # initialize model
        BaseMaskedModel.__init__(self, **base_kwargs)
        MlpMixer.__init__(self, **model_kwargs)
        self._init_mask_layer()
        self._init_classifier(self.num_features, self.num_classes)

        # convert blocks to masked blocks
        mask_kwargs.update({'mask_alg': mask_alg})
        for i in range(len(self.blocks)):
            mask_attention = self.mask_attention and (i in self.mask_layer)
            self.blocks[i] = MaskedMLPMixerBlock(self.blocks[i], mask_attention=mask_attention, **mask_kwargs)

        self.apply(self.init_weights)

    def forward(self, x):
        return BaseMaskedModel.forward(self, x)

    @property
    def depth(self):
        return len(self.blocks)

    def get_all_blocks(self):
        ret = []  # (layer_idx, block_idx, block)
        for i, blk in enumerate(self.blocks):
            ret.append([i, 0, blk])
        return ret

    def forward_features(self, x, patch_class=None):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, patch_class)
        x = self.norm(x)

        x_cls = x.mean(dim=1)
        x_aux = {'tok': x}

        if self.token_label:
            return x_cls, x_aux
        else:
            return x_cls


class MaskedMLPMixerBlock(BaseMaskedBlock):
    """Mixer block with MaskedPatchMlp"""

    def __init__(self, block, **kwargs):
        super().__init__()
        # copy attributes of block
        self.norm1 = block.norm1
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp_channels = block.mlp_channels

        # modify attention to the masked attention
        self.mlp_tokens = MaskedPatchMlp(block.mlp_tokens, **kwargs)
        self.add_masked_module('mlp_tokens', self.mlp_tokens)

    def forward(self, x, patch_class=None):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2), patch_class).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MaskedPatchMlp(BaseMaskedModule):
    """PatchMLP with reweighting mask"""

    def __init__(self, mlp, mask_alg='linearize', **kwargs):
        super().__init__(num_heads=1, **kwargs)
        self.mask_alg = mask_alg

        # copy attributes of mlp
        self.fc1 = mlp.fc1
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.fc2 = mlp.fc2
        self.drop2 = mlp.drop2

    @property
    def hparam_dict(self):
        d = super().hparam_dict
        if self.mask_attention:
            d['mask_alg'] = self.mask_alg
        return d

    def forward(self, x, patch_class=None):
        if self.mask_attention:
            mask = self.get_mask(patch_class).squeeze(1)  # (B, N, N)

            w1 = self.fc1.weight  # (M, N)
            w2 = self.fc2.weight  # (N, M)
            b1 = self.fc1.bias  # (M)
            b2 = self.fc2.bias  # (N)

            if self.mask_alg == 'linearize':
                linear = (w2 @ w1).unsqueeze(0)  # (1, N, N)
                bias = b1 @ w2.t() + b2  # (N)
                out_mlp = self.forward_base(x)
                out_linear = (x @ linear.transpose(1, 2) + bias)
                res = out_mlp - out_linear
            else:
                raise ValueError

            x = x @ (linear * mask).transpose(1, 2) + bias + res  # (B, C, N)
        else:
            x = self.forward_base(x)  # (B, C, N)

        return x

    def forward_base(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@register_model
def mixer_s32(pretrained=False, **kwargs):
    model = MaskedMLPMixer(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixer_s16(pretrained=False, **kwargs):
    model = MaskedMLPMixer(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixer_b32(pretrained=False, **kwargs):
    model = MaskedMLPMixer(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixer_b16(pretrained=False, **kwargs):
    model = MaskedMLPMixer(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model.default_cfg = _cfg()
    return model

