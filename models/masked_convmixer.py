# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange

from timm.models.convmixer import ConvMixer, _cfg
from timm.models.registry import register_model

from .masked_base import BaseMaskedModel, BaseMaskedBlock, BaseMaskedModule


__all__ = [
    'convmixer_s32', 'convmixer_s16',
]


class MaskedConvMixer(BaseMaskedModel, ConvMixer):
    """ConvMixer with Masked weight"""

    def __init__(self, share_conv=True, **all_kwargs):
        base_kwargs, model_kwargs, mask_kwargs = BaseMaskedModel.parse_kwargs(**all_kwargs)

        # initialize model
        BaseMaskedModel.__init__(self, **base_kwargs)
        ConvMixer.__init__(self, **model_kwargs)
        self._init_mask_layer()
        self._init_classifier(self.num_features, self.num_classes)

        # convert blocks to masked blocks
        mask_kwargs.update({'share_conv': share_conv})
        for i in range(len(self.blocks)):
            mask_attention = self.mask_attention and (i in self.mask_layer)
            self.blocks[i] = MaskedConvMixerBlock(self.blocks[i], mask_attention=mask_attention, **mask_kwargs)

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
        x = rearrange(x, 'B C H W -> B (H W) C')

        x_cls = x.mean(dim=1)
        x_aux = {'tok': x}

        if self.token_label:
            return x_cls, x_aux
        else:
            return x_cls


class MaskedConvMixerBlock(BaseMaskedBlock):
    """ConvMixer block with MaskedPatchConv"""

    def __init__(self, block, **kwargs):
        super().__init__()
        # modify conv to masked conv
        block[0].fn[0] = MaskedPatchConv(block[0].fn[0], **kwargs)
        self.block = block
        self.add_masked_module('conv_tokens', self.block[0].fn[0])

    def forward(self, x, patch_class=None):
        # custom block 0
        x_orig = x
        x = self.block[0].fn[0](x, patch_class)
        for blk in self.block[0].fn[1:]:
            x = blk(x)
        x = x + x_orig

        # remaining blocks
        for blk in self.block[1:]:
            x = blk(x)
        return x


class MaskedPatchConv(BaseMaskedModule):
    """Conv2D with reweighting mask"""

    def __init__(self, conv, share_conv=True, **kwargs):
        super().__init__(num_heads=1, **kwargs)
        self.share_conv = share_conv
        if self.share_conv:
            # share weights over dimensions
            self.conv = nn.Conv2d(1, 1, conv.kernel_size, groups=1, padding="same")
        else:
            assert self.mask_attention is False
            self.conv = conv  # use original conv

    @property
    def hparam_dict(self):
        d = super().hparam_dict
        d['share_conv'] = self.share_conv
        return d

    def forward(self, x, patch_class=None):
        B, C, H, W = x.shape

        if self.mask_attention:
            conv = self.conv.weight.squeeze()  # (K, K)
            bias = self.conv.bias  # (1)

            linear = self.linearize_conv(conv, (H, W))  # (N, N)
            linear = linear.reshape(1, H*W, H*W)  # (1, N, N)
            mask = self.get_mask(patch_class).squeeze(1)  # (B, N, N)

            x = rearrange(x, 'B C H W -> B C (H W)')
            x = x @ (linear * mask).transpose(1, 2) + bias
            x = rearrange(x, 'B C (H W) -> B C H W', H=H)
        elif self.share_conv:
            x = rearrange(x, 'B C H W -> (B C) 1 H W')  # use 1-dim conv
            x = rearrange(self.conv(x), '(B C) 1 H W -> B C H W', B=B)
        else:
            x = self.conv(x)

        return x

    # reference code: https://stackoverflow.com/questions/56702873/
    # is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
    def linearize_conv(self, kernel, input_size):
        """Get linearized conv (assume padding="same")"""
        k_h, k_w = kernel.shape
        i_h, i_w = input_size
        device = kernel.device

        pad_h = (k_h - 1) // 2  # padding for height
        pad_w = (k_w - 1) // 2  # padding for width

        # construct 1d conv toeplitz matrices for each row of the kernel
        toeplitz = []
        for i in range(k_h):
            t = torch.zeros((i_w, i_w), device=device)
            for j in range(i_w):
                j_min = max(j - pad_w, 0)
                j_max = min(j + k_w - pad_w, i_w)
                j_min_gap = j_min - (j - pad_w)
                j_max_gap = (j + k_w - pad_w) - j_max
                t[j, j_min:j_max] = kernel[i][j_min_gap:k_w - j_max_gap]
            toeplitz.append(t)

        # construct toeplitz matrix of toeplitz matrices (for padding="same")
        W = torch.zeros((i_h, i_w, i_h, i_w), device=device)

        for i, B in enumerate(toeplitz):
            for j in range(i_h):
                if 0 <= i + j - pad_h < i_h:
                    W[j, :, i + j - pad_h, :] = B

        W = W.reshape((i_h * i_w, i_h * i_w))  # resize to N x N

        return W


@register_model
def convmixer_s32(pretrained=False, **kwargs):
    """Analogy of mlp-mixer-s32"""
    model = MaskedConvMixer(dim=512, depth=8, kernel_size=9, patch_size=32, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def convmixer_s16(pretrained=False, **kwargs):
    """Analogy of mlp-mixer-s16"""
    model = MaskedConvMixer(dim=512, depth=8, kernel_size=9, patch_size=16, **kwargs)
    model.default_cfg = _cfg()
    return model

