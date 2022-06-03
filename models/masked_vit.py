# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.convit import ConViT
from timm.models.registry import register_model

from .masked_base import BaseMaskedModel, BaseMaskedBlock, BaseMaskedModule

import copy
from einops import rearrange
from einops.layers.torch import Rearrange


__all__ = [
    'deit_t', 'deit_s', 'deit_b',
]


class MaskedViT(BaseMaskedModel):
    """ViT with MaskedAttention"""
    def __new__(cls, use_cls_token=True, **all_kwargs):
        base_kwargs, model_kwargs, mask_kwargs = BaseMaskedModel.parse_kwargs(**all_kwargs)
        attention_type = base_kwargs.get('attention_type', 'vit')
        mixin = {'vit': VisionTransformer, 'convit': ConViT, 'coat': VisionTransformer}[attention_type]
        cls = type(cls.__name__ + '+' + mixin.__name__, (cls, mixin), {})
        return super(MaskedViT, cls).__new__(cls)

    def __init__(self, use_cls_token=True, **all_kwargs):
        base_kwargs, model_kwargs, mask_kwargs = BaseMaskedModel.parse_kwargs(**all_kwargs)
        self.use_cls_token = use_cls_token

        # initialize model
        BaseMaskedModel.__init__(self, **base_kwargs)

        print(f'model_kwargs: {model_kwargs}')

        if self.attention_type == 'vit':
            # Remove ConViT parameters
            model_kwargs.pop('local_up_to_layer')
            model_kwargs.pop('locality_strength')
            VisionTransformer.__init__(self, **model_kwargs)

        elif self.attention_type == 'coat':
            # Remove ConViT parameters
            model_kwargs.pop('local_up_to_layer')
            model_kwargs.pop('locality_strength')
            VisionTransformer.__init__(self, **model_kwargs)

        elif self.attention_type == 'convit':
            embed_dim = model_kwargs.get('embed_dim', 192)
            num_heads = model_kwargs.get('num_heads', 3)
            model_kwargs['embed_dim'] = int(embed_dim / num_heads)

            ConViT.__init__(self, **model_kwargs)
            self.local_up_to_layer = model_kwargs.get('local_up_to_layer', 10)
        else:
            raise ValueError

        self._init_mask_layer()
        self._init_classifier(self.num_features, self.num_classes)

        # convert blocks to masked blocks
        mask_kwargs_no_cls = copy.deepcopy(mask_kwargs)
        mask_kwargs_no_cls.update({'use_cls_token': False})
        mask_kwargs_cls = copy.deepcopy(mask_kwargs)
        mask_kwargs_cls.update({'use_cls_token': True})

        if self.attention_type == 'vit':
            for i in range(len(self.blocks)):
                mask_attention = self.mask_attention and (i in self.mask_layer)
                self.blocks[i] = MaskedBlock(self.blocks[i], mask_attention=mask_attention, attention_type='vit', **mask_kwargs)

        elif self.attention_type == 'coat':
            self.cls_token = None
            self.use_cls_token = False
            self.pos_embed = None
            for i in range(len(self.blocks)):
                mask_attention = self.mask_attention and (i in self.mask_layer)
                self.blocks[i] = MaskedBlock(self.blocks[i], mask_attention=mask_attention, attention_type='coat', **mask_kwargs)

        elif self.attention_type == 'convit':
            for i in range(len(self.blocks)):
                mask_attention = self.mask_attention and (i in self.mask_layer)
                if i >= self.local_up_to_layer: # use MHSA 
                    self.blocks[i] = MaskedBlock(self.blocks[i], mask_attention=mask_attention, attention_type='convit', **mask_kwargs_cls)
                else: # use GPSA
                    self.blocks[i] = MaskedBlock(self.blocks[i], mask_attention=mask_attention, attention_type='convit', **mask_kwargs_no_cls)
        else:
            raise ValueError

        self.apply(self._init_weights)

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

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward_features(self, x, patch_class=None):
        if self.attention_type == 'vit':
            x = self.prepare_tokens(x)
            for blk in self.blocks:
                x = blk(x, patch_class)
            x = self.norm(x)

        elif self.attention_type == 'coat':
            B = x.shape[0]
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x, patch_class)
            x = self.norm(x)

        elif self.attention_type == 'convit':
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            if self.use_pos_embed:
                x = x + self.pos_embed
            x = self.pos_drop(x)
            for u, blk in enumerate(self.blocks):
                if u == self.local_up_to_layer:
                    x = torch.cat((cls_tokens, x), dim=1)
                x = blk(x, patch_class)
            x = self.norm(x)

        if self.use_cls_token:
            x_cls = x[:, 0]
            x_aux = {'tok': x[:, 1:]}
        else:
            x_cls = x.mean(dim=1)
            x_aux = {'tok': x}

        if self.token_label:
            return x_cls, x_aux
        else:
            return x_cls

    def get_attention(self, x, patch_class=None, layer=-1):
        layer = layer if layer != -1 else self.depth
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < layer - 1:
                x = blk(x, patch_class)
            else:
                x = blk.attn.get_attention(blk.norm1(x))
                break
        return x

    def get_attention_after_masking(self, x, patch_class=None, layer=-1):
        layer = layer if layer != -1 else self.depth
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < layer - 1:
                x = blk(x, patch_class)
            else:
                x = blk.attn.get_attention_after_masking(blk.norm1(x), patch_class)
                break
        return x

    def get_masking_result(self, x, patch_class=None, layer=-1):
        layer = layer if layer != -1 else self.depth
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < layer - 1:
                x = blk(x, patch_class)
            else:
                x = blk.attn.get_masking_result(blk.norm1(x), patch_class)
                break
        return x


class MaskedBlock(BaseMaskedBlock):
    """ViT block with MaskedAttention"""

    def __init__(self, block, attention_type='vit', **kwargs):
        super().__init__()
        # copy attributes of block
        self.norm1 = block.norm1
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp

        # modify attention to the masked attention
        if attention_type == 'vit':
            self.attn = MaskedAttention(block.attn, **kwargs)

        elif attention_type == 'coat':
            dim = 192
            self.attn = RelativeAttention(block.attn, **kwargs)
            self.attn = PreNorm(dim, self.attn, nn.LayerNorm)

        elif attention_type == 'convit':
            self.use_gpsa = block.use_gpsa
            if self.use_gpsa:
                self.attn = MaskedGPSA(block.attn, **kwargs)
            else:
                self.attn = MaskedAttention(block.attn, **kwargs)
        else:
            raise ValueError

        self.attention_type = attention_type
        self.add_masked_module('attn', self.attn)

    def forward(self, x, patch_class=None):
        x = x + self.drop_path(self.attn(self.norm1(x), patch_class))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAttention(BaseMaskedModule):
    """Attention with reweighting mask"""

    def __init__(self, attn, use_cls_token=True, **kwargs):
        super().__init__(num_heads=attn.num_heads, **kwargs)
        self.use_cls_token = use_cls_token

        # copy attributes of attention
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x, patch_class=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # apply mask
        if self.mask_attention:
            mask = self.get_mask(patch_class)  # (B, H, N, N)

            if self.use_cls_token:  # pad ones for [CLS] token
                row = torch.ones((B, self.num_heads, 1, N - 1), device=mask.device)
                col = torch.ones((B, self.num_heads, N, 1), device=mask.device)
                mask = torch.cat([row, mask], dim=2)
                mask = torch.cat([col, mask], dim=3)

            attn = attn * mask
            attn = attn / attn.sum(dim=-1, keepdim=True)  # (B, H, N, N)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_after_masking(self, x, patch_class=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # apply mask
        if self.mask_attention:
            mask = self.get_mask(patch_class)  # (B, H, N, N)

            if self.use_cls_token:  # pad ones for [CLS] token
                row = torch.ones((B, self.num_heads, 1, N - 1), device=mask.device)
                col = torch.ones((B, self.num_heads, N, 1), device=mask.device)
                mask = torch.cat([row, mask], dim=2)
                mask = torch.cat([col, mask], dim=3)

            attn = attn * mask
            attn = attn / attn.sum(dim=-1, keepdim=True)  # (B, H, N, N)

        attn = self.attn_drop(attn)

        return attn

    def get_masking_result(self, x, patch_class=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # apply mask
        if self.mask_attention:
            mask = self.get_mask(patch_class)  # (B, H, N, N)

            if self.use_cls_token:  # pad ones for [CLS] token
                row = torch.ones((B, self.num_heads, 1, N - 1), device=mask.device)
                col = torch.ones((B, self.num_heads, N, 1), device=mask.device)
                mask = torch.cat([row, mask], dim=2)
                mask = torch.cat([col, mask], dim=3)

        return mask


# For ConViT
class MaskedGPSA(BaseMaskedModule):
    def __init__(self, attn, use_cls_token=True, **kwargs):
        super().__init__(num_heads=attn.num_heads, **kwargs)
        self.use_cls_token = use_cls_token

        # copy attributes of attention
        self.num_heads = attn.num_heads
        self.dim = attn.dim
        self.head_dim = self.dim // self.num_heads

        self.scale = attn.scale
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.pos_proj = attn.pos_proj
        self.proj_drop = attn.proj_drop
        self.gating_param = attn.gating_param

        self.locality_strength = attn.locality_strength

        self.qk = attn.qk
        self.v = attn.v

        self.rel_indices = attn.rel_indices

    def forward(self, x, patch_class=None):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)
        attn = self.get_attention(x, patch_class)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x, patch_class=None):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)

        # apply mask
        if self.mask_attention:
            mask = self.get_mask(patch_class)  # (B, H, N, N)

            if self.use_cls_token:  # pad ones for [CLS] token
                row = torch.ones((B, self.num_heads, 1, N - 1), device=mask.device)
                col = torch.ones((B, self.num_heads, N, 1), device=mask.device)
                mask = torch.cat([row, mask], dim=2)
                mask = torch.cat([col, mask], dim=3)

            attn = attn * mask
            attn = attn / attn.sum(dim=-1, keepdim=True)  # (B, H, N, N)

        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= self.locality_strength

    def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)


# For CoaT
class RelativeAttention(BaseMaskedModule):
    def __init__(self, attn, use_cls_token=True, **kwargs):
        super().__init__(num_heads=attn.num_heads, **kwargs)

        dim = attn.qkv.weight.shape[1]
        num_heads = attn.num_heads
        self.num_heads = num_heads
        dim_head = dim // num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.ih, self.iw = (224 // 16, 224 // 16)
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), num_heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = attn.qkv

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x, patch_class=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    @property
    def hparam_dict(self):
        d = dict()
        d['mask_attn'] = self.fn.mask_attention
        if self.fn.mask_attention:
            d['patch_dist'] = self.fn.patch_dist
            d['mask_func'] = self.fn.mask_func
        return d

    @property
    def param_dict(self):
        d = dict()
        if self.fn.mask_attention:
            d['mask_scale'] = self.fn.mask_scale.view(-1).tolist()
            d['dist_stats'] = self.fn.dist_stats.stats()
            d['mask_stats'] = self.fn.mask_stats.stats()
        return d

    @property
    def hparam_info(self):
        info = ', '.join(['{}: {}'.format(k, v) for (k, v) in self.fn.hparam_dict.items()])
        return info

    @property
    def param_info_list(self):
        info = []  # list of params
        for name, param in self.fn.param_dict.items():
            _info = '{}: '.format(name) + ', '.join(['{:.3f}'.format(p) for p in param])
            info.append(_info)
        return info

    @property
    def dummy(self):
        dummy = 0
        for p in self.fn.parameters():
            dummy = dummy + p.mean() * 0
        return dummy

    def forward(self, x, patch_class=None, **kwargs):
        return self.fn(self.norm(x), patch_class=patch_class, **kwargs)


@register_model
def deit_t(pretrained=False, **kwargs):
    model = MaskedViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_up_to_layer=10, locality_strength=1.0, # for ConViT
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_s(pretrained=False, **kwargs):
    model = MaskedViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_up_to_layer=10, locality_strength=1.0, # for ConViT
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_b(pretrained=False, **kwargs):
    model = MaskedViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
