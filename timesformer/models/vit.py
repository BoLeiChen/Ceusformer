# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat
from timesformer.cupy_layers.aggregation_zeropad import LocalConvolution
from .layers import get_act_layer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

# scale the pos to (0 ~ 1)
def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return  x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(32, 32)):
    assert seq_len == grid_size[0] * grid_size[1]

    # record the pos of each grid according to the form of region box
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()

    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)

    px_max = px_min + 1
    py_max = py_min + 1

    # scale pos
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])

    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])

    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)

class S_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 768        768 * 3
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(num_heads)])
    def forward(self, x):

        xt = x[:, 1:, :]      #(32,196,768)
        relative_geometry_embeddings = BoxRelationalEmbedding(xt)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        B, N, C = x.shape     # 32  197  768
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #(3,32,12,197,64)
           q, k, v = qkv[0], qkv[1], qkv[2]      #(32,12,197,64)
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale    #(32,12,197,197)
        size = list(attn.size())
        t1 = torch.zeros([size[0], size[1], size[2] - 1, 1]).cuda()
        t2 = torch.zeros([size[0], size[1], 1, size[3]]).cuda()
        rgw = torch.log(torch.clamp(relative_geometry_weights, min=1e-6))
        rgw = torch.cat((t1, rgw), 3)
        rgw = torch.cat((t2, rgw), 2)
        attn = rgw + attn
        attn = attn.softmax(dim=-1)    #(32,12,197,197)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class S_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = S_Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):

        num_spatial_tokens = (x.size(1) - 1) // T    # 196
        H = num_spatial_tokens // W                  # 14

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:] #(4,1568,768)
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)    #(784,8,768)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt    #(32,197,768)
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)

            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=8, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
        # self.norm1 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        # self.CoTNet = CotLayer(dim, 3)
        self.S_attn = S_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=[1, 1], dilation=1, ceil_mode=False)
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            x1 = xt[:, 0, :].unsqueeze(1)
            x2 = xt[:, 1, :].unsqueeze(1)
            x3 = xt[:, 2, :].unsqueeze(1)
            x4 = xt[:, 3, :].unsqueeze(1)
            x5 = xt[:, 4, :].unsqueeze(1)
            x6 = xt[:, 5, :].unsqueeze(1)
            x7 = xt[:, 6, :].unsqueeze(1)
            x8 = xt[:, 7, :].unsqueeze(1)
            x9 = xt[:, 8, :].unsqueeze(1)

            x_c8 = torch.cat([x2 - x1, x2 - x1, x3 - x2, x4 - x3, x5 - x4, x6 - x5, x7 - x6, x8 - x7, x9 - x8], 1)
            xt = x_c8

            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)

            conv_ = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.S_attn(self.norm3(conv_)))
            ###############################
            # xs = self.norm1(xs)
            # b, hw, c = xs.size()
            # xs_cbl = rearrange(xs, 'b (h w) c  -> b c h w', b=b, h=int(hw ** 0.5), w=int(hw ** 0.5), c=c)
            # conv_ = self.CoTNet(xs_cbl)
            # conv_ = rearrange(conv_, 'b c h w  -> b (h w) c', b=b, h=int(hw ** 0.5), w=int(hw ** 0.5), c=c)
            # conv_ = torch.cat((cls_token, conv_), 1)
            #
            # # res_spatial = conv_
            # res_spatial = self.drop_path(self.S_attn(self.norm3(conv_)))
            ################################

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.num_frames = 9
        self.patch_size = patch_size
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.mask_gen = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)        #(B, f * 256 + 1, 768)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)     # (b, 784 + 1, 768)
        return x

    def forward(self, x):
        # f = x.size(2)
        x = self.forward_features(x)

        mask_ceus = x[:, 1:, :]
        B, N, D = mask_ceus.size()
        mask_ceus = mask_ceus.view(B, N, 3, self.patch_size, self.patch_size)
        mask_ceus = mask_ceus.permute(0, 2, 1, 3, 4).contiguous().view(B * self.num_frames, 3, int((N / self.num_frames) ** 0.5), self.patch_size,
                                                             int((N / self.num_frames) ** 0.5), self.patch_size)
        mask_ceus = mask_ceus.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 3 * self.num_frames, int((N / self.num_frames) ** 0.5) * self.patch_size,
                                                          int((N / self.num_frames) ** 0.5) * self.patch_size)
        f_idx = int((self.num_frames - 1) / 2 * 3)
        mask_ceus = self.mask_gen(mask_ceus[:,f_idx:f_idx + 3])

        x = self.head(x[:, 0])

        return x, mask_ceus

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time', pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=False
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=4, num_heads=8, mlp_ratio=2, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x, mask = self.model(x)

        results = {"seg_ceus": mask, "label": x}
        return results
