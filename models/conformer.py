import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy
from timm.models.layers import DropPath, trunc_normal_
from timesformer.models.vit import PatchEmbed, CotLayer, S_Attention
from einops import rearrange
import math

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, _ = x1.size()
        seq_len2 = x2.size()[1]

        # q1(batch_size, num_heads, seq_len1, k_dim)
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        # k2(batch_size, num_heads, k_dim, seq_len2)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        # v2(batch_size, num_heads, seq_len2, v_dim)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        # attention(batch_size, num_heads, seq_len1, seq_len2)
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = F.softmax(attention, dim=1)
        # output(batch_size, num_heads, seq_len1, v_dim)=>(batch_size, seq_len1, num_heads*v_dim)
        output = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # output(batch_size, seq_len1, in_dim1)
        output = self.proj_o(output)

        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block_ST(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
        self.norm1 = norm_layer(dim)
        # self.norm3 = norm_layer(dim)
        self.CoTNet = CotLayer(dim, 3)
        # self.S_attn = S_Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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
            # x8 = xt[:, 7, :].unsqueeze(1)
            # x9 = xt[:, 8, :].unsqueeze(1)
            # x10 = xt[:, 9, :].unsqueeze(1)
            # x11 = xt[:, 10, :].unsqueeze(1)
            # x12 = xt[:, 11, :].unsqueeze(1)
            # x13 = xt[:, 12, :].unsqueeze(1)
            # x14 = xt[:, 13, :].unsqueeze(1)
            # x15 = xt[:, 14, :].unsqueeze(1)
            # x16 = xt[:, 15, :].unsqueeze(1)
            # x17 = xt[:, 16, :].unsqueeze(1)

            # x_c8 = torch.cat([x2 - x1, x2 - x1, x3 - x2, x4 - x3, x5 - x4, x6 - x5, x7 - x6, x8 - x7, x9 - x8], 1)
            x_c8 = torch.cat([x2 - x1, x2 - x1, x3 - x2, x4 - x3, x5 - x4, x6 - x5, x7 - x6], 1)
            # x_c8 = torch.cat([x2 - x1, x2 - x1, x3 - x2, x4 - x3, x5 - x4], 1)

            # xt = xt + x_c8
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

            ###############################
            xs = self.norm1(xs)
            b, hw, c = xs.size()
            xs_cbl = rearrange(xs, 'b (h w) c  -> b c h w', b=b, h=int(hw ** 0.5), w=int(hw ** 0.5), c=c)
            conv_ = self.CoTNet(xs_cbl)
            conv_ = rearrange(conv_, 'b c h w  -> b (h w) c', b=b, h=int(hw ** 0.5), w=int(hw ** 0.5), c=c)
            conv_ = torch.cat((cls_token, conv_), 1)

            res_spatial = conv_
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

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class FrontDoorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ll_self_attn = Attention(768, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1)
        self.lg_cross_attn = CrossAttention(768, 768, 1024, 1024, 1)
        # self.ln = BertLayerNorm(768, eps=1e-12)

        self.aug_linear = nn.Linear(768, 1)
        self.ori_linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_feats, global_feats):
        '''
        :local_feats: input's hidden_states
        :global_feats: KMeans's hidden_states
        '''
        # ll_feats = self.ll_self_attn(local_feats)
        # lg_feats = self.lg_cross_attn(local_feats, global_feats)
        # out_feats = self.ln(ll_feats + lg_feats)
        #
        # aug_linear_weight = self.aug_linear(out_feats)
        # ori_linear_weight = self.ori_linear(local_feats)
        # aug_weight = self.sigmoid(aug_linear_weight + ori_linear_weight)
        # out_feats = torch.mul(aug_weight, out_feats) + torch.mul((1 - aug_weight), local_feats)

        lg_feats = self.lg_cross_attn(local_feats, global_feats)

        aug_linear_weight = self.aug_linear(lg_feats)
        ori_linear_weight = self.ori_linear(local_feats)
        aug_weight = self.sigmoid(aug_linear_weight + ori_linear_weight)
        out_feats = torch.mul(aug_weight, lg_feats) + torch.mul((1 - aug_weight), local_feats)

        return out_feats

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), num_frames=3):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.num_frames = num_frames

        # self.confounder = numpy.load('/home/cbl/Ceusformer/confounder/confounder.npy', allow_pickle=True).item()
        # self.cross_attn = CrossAttention(768, 768, 1024, 1024, 1)
        # self.front_door_adj = FrontDoorEncoder()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        # back_confounder = torch.from_numpy(numpy.concatenate((self.confounder["bus_back_mask"], self.confounder["bus_back_ROI"],
        #                                                       self.confounder["bus_back_bg"]), axis=0)).unsqueeze(0).to('cuda:0').to(torch.float32)
        # B, _, C = x.shape
        # back_confounder = back_confounder.repeat(B,1,1)
        # x_after_back = self.cross_attn(x, back_confounder)
        # x = x + x_after_back
        # front_confounder = torch.from_numpy(self.confounder["bus_front"]).unsqueeze(0).to('cuda:0').to(torch.float32)
        # front_confounder = front_confounder.repeat(B,1,1)
        # x = self.front_door_adj(x, front_confounder)
        #
        x = x.repeat(1, self.num_frames, 1)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), num_frames=3):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.num_frames = num_frames
        self.conv_project = nn.Conv2d(inplanes * num_frames, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

        # self.confounder = numpy.load('/home/cbl/Ceusformer/confounder/confounder.npy', allow_pickle=True).item()
        # self.cross_attn = CrossAttention(768, 768, 1024, 1024, 1)
        # self.front_door_adj = FrontDoorEncoder()


    def forward(self, x, H, W):
        B, _, C = x.shape

        # back_confounder = torch.from_numpy(numpy.concatenate((self.confounder["ceus_back_mask"], self.confounder["ceus_back_ROI"],
        #                                                       self.confounder["ceus_back_bg"]), axis=0)).unsqueeze(0).to('cuda:0').to(torch.float32)
        # back_confounder = back_confounder.repeat(B,1,1)
        # x_after_back = self.cross_attn(x[:, 1:, :], back_confounder)
        # x_after_back = torch.cat([x[:, 0:1, :], x_after_back], dim=1)
        # x = x + x_after_back
        # front_confounder = torch.from_numpy(self.confounder["ceus_front"]).unsqueeze(0).to('cuda:0').to(torch.float32)
        # front_confounder = front_confounder.repeat(B,1,1)
        # x_ = self.front_door_adj(x[:, 1:, :], front_confounder)
        # x = torch.cat([x[:, 0:1, :], x_], dim=1)

        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C * self.num_frames, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, num_frames=3, attention_type="divided_space_time"):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride, num_frames=num_frames)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride, num_frames=num_frames)

        # self.trans_block = Block(
        #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.trans_block = Block_ST(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attention_type=attention_type)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t, B, T, W):
        x, x2 = self.cnn_block(x)

        _, _, h, w = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t, B, T, W)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, h // self.dw_stride, w // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class CNN_Decoder(nn.Module):
    def __init__(self):
        super(CNN_Decoder, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (B, 512, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 256, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 128, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 64, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),  # (B, 3, 256, 256)
        )

    def forward(self, x):
        return self.deconv_layers(x)


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=2, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, img_size=256, num_frames=3, attention_type='divided_space_time', depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.1):

        # Transformer
        super().__init__()

        # self.total_f = []
        # self.count = 0

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        # self.trans_norm = nn.LayerNorm(embed_dim)
        # self.trans_cls_head = nn.Linear(embed_dim, 1)
        # self.trans_cls_head = nn.Linear(embed_dim, 32)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), 1)
        # self.label_pred = nn.Linear(32 * 2, 1)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attention_type = 'divided_space_time'
        self.num_frames = num_frames
        self.patch_size = patch_size
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)
        self.linear_proj = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * 3)
        self.mask_gen = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
        # self.mask_gen = nn.Conv2d(in_channels=3 * self.num_frames, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.CNNDecoder = CNN_Decoder()


        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        # self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
        #                      )
        self.trans_1 = Block_ST(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attention_type=attention_type)

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, num_frames=num_frames, attention_type=attention_type
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, num_frames=num_frames, attention_type=attention_type
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion, num_frames=num_frames, attention_type=attention_type
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):

        us = x[:, self.num_frames]
        ceus = x[:, 0:self.num_frames].transpose(1,2)
        # ceus = x[:, self.num_frames:].transpose(1, 2)

        # x1 = ceus[:, :, 0].unsqueeze(2)
        # x2 = ceus[:, :, 1].unsqueeze(2)
        # x3 = ceus[:, :, 2].unsqueeze(2)
        # x4 = ceus[:, :, 3].unsqueeze(2)
        # x5 = ceus[:, :, 4].unsqueeze(2)
        # x6 = ceus[:, :, 5].unsqueeze(2)
        # x7 = ceus[:, :, 6].unsqueeze(2)
        # x8 = ceus[:, :, 7].unsqueeze(2)
        # x9 = ceus[:, :, 8].unsqueeze(2)
        #
        # ceus = torch.cat([x2 - x1, x2 - x1, x3 - x2, x4 - x3, x5 - x4, x6 - x5, x7 - x6, x8 - x7, x9 - x8], 2)

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(us))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        B = ceus.size(0)
        x_t, T, W = self.patch_embed(ceus)
        cls_tokens = self.cls_token.expand(x_t.size(0), -1, -1)
        x_t = torch.cat((cls_tokens, x_t), dim=1)
        if x_t.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x_t.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x_t.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x_t = x_t + new_pos_embed
        else:
            x_t = x_t + self.pos_embed
        x_t = self.pos_drop(x_t)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x_t[:B, 0, :].unsqueeze(1)
            x_t = x_t[:,1:]
            x_t = rearrange(x_t, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x_t = x_t + new_time_embed
            else:
                x_t = x_t + self.time_embed
            x_t = self.time_drop(x_t)
            x_t = rearrange(x_t, '(b n) t m -> b (n t) m',b=B,t=T)
            x_t = torch.cat((cls_tokens, x_t), dim=1)        #(B, f * 256 + 1, 768)

        x_t = self.trans_1(x_t, B, T, W)
        
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t, B, T, W)     #x: (B, 256, 128, 128)  x_t: (B, 9217, 384)

        mask_ceus = x_t[:, 1:, :]
        mask_ceus = self.linear_proj(mask_ceus)

        # save_f = torch.mean(torch.mean(mask_ceus, dim=1),dim=0)
        # self.total_f.append(save_f.unsqueeze(0))
        # self.count += 1
        # print(len(self.total_f))
        # print(self.count)
        #
        # if self.count >= 237:
        #     avg_f = torch.mean(torch.cat(self.total_f, dim=0), dim=0)
        #     self.total_f = []
        #     self.count = 0
        #     print(avg_f)
        #     print(avg_f.size())
        #     avg_f = avg_f.data.cpu().numpy()
        #     numpy.save("/home/cbl/Ceusformer/global_front_b.npy", avg_f)

        B, N, D = mask_ceus.size()
        mask_ceus = mask_ceus.view(B, N, 3, self.patch_size, self.patch_size)
        mask_ceus = mask_ceus.permute(0, 2, 1, 3, 4).contiguous().view(B * self.num_frames, 3, int((N / self.num_frames) ** 0.5), self.patch_size,
                                                             int((N / self.num_frames) ** 0.5), self.patch_size)
        mask_ceus = mask_ceus.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 3 * self.num_frames, int((N / self.num_frames) ** 0.5) * self.patch_size,
                                                          int((N / self.num_frames) ** 0.5) * self.patch_size)
        f_idx = int((self.num_frames - 1) / 2 * 3)
        mask_ceus = self.mask_gen(mask_ceus[:,f_idx:f_idx + 3])
        # mask_ceus = self.mask_gen(mask_ceus)
        results = {"seg_ceus": mask_ceus}

        mask_us = self.CNNDecoder(x)
        results["seg_us"] = mask_us

        # results = {}
        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)
        results["label"] = conv_cls

        # trans classification
        # x_t = self.trans_norm(x_t[:, 0])
        # tran_cls = self.trans_cls_head(x_t)
        # results["label"] = tran_cls

        # label_p = self.label_pred(torch.cat([conv_cls, tran_cls], dim=1))
        # results["label"] = label_p

        return results
