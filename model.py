import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np


# from .MBConv import MBConvBlock
# from .MHSA import MHSABlock, Mlp
# from .meta_encoder import ResNormLayer
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 33, 'input_size': (3, 384, 384), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'MetaFG_0': _cfg(),
    'MetaFG_1': _cfg(),
    'MetaFG_2': _cfg(),
}


def make_blocks(stage_index, depths, embed_dims, img_size, dpr, extra_token_num=1, num_heads=8, mlp_ratio=4.,
                stage_type='conv'):
    stage_name = f'stage_{stage_index}'
    blocks = []
    for block_idx in range(depths[stage_index]):
        stride = 2 if block_idx == 0 and stage_index != 1 else 1
        in_chans = embed_dims[stage_index] if block_idx != 0 else embed_dims[stage_index - 1]
        out_chans = embed_dims[stage_index]
        image_size = img_size if block_idx == 0 or stage_index == 1 else img_size // 2
        drop_path_rate = dpr[sum(depths[1:stage_index]) + block_idx]
        if stage_type == 'conv':
            blocks.append(MBConvBlock(ksize=3, input_filters=in_chans, output_filters=out_chans,
                                      image_size=image_size, expand_ratio=int(mlp_ratio), stride=stride,
                                      drop_connect_rate=drop_path_rate))
        elif stage_type == 'mhsa':
            blocks.append(MHSABlock(input_dim=in_chans, output_dim=out_chans,
                                    image_size=image_size, stride=stride, num_heads=num_heads,
                                    extra_token_num=extra_token_num,
                                    mlp_ratio=mlp_ratio, drop_path=drop_path_rate))
        else:
            raise NotImplementedError("We only support conv and mhsa")
    return blocks


class MetaFG_Meta(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=33,
                 conv_embed_dims=[64, 96, 192], attn_embed_dims=[384, 768],
                 conv_depths=[2, 2, 3], attn_depths=[5, 2], num_heads=32, extra_token_num=3, mlp_ratio=4.,
                 conv_norm_layer=nn.BatchNorm2d, attn_norm_layer=nn.LayerNorm,
                 conv_act_layer=nn.ReLU, attn_act_layer=nn.GELU,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 add_meta=True, meta_dims=[4, 3], mask_prob=1.0, mask_type='linear',
                 only_last_cls=False,
                 use_checkpoint=False):
        super().__init__()
        self.only_last_cls = only_last_cls
        self.img_size = img_size
        self.num_classes = num_classes
        self.add_meta = add_meta
        self.meta_dims = meta_dims
        self.cur_epoch = -1
        self.total_epoch = -1
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        self.attn_embed_dims = attn_embed_dims
        self.extra_token_num = extra_token_num
        if self.add_meta:
            #             assert len(meta_dims)==extra_token_num-1
            for ind, meta_dim in enumerate(meta_dims):
                meta_head_1 = nn.Sequential(
                    nn.Linear(meta_dim, attn_embed_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(attn_embed_dims[0]),
                    ResNormLayer(attn_embed_dims[0]),
                ) if meta_dim > 0 else nn.Identity()
                meta_head_2 = nn.Sequential(
                    nn.Linear(meta_dim, attn_embed_dims[1]),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(attn_embed_dims[1]),
                    ResNormLayer(attn_embed_dims[1]),
                ) if meta_dim > 0 else nn.Identity()
                setattr(self, f"meta_{ind + 1}_head_1", meta_head_1)
                setattr(self, f"meta_{ind + 1}_head_2", meta_head_2)

        stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(conv_depths[1:] + attn_depths))]
        # stage_0
        self.stage_0 = nn.Sequential(*[
            nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
            conv_norm_layer(stem_chs[0]),
            conv_act_layer(inplace=True),
            nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
            conv_norm_layer(stem_chs[1]),
            conv_act_layer(inplace=True),
            nn.Conv2d(stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False)])
        self.bn1 = conv_norm_layer(conv_embed_dims[0])
        self.act1 = conv_act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage_1
        self.stage_1 = nn.ModuleList(
            make_blocks(1, conv_depths + attn_depths, conv_embed_dims + attn_embed_dims, img_size // 4,
                        dpr=dpr, num_heads=num_heads, extra_token_num=extra_token_num, mlp_ratio=mlp_ratio,
                        stage_type='conv'))
        # stage_2
        self.stage_2 = nn.ModuleList(
            make_blocks(2, conv_depths + attn_depths, conv_embed_dims + attn_embed_dims, img_size // 4,
                        dpr=dpr, num_heads=num_heads, extra_token_num=extra_token_num, mlp_ratio=mlp_ratio,
                        stage_type='conv'))

        # stage_3
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[0]))
        self.stage_3 = nn.ModuleList(
            make_blocks(3, conv_depths + attn_depths, conv_embed_dims + attn_embed_dims, img_size // 8,
                        dpr=dpr, num_heads=num_heads, extra_token_num=extra_token_num, mlp_ratio=mlp_ratio,
                        stage_type='mhsa'))
        # stage_4
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[1]))
        self.stage_4 = nn.ModuleList(
            make_blocks(4, conv_depths + attn_depths, conv_embed_dims + attn_embed_dims, img_size // 16,
                        dpr=dpr, num_heads=num_heads, extra_token_num=extra_token_num, mlp_ratio=mlp_ratio,
                        stage_type='mhsa'))
        self.norm_2 = attn_norm_layer(attn_embed_dims[1])

        # Aggregate
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(*[Mlp(in_features=attn_embed_dims[0], out_features=attn_embed_dims[1]),
                                           attn_norm_layer(attn_embed_dims[1])])
            self.aggregate = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.norm = attn_norm_layer(attn_embed_dims[1])
            self.norm_1 = attn_norm_layer(attn_embed_dims[0])
        # Classifier head
        self.head = nn.Linear(attn_embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token_1, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             fan_out //= m.groups
        #             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #             if m.bias is not None:
        #                 m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token_1', 'cls_token_2'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, meta=None):
        B = x.shape[0]
        extra_tokens_1 = [self.cls_token_1]
        extra_tokens_2 = [self.cls_token_2]
        if self.add_meta:
            assert meta != None, 'meta is None'
            if len(self.meta_dims) > 1:
                metas = torch.split(meta, self.meta_dims, dim=1)
            else:
                metas = (meta,)
            for ind, cur_meta in enumerate(metas):
                meta_head_1 = getattr(self, f"meta_{ind + 1}_head_1")
                meta_head_2 = getattr(self, f"meta_{ind + 1}_head_2")
                meta_1 = meta_head_1(cur_meta)
                meta_1 = meta_1.reshape(B, -1, self.attn_embed_dims[0])
                meta_2 = meta_head_2(cur_meta)
                meta_2 = meta_2.reshape(B, -1, self.attn_embed_dims[1])
                extra_tokens_1.append(meta_1)
                extra_tokens_2.append(meta_2)

        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for blk in self.stage_1:
            x = blk(x)
        for blk in self.stage_2:
            x = blk(x)
        H0, W0 = self.img_size // 8, self.img_size // 8
        for ind, blk in enumerate(self.stage_3):
            if ind == 0:
                x = blk(x, H0, W0, extra_tokens_1)
            else:
                x = blk(x, H0, W0)
        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)

        x = x[:, self.extra_token_num:, :]
        H1, W1 = self.img_size // 16, self.img_size // 16
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        for ind, blk in enumerate(self.stage_4):
            if ind == 0:
                x = blk(x, H1, W1, extra_tokens_2)
            else:
                x = blk(x, H1, W1)
        cls_2 = x[:, :1, :]
        cls_2 = self.norm_2(cls_2)
        if not self.only_last_cls:
            cls = torch.cat((cls_1, cls_2), dim=1)  # B,2,C
            cls = self.aggregate(cls).squeeze(dim=1)  # B,C
            cls = self.norm(cls)
        else:
            cls = cls_2.squeeze(dim=1)
        return cls

    def forward(self, x, meta=None):
        if meta is not None:
            if self.mask_type == 'linear':
                cur_mask_prob = self.mask_prob - self.cur_epoch / self.total_epoch
            else:
                cur_mask_prob = self.mask_prob
            if cur_mask_prob != 0 and self.training:
                mask = torch.ones_like(meta)
                mask_index = torch.randperm(meta.size(0))[:int(meta.size(0) * cur_mask_prob)]
                mask[mask_index] = 0
                meta = mask * meta
        x = self.forward_features(x, meta)
        x = self.head(x)
        return x


@register_model
def MetaFG_meta_0(pretrained=True, **kwargs):
    model = MetaFG_Meta(img_size=384, conv_embed_dims=[64, 96, 192], attn_embed_dims=[384, 768], meta_dims=[4,3],
                        conv_depths=[2, 2, 3], attn_depths=[5, 2], num_heads=8, mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_0']
    if pretrained:
        model = model.load_state_dict(torch.load('pretrain/metafg_1_1k_384.pth'))
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def MetaFG_meta_1(pretrained=False, **kwargs):
    model = MetaFG_Meta(img_size=384, num_classes=1000, conv_embed_dims=[64, 96, 192], attn_embed_dims=[384, 768],
                        conv_depths=[2, 2, 6], attn_depths=[14, 2], num_heads=8, mlp_ratio=4., meta_dims=[4,3], **kwargs)
    model.default_cfg = default_cfgs['MetaFG_1']
    if pretrained:
        model = model.load_state_dict(torch.load('pretrain/metafg_1_1k_384.pth'))
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def MetaFG_meta_2(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims=[128, 128, 256], attn_embed_dims=[512, 1024],
                        conv_depths=[2, 2, 6], attn_depths=[14, 2], num_heads=8, mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_2']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


"    "
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    return partial(Conv2dStaticSamePadding, image_size=image_size)


def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """
    计算出 Conv2dSamePadding with a stride.
    """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# #MBConvBlock
class MBConvBlock(nn.Module):
    '''
    层 ksize3*3 输入32 输出16  conv1  stride步长1
    '''

    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, image_size=224,
                 drop_connect_rate=0.):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride
        self._drop_connect_rate = drop_connect_rate
        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution
        k = self._kernel_size
        s = self._stride
        self._depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup,
                                         kernel_size=k, stride=s, padding=1, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # Squeeze and Excitation layer, if desired
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)
        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if self._drop_connect_rate != 0:
                x = drop_connect(x, p=self._drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Relative_Attention(nn.Module):
    def __init__(self, dim, img_size, extra_token_num=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.extra_token_num = extra_token_num
        head_dim = dim // num_heads
        self.img_size = img_size  # h,w
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias,add cls_token bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * img_size[0] - 1) * (2 * img_size[1] - 1) + 1, num_heads))  # 2*h-1 * 2*w-1 + 1, nH

        # get pair-wise relative position index for each token
        coords_h = torch.arange(self.img_size[0])
        coords_w = torch.arange(self.img_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += self.img_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.img_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.img_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # h*w, h*w
        relative_position_index = F.pad(relative_position_index, (extra_token_num, 0, extra_token_num, 0))
        relative_position_index = relative_position_index.long()
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, ):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.img_size[0] * self.img_size[1] + self.extra_token_num,
            self.img_size[0] * self.img_size[1] + self.extra_token_num, -1)  # h*w+1,h*w+1,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w+1, h*w+1
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MHSABlock(nn.Module):
    def __init__(self, input_dim, output_dim, image_size, stride, num_heads, extra_token_num=1, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if stride != 1:
            self.patch_embed = OverlapPatchEmbed(patch_size=3, stride=stride, in_chans=input_dim, embed_dim=output_dim)
            self.img_size = image_size // 2
        else:
            self.patch_embed = None
            self.img_size = image_size
        self.img_size = to_2tuple(self.img_size)

        self.norm1 = norm_layer(output_dim)
        self.attn = Relative_Attention(
            output_dim, self.img_size, extra_token_num=extra_token_num, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(output_dim)
        mlp_hidden_dim = int(output_dim * mlp_ratio)
        self.mlp = Mlp(in_features=output_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, extra_tokens=None):
        if self.patch_embed is not None:
            x, _, _ = self.patch_embed(x)

            extra_tokens = [token.expand(x.shape[0], -1, -1) for token in extra_tokens]
            extra_tokens.append(x)
            x = torch.cat(extra_tokens, dim=1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H // 2, W // 2))
        return x


import torch.nn as nn


class ResNormLayer(nn.Module):
    def __init__(self, linear_size, ):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


if __name__ == "__main__":
    x = torch.randn([2, 3, 224, 224])
    meta = torch.randn([2, 7])  # first 3 is geom last 4 is time
    model = MetaFG_Meta()
    # import ipdb;ipdb.set_trace()
    output = model(x, meta)
    print(output.shape)