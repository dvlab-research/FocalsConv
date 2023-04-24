import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SubMConv3d
import torch
from torch import nn, einsum

from ..registry import BACKBONES
from ..utils import build_norm_layer
from functools import partial

def conv(in_planes, out_planes, kernel_size=3, stride=1, indice_key=None, bias=True):
    """convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=int(kernel_size//2),
        bias=bias,
        indice_key=indice_key,
    )

def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                    )

        self.conv3x3_1 = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=int(kernel_size//2)-1,
                                        bias=bias,
                                        dilation=int(kernel_size//2)-1,
                                        indice_key=indice_key+'conv_3x3_1',
                                    )
        self._indice_list = []

        if kernel_size==7:
            _list = [0, 3, 4, 7]
        elif kernel_size==5:
            _list = [0, 2, 3, 5]
        else:
            raise ValueError('Unknown kernel size %d'%kernel_size)
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    b = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data)
        x_conv_block = self.block(x_conv)
        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)
        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block


class SpatialGroupConvV2(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, dilation=1, _type='A'):
        super(SpatialGroupConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_size==3:
            kernel_size = 7
        _list = [0, int(kernel_size//2), int(kernel_size//2)+1, 7]
        self.group_map = torch.zeros((3**3, int(kernel_size//2)**3)) - 1
        _num = 0
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    _pos = a.sum()
                    self.group_map[_num][:_pos] = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    _num += 1
        self.group_map = self.group_map.int()
        position_embedding = True
        self.block = spconv.SpatialGroupConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size, 3,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        dilation=dilation,
                                        indice_key=indice_key,
                                        algo=ConvAlgo.Native,
                                        position_embedding=position_embedding,
                                    )
        if position_embedding:
            trunc_normal_(self.block.position_embedding, std=0.02)

    def forward(self, x_conv):
        x_conv = self.block(x_conv, group_map=self.group_map.to(x_conv.features.device))
        return x_conv


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
        conv_type='common',
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        if conv_type=="spatialgroupconv":
            conv_func = SpatialGroupConv
        elif conv_type == 'spatialgroupconvv2':
            conv_func = SpatialGroupConvV2
        elif conv_type=='common':
            conv_func = conv
        else:
            raise ValueError('Unknown conv type %s.'%conv_type)

        self.conv1 = conv_func(inplanes, planes, kernel_size, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(True)
        self.conv2 = conv_func(planes, planes, kernel_size, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


@BACKBONES.register_module
class SpMiddleResNetFHDLargeKernel(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDLargeKernel", **kwargs
    ):
        super(SpMiddleResNetFHDLargeKernel, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        conv_types = kwargs.get('conv_types', ['common', 'common', 'common', 'common'])
        kernel_sizes = kwargs.get('kernel_sizes', [3, 3, 3, 3])
        kernel_sizes_downsample = kwargs.get('kernel_sizes_downsample', [3, 3])

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.fuse = kwargs.get('fuse', False)
        if self.fuse:
            self.conv = SubMConv3d(16, 16, 3, bias=False, indice_key="conv_img")
            self.bn = build_norm_layer(norm_cfg, 16)[1]
            self.relu = nn.ReLU(inplace=True)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res_input"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, kernel_size=kernel_sizes[0], norm_cfg=norm_cfg, indice_key="res0", 
                conv_type=conv_types[0]),
            SparseBasicBlock(16, 16, kernel_size=kernel_sizes[0], norm_cfg=norm_cfg, indice_key="res0", 
                conv_type=conv_types[0]),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, kernel_sizes_downsample[0], 2, padding=int(kernel_sizes_downsample[0]//2), bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, kernel_size=kernel_sizes[1], norm_cfg=norm_cfg, indice_key="res1", 
                conv_type=conv_types[1]),
            SparseBasicBlock(32, 32, kernel_size=kernel_sizes[1], norm_cfg=norm_cfg, indice_key="res1", 
                conv_type=conv_types[1]),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, kernel_sizes_downsample[1], 2, padding=int(kernel_sizes_downsample[1]//2), bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, kernel_size=kernel_sizes[2], norm_cfg=norm_cfg, indice_key="res2", 
                conv_type=conv_types[2]),
            SparseBasicBlock(64, 64, kernel_size=kernel_sizes[2], norm_cfg=norm_cfg, indice_key="res2", 
                conv_type=conv_types[2]),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, kernel_size=kernel_sizes[3], norm_cfg=norm_cfg, indice_key="res3", 
                conv_type=conv_types[3]),
            SparseBasicBlock(128, 128, kernel_size=kernel_sizes[3], norm_cfg=norm_cfg, indice_key="res3", 
                conv_type=conv_types[3]),
        )


        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(True),
        )

    def forward(self, voxel_features, batch_dict, coors, batch_size, input_shape, fuse_func=None):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)

        if self.fuse:
            x_conv1 = self.conv(x_conv1)
            x_conv1 = fuse_func(batch_dict, encoded_voxel=x_conv1, layer_name="layer1")
            x_conv1 = x_conv1.replace_feature(self.bn(x_conv1.features))
            x_conv1 = x_conv1.replace_feature(self.relu(x_conv1.features))

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features, 0