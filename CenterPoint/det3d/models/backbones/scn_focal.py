import numpy as np
try:
    import spconv.pytorch as spconv 
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except: 
    import spconv 
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from functools import partial

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .focal_sparse_conv.focal_sparse_conv import FocalSparseConv


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, gt_boxes=None):
        loss_box_of_pts = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, _loss = module(input, gt_boxes)
                loss_box_of_pts += _loss
            else:
                input = module(input)
        return input, loss_box_of_pts


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
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


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
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
class SpMiddleResNetFHDFocal(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHDFocal", **kwargs
    ):
        super(SpMiddleResNetFHDFocal, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        use_img = kwargs.get('USE_IMG', False)
        topk = kwargs.get('TOPK', True)
        threshold = kwargs.get('THRESHOLD', 0.5)
        skip_loss = kwargs.get('SKIP_LOSS', False)
        mask_multi = kwargs.get('MASK_MULTI', True)
        enlarge_voxel_channels = kwargs.get('ENLARGE_VOXEL_CHANNELS', -1)
        self.use_img = use_img

        if use_img:
            self.conv_focal_multimodal = FocalSparseConv(16, 16, voxel_stride=1, norm_cfg=norm_cfg, padding=1, 
                                                    indice_key='spconv_focal_multimodal', skip_loss=skip_loss, 
                                                    mask_multi=mask_multi, topk=topk, threshold=threshold, use_img=True)

        special_spconv_fn = partial(FocalSparseConv, skip_loss=skip_loss, enlarge_voxel_channels=enlarge_voxel_channels, 
                                                    mask_multi=mask_multi, topk=topk, threshold=threshold)
        special_conv_list = kwargs.get('SPECIAL_CONV_LIST', [])

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            
            special_spconv_fn(16, 16, kernel_size=3, voxel_stride=1, norm_cfg=norm_cfg, padding=1, indice_key='focal0') 
            if 1 in special_conv_list else None,
        )

        self.conv2 = SparseSequentialBatchdict(
            spconv.SparseSequential(SparseConv3d(16, 32, 3, 2, padding=1, bias=False),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True)),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),

            special_spconv_fn(32, 32, kernel_size=3, voxel_stride=2, norm_cfg=norm_cfg, padding=1, indice_key='focal1')
            if 2 in special_conv_list else None,

        )

        self.conv3 = SparseSequentialBatchdict(
            spconv.SparseSequential(SparseConv3d(32, 64, 3, 2, padding=1, bias=False),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True)),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),

            special_spconv_fn(64, 64, kernel_size=3, voxel_stride=4, norm_cfg=norm_cfg, padding=1, indice_key='focal2')
            if 3 in special_conv_list else None,
        )

        self.conv4 = SparseSequentialBatchdict(
            spconv.SparseSequential(SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1], bias=False),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True)),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, batch_dict, coors, batch_size, input_shape, fuse_func=None):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        loss_box_of_pts = 0

        x = self.conv_input(ret)

        x_conv1, _loss = self.conv1(x, batch_dict)
        loss_box_of_pts += _loss

        if self.use_img:
            x_conv1, _loss = self.conv_focal_multimodal(x_conv1, batch_dict, fuse_func)
            loss_box_of_pts = loss_box_of_pts + _loss

        x_conv2, _loss = self.conv2(x_conv1, batch_dict)
        loss_box_of_pts += _loss

        x_conv3, _loss = self.conv3(x_conv2, batch_dict)
        loss_box_of_pts += _loss

        x_conv4, _loss = self.conv4(x_conv3, batch_dict)
        loss_box_of_pts += _loss

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

        return ret, multi_scale_voxel_features, loss_box_of_pts
