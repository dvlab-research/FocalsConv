import torch
import torch.nn as nn
import spconv.pytorch as spconv
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import split_voxels, FocalLoss
from pcdet.utils import common_utils
from spconv.core import ConvAlgo
import time
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
from spconv.pytorch.cppcore import torch_tensor_to_tv
from spconv.utils import Point2VoxelGPU3d
#from pcdet.ops.check_repeat.check_repeat_utils import CheckRepeat
from spconv.pytorch.spatial import RemoveDuplicate

class FocalSparseConvCUDA(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, norm_fn=None, indice_key=None,
                image_channel=3, kernel_size=3, padding=1, mask_multi=False, use_img=False,
                topk=False, threshold=0.5, skip_mask_kernel=False, enlarge_voxel_channels=-1, 
                point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                voxel_size = [0.1, 0.05, 0.05]):
        super(FocalSparseConvCUDA, self).__init__()

        # SubMConv3d, FocalsConv3d
        self.conv = spconv.FocalsConv3d(inplanes, planes, kernel_size=kernel_size, stride=1, bias=False, indice_key=indice_key, algo=ConvAlgo.Native)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        offset_channels = kernel_size**3

        self.topk = topk
        self.threshold = threshold
        self.voxel_stride = voxel_stride
        self.focal_loss = FocalLoss()
        self.mask_multi = mask_multi
        self.skip_mask_kernel = skip_mask_kernel
        self.use_img = use_img
        #self.checkrepeat = CheckRepeat()
        self.time = [0, 0, 0, 0]
        self.indice_key = indice_key

        voxel_channel = enlarge_voxel_channels if enlarge_voxel_channels>0 else inplanes
        in_channels = image_channel + voxel_channel if use_img else voxel_channel

        self.conv_enlarge = spconv.SparseSequential(spconv.SubMConv3d(inplanes, enlarge_voxel_channels, 
            kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key+'_enlarge'),
            norm_fn(enlarge_voxel_channels),
            nn.ReLU(True)) if enlarge_voxel_channels>0 else None

        self.conv_imp = spconv.SubMConv3d(in_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key+'_imp')

        _step = int(kernel_size//2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda()
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.remove_repeat_spconv = RemoveDuplicate()

    def construct_multimodal_features(self, x, x_rgb, batch_dict, fuse_sum=False):
        """
            Construct the multimodal features with both lidar sparse features and image features.
            Args:
                x: [N, C] lidar sparse features
                x_rgb: [b, c, h, w] image features
                batch_dict: input and output information during forward
                fuse_sum: bool, manner for fusion, True - sum, False - concat

            Return:
                image_with_voxelfeatures: [N, C] fused multimodal features
        """
        batch_index = x.indices[:, 0]
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]

        if not x_rgb.shape == batch_dict['images'].shape:
            x_rgb = nn.functional.interpolate(x_rgb, (h, w), mode='bilinear')

        image_with_voxelfeatures = []
        voxels_2d_int_list = []
        filter_idx_list = []
        for b in range(batch_size):
            x_rgb_batch = x_rgb[b]

            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index==b]
            voxel_features_sparse = x.features[batch_index==b]

            # Reverse the point cloud transformations to the original coords.
            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1

            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())

            voxels_2d_int = torch.Tensor(voxels_2d).to(x_rgb_batch.device).long()

            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], x_rgb_batch.shape[0]), device=x_rgb_batch.device)
            image_features_batch[filter_idx] = x_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            if fuse_sum:
                image_with_voxelfeature = image_features_batch + voxel_features_sparse
            else:
                image_with_voxelfeature = torch.cat([image_features_batch, voxel_features_sparse], dim=1)

            image_with_voxelfeatures.append(image_with_voxelfeature)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures)
        return image_with_voxelfeatures

    def _gen_sparse_features(self, x, imps_3d, batch_dict, voxels_3d):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                imps_3d: [N, kernelsize**3], the predicted importance values
                batch_dict: input and output information during forward
                voxels_3d: [N, 3], the 3d positions of voxel centers
        """
        batch_size = x.batch_size
        add_features = []
        add_indices = []
        add_masks = []

        box_of_pts_cls_targets = []
        mask_voxels = []
        mask_kernel_list = []

        for b in range(batch_size):
            if self.training:
                index = x.indices[:, 0]
                batch_index = index==b
                mask_voxel = imps_3d[batch_index, -1].sigmoid()
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :-1].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch>=0)

            selected_features, selected_indices, mask_kernel_fore = split_voxels(x, b, imps_3d, voxels_3d, self.kernel_offsets, 
                                    mask_multi=self.mask_multi, topk=self.topk, threshold=self.threshold, only_return_add=True)

            add_features.append(selected_features)
            add_indices.append(selected_indices)
            add_masks.append(mask_kernel_fore)

        add_features = torch.cat(add_features, dim=0)
        add_indices = torch.cat(add_indices, dim=0)
        add_masks = torch.cat(add_masks, dim=0)
        
        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return add_features, add_indices, add_masks, loss_box_of_pts

        x_fore = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        x_back = spconv.SparseConvTensor(voxel_features_back, voxel_indices_back, x.spatial_shape, x.batch_size)

        return x_fore, x_back, loss_box_of_pts, mask_kernel

    def combine_out2(self, x_fore, add_features, add_indices, remove_repeat=False, replace_features=True):
        #x_fore_features = torch.cat([x_fore.features, add_features], dim=0)
        x_fore_indices = torch.cat([x_fore.indices, add_indices], dim=0)
        x_fore_features = torch.cat([x_fore.features, torch.zeros((add_indices.shape[0], x_fore.features.shape[-1]), device=x_fore.features.device)])
        x_fore = x_fore.replace_feature(x_fore_features)
        x_fore.indices = x_fore_indices
        #im_mask = x_fore_features.abs().sum(-1)>0
        return x_fore #, im_mask

    def _unique(self, out):
        features, indices = out.features, out.indices
        X, Y, Z = out.spatial_shape
        idx = indices
        idx_sum = idx.select(1, 0) * X*Y*Z + idx.select(1, 1) * Y*Z + idx.select(1, 2) * Z + idx.select(1, 3)
        _, ind = idx_sum.sort()
        indices = indices[ind] 
        features = features[ind]
        #features, indices = features.flip([0]), indices.flip([0])
        _unique, inverse = torch.unique_consecutive(indices, return_inverse=True, dim=0)

        if _unique.shape[0] < indices.shape[0]:
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
            features_new.index_add_(0, inverse.long(), features)
            features = features_new
            perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
            indices = indices[perm_].int()
        out = out.replace_feature(features)
        out.indices = indices
        return out

    def forward(self, x, batch_dict, x_rgb=None):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        if self.use_img:
            features_multimodal = self.construct_multimodal_features(x, x_rgb, batch_dict)
            x_predict = spconv.SparseConvTensor(features_multimodal, x.indices, x.spatial_shape, x.batch_size)
        else:
            x_predict = self.conv_enlarge(x) if self.conv_enlarge else x

        imps_3d = self.conv_imp(x_predict).features

        add_features, add_indices, add_masks, loss_box_of_pts = self._gen_sparse_features(x, imps_3d, batch_dict, voxels_3d)

        if not self.skip_mask_kernel:
            add_features *= add_masks.unsqueeze(-1) 

        add_indices = torch.unique(add_indices, dim=0)

        out = self.combine_out2(x, add_features, add_indices, remove_repeat=True)

        if out.batch_size >1:
            out_list = []
            for b in range(out.batch_size):
                batch_index = out.indices[:, 0] == b
                ori_feat_num = (x.indices[:, 0] == b).sum()
                x_ = spconv.SparseConvTensor(out.features[batch_index], out.indices[batch_index], out.spatial_shape, out.batch_size)
                out_ = self.conv(x_, ori_feat_num=ori_feat_num) #x_.features.shape[0])
                out_list.append(out_)
            out = out.replace_feature(torch.cat([out_.features for out_ in out_list]))
            out.indices = torch.cat([out_.indices for out_ in out_list])
        else:
            out = self.conv(out, ori_feat_num=int(x.features.shape[0]))
        out = self._unique(out)

        if self.use_img:
            out = out.replace_feature(self.construct_multimodal_features(out, x_rgb, batch_dict, True))

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        return out, batch_dict, loss_box_of_pts
