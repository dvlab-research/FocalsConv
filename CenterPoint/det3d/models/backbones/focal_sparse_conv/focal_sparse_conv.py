import torch
import torch.nn as nn
import spconv.pytorch as spconv
from det3d.models.utils import build_norm_layer
from det3d.models.model_utils.basic_block_2d import BasicBlock2D
from det3d.models.backbones.focal_sparse_conv.utils import split_voxels, check_repeat, FocalLoss


class FocalSparseConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, norm_cfg=None, indice_key=None, kernel_size=3, padding=1, 
                skip_loss=False, mask_multi=False, topk=False, threshold=0.5, enlarge_voxel_channels=-1, use_img=False, 
                point_cloud_range=[-5.0, -54, -54, 3.0, 54, 54], voxel_size=[0.2, 0.075, 0.075]):
        super(FocalSparseConv, self).__init__()

        self.conv = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=1, bias=False, indice_key=indice_key)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(True)


        # TODO: max_additional_voxels * 4 -- (x + y + z + whether to use)
        offset_channels = kernel_size**3
        self.topk = topk
        self.threshold = threshold
        self.voxel_stride = voxel_stride
        self.focal_loss = FocalLoss()
        self.mask_multi = mask_multi
        self.skip_loss = skip_loss
        self.use_img = use_img

        self.conv_enlarge = spconv.SparseSequential(spconv.SubMConv3d(inplanes, enlarge_voxel_channels, kernel_size=3, 
                                    stride=1, padding=1, bias=False, indice_key=indice_key+'_enlarge'),
                                    build_norm_layer(norm_cfg, enlarge_voxel_channels)[1], nn.ReLU()) if enlarge_voxel_channels>0 else None
        in_channels = enlarge_voxel_channels if enlarge_voxel_channels>0 else inplanes

        self.conv_imp = spconv.SubMConv3d(in_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key+'_imp')
        
        _step = int(kernel_size//2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda()
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()

    def _gen_sparse_features(self, x, imp3_3d, voxels_3d, gt_boxes=None):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                imps_3d: [N, kernelsize**3], the predicted importance values
                voxels_3d: [N, 3], the 3d positions of voxel centers
                gt_boxes: for focal loss calculation
        """
        index = x.indices[:, 0]
        batch_size = x.batch_size
        voxel_features_fore = []
        voxel_indices_fore = []
        voxel_features_back = []
        voxel_indices_back = []

        box_of_pts_cls_targets = []
        mask_voxels = []

        loss_box_of_pts = 0
        for b in range(batch_size):

            if self.training and not self.skip_loss:
                gt_boxes_batch = gt_boxes[b, :, :-1]
                gt_boxes_batch_idx = (gt_boxes_batch**2).sum(-1)>0
                gt_boxes_centers_batch = gt_boxes_batch[gt_boxes_batch_idx, :3]
                gt_boxes_sizes_batch = gt_boxes_batch[gt_boxes_batch_idx, 3:6]

                index = x.indices[:, 0]
                batch_index = index==b
                mask_voxel = imp3_3d[batch_index, -1].sigmoid()
                mask_voxels.append(mask_voxel)
                voxels_3d_batch = voxels_3d[batch_index]
                dist_voxels_to_gtboxes = (voxels_3d_batch[:, self.inv_idx].unsqueeze(1).repeat(1, gt_boxes_centers_batch.shape[0], 1) - gt_boxes_centers_batch.unsqueeze(0)).abs()
                offsets_dist_boundry = dist_voxels_to_gtboxes - gt_boxes_sizes_batch.unsqueeze(0)
                inboxes_voxels = ~torch.all(~torch.all(offsets_dist_boundry<=0, dim=-1), dim=-1)
                box_of_pts_cls_targets.append(inboxes_voxels)

            features_fore, indices_fore, features_back, indices_back = split_voxels(x, b, imp3_3d, voxels_3d, self.kernel_offsets, mask_multi=self.mask_multi, topk=self.topk, threshold=self.threshold)

            voxel_features_fore.append(features_fore)
            voxel_indices_fore.append(indices_fore)
            voxel_features_back.append(features_back)
            voxel_indices_back.append(indices_back)

        voxel_features_fore = torch.cat(voxel_features_fore+voxel_features_back, dim=0)
        voxel_indices_fore = torch.cat(voxel_indices_fore+voxel_indices_back, dim=0)
        out = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        
        if self.training and not self.skip_loss:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts += self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())
        return out, loss_box_of_pts

    def forward(self, x, batch_dict, fuse_func=None):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        x_predict = self.conv_enlarge(x) if self.conv_enlarge else x
        if self.use_img:
            x_predict = fuse_func(batch_dict, encoded_voxel=x_predict, layer_name="layer1")
        imp3_3d = self.conv_imp(x_predict).features

        out, loss_box_of_pts = self._gen_sparse_features(x, imp3_3d, voxels_3d, batch_dict['gt_boxes'] if self.training else None)
        out = self.conv(out)

        if self.use_img:
            out = fuse_func(batch_dict, encoded_voxel=out, layer_name="layer1")

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        return out, loss_box_of_pts
