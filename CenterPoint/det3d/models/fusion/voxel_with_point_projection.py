import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.models.registry import FUSION
from det3d.models.model_utils.basic_block_1d import BasicBlock1D
from .point_to_image_projection import Point2ImageProjection

@FUSION.register_module
class VoxelWithPointProjection(nn.Module):
    def __init__(self, fuse_mode, interpolate, voxel_size, pc_range, image_list, image_scale=1, depth_thres=0, double_flip=False, layer_channel=None):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_projector = Point2ImageProjection(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)

    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if self.fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat += fuse_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0).contiguous()) / 2
        elif self.fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0).contiguous()], dim=0)
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0).contiguous()
        else:
            raise NotImplementedError
        
        return voxel_feat

    def forward(self, batch_dict, encoded_voxel=None, layer_name=None, img_conv_func=None):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
            encoded_voxel: (N, C), Sparse Voxel featuress
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
            voxel_features: (N, C), Sparse Image voxel features
        """
        for cam_key in self.image_list:
            cam_key = cam_key.lower()
            # Generate sampling grid for frustum volume
            projection_dict = self.point_projector(voxel_coords=encoded_voxel.indices.float(),
                                                   image_scale=self.image_scale,
                                                   batch_dict=batch_dict, 
                                                   cam_key=cam_key)

            # check 
            if encoded_voxel is not None:
                in_bcakbone = True
            else:
                in_bcakbone = False
                encoded_voxel = batch_dict['encoded_spconv_tensor']
            batch_size = len(batch_dict['image_shape'][cam_key])
            if not self.training and self.double_flip:
                batch_size = batch_size * 4
            for _idx in range(batch_size): #(len(batch_dict['image_shape'][cam_key])):
                _idx_key = _idx//4 if self.double_flip else _idx
                image_feat = batch_dict['img_feat'][layer_name+'_feat2d'][cam_key][_idx_key]
                if img_conv_func:
                    image_feat = img_conv_func(image_feat.unsqueeze(0))[0]
                raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
                feat_shape = image_feat.shape[-2:]
                if self.image_interp:
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2], mode='bilinear')[0]
                index_mask = encoded_voxel.indices[:,0]==_idx
                voxel_feat = encoded_voxel.features[index_mask]
                image_grid = projection_dict['image_grid'][_idx]
                voxel_grid = projection_dict['batch_voxel'][_idx]
                point_mask = projection_dict['point_mask'][_idx]
                image_depth = projection_dict['image_depths'][_idx]
                # temporary use for validation
                # point_mask[len(voxel_feat):] -> 0 for batch construction
                voxel_mask = point_mask[:len(voxel_feat)]
                if self.training and 'overlap_mask' in batch_dict.keys():
                    overlap_mask = batch_dict['overlap_mask'][_idx]
                    is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                    if 'depth_mask' in batch_dict.keys():
                        depth_mask = batch_dict['depth_mask'][_idx]
                        depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                        is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                        is_overlap = is_overlap & (~is_inrange)

                    image_grid = image_grid[~is_overlap]
                    voxel_grid = voxel_grid[~is_overlap]
                    point_mask = point_mask[~is_overlap]
                    voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
                if not self.image_interp:
                    image_grid = image_grid.float()
                    image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                    image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                    image_grid = image_grid.long()

                voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask], 
                                                    image_grid[point_mask], layer_name)

                encoded_voxel.features[index_mask] = voxel_feat

        return encoded_voxel
