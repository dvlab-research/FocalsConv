import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')

from ..utils import transform_utils


class Point2ImageProjection(nn.Module):

    def __init__(self, voxel_size, pc_range, depth_thres={}, double_flip=False, device="cuda"):
        """
        Initializes Grid Generator for frustum features
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.dtype = torch.float32
        self.voxel_size = torch.as_tensor(voxel_size, dtype=self.dtype)
        self.pc_range = pc_range
        self.depth_thres = depth_thres
        self.device = device
        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.double_flip = double_flip
        # self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Add offsets to center of voxel
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype,
                                 device=self.device)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_coords, batch_dict, cam_key):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            batch_dict
                voxel_coords: (B, N, 3), Voxel sampling grid
                grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 3), Camera intrinsic matrix
        Returns:
            image_grid: (B, N, 2), Image sampling grid
            image_depths: (B, N), Depth sampling grid
        """
        calib_key = cam_key.lstrip('cam_')
        lidar_to_cam=batch_dict['calib']['lidar2cam_'+calib_key]
        cam_to_img=batch_dict['calib']['cam_intrinsic_'+calib_key]
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = self.grid_to_lidar.to(lidar_to_cam.device)  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam.float()  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 3)

        # Transform to LiDAR
        voxel_coords = voxel_coords[:,[0,3,2,1]] # B,Z,Y,X -> B,X,Y,Z
        if not self.training and self.double_flip:
            batch_idx = voxel_coords[:, 0]
            B = B * 4
            for idx in range(B):
                if idx%4==1:
                    voxel_coords[batch_idx==idx, 2] = (self.pc_range[3] - self.pc_range[0])/self.voxel_size[0] - voxel_coords[batch_idx==idx, 2] - 1
                if idx%4==2:
                    voxel_coords[batch_idx==idx, 1] = (self.pc_range[4] - self.pc_range[1])/self.voxel_size[1] - voxel_coords[batch_idx==idx, 1] - 1
                if idx%4==3:
                    voxel_coords[batch_idx==idx, 1] = (self.pc_range[3] - self.pc_range[0])/self.voxel_size[0] - voxel_coords[batch_idx==idx, 1] - 1
                    voxel_coords[batch_idx==idx, 2] = (self.pc_range[4] - self.pc_range[1])/self.voxel_size[1] - voxel_coords[batch_idx==idx, 2] - 1

        point_grid = transform_points(trans_01=V_G.unsqueeze(0), points_1=voxel_coords[:,1:].unsqueeze(0))
        point_grid = point_grid.squeeze()

        batch_idx = voxel_coords[:,0]
        point_count = batch_idx.unique(return_counts=True)
        batch_voxel = torch.zeros(B, max(point_count[1]), 3).to(lidar_to_cam.device)
        point_inv = torch.zeros(B, max(point_count[1]), 3).to(lidar_to_cam.device)
        batch_mask = torch.zeros(B, max(point_count[1])).to(lidar_to_cam.device)
        for _idx in range(B):
            # project points to the non-augment one
            '''
            if self.training and 'aug_matrix' in batch_dict.keys():
                aug_mat_inv = torch.inverse(batch_dict['aug_matrix'][_idx])
            else:
                aug_mat_inv = torch.eye(3).to(lidar_to_cam.device)
            point_inv[_idx,:point_count[1][_idx]] = torch.matmul(aug_mat_inv, point_grid[batch_idx==_idx].t()).t()       
            '''
            point_grid_batch = point_grid[batch_idx==_idx]
            if self.training and 'aug_matrix_inv' in batch_dict.keys():
                aug_matrix_inv = batch_dict['aug_matrix_inv'][_idx]
                for aug_type in ['translate', 'rescale', 'rotate', 'flip']:
                    if aug_type in aug_matrix_inv:
                        if aug_type == 'translate':
                            point_grid_batch += torch.Tensor(aug_matrix_inv[aug_type]).to(lidar_to_cam.device)
                        else:
                            point_grid_batch = point_grid_batch @ torch.Tensor(aug_matrix_inv[aug_type]).to(lidar_to_cam.device)
            point_inv[_idx,:point_count[1][_idx]] = point_grid_batch
            batch_voxel[_idx,:point_count[1][_idx]] = voxel_coords[batch_idx==_idx][:,1:]
            batch_mask[_idx,:point_count[1][_idx]] = 1

        # Transform to camera frame
        if self.double_flip:
            C_V_new = []
            I_C_new = []
            for i in range(C_V.shape[0]):
                for j in range(4):
                    C_V_new.append(C_V[i].unsqueeze(0))
                    I_C_new.append(I_C[i].unsqueeze(0))
            C_V = torch.cat(C_V_new, dim=0)
            I_C = torch.cat(I_C_new, dim=0)

        camera_grid = transform_points(trans_01=C_V, points_1=point_inv)
        image_depths = camera_grid[...,2].clone()
        # Project to image
        image_grid = transform_utils.camera_to_image(project=I_C, points=camera_grid)

        return image_grid.long(), image_depths, batch_voxel.long(), batch_mask


    def forward(self, voxel_coords, image_scale, batch_dict, cam_key):
        """
        Generates sampling grid for frustum features
        Args:
            voxel_coords: (N, 4), Voxel coordinates
            batch_dict:
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            projection_dict: 
                image_grid: (B, N, 2), Image coordinates in X,Y of image plane
                image_depths: (B, N), Image depth
                batch_voxel: (B, N, 3), Voxel coordinates in X,Y,Z of point plane
                point_mask: (B, N), Useful points indictor
        """
        image_grid, image_depths, batch_voxel, batch_mask = self.transform_grid(voxel_coords=voxel_coords, 
                                                                                batch_dict=batch_dict,
                                                                                cam_key=cam_key)
        # Rescale Image grid
        image_grid = (image_scale * image_grid.float()).long()

        # Drop points out of range
        image_shape = batch_dict["image_shape"][cam_key].to(image_grid.device)
        if self.double_flip:
            image_shape_new = []
            for i in range(image_shape.shape[0]):
                for j in range(4):
                    image_shape_new.append(image_shape[i].unsqueeze(0))
            image_shape = torch.cat(image_shape_new, dim=0)

        point_mask = (image_grid[...,0]>0) & (image_grid[...,0]<image_shape[:,1].unsqueeze(-1)) & \
                     (image_grid[...,1]>0) & (image_grid[...,1]<image_shape[:,0].unsqueeze(-1)) & \
                     (image_depths > self.depth_thres[cam_key.upper()])
        point_mask = point_mask & batch_mask.bool()
        image_grid[~point_mask] = 0
        image_depths[~point_mask] = 0
        batch_voxel[~point_mask] = 0
        projection_dict = {}
        projection_dict['image_grid'] = image_grid
        projection_dict['image_depths'] = image_depths
        projection_dict['batch_voxel'] = batch_voxel
        projection_dict['point_mask'] = point_mask

        '''
        for image_idx in range(2):
            image_test = batch_dict["images"][cam_key][image_idx].cpu().numpy()
            image_test = (image_test * 255).astype(np.uint8())

            image_test = np.ascontiguousarray(image_test)
            for _point in image_grid[image_idx]:
                if _point[0] > 0 and _point[1] > 0:
                    circle_coord = tuple(_point.cpu().numpy())
                    cv2.circle(image_test, circle_coord, 3, (0,255,0), -1)

            cv2.imwrite('image_test_{}.png'.format(image_idx), image_test)
        from IPython import embed; embed()
        raise ValueError('Stop.')
        '''

        # debug use
        # import ipdb; ipdb.set_trace()

        # for image_idx in range(2):
        #     image_test = batch_dict["images"][cam_key][image_idx].cpu().numpy()
        #     image_test = (image_test * 255).astype(np.uint8())

        #     image_test = np.ascontiguousarray(image_test)
        #     for _point in image_grid[image_idx]:
        #         if _point[0] > 0 and _point[1] > 0:
        #             circle_coord = tuple(_point.cpu().numpy())
        #             cv2.circle(image_test, circle_coord, 3, (0,255,0), -1)

        #     # debug use
        #     import ipdb; ipdb.set_trace()

        #     cv2.imwrite('image_test_{}.png'.format(image_idx), image_test)

        return projection_dict
