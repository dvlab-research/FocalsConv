import numpy as np
import os.path as osp
from skimage import io
import cv2

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from det3d.datasets.nuscenes.nusc_common import get_lidar2cam_matrix, view_points
from ..registry import PIPELINES
from .loading import get_image
import copy, random

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        # self.no_augmentation = cfg.get('no_augmentation', False)
        self.augmentation = cfg.get('augmentation', ['db_sample', 'flip', 'rotate', 'rescale', 'translate'])
        self.with_info = cfg.get('with_info', False) # load nuScenes information
        self.sample_method = cfg.get('sample_method', 'by_depth')
        self.keep_raw = cfg.get('keep_raw', True)
        if len(self.augmentation) == 0:
            print("NO AUGMENTATION.....")

    def _warp_images(self, res, points_before_aug, points_after_aug):
        cam_images = res['cam']
        image_shape = res['image_shape']
        ransacReprojThreshold = 10
        sample_points = 1000

        for _key in cam_images:
            lidar2cam, cam_intrinsic = res['calib']['lidar2cam_%s'%_key.lstrip('cam_')], res['calib']['cam_intrinsic_%s'%_key.lstrip('cam_')]
            # get points_img before
            points_3d_before_aug = np.concatenate([points_before_aug[:, :3], np.ones((points_before_aug.shape[0], 1))], axis=-1)
            points_cam_before_aug = (points_3d_before_aug @ lidar2cam.T).T
            point_img_before_aug = view_points(points_cam_before_aug[:3, :], np.array(cam_intrinsic), normalize=True)
            point_img_before_aug = point_img_before_aug.transpose()[..., :2] * res['image_scale']

            # get points_img
            points_3d = np.concatenate([points_after_aug[:, :3], np.ones((points_after_aug.shape[0], 1))], axis=-1)
            points_cam = (points_3d @ lidar2cam.T).T
            point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
            point_img = point_img.transpose()[..., :2] * res['image_scale']

            #_mask_before_aug = (point_img_before_aug[:, 0] > 0) * (point_img_before_aug[:, 1] > 0) * (point_img_before_aug[:, 0] < image_shape[_key][1]) * (point_img_before_aug[:, 1] < image_shape[_key][0])
            _mask = (point_img[:, 0] > 0) * (point_img[:, 1] > 0) * (point_img[:, 0] < image_shape[_key][1]) * (point_img[:, 1] < image_shape[_key][0])
            point_img_key_before_aug = point_img_before_aug[_mask]
            point_img_key = point_img[_mask]
            indexes = random.sample(range(point_img_key_before_aug.shape[0]), sample_points)
            H, status = cv2.findHomography(point_img_key[indexes], point_img_key_before_aug[indexes], cv2.RANSAC, ransacReprojThreshold)
            cam_images[_key] = cv2.warpPerspective(cam_images[_key], H, (cam_images[_key].shape[1], cam_images[_key].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return cam_images

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            if not anno_dict is None:
                gt_dict = {
                    "gt_boxes": anno_dict["boxes"],
                    "gt_names": np.array(anno_dict["names"]).reshape(-1),
                }
            else:
                gt_dict = {
                    "gt_boxes": np.zeros((1, 9)),
                    "gt_names": np.array(['car']).reshape(-1),
                }

        if self.mode == "train" and len(self.augmentation) > 0:
            if self.with_info:
                data_info = res["data_info"]
                cam_images = res["cam"]
            else:
                data_info, cam_images = None, None

            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler and 'db_sample' in self.augmentation:
                # With image info for sample
                if self.with_info:
                    # Transform points
                    sample_coords = box_np_ops.rbbox3d_to_corners(gt_dict["gt_boxes"])
                    sample_record = data_info.get('sample', res['metadata']['token'])
                    pointsensor_token = sample_record['data']['LIDAR_TOP']
                    crop_img_list = [[] for _ in range(len(sample_coords))]
                    # Crop images from raw images
                    for _key in cam_images:
                        cam_key = _key.upper()
                        camera_token = sample_record['data'][cam_key]
                        cam = data_info.get('sample_data', camera_token)
                        lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                        points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                        points_cam = (points_3d @ lidar2cam.T).T
                        # Filter useless boxes according to depth
                        cam_mask = (points_cam[2] > 0).all(axis=0)
                        cam_count = cam_mask.nonzero()[0]
                        if cam_mask.sum() == 0:
                            continue
                        points_cam = points_cam[...,cam_mask].reshape(4, -1)
                        point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                        point_img = point_img.reshape(3, 8, -1)
                        point_img = point_img.transpose()[...,:2]
                        minxy = np.min(point_img, axis=-2)
                        maxxy = np.max(point_img, axis=-2)
                        bbox = np.concatenate([minxy, maxxy], axis=-1)
                        bbox = (bbox * res['image_scale']).astype(np.int32)
                        bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=res['image_shape'][_key][1]-1)
                        bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=res['image_shape'][_key][0]-1)
                        cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                        if cam_mask.sum() == 0:
                            continue
                        cam_count = cam_count[cam_mask]
                        bbox = bbox[cam_mask]

                        for _idx, _box in enumerate(bbox):
                            crop_img_list[_idx] = cam_images[_key][_box[1]:_box[3],_box[0]:_box[2]]


                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None,
                    gt_token=res['metadata']['token'],
                    data_info=data_info,
                    cam_images=cam_images,
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_points_idx = sampled_dict["points_idx"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    raw_gt_box_num = len(gt_dict["gt_boxes"])
                    raw_point_idx = -1 * np.ones(len(points), dtype=np.int64)
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    # points = np.concatenate([sampled_points, points], axis=0)
                    points = np.concatenate([points, sampled_points], axis=0)
                    points_idx = np.concatenate([raw_point_idx, sampled_points_idx], axis=0)

                    if self.with_info:
                        # Transform points
                        sample_coords = box_np_ops.rbbox3d_to_corners(gt_dict["gt_boxes"])
                        sample_crops = crop_img_list + sampled_dict['img_crops']
                        if not self.keep_raw:
                            points_coords = points[:,:4].copy()
                            points_coords[:,-1] = 1
                            point_keep_mask = np.ones(len(points_coords)).astype(np.bool)
                        # Paste image according to sorted strategy
                        for _key in cam_images:
                            cam_key = _key.upper()
                            camera_token = sample_record['data'][cam_key]
                            cam = data_info.get('sample_data', camera_token)
                            lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                            points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                            points_cam = (points_3d @ lidar2cam.T).T
                            depth = points_cam[2]
                            cam_mask = (depth > 0).all(axis=0)
                            cam_count = cam_mask.nonzero()[0]
                            if cam_mask.sum() == 0:
                                continue
                            depth = depth.mean(0)[cam_mask]
                            points_cam = points_cam[...,cam_mask].reshape(4, -1)
                            point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                            point_img = point_img.reshape(3, 8, -1)
                            point_img = point_img.transpose()[...,:2]
                            minxy = np.min(point_img, axis=-2)
                            maxxy = np.max(point_img, axis=-2)
                            bbox = np.concatenate([minxy, maxxy], axis=-1)
                            bbox = (bbox * res['image_scale']).astype(np.int32)
                            bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=res['image_shape'][_key][1]-1)
                            bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=res['image_shape'][_key][0]-1)
                            cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                            if cam_mask.sum() == 0:
                                continue
                            depth = depth[cam_mask]
                            if 'depth' in self.sample_method:
                                paste_order = depth.argsort()
                                paste_order = paste_order[::-1]
                            else:
                                paste_order = np.arange(len(depth), dtype=np.int64)
                            cam_count = cam_count[cam_mask][paste_order]
                            bbox = bbox[cam_mask][paste_order]

                            paste_mask = -255 * np.ones(res['image_shape'][_key][:2], dtype=np.int64)
                            fg_mask = np.zeros(res['image_shape'][_key][:2], dtype=np.int64)
                            for _count, _box in zip(cam_count, bbox):
                                img_crop = sample_crops[_count]
                                if len(img_crop) == 0: continue
                                img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                                cam_images[_key][_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                                paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
                                # foreground area of original point cloud in image plane
                                if _count < raw_gt_box_num:
                                    fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                            # calculate modify mask
                            if not self.keep_raw:
                                points_cam = (points_coords @ lidar2cam.T).T
                                cam_mask = points_cam[2] > 0
                                if cam_mask.sum() == 0:
                                    continue
                                point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                                point_img = point_img.transpose()[...,:2]
                                point_img = (point_img * res['image_scale']).astype(np.int64)
                                cam_mask = (point_img[:,0] > 0) & (point_img[:,0] < res['image_shape'][_key][1]) & \
                                           (point_img[:,1] > 0) & (point_img[:,1] < res['image_shape'][_key][0]) & cam_mask
                                point_img = point_img[cam_mask]
                                new_mask = paste_mask[point_img[:,1], point_img[:,0]]==(points_idx[cam_mask]+raw_gt_box_num)
                                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_gt_box_num)
                                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                                raw_mask = raw_fg[point_img[:,1], point_img[:,0]] | raw_bg[point_img[:,1], point_img[:,0]]
                                keep_mask = new_mask | raw_mask
                                point_keep_mask[cam_mask] = point_keep_mask[cam_mask] & keep_mask

                        # Replace the original images
                        res['cam'] = cam_images
                        # remove overlaped LIDAR points
                        if not self.keep_raw:
                            points = points[point_keep_mask]


            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            res['aug_matrix_inv'] = {}
            if 'flip' in self.augmentation:
                gt_dict["gt_boxes"], points, enable_xy = prep.random_flip_both(gt_dict["gt_boxes"], points, return_enable_xy=True)
                flip_mat_T_inv = np.array(
                    [[[1, -1][enable_xy[1]], 0, 0], [0, [1, -1][enable_xy[0]], 0], [0, 0, 1]],
                    dtype=points.dtype,
                )
                res['aug_matrix_inv']['flip'] = flip_mat_T_inv

            if 'rotate' in self.augmentation:
                gt_dict["gt_boxes"], points, noise_rotation = prep.global_rotation(
                    gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise, return_rot_noise=True
                )
                rot_sin = np.sin(noise_rotation*-1)
                rot_cos = np.cos(noise_rotation*-1)
                rot_mat_T_inv = np.array(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                    dtype=points.dtype,
                )
                res['aug_matrix_inv']['rotate'] = rot_mat_T_inv

            if 'rescale' in self.augmentation:
                '''
                points_before_aug = copy.deepcopy(points)
                gt_dict["gt_boxes"], points, scale_noise = prep.global_scaling_v2(
                    gt_dict["gt_boxes"], points, *self.global_scaling_noise, return_scale_noise=True
                )
                points_after_aug = copy.deepcopy(points)
                res['cam'] = self._warp_images(res, points_before_aug, points_after_aug)
                '''

                gt_dict["gt_boxes"], points, scale_noise = prep.global_scaling_v2(
                    gt_dict["gt_boxes"], points, *self.global_scaling_noise, return_scale_noise=True
                )
                scale_mat_T_inv = np.array(
                    [[1/scale_noise, 0, 0], [0, 1/scale_noise, 0], [0, 0, 1/scale_noise]],
                    dtype=points.dtype,
                )
                res['aug_matrix_inv']['rescale'] = scale_mat_T_inv

            if 'translate' in self.augmentation:
                gt_dict["gt_boxes"], points, translate_mat_T = prep.global_translate_(
                    gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std, return_noise_T=True
                )
                translate_mat_T_inv = translate_mat_T * -1
                res['aug_matrix_inv']['translate'] = translate_mat_T_inv

        elif self.mode == "train" and len(self.augmentation) == 0:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip #and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"] 
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info



# debug use
# import ipdb; ipdb.set_trace()
# debug use
# image_test = (cam_images[_key] * 255).astype(np.uint8)[...,[2,1,0]]
# image_test = np.ascontiguousarray(image_test)
# cv2.rectangle(image_test, tuple(_box[:2]), tuple(_box[2:]), (0,0,255), 2)
# points_3d = points[:,:4].copy()
# points_3d[:,-1] = 1
# points_cam = lidar2cam @ points_3d.T
# depth = points_cam[2,:]
# point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
# point_img = point_img.transpose()[:,:2]
# point_img = (point_img * res['image_scale']).astype(np.int)
# _mask = (depth > 0) & (point_img[:,0] > 0) & (point_img[:,0] < res['image_shape'][_key][1]-1) & \
#         (point_img[:,1] > 0) & (point_img[:,1] < res['image_shape'][_key][0]-1)

# point_img = point_img[_mask]
# for _point in point_img:
#     circle_coord = tuple(_point)
#     cv2.circle(image_test, circle_coord, 3, (0,255,0), -1)

# cv2.imwrite('image_test.png', image_test)
