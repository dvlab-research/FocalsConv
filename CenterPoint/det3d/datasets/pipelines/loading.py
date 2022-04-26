import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os
from skimage import io
import cv2
from ..registry import PIPELINES
from ..nuscenes.nusc_common import get_lidar2cam_matrix

try:
    from nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.data_classes import LidarPointCloud
except:
    print("nuScenes devkit not Found!")

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points

def get_image(path):
    """
    Loads image for a sample. Copied from PCDet.
    Args:
        idx: int, Sample index
    Returns:
        image: (H, W, 3), RGB Image
    """
    assert osp.exists(path)
    image = io.imread(path)
    image = image.astype(np.float32)
    image /= 255.0
    return image

def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudImageFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.image_scale = kwargs.get("image_scale", 1)
        self.image_list = kwargs.get("image_list", [])
        self.with_info = kwargs.get("with_info", False)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

            # add extra data
            if self.with_info:
                data_info = res['data_info']
                sample_token = res['metadata']['token']
                sample_record = data_info.get('sample', sample_token)
                res['calib'], res['image_shape'] = {}, {}
                for cam_channel in self.image_list:
                    camera_token = sample_record['data'][cam_channel]
                    cam = data_info.get('sample_data', camera_token)
                    cam_img = get_image(osp.join(data_info.dataroot, cam['filename']))
                    if self.image_scale != 1:
                        new_shape = [int(cam_img.shape[1]*self.image_scale), int(cam_img.shape[0]*self.image_scale)]
                        cam_img = cv2.resize(cam_img, new_shape)
                    res['image_scale'] = self.image_scale
                    res['cam'][cam_channel.lower()] = cam_img
                    res['image_shape'][cam_channel.lower()] = cam_img.shape
                    pointsensor_token = sample_record['data']['LIDAR_TOP']
                    lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                    cam_key = 'lidar2cam'+cam_channel.lstrip('CAM').lower()
                    intri_key = 'cam_intrinsic'+cam_channel.lstrip('CAM').lower()
                    res['calib'][cam_key] = lidar2cam
                    res['calib'][intri_key] = cam_intrinsic
                    
                    # debug use
                    # import ipdb; ipdb.set_trace()
                    # import cv2

                    # image_test = res['cam'][cam_channel.lower()]
                    # image_test = (image_test * 255).astype(np.uint8())
                    # image_test = np.ascontiguousarray(image_test)

                    # point_img = points.copy()
                    # point_img[:,-1] = 1
                    # point_img = np.dot(lidar2cam, point_img.transpose())
                    # depth = point_img[2,:].copy()
                    # point_img = self.view_points(point_img[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
                    
                    # _mask = (point_img[0] > 1) & (point_img[1] > 1) & (point_img[0] < image_test.shape[1]-1) & (point_img[1] < image_test.shape[0]-1) & (depth > 1)
                    # point_img = point_img.transpose()[_mask,:2].astype(np.int)

                    # for _point in point_img:
                    #     if _point.sum() > 0:
                    #         circle_coord = tuple(_point)
                    #         cv2.circle(image_test, circle_coord, 3, (0,255,0), -1)

                    # import ipdb; ipdb.set_trace()

                    # cv2.imwrite('image_test.png', image_test)

        else:
            raise NotImplementedError

        return res, info


    def view_points(self, points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        # copied from https://github.com/nutonomy/nuscenes-devkit/
        # only for debug use
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
