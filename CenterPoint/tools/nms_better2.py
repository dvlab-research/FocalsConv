import argparse
import copy
import json
import os
import sys

import numpy as np
import pickle
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box, RadarPointCloud
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.geometry_utils import points_in_box
from functools import reduce
from tqdm import tqdm
from det3d.core import box_torch_ops
from collections import defaultdict
import torch
from det3d.datasets.nuscenes.nusc_common import cls_attr_dist
import operator
from det3d.core.bbox.box_np_ops import limit_period

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument('--paths', help='delimited list input of prediction paths for ensemble, separate by ,', type=str)
    parser.add_argument("--data_root", type=str, default="data/nuScenes")

    args = parser.parse_args()

    args.paths = [str(item) for item in args.paths.split(',')]
    return args


def get_sample_data(pred):
    box_list = []
    score_list = []
    pred = pred.copy()

    for item in pred:
        box =  Box(item['translation'], item['size'], Quaternion(item['rotation']),
                   velocity=item['velocity']+[0], name=item['detection_name'])
        score_list.append(item['detection_score'])
        box_list.append(box)

    top_boxes = reorganize_boxes(box_list)
    top_scores = np.array(score_list).reshape(-1)

    return top_boxes, top_scores

def reorganize_boxes(box_lidar_nusc):
    rots = []
    centers = []
    wlhs = []
    vels = []
    for i, box_lidar in enumerate(box_lidar_nusc):
        v = np.dot(box_lidar.rotation_matrix, np.array([1, 0, 0]))
        rot = np.arctan2(v[1], v[0])

        rot = -rot- np.pi / 2
        rots.append(rot)
        centers.append(box_lidar.center)
        wlhs.append(box_lidar.wlh)
        vels.append(box_lidar.velocity)

    rots = np.asarray(rots)
    centers = np.asarray(centers)
    wlhs = np.asarray(wlhs)
    vels = np.asarray(vels)[:, :2]
    gt_boxes_lidar = np.concatenate([centers.reshape(-1,3), wlhs.reshape(-1,3), vels, rots[..., np.newaxis].reshape(-1,1) ], axis=1)

    return gt_boxes_lidar

def reorganize_pred_by_class(pred):
    ret_dicts = defaultdict(list)
    for item in pred:
        ret_dicts[item['detection_name']].append(item)

    return ret_dicts

def concatenate_list(lists):
    ret = []
    for l in lists:
        ret += l

    return ret

NAME_TO_THRESH = {
    'traffic_cone': 0.05,
    'bicycle': 0.15,
    'bus': 0.25,
    'barrier': 0.1,
    'car': 0.1,
    'construction_vehicle': 0.1,
    'motorcycle': 0.1,
    'pedestrian': 0.1,
    'trailer': 0.1,
    'truck': 0.1
}

"""WEIGHT = {
    0:{
       'car':0.8,
       'truck':0.8,
       'bus':0.2,
       'trailer':0.2,
       'construction_vehicle':0.6,
       'pedestrian':1,
       'motorcycle':1,
       'bicycle':1,
       'traffic_cone':1,
       'barrier':0.8
    },
    1:{
       'car':1,
       'truck':1,
       'bus':1,
       'trailer':0.8,
       'construction_vehicle':0.2,
       'pedestrian':0.8,
       'motorcycle':0.8,
       'bicycle':0.8,
       'traffic_cone':0.8,
       'barrier':1
    },
    2:{
       'car':1,
       'truck':0.4,
       'bus':0.6,
       'trailer':0.4,
       'construction_vehicle':0.8,
       'pedestrian':0.6,
       'motorcycle':0.6,
       'bicycle':0.6,
       'traffic_cone':0.6,
       'barrier':0.6
    },
    3:{
       'car':1,
       'truck':0.6,
       'bus':0.8,
       'trailer':0.6,
       'construction_vehicle':1,
       'pedestrian':0.4,
       'motorcycle':0.4,
       'bicycle':0.4,
       'traffic_cone':0.4,
       'barrier':0.4
    },
    4:{
       'car':0.6,
       'truck':0.2,
       'bus':0.4,
       'trailer':1,
       'construction_vehicle':0.4,
       'pedestrian':0.2,
       'motorcycle':0.2,
       'bicycle':0.2,
       'traffic_cone':0.2,
       'barrier':0.2 
    },
}
"""
WEIGHT = {
    0:{
        'car':0.9,
        'truck':0.9,
        'bus':0.6,
        'trailer':0.6,
        'construction_vehicle':0.8,
        'pedestrian':1,
        'motorcycle':1,
        'bicycle':1,
        'traffic_cone':1,
        'barrier':0.9
    },
    1:{
        'car':1,
        'truck':1,
        'bus':1,
        'trailer':0.9,
        'construction_vehicle':0.6,
        'pedestrian':0.9,
        'motorcycle':0.9,
        'bicycle':0.9,
        'traffic_cone':0.9,
        'barrier':1
    },
    2:{
        'car':1,
        'truck':0.7,
        'bus':0.8,
        'trailer':0.7,
        'construction_vehicle':0.9,
        'pedestrian':0.8,
        'motorcycle':0.8,
        'bicycle':0.8,
        'traffic_cone':0.8,
        'barrier':0.8
    },
    3:{
        'car':1,
        'truck':0.8,
        'bus':0.9,
        'trailer':0.8,
        'construction_vehicle':1,
        'pedestrian':0.7,
        'motorcycle':0.7,
        'bicycle':0.7,
        'traffic_cone':0.7,
        'barrier':0.7
    },
    4:{
        'car':0.8,
        'truck':0.6,
        'bus':0.7,
        'trailer':1,
        'construction_vehicle':0.7,
        'pedestrian':0.6,
        'motorcycle':0.6,
        'bicycle':0.6,
        'traffic_cone':0.6,
        'barrier':0.6
    },

}

def main():
    args = parse_args()
    # read init files
    preds = []
    for idx, path in enumerate(args.paths):
        with open(path, 'rb') as f:
            pred=json.load(f)['results']

        new_pred = {}
        for k, val in pred.items():
            new_annos = []
            for box in val:
                name = box['detection_name']
                box['detection_score'] *= 1.0 #WEIGHT[idx][name]

                new_annos.append(box)

            new_pred[k] = new_annos

        preds.append(new_pred)

    merged_predictions = {}
    for token in preds[0].keys():
        annos = [pred[token] for pred in preds]
        merged_predictions[token] = concatenate_list(annos)

    predictions = merged_predictions

    print("Finish Merging")

    mode = "val"

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    for sample_token, prediction in tqdm(predictions.items()):
        annos = []
        scores = []
        # reorganize pred by class
        pred_dicts = reorganize_pred_by_class(prediction)

        for name, pred in pred_dicts.items():
            # in global coordinate
            top_boxes, top_scores = get_sample_data(pred)

            with torch.no_grad():
                top_boxes_tensor = torch.from_numpy(top_boxes)
                boxes_for_nms = top_boxes_tensor[:, [0, 1, 3, 4, -1]]
                top_scores_tensor = torch.from_numpy(top_scores)

                # change this to a nms function that works with the coordinate.
                # I remembered there is some problem with the current pcdet nms
                # used in this codebase but I don't remember how I solve this..
                # probably try the nms in det3d repo
                selected = box_torch_ops.rotate_nms_pcdet(top_boxes_tensor.cuda().float(), top_scores_tensor, 
                                        thresh=NAME_TO_THRESH[name],
                                        pre_maxsize=None,
                                        ).numpy()

            pred = [pred[s] for s in selected]
            annos.extend(pred)
            scores.extend([top_scores[s] for s in selected])

        scores = np.asarray(scores)
        order = np.argsort(scores, axis=-1)[::-1][:500]

        annos = [annos[i] for i in order]

        assert len(annos) <= 500
        nusc_annos["results"].update({sample_token: annos})

    nusc_annos["meta"] = {
        "use_camera": True,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'result.json'), "w") as f:
        json.dump(nusc_annos, f)

    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=True)
    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=os.path.join(args.work_dir, 'result.json'),
        eval_set='val',
        output_dir=args.work_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0,)


if __name__ == "__main__":
    main()
