from ..registry import DETECTORS
from .. import builder
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelFocal(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        network2d=None,
        fusion=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        noise_rotation=None,
    ):
        super(VoxelFocal, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        if network2d is None or fusion is None:
            self.network2d = None
            self.fusion = None
        else:
            self.network2d = builder.build_network2d(network2d)
            self.fusion = builder.build_fusion(fusion)
        self.noise_rotation = noise_rotation
        if not self.noise_rotation is None:
            self.noise_rotation = noise_rotation * -1
        
    def extract_feat(self, data, batch_dict):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature, loss_box_of_pts = self.backbone(
            input_features, batch_dict, data["coors"], data["batch_size"], data["input_shape"],
            fuse_func=self.fusion
        )
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature, loss_box_of_pts

    def extract_feat2d(self, data):
        img_feature = {}
        for single_view in data.keys():
            single_result = self.network2d(data[single_view])
            for layer in single_result.keys():
                if layer not in img_feature:
                    img_feature[layer] = {}
                img_feature[layer][single_view] = single_result[layer]

        return img_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        batch_dict = {}
        if self.network2d is not None and self.fusion is not None:
            batch_dict["images"] = example['cam']
            batch_dict['image_shape'] = example["image_shape"]
            batch_dict['calib'] = example['calib']
            batch_dict['img_feat'] = self.extract_feat2d(example['cam'])
            if 'aug_matrix_inv' in example:
                batch_dict['aug_matrix_inv'] = example['aug_matrix_inv']

        if self.training:
            batch_dict['gt_boxes'] = example['gt_boxes_and_cls'][:, :, :7]

        x, _, loss_box_of_pts = self.extract_feat(data, batch_dict)
        preds = self.bbox_head(x)

        if return_loss:
            loss_dict = self.bbox_head.loss(example, preds)
            loss_dict['loss'] = [_loss + loss_box_of_pts/len(loss_dict['loss']) for _loss in loss_dict['loss']]
            return loss_dict
            #return self.bbox_head.loss(example, preds)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            if not self.noise_rotation is None:
                import numpy as np
                from det3d.core.bbox.box_np_ops import rotation_points_single_angle
                box3d_lidar = [boxes_batch['box3d_lidar'] for boxes_batch in boxes]
                for i, boxes_batch in enumerate(box3d_lidar):
                    boxes_batch[:, :3] = torch.Tensor(rotation_points_single_angle(
                        boxes_batch[:, :3].cpu().numpy(), self.noise_rotation, axis=2
                    )).to(boxes_batch.device)
                    if boxes_batch.shape[1] > 7:
                        boxes_batch[:, 6:8] = torch.Tensor(rotation_points_single_angle(
                            np.hstack([boxes_batch[:, 6:8].cpu().numpy(), np.zeros((boxes_batch.shape[0], 1))]),
                            self.noise_rotation,
                            axis=2,
                        ))[:, :2].to(boxes_batch.device)
                        boxes_batch[:, -1] += self.noise_rotation
                    boxes[i]['box3d_lidar'] = boxes_batch
            return boxes #self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, voxel_feature = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None 
