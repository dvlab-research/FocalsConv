import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NETWORK2D
from ..model_utils.basic_block_2d import BasicBlock2D
from . import ifn

@NETWORK2D.register_module
class PyramidFeat2D(nn.Module):

    def __init__(self, optimize, model_cfg):
        """
        Initialize 2D feature network via pretrained model
        Args:
            model_cfg: EasyDict, Dense classification network config
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.is_optimize = optimize

        # Create modules
        self.ifn = getattr(ifn, model_cfg['name'])(
            num_classes=model_cfg.num_class,
            backbone_name=model_cfg.backbone,
            **model_cfg.args
        )
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.channel_reduce["in_channels"]):
            _channel_out = model_cfg.channel_reduce["out_channels"][_idx]
            self.out_channels[model_cfg.args['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.channel_reduce["kernel_size"][_idx],
                         "stride": model_cfg.channel_reduce["stride"][_idx],
                         "bias": model_cfg.channel_reduce["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

    def get_output_feature_dim(self):
        return self.out_channels

    def forward(self, images):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            images: (N, H_in, W_in, 3), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        batch_dict = {}
        images = images.permute(0, 3, 1, 2).contiguous()
        ifn_result = self.ifn(images)

        for _idx, _layer in enumerate(self.model_cfg.args['feat_extract_layer']):
            image_features = ifn_result[_layer]
            # Channel reduce
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)

            batch_dict[_layer+"_feat2d"] = image_features
        
        if self.training:
            # detach feature from graph if not optimize
            if "logits" in ifn_result:
                ifn_result["logits"].detach_()
            if not self.is_optimize:
                image_features.detach_()

        return batch_dict

    def get_loss(self):
        """
        Gets loss
        Args:
        Returns:
            loss: (1), Network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        return None, None
