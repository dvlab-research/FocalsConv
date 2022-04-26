from collections import OrderedDict
from pathlib import Path
from torch import hub

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class ClsTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        
        return_layers = {_layer:_layer for _layer in feat_extract_layer}
        self.model.return_layers.update(return_layers)


    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        # Get model
        model = constructor(pretrained=False,
                            progress=False,
                            num_classes=self.num_classes)
        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            
            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                                                          pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model.cuda()

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        if "classifier.4.weight" in pretrained_dict and "classifier.4.weight" in model_dict:
            model_num_classes = model_dict["classifier.4.weight"].shape[0]
            pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
            if model_num_classes != pretrained_num_classes:
                pretrained_dict.pop("classifier.4.weight")
                pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """

        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        features = self.model(x)
        for _layer in self.feat_extract_layer:
            result[_layer] = features[_layer]

        result["logits"] = features["output"]
        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)
        return x.cuda()
