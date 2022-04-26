from det3d.models.backbones.resnet import resnet50, resnet101
from .cls_template import ClsTemplate


class ClsResNet(ClsTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes ResNet model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50]
        """
        if backbone_name == "ResNet50":
            constructor = resnet50
        elif backbone_name == "ResNet101":
            constructor = resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
