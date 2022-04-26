import torchvision

from .seg_template import SegTemplate


class SemDeepLabV3(SegTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes SemDeepLabV3 model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50/ResNet101]
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
