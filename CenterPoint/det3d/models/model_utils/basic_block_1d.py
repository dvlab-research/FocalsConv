import torch.nn as nn


class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x
