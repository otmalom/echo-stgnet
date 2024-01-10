import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseMapToNode(nn.Module):
    """Base MapToNode, only using feat4"""
    def __init__(self, in_channels, num_points):
        super(BaseMapToNode, self).__init__()
        in_channels = in_channels[-1]
        self.num_points = num_points
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv_to_node = nn.Sequential(
            nn.Conv2d(256, num_points*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_points*4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_out(x)
        x = self.conv_to_node(x)
        x = x.view(x.size(0), self.num_points, -1)

        return x
