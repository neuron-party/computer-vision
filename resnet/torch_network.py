import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch bottleneck architecture applies stride in conv3x3 with stride=2 and padding=1
# other architectures apply stride in 1st conv1x1 with stride=2 and padding=0

# apply stride (image shape reduction) in first bottleneck block of each layer

# bottleneck architecture: 
    # conv1x1 dimension reduction
    # conv3x3 image shape reduction
    # conv1x1 dimension expansion

class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, expansion, stride, padding, downsample=False):
        super().__init__()
        self.expansion = expansion
        
        # layers
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Identity()
        if downsample:
            self.downsample = nn.Conv2d(in_channels=in_dim, out_channels=out_dim * self.expansion, kernel_size=1, stride=stride)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        identity = self.downsample(x) # identity mapping
        
        out += identity
        out = self.relu(out)
        
        return out