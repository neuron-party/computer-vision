import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
- pytorch bottleneck architecture applies stride in conv3x3 with stride=2 and padding=1
- other architectures apply stride in 1st conv1x1 with stride=2 and padding = 0

- apply downsampling in first bottleneck block of each layer
    - expand image dimension 
    - downsize image shape

- pytorch bottleneck design:
    - conv1x1: dimension reduction
    - conv3x3: image downsizing
    - conv1x1: dimension restoration
     
- other bottleneck designs:
    - conv1x1: dimension reduction and image downsizing
    - conv3x3: ...
    - conv1x1: dimension restoration
'''


# Bottleneck - dimension expansion, compression, expansion (bottleneck structure), conv1x1 -> conv3x3 -> conv1x1
class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, expansion, stride, padding, downsample=False):
        super().__init__()
        self.expansion = expansion
        self.downsample = downsample
        
        # layers
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=padding, bias=False) # image downsizing
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * expansion)
            )
        
    def forward(self, x):
        identity = x # identity mapping
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample: # image downsample
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    
# Block (Basic Block): no expansion factor, conv3x3 and conv3x3
class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, downsample=False):
        super().__init__()
        self.downsample = downsample
        
        # layers
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=stride, bias=False) # image downsizing
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=stride, bias=False), # image downsizing
                nn.BatchNorm2d(out_dim)
            )
            
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
# to do:
# 1. stochastic depth
# 2. zero init
# 3. other upgrades when applicable

