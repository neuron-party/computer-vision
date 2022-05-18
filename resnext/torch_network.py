import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Different types of grouped convolution architectures in https://arxiv.org/pdf/1611.05431.pdf
1. Aggregated residual transformations
2. Block equivalent 
3. Wider and sparsely connected module
    [128, 256, 512, 1024]
'''


class Bottleneck1(nn.Module):
    def __init__(self, in_dim, out_dim, width_per_group, num_groups, expansion, stride, padding, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # 4
        
        self.width = int((out_dim * 2) / num_groups)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        
        self.branches = self._build_branches()
        
        if downsample: # downsample with 1x1 convolution, stride=stride, no padding
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * expansion)
            )
            
    def forward(self, x):
        identity = x
        aggregate = self.branches[0](x)
        for i in range(1, self.num_groups):
            aggregate += self.branches[i](x)
            
        if self.downsample:
            identity = self.downsample(x)
            
        out = aggregate + identity
        out = F.relu(out)
        return out
        
    def _build_branches(self):
        branches = []
        for i in range(self.num_groups):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_dim, out_channels=self.width, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=self.stride, padding=self.padding, bias=False), # downsizing in layer2+
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=self.width, out_channels=self.out_dim * self.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.out_dim * self.expansion)
                )
            )
            
        return branches
    

class Bottleneck2(nn.Module):
    def __init__(self, in_dim, out_dim, width_per_group, num_groups, expansion, stride, padding, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # 4
        
        self.width = int((out_dim * 2) / num_groups)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        
        cat_dim = int(out_dim * (width_per_group / 64.0) * num_groups)
        
        self.branches = self._build_branches()
        self.conv3 = nn.Conv2d(in_channels=cat_dim, out_channels=out_dim * expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * expansion)
        
        if downsample: # downsample with 1x1 convolution, stride=stride, no padding
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * expansion)
            )
            
    def forward(self, x):
        identity = x
        cat = []
        for branch in self.branches: # branching to depth 4 filters before catting and passing into final convolution layer
            cat.append(branch(x))
            
        cat = torch.cat(cat, axis=1) # [B, grouped_convolution_dim, h, w]
        out = self.bn3(self.conv3(cat))
            
        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        return out
        
    def _build_branches(self):
        branches = []
        for i in range(self.num_groups):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_dim, out_channels=self.width, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=self.stride, padding=self.padding, bias=False), # downsizing in layer2+
                    nn.BatchNorm2d(self.width)
                )
            )
            
        return branches
    
    
class Bottleneck3(nn.Module):
    def __init__(self, in_dim, out_dim, width_per_group, num_groups, expansion, stride, padding, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # 4
        
        width = int(out_dim * (width_per_group / 64.0) * num_groups) # [128, 256, 512, 1048]\
        bottleneck_width = int((out_dim * 2) / num_groups)
        
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=padding, bias=False) # downsizing
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_dim * expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * expansion)
        self.relu = nn.ReLU(inplace=True)
        
        if downsample: # downsample with 1x1 convolution, stride=stride, no padding
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * expansion)
            )
            
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out