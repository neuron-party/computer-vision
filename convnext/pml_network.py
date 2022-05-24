import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layernorm2d import LayerNorm2d

# pml models currently do not have 1-1 loading with pretrained weights 
# most convnext models do not follow the original paper architecture exactly 
# conv1x1 layers are replaced with linear layers + input permutations 


class CNBlock(nn.Module): # 96, 192, 384, 768 / 3 3 9 3
    '''
    Inverted bottlenecks
    Applies downsampling after residual connections -> image downsizing occurs after each CN block
        In ResNet, downsampling is applied to the identity mapping in the first bottleneck layer
        prior to the residual addition operation
    '''
    def __init__(self, in_dim, out_dim, expansion, downsample=False):
        super().__init__()
        
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=7, stride=1, padding=3, groups=in_dim, bias=True)
        self.ln = LayerNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim * expansion, kernel_size=1, stride=1, bias=True)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(in_channels=in_dim * expansion, out_channels=in_dim, kernel_size=1, stride=1, bias=True)
        # todo: stochastic depth
        
        # image downsizing and depth expansion
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2, bias=True)
        )
        
    def forward(self, x):
        identity = x
        
        out = self.ln(self.conv1(x))
        out = self.gelu(self.conv2(out))
        out = self.conv3(out)
        
        out += identity
        
        if self.downsample:
            out = self.downsample(out)
        
        return out