import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DyHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.scale_aware_attention = ScaleAwareAttention(num_layers=3) # L = 3
        self.spatial_aware_attention = SpatialAwareAttention()
        self.task_aware_attention = TaskAwareAttention(in_channels=256)
        
    def forward(self, x):
        x = self.scale_aware_attention(x)
        x = self.spatial_aware_attention(x)
        x = self.task_aware_attention(x)
        return x
    

class ScaleAwareAttention(nn.Module):
    '''
    AvgPool2d -> Conv 1x1 -> ReLU -> Hard Sigmoid
    Input: Tensor of shape [b, L, S, C]
    '''
    def __init__(self, num_layers):
        super().__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels=num_layers, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.hs = nn.Hardsigmoid()
        
    def forward(self, x):
        pi_L = self.avgpool(x) # [b, L, S, C]
        pi_L = self.conv(pi_L) # [b, 1, S, C]
        pi_L = self.hs(self.relu(pi_L)) # [b, 1, S, C]
        out = x * pi_L # [b, L, S, C]
        return out
    

class SpatialAwareAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        offset_weights = self.conv(x) # [b, 27, S, C]
        offset, weights = offset_weights[:, :18, :, :], offset_weights[:, 18:, :, :]
        weights = torch.sigmoid(weights)
        
        x = self.deform_conv(x, offset, weights)
        return x
    
    
class TaskAwareAttention(nn.Module):
    '''
    AdaptiveAvgPool2d -> FC -> ReLU -> FC -> ReLU -> Normalize -> Dynamic ReLU
    Input: Tensor of shape [b, L, S, C], transformed by scale and spatial awareness attention layers
    
    Transformations:
        AvgPool reduces the L & S dimensions to 1
        Linear1 projects 256 -> 64
        Linear2 projects 64 -> 1024
        Split across 4 channels [256, 256, 256, 256]
        Normalize between -1, 1
    '''
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        
        # Following parameters inferred from the paper/github 
            # in_channels/out_channels = 256
            # linear1 in_features = in_channels; out_features = in_channels // 4
            # linear2 in_features = in_channels // 4; out_features = in_channels * 4
            # reduction = 4
            
        self.in_channels = in_channels
            
        self.init_alpha = [1.0, 0.0]
        self.init_beta = [0.0, 0.0]
        self.lambda_alpha = 1.0
            
        hidden_dim, out_dim = in_channels // reduction, in_channels * 4
        self.linear1 = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.normalize = h_sigmoid() # taken from github directly
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = x
        # x: [b, L, S, C]
        x = x.permute(0, 3, 1, 2) # [b, C, L, S]
        x = self.avgpool(x) # [b, C, 1, 1]
        x = x.flatten(start_dim=1) # [b, C]
        x = self.linear2(self.relu(self.linear1(x))) # [b, C * 4]
        a1, a2, b1, b2 = torch.split(x, self.in_channels, dim=1) # [b, C], [b, C], [b, C], [b, C]
        
        a1 = (a1 - 0.5) * self.lambda_alpha + self.init_alpha[0]
        a2 = (a2 - 0.5) * self.lambda_alpha + self.init_alpha[1]
        b1 = b1 - 0.5 + self.init_beta[0]
        b2 = b2 - 0.5 + self.init_beta[1]
        
        out = torch.max(a1 * y + b1, a2 * y + b2)
        return out
    
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6