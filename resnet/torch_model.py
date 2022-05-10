import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.torch_network import *

class ResNet(nn.Module):
    def __init__(self, **params):
        super().__init__()
        
        self.dim = params['dim']
        self.expansion = params['expansion']
        self.num_classes = params['num_classes']
        
        # parameters from resnet paper
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # bottleneck layers
        self.layer1 = self._build_layer(num_layers=3, out_dim=64, stride=1, padding=1) 
        self.layer2 = self._build_layer(num_layers=4, out_dim=128, stride=2, padding=1)
        self.layer3 = self._build_layer(num_layers=6, out_dim=256, stride=2, padding=1)
        self.layer4 = self._build_layer(num_layers=3, out_dim=512, stride=2, padding=1)
        
        # pool/fc
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.adapool(x)
        x = x.flatten(1)
        out = self.fc(x)
        
        return out
         
    def _build_layer(self, num_layers, out_dim, stride, padding):
        '''
        layer1: (no image reduction)
            in_dim: 64
            out_dim: 64
            stride: 1
            padding: 0
        
        layer2: 56 -> 28
            in_dim: 256
            out_dim: 128
            stride: 2
            padding: 1
            
        layer3: 28 -> 14
            in_dim: 516
            out_dim: 256
            stride; 2
            padding: 1
        
        layer4: 14 -> 7
            in_dim: 1028
            out_dim: 512
            stride: 2
            padding: 1
            
        AdaptiveAvgPool2d: 7 -> 1
        '''
        # apply stride (image shape reduction) in first bottleneck block of each layer
        layers = []
        layers.append(
            Bottleneck(in_dim=self.dim, out_dim=out_dim, expansion=self.expansion, stride=stride, padding=padding, downsample=True)
        )
        
        self.dim = self.expansion * out_dim # dimension expansion
        for i in range(1, num_layers):
            layers.append(
                Bottleneck(in_dim=self.dim, out_dim=out_dim, expansion=self.expansion, stride=1, padding=padding)
            )
            
        return nn.Sequential(*layers)