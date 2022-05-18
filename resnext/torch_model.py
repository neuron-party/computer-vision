import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnext.torch_network import *


class ResNext(nn.Module):
    def __init__(self, **params):
        super().__init__()
        
        self.dim = params['dim']
        self.expansion = params['expansion']
        self.in_channels = params['in_channels']
        self.num_classes = params['num_classes']
        self.num_layers = params['num_layers']
        self.layer_dims = params['layer_dims']
        self.groups = params['groups']
        self.width_per_group = params['width_per_group']
        
        self.gct = params['group_conv_type'] # 3 different types of grouped convolutions in official research paper
        
        # static parameters from resnet paper
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # bottleneck layers
        self.layer1 = self._build_layer(num_layers=self.num_layers[0], out_dim=self.layer_dims[0], stride=1, padding=1, gct=self.gct) 
        self.layer2 = self._build_layer(num_layers=self.num_layers[1], out_dim=self.layer_dims[1], stride=2, padding=1, gct=self.gct)
        self.layer3 = self._build_layer(num_layers=self.num_layers[2], out_dim=self.layer_dims[2], stride=2, padding=1, gct=self.gct)
        self.layer4 = self._build_layer(num_layers=self.num_layers[3], out_dim=self.layer_dims[3], stride=2, padding=1, gct=self.gct)
        
        # pool/fc
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.dim, self.num_classes)
        
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
         
    def _build_layer(self, num_layers, out_dim, stride, padding, gct):
        layers = []
        
        if gct == 1: # AGGREGATION
            layers.append(
                Bottleneck1(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                            expansion=self.expansion, stride=stride, padding=padding, downsample=True)
            )

            self.dim = self.expansion * out_dim # dimension expansion
            for i in range(1, num_layers):
                layers.append(
                    Bottleneck1(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                                expansion=self.expansion, stride=1, padding=1)
                )
                
        elif gct == 2: # BRANCHING + CONCATENATION
            layers.append(
                Bottleneck2(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                            expansion=self.expansion, stride=stride, padding=padding, downsample=True)
            )

            self.dim = self.expansion * out_dim # dimension expansion
            for i in range(1, num_layers):
                layers.append(
                    Bottleneck2(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                                expansion=self.expansion, stride=1, padding=1)
                )
            
        else: # WIDER RESNET 
            layers.append(
                Bottleneck3(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                            expansion=self.expansion, stride=stride, padding=padding, downsample=True)
            )

            self.dim = self.expansion * out_dim # dimension expansion
            for i in range(1, num_layers):
                layers.append(
                    Bottleneck3(in_dim=self.dim, out_dim=out_dim, width_per_group=self.width_per_group, num_groups=self.groups, 
                                expansion=self.expansion, stride=1, padding=1)
                )
        
        return nn.Sequential(*layers)
        