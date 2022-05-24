import torch
import torch.nn as nn
import torch.nn.functional as F
from convnext.pml_network import *
from utils.layernorm2d import LayerNorm2d


class ConvNext(nn.Module):
    def __init__(self, **params):
        super().__init__()
        
        self.in_channels = params['in_channels']
        self.num_classes = params['num_classes']
        self.num_layers = params['num_layers']
        self.layer_dims = params['layer_dims']
        self.expansion = params['expansion']
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.layer_dims[0], kernel_size=4, stride=4, bias=True)
        self.ln = LayerNorm2d(self.layer_dims[0])

        self.layer1 = self._build_layers(self.layer_dims[0], self.layer_dims[1], self.expansion, self.num_layers[0])
        self.layer2 = self._build_layers(self.layer_dims[1], self.layer_dims[2], self.expansion, self.num_layers[1])
        self.layer3 = self._build_layers(self.layer_dims[2], self.layer_dims[3], self.expansion, self.num_layers[2])
        self.layer4 = self._build_layers(self.layer_dims[3], self.layer_dims[3], self.expansion, self.num_layers[3], downsample=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.layer_dims[3], self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        
        out = self.classifier(x)
        
        return out
        
    def _build_layers(self, in_dim, out_dim, expansion, num_layers, downsample=True):
        layers = []
        
        for i in range(num_layers - 1):
            layers.append(CNBlock(in_dim=in_dim, out_dim=out_dim, expansion=expansion, downsample=False))
        
        if downsample: # no downsampling on last layer
            layers.append(CNBlock(in_dim=in_dim, out_dim=out_dim, expansion=expansion, downsample=True))
        
        return nn.Sequential(*layers)
        