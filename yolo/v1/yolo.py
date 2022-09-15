import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from darknet import *


class YOLOv1(nn.Module):
    def __init__(self, grid_size, num_bboxes, num_classes):
        super().__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        
        self.darknet = Darknet()
        self.darknet.fc = nn.Identity() # remove fc layer
        
        # YOLO has 4 additional convolutional layers
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(1024)
        
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        '''
        Input: Tensor of shape [b, 3, 448, 448]
        Output: Tensor of shape [b, 7, 7, 30] 
            30 -> [2 * 5 + 20] : 2 bounding box predictions per cell with x, y, w, h, conf and 20 class probabilities for classification
        '''
        x = self.darknet(x)
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        x = self.fc(x)
        
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x