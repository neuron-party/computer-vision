'''Pretty slow at writing these... each network has its own unique detector :/'''
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from darknet import *
from yolo import *


class YOLOv1_Detector:
    def __init__(self, grid_size, num_bboxes, num_classes, model, prob_threshold=0.5):
        
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.prob_threshold = prob_threshold
        self.cell_size = 1.0 / self.S
        
        self.yolo = model
    
    def detect(self, img, img_size=448):
        '''
        Inputs: 
            RGB/BGR image (depending on if you read the image with OpenCV or some other library) of shape [448, 448, 3]
        Outputs:
            Dict: {'bounding_boxes':, 'class':, 'confidence'}
        '''
        # do some preprocessing here (code later)
        pred = self.model(img)
        
    def decode(self, x: torch.Tensor):
        '''
        Input: Tensor of shape [1, 7, 7, 30]
        Output: List: [bounding_boxes, class, confs]
        '''
        bounding_boxes, classes, confs = [], [], []
        
        # probably a more optimized way to do this with tensor operations
        for i in range(self.S):
            for j in range(self.S):
                class_prob, class_label = torch.max(x[:, :, 10:], 0)
                
                for k in range(self.B):
                    # do not calculate and return bounding boxes for low confidence predictions
                    conf = x[i, j, (k*5) + 4]
                    prob_threshold = conf * class_prob
                    if prob_threshold < self.prob_threshold:
                        continue
                    
                    # "we parameterize the x, y coordinates to be offsets of a particular grid cell location so they are bounded between 0 and 1"
                    bbox = x[i, j, (k*5):(k*5 + 4)] 
                    x0_y0_norm = torch.FloatTensor([i, j]) * self.cell_size # normalized offset for the top left corner
                    xy = bbox[:2] * self.cell_size + x0_y0_norm # normalized center of the bbox
                    wh = bbox[2:] 
                    
                    bounding_box = torch.cat([ # [x1, y1, x2, y2] - top left corner & bottom right corner
                        xy - 0.5 * wh,
                        xy + 0.5 * wh
                    ])
                    
                    bounding_boxes.append(bounding_box)
                    classes.append(class_label)
                    confs.append(conf)
                    
        
        bounding_boxes = torch.stack(bounding_boxes, 0) if len(bounding_boxes) > 1 else torch.Tensor([])
        classes = torch.Tensor(classes)
        confs = torch.Tensor(confs)
           
        return bounding_boxes, classes, confs # return empty tensors if there are no objects detected                  