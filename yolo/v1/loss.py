import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Loss(nn.Module):
    '''
    Loss function as described in https://arxiv.org/pdf/1506.02640.pdf
    Inputs: Tensors of size [b, S, S, B * 5 + C]
    Output: scalar loss
    
    More details regarding the loss function in the forward function
    '''
    def __init__(self):
        super().__init__()
        self.S = 7
        self.B = 2
        self.C = 20
        self.N = self.B * 5 + self.C
        
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        # adapted from motokimura's github, in my implementation the bounding boxes will always have size(0) == 1
        bbox1, bbox2 = bbox1.unsqueeze(0), bbox2.unsqueeze(0)
        
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou
        
    def forward(self, pred, target):
        '''
        Note: Loss in each cell is only computed for the bounding box predictor that is 'responsible', i.e has the highest IOU 
              The Loss function only penalizes classification error if an object is present in that grid cell

        λ_coord * Summation [0, S^2) Summation [0, B): [(x - x_hat)^2 + (y - y_hat)^2] 
            + λ_coord * Summation [0, S^2) Summation [0, B): [(sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2] 
                + Summation [0, S^2) Summation [0, B): [(C - C_hat)^2]
                    + λ_noobj * Summation [0, S^2) Summation [0, B): [(C - C_hat)^2]
                        + Summation [0, S^2) Summation [0, num_classes): [(p(c) - p(c_hat)^2]
        '''
        batch_size = pred.size(0)
        coord_mask = target[:, :, :, 4] > 0
        noobj_mask = target[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        coord_pred = pred[coord_mask].view(-1, self.N)
        coord_target = target[coord_mask].view(-1, self.N)
        
        coord_pred = coord_pred[:, :10].view(-1, self.B, 5)
        coord_target = coord_target[:, :10].view(-1, self.B, 5)
        
        # add some assertions i.e coord_pred.size(0) cannot be greater than batch_size * self.S ** 2
        noobj_pred = pred[noobj_mask].view(-1, self.N)
        noobj_target = target[noobj_mask].view(-1, self.N)
        
        noobj_pred = noobj_pred[:, :10].view(-1, self.B, 5)
        noobj_target = noobj_target[:, :10].view(-1, self.B, 5)

        coord_class_prob_pred = coord_pred[:, 10:]
        coord_class_prob_target = coord_target[:, 10:]

        # loss for lines 1 - 3
        bbox_loss = 0
        for i in range(min(self.S ** 2, coord_pred.size(0))): # only calculate bounding box loss for cells with objects
            info = torch.empty((2, 4)) # use to determine the predictor bounding box

            for j in range(self.B):
                # rescale normalized bounding boxes for the image
                coord_pred[i, j, :2] = coord_pred[i, j, :2] / float(self.S) - 0.5 * coord_pred[i, j, 2:4]
                coord_pred[i, j, 2:4] = coord_pred[i, j, :2] / float(self.S) + 0.5 * coord_pred[i, j, 2:4]

                coord_target[i, j, :2] = coord_target[i, j, :2] / float(self.S) - 0.5 * coord_target[i, j, 2:4]
                coord_target[i, j, 2:4] = coord_target[i, j, :2] / float(self.S) + 0.5 * coord_target[i, j, 2:4]

                l1 = self.lambda_coord * (coord_target[i, j, 0] - coord_pred[i, j, 0]) ** 2 + (coord_target[i, j, 1] - coord_pred[i, j, 1]) ** 2
                l2 = self.lambda_coord * (coord_target[i, j, 2] ** 0.5 - coord_pred[i, j, 2] ** 0.5) ** 2 + (coord_target[i, j, 3] ** 0.5 - coord_pred[i, j, 3] ** 0.5) ** 2
                l3 = (coord_target[i, j, 4] - coord_pred[i, j, 4]) ** 2

                iou = self.compute_iou(coord_pred[i, j, :4], coord_target[i, j, :4])
                
                info[j, 0], info[j, 1], info[j, 2], info[j, 3] = l1, l2, l3, iou

            bbox_loss += sum(info[info[:, 3].argmax()])

        # loss for line 4
        noobj_loss = self.lambda_noobj * F.mse_loss(noobj_pred[:, :, 4], noobj_target[:, :, 4], reduction='sum')
        
        # loss for line 5
        class_prob_loss = F.mse_loss(coord_class_prob_pred, coord_class_prob_target, reduction='sum')
        
        total_loss = bbox_loss + noobj_loss + class_prob_loss
        return total_loss