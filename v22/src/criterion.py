import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-12

class StackedHeatmapLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(StackedHeatmapLoss, self).__init__()
        
        self.alpha, self.beta = alpha, beta
        
    def forward(self, estimated_heatmap, target_heatmap, target_pointmap, num_object):
        """
        Args:
            estimated_heatmap (S, batch_size, 1, H//R, W//R):
            target_heatmap (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        alpha, beta = self.alpha, self.beta
        
        estimated_heatmap = estimated_heatmap.squeeze(dim=2)
        
        loss = torch.pow((1-target_heatmap), beta) * torch.pow(estimated_heatmap, alpha) * torch.log(1-estimated_heatmap+EPS) + target_pointmap * torch.pow((1-estimated_heatmap), alpha) * torch.log(estimated_heatmap+EPS)
        loss = - loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean()
        
        return loss
        
class StackedLocalOffsetLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(StackedLocalOffsetLoss, self).__init__()
        
    def forward(self, estimated_local_offset, target_local_offset, target_pointmap, num_object):
        """
        Args:
            estimated_local_offset (S, batch_size, 1, H//R, W//R):
            target_local_offset (batch_size, H, W):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        target_pointmap * (estimated_local_offset - target_local_offset)
        
        
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean()
        
        return loss
        
class StackedL1Loss(nn.Module):
    def __init__(self):
        super(StackedL1Loss, self).__init__()
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        Args:
            estimated (S, batch_size, 1, H//R, W//R):
            target (batch_size, H, W):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        estimated = estimated.squeeze(dim=2)
        loss = target_pointmap * (torch.abs(estimated - target))
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
        return loss
    
class StackedDepthLoss(nn.Module):
    def __init__(self):
        super(StackedDepthLoss, self).__init__()
        
    def forward(self, estimated_depth, target_depth, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated_depth (S, batch_size, 1, H//R, W//R):
            target_depth (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        estimated_depth = estimated_depth.squeeze(dim=2)
        
        loss = target_pointmap * torch.abs(estimated_depth - target_depth)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
        return loss
        
class StackedSinLoss(nn.Module):
    def __init__(self):
        super(StackedSinLoss, self).__init__()
        
    def forward(self, estimated_sin, target_sin, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated_sin (S, batch_size, 1, H//R, W//R):
            target_sin (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        estimated_sin = estimated_sin.squeeze(dim=2)
        
        loss = target_pointmap * torch.abs(estimated_sin - target_sin)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
        return loss
        
class StackedAngleLoss(nn.Module):
    def __init__(self, metrics='cos_sim'):
        super(StackedAngleLoss, self).__init__()
        
        self.metrics = metrics
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated (S, batch_size, 1, H//R, W//R):
            target (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """

        estimated = estimated.squeeze(dim=2)
        loss = target_pointmap * torch.abs(estimated - target) # loss: (S, batch_size, H//R, W//R)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
    
        loss = loss.mean(dim=0)
        
        return loss
        
class StackedYawLoss(nn.Module):
    def __init__(self):
        super(StackedYawLoss, self).__init__()
        
    def forward(self, estimated_sin, target_sin, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated_sin (S, batch_size, 1, H//R, W//R):
            target_sin (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        estimated_sin = estimated_sin.squeeze(dim=2)
        
        loss = target_pointmap * torch.abs(estimated_sin - target_sin)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
        return loss
        
class StackedCategoricalLoss(nn.Module):
    def __init__(self, num_category):
        super(StackedCategoricalLoss, self).__init__()
        
        self.num_category = num_category
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated (S, batch_size, num_category, H//R, W//R):
            target (batch_size, num_category, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        Q = torch.exp(estimated).sum(dim=2) # Q (S, batch_size, H//R, W//R)
        Q = torch.log(Q) # Q (S, batch_size, H//R, W//R)
        
        loss = - (target * estimated).sum(dim=2) + Q # loss (S, batch_size, H//R, W//R)
        loss = target_pointmap * loss # loss (S, batch_size, H//R, W//R)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1) # (batch_size, )
        loss = loss.mean(dim=0)
        
        return loss

class StackedPitchOffsetLoss(nn.Module):
    def __init__(self, num_category):
        super(StackedPitchOffsetLoss, self).__init__()
        
        self.num_category = num_category
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        Args:
            estimated (S, batch_size, num_category, H//R, W//R):
            target (batch_size, num_category, 2, H//R, W//R): For 3rd dimension, channel 0 is category and channel 1 is in-bin offset
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        num_category = self.num_category
        
        target_category = target[:, :, 0, :, :] # (batch_size, num_category, H//R, W//R)
        target_pitch_offset = (math.pi/num_category) * torch.tanh(target[:, :, 1, :, :]) # (batch_size, num_category, H//R, W//R)
        loss = torch.abs(estimated - target_pitch_offset) # (S, batch_size, num_category, H//R, W//R)
        loss = (target_category * loss).sum(dim=2) # (S, batch_size, H//R, W//R)
        loss = target_pointmap * loss # (S, batch_size, H//R, W//R)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1) # (batch_size, )
        loss = loss.mean(dim=0)
        
        return loss
        
class StackedCategoricalPitchLoss(nn.Module):
    def __init__(self, num_category, gamma=1.0):
        super(StackedCategoricalPitchLoss, self).__init__()
        
        self.num_category = num_category
        self.gamma = gamma
        self.categorical_loss = StackedCategoricalLoss(num_category=num_category)
        self.offset_loss = StackedPitchOffsetLoss(num_category=num_category)
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        Args:
            estimated (S, batch_size, num_category, 2, H//R, W//R):
            target (batch_size, num_category, 2, H//R, W//R): 0 is category, 1 is offset
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        categorical_loss = self.categorical_loss(estimated[:, :, :, 0, :, :], target[:, :, 0, :, :], target_pointmap, num_object)
        offset_loss = self.offset_loss(estimated[:, :, :, 1, :, :], target, target_pointmap, num_object)
        
        loss = self.gamma * categorical_loss + (1-self.gamma) * offset_loss
        
        return loss
         
class StackedRollLoss(nn.Module):
    def __init__(self):
        super(StackedRollLoss, self).__init__()
        
    def forward(self, estimated_sin, target_sin, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated_sin (S, batch_size, 1, H//R, W//R):
            target_sin (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        estimated_sin = estimated_sin.squeeze(dim=2)
        
        loss = target_pointmap * torch.abs(estimated_sin - target_sin)
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
        return loss

class Stacked3dObjectDetectionLoss(nn.Module):
    def __init__(self, num_stacks):
        super(Stacked3dObjectDetectionLoss, self).__init__()
        
        self.num_stacks = num_stacks
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target):
        """
        Args:
            input (batch_size, S, C, H, W): estimated features
            target (batch_size, C, H, W): target features
        """
        target = target.unsqueeze(dim=1)
        
        return self.l1_loss(input, target)
        
    
