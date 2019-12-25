import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-8

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
            num_object (batch_size, ):
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
            num_object (batch_size, ):
        """
        target_pointmap * (estimated_local_offset - target_local_offset)
        
        
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean()
    
class StackedDepthLoss(nn.Module):
    def __init__(self):
        super(StackedDepthLoss, self).__init__()
        
    def forward(self, estimated_depth, target_depth, target_pointmap, num_object):
        """
        Args:
            estimated_depth (S, batch_size, 1, H//R, W//R):
            target_depth (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        estimated_depth = estimated_depth.squeeze(dim=2)
        
        loss = target_pointmap * torch.abs(1 / (torch.sigmoid(estimated_depth)) - 1 - target_depth)
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
        
    
