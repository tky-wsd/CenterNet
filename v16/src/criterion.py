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
        
    def forward(self, estimated, target_angle, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated (S, batch_size, 2, H//R, W//R):
            target_angle (batch_size, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, 1):
        """
        target_cos = torch.cos(target_angle).unsqueeze(dim=1)
        target_sin = torch.sin(target_angle).unsqueeze(dim=1)
        target = torch.cat((target_cos, target_sin), dim=1) # target: (batch_size, 2, H//R, W//R)
        
        if self.metrics == 'cos_sim':
            loss = - target_pointmap * (estimated*target).sum(dim=2) # loss: (S, batch_size, H//R, W//R)
            loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        else:
            target_pointmap = target_pointmap.unsqueeze(dim=1)
            loss = target_pointmap * torch.abs(estimated - target) # loss: (S, batch_size, 2, H//R, W//R)
            loss = loss.mean(dim=0).sum(dim=1).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
            
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
        
class StackedPitchLoss(nn.Module):
    def __init__(self):
        super(StackedPitchLoss, self).__init__()
        
    def forward(self, estimated, target, target_pointmap, num_object):
        """
        L1 Loss
        Args:
            estimated (S, batch_size, 2, H//R, W//R):
            target (batch_size, 2, H//R, W//R):
            target_pointmap (batch_size, H//R, W//R):
            num_object (batch_size, ):
        """
        loss = - (target[:, 0, :, :] * torch.log(estimated[:, :, 0, :, :] + EPS) + (1 - target[:, 0, :, :]) * torch.log(1 - estimated[:, :, 0, :, :] + EPS))
        
        loss = target_pointmap * torch.abs(estimated[:, :, 1, :, :] - target[:, 1, :, :])
        loss = loss.mean(dim=0).sum(dim=1).sum(dim=1) / num_object.squeeze(dim=1)
        loss = loss.mean(dim=0)
        
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
        
    
