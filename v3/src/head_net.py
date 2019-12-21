import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapNet(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(HeatmapNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = self.sigmoid(x)
        
        return output
        
class ObjectSizeNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(ObjectSizeNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = self.tanh(x)
    
        return output

class DepthNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DepthNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
    
    def forward(self, input):
        output = self.bottleneck_conv2d(input)

        return output
