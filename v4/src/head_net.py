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
    def __init__(self, in_channels, out_channels=1, fixed_height=None, fixed_width=None):
        super(DepthNet, self).__init__()
        
        if fixed_height is not None and fixed_width is not None:
            self.potential_map = nn.Parameter(torch.Tensor(1, fixed_height, fixed_width))
            self.bottleneck_conv2d = nn.Conv2d(in_channels+1, out_channels, kernel_size=(1,1))
        else:
            self.potential_map = None
            self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.relu = nn.ReLU()
    
    def forward(self, input):
        if self.potential_map is not None:
            batch_size, _, _, _ = input.size()
            potential_map = self.potential_map.unsqueeze(dim=0)
            potential_map = potential_map.expand(batch_size, -1, -1, -1)
            x = torch.cat((input, potential_map), dim=1)
            x = self.bottleneck_conv2d(x)
        else:
            x = self.bottleneck_conv2d(input)
        
        output = self.relu(x)

        return output
