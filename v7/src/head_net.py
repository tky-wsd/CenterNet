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
    def __init__(self, in_channels, bottleneck_channels, out_channels=2):
        super(ObjectSizeNet, self).__init__()
        
        self.bottleneck_conv2d_in = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=(1,1))
        self.bottleneck_conv2d_out = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        x = self.bottleneck_conv2d_in(input)
        x = self.bottleneck_conv2d_out(x)
        output = self.tanh(x)
    
        return output

class DepthNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DepthNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
    
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = 1 / torch.sigmoid(x) - 1

        return output
        
class YawNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(YawNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = self.tanh(x)
        
        return output
        
class PitchNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(PitchNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output_is_front = self.sigmoid(x[:, 0:1, :, :])
        output_cos = self.tanh(x[:, 1:2, :, :])
        output = torch.cat((output_is_front, output_cos), dim=1)
        
        return output
        
class RollNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(RollNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = self.tanh(x)
        
        return output
