import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet import EncoderBlock, DecoderBlock
from conv import BottleneckConv2d

EPS=1e-9

class HeatmapNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(HeatmapNet, self).__init__()
    
        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = self.sigmoid(x)
        
        return output
        
class XNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(XNet, self).__init__()
    
        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W):
        """
        x = self.bottleneck_conv2d(input)
        output = 50 * self.tanh(x)
        
        return output
        
        
class YNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(YNet, self).__init__()
    
        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W):
        """
        x = self.bottleneck_conv2d(input)
        x = self.relu(x)
        output = torch.sqrt(x)
        
        return output

class DepthNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DepthNet, self).__init__()

        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
    
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = 1 / (torch.sigmoid(x)+EPS)-1

        return output
        
class AngleNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(AngleNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        """
        Predict angle; from -pi to pi
        """
        x = self.bottleneck_conv2d(input)
        x = self.tanh(x)
        output = math.pi * x
        
        return output
        
class PitchNet(nn.Module):
    def __init__(self, in_channels, out_channels=8):
        """
        First 4 channels mean bin; -pi/2<=theta<=pi/2, 0<=theta<=pi, pi/2<=theta<=3pi/2, pi<=theta<=2pi
        Second 4 channels mean bin offset from; 0, pi/2, 3pi/2, 2pi
        -> Thus, out_channels is 4+4=8
        """
        super(PitchNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        batch_size, out_channels, H, W = x.size()
        
        assert out_channels == 8, "out_channels is expected 8, given {}".format(out_channels)
        
        return output

class RollNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(RollNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        """
        Predict sin(roll) only
        """
        x = self.bottleneck_conv2d(input)
        output = self.tanh(x)
        
        return output
        
class CategoricalPitchNet(nn.Module):
    def __init__(self, in_channels, out_channels=8):
        """
        First 4 channels mean bin; -pi/4<=theta<=pi/4, pi/4<=theta<=3*pi/4, 3*pi/4<=theta<=5*pi/4, -3*pi/4<=theta<=-pi/4
        Second 4 channels mean bin offset from; 0, pi/2, pi, -pi/2
        -> Thus, out_channels is 4+4=8
        """
        super(CategoricalPitchNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        batch_size, out_channels, H, W = x.size()
        
        assert out_channels == 8, "out_channels is expected 8, given {}".format(out_channels)
        
        output = x.view(batch_size, out_channels//2, 2, H, W)
        
        return output
