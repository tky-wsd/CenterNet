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

class DummyHeatmapNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(HeatmapNet, self).__init__()
        
        self.encoder = EncoderBlock(in_channels, in_channels, batch_norm=False, relu=False)
        self.decoder = DecoderBlock(in_channels, in_channels, batch_norm=False, relu=False)
        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        
        _, _, H_original, W_original = input.size()
        _, _, H, W = x.size()
        H_pad_left, W_pad_left = (H_original-H)//2, (W_original-W)//2
        H_pad_right, W_pad_right = H_original-H-H_pad_left, W_original-W-W_pad_left
        
        x = F.pad(x, (W_pad_left, W_pad_right, H_pad_left, H_pad_right))
        x = self.bottleneck_conv2d(x)
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

        self.bottleneck_conv2d = BottleneckConv2d(in_channels, out_channels)
    
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        output = 1 / (torch.sigmoid(x)+EPS)-1

        return output
        
class AngleNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(AngleNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.bottleneck_conv2d(input)
        x = self.tanh(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        output = x / (norm + EPS)
        
        return output
        
class YawNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(YawNet, self).__init__()
        
        self.bottleneck_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        """
        Predict sin(yaw)
        """
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
