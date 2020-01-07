import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckConv2d, self).__init__()
        
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, input):
        outputs = self.net(input)
        
        return outputs

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.channel_wise_conv2d = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=(3,3), stride=(2,2))
        self.point_wise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, input):
        x = F.pad(input, (0,1,0,1))
        x = self.channel_wise_conv2d(x)
        output = self.point_wise_conv2d(x)
        
        return output
        
class DepthwiseSeparableDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableDeconv2d, self).__init__()
        
        self.point_wise_deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.channel_wise_deconv2d = nn.ConvTranspose2d(out_channels, out_channels, groups=out_channels, kernel_size=(3,3), stride=(2,2))
        
    def forward(self, input):
        x = self.point_wise_deconv2d(input)
        output = self.channel_wise_deconv2d(x)

        return output
