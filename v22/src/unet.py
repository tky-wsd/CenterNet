import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import BottleneckConv2d, DepthwiseSeparableConv2d, DepthwiseSeparableDeconv2d, DownConv2d

class StackedUNet(nn.Module):
    def __init__(self, head_list, num_stacks, in_channels, channel_list, separable=True, dilated=False, batch_norm=False, head_relu=False, potential_map=False):
        super(StackedUNet, self).__init__()
        
        self.num_stacks = num_stacks
        
        self.preprocess = Preprocess(in_channels, channel_list[0])
        
        net = []
        for idx in range(num_stacks):
            net.append(UNet(head_list, channel_list, separable=separable, dilated=dilated, batch_norm=batch_norm, head_relu=head_relu, potential_map=potential_map))
            
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        outputs = []
        
        x = self.preprocess(input)
        
        for idx in range(self.num_stacks):
            x, output_intermediate = self.net[idx](x)
                
            outputs.append(output_intermediate)
        
        return outputs

class UNet(nn.Module):
    def __init__(self, head_list, channel_list, separable=True, dilated=False, batch_norm=False, head_relu=False, potential_map=False):
        super(UNet, self).__init__()
        
        self.head_list = head_list
        self.head_relu = head_relu
        self.potential_map = potential_map
        
        self.encoder = Encoder(channel_list, separable=separable, dilated=dilated, batch_norm=batch_norm)
        self.bottleneck_conv2d = BottleneckConv2d(channel_list[-1], channel_list[-1])
        self.decoder = Decoder(channel_list[::-1], separable=separable, batch_norm=batch_norm)
        self.point_wise_conv2d = nn.Conv2d(channel_list[0], channel_list[0], kernel_size=(1,1), stride=(1,1))
        
        bottleneck_conv2d_intermediate = []
        head_net = []
        
        if potential_map:
            potential_map_channel = 2
        else:
            potential_map_channel = 0
        
        for head, num_in_features, head_module in head_list:
            bottleneck_conv2d_intermediate.append(BottleneckConv2d(channel_list[0]+potential_map_channel, num_in_features))
            head_net.append(head_module)
        
        self.bottleneck_conv2d_intermediate = nn.ModuleList(bottleneck_conv2d_intermediate)
        self.head_net = nn.ModuleList(head_net)
        
    def forward(self, input):
        residual = input
        x, skips = self.encoder(input)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skips)
        
        batch_size, _, H_original, W_original = residual.size()
        _, _, H, W = x.size()
        H_pad_left, W_pad_left = (H_original-H)//2, (W_original-W)//2
        H_pad_right, W_pad_right = H_original-H-H_pad_left, W_original-W-W_pad_left
        x = F.pad(x, (W_pad_left, W_pad_right, H_pad_left, H_pad_right))
        
        x = x + residual
        x = self.point_wise_conv2d(x)
        output = x
        
        if self.potential_map:
            potential_h, potential_w = torch.meshgrid([torch.linspace(start=0.0, end=1.0, steps=H_original), torch.linspace(start=0.0, end=1.0, steps=W_original)])
            potential_h = potential_h.unsqueeze(dim=0)
            potential_w = potential_w.unsqueeze(dim=0)
            potential_h = potential_h.expand(batch_size, -1, -1, -1).detach()
            potential_w = potential_w.expand(batch_size, -1, -1, -1).detach()
            
            potential_h = potential_h.to(x.device)
            potential_w = potential_w.to(x.device)
            
            x = torch.cat((x, potential_h, potential_w), dim=1)
    
        output_intermediate = {}
        
        for idx, (head, _, _) in enumerate(self.head_list):
            x_intermediate = self.bottleneck_conv2d_intermediate[idx](x)
            
            if self.head_relu:
                x_intermediate = torch.relu(x_intermediate)
                
            output_intermediate[head] = self.head_net[idx](x_intermediate)
    
        return output, output_intermediate


class Encoder(nn.Module):
    def __init__(self, channel_list, separable=True, dilated=False, batch_norm=False):
        super(Encoder, self).__init__()
        
        self.num_blocks = len(channel_list) - 1
        
        net = []
        
        for idx in range(self.num_blocks):
            if dilated:
                dilation = 2**idx
            else:
                dilation = 1
            net.append(EncoderBlock(channel_list[idx], channel_list[idx+1], separable=separable, dilation=dilation, batch_norm=batch_norm))
            
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        skips = []
        
        x = input
        
        for idx in range(self.num_blocks):
            x = self.net[idx](x)
            if idx != self.num_blocks-1:
                skips.append(x)
            
        output = x
        
        return output, skips


class Decoder(nn.Module):
    def __init__(self, channel_list, separable=True, batch_norm=False):
        super(Decoder, self).__init__()
        
        self.num_blocks = len(channel_list) - 1
        
        net = []
        
        for idx in range(self.num_blocks):
            net.append(DecoderBlock(channel_list[idx], channel_list[idx+1], separable=separable, batch_norm=batch_norm))
            
        self.net = nn.Sequential(*net)
        
    def forward(self, input, skips):
        x = input
        
        for idx in range(self.num_blocks):
            x = self.net[idx](x)
            if idx != self.num_blocks-1:
                _, _, H_original, W_original = skips[-idx-1].size()
                _, _, H, W = x.size()
                H_pad_left, W_pad_left = (H_original-H)//2, (W_original-W)//2
                H_pad_right, W_pad_right = H_original-H-H_pad_left, W_original-W-W_pad_left
                x = F.pad(x, (W_pad_left, W_pad_right, H_pad_left, H_pad_right))
                x = x + skips[-idx-1]
            
        output = x
        
        return output

        
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, separable=True, batch_norm=False, relu=True):
        super(EncoderBlock, self).__init__()
        
        if separable:
            self.conv2d = DepthwiseSeparableConv2d(in_channels, out_channels, dilation=dilation)
        else:
            self.conv2d = DownConv2d()
    
        if batch_norm:
            self.norm2d = nn.BatchNorm2d(out_channels)
        else:
            self.norm2d = None
            
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        
    def forward(self, input):
        x = self.conv2d(input)
        
        if self.norm2d is not None:
            x = self.norm2d(x)
        if self.relu is not None:
            x = self.relu(x)
        
        output = x
        
        return output


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, separable=True, batch_norm=False, relu=True):
        super(DecoderBlock, self).__init__()
        
        if separable:
            self.deconv2d = DepthwiseSeparableDeconv2d(in_channels, out_channels)
        else:
            self.deconv2d = UpConv2d(in_channels, out_channels)
        
        if batch_norm:
            self.norm2d = nn.BatchNorm2d(out_channels)
        else:
            self.norm2d = None
            
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        
    def forward(self, input):
        x = self.deconv2d(input)
        
        if self.norm2d is not None:
            x = self.norm2d(x)
        
        if self.relu is not None:
            x = self.relu(x)
        
        output = x
        
        return output
        
class Preprocess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Preprocess, self).__init__()
        
        self.down_sampling = nn.Sequential(DownConv2d(in_channels, out_channels//4), DownConv2d(out_channels//4, out_channels//2), nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), DownConv2d(out_channels//2, out_channels))
        
    def forward(self, input):
        output = self.down_sampling(input)
        
        return output
    
        
if __name__ == '__main__':
    batch_size = 4
    C, H, W = 3, 1024, 2048
    head_list = []
    channel_list = [64, 128, 128, 256, 256]
    output_bottleneck_channels = 256
    
    head_list = []
    
    num_out_features = 1
    head_list.append(('heatmap', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
    
    unet = StackedUNet(head_list, num_stacks=2, in_channels=C, channel_list=channel_list)
    
    print(unet)
    
    input = torch.randn(batch_size, C, H, W)
    
    outputs = unet(input)
    
    for output in outputs:
        for head in output.keys():
            print(head, output[head].size())
    
    
    
        

