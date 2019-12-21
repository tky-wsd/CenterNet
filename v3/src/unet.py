import torch
import torch.nn as nn
import torch.nn.functional as F

from head_net import HeatmapNet, DepthNet

class StackedUNet(nn.Module):
    def __init__(self, head_list, num_stacks, in_channels, channel_list, hidden_channel=256):
        super(StackedUNet, self).__init__()
        
        self.num_stacks = num_stacks
        
        self.preprocess = Preprocess(in_channels, channel_list[0])
        
        net = []
        for idx in range(num_stacks):
            net.append(UNet(head_list, channel_list, hidden_channel=hidden_channel))
            
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        outputs = []
        
        x = self.preprocess(input)
        
        for idx in range(self.num_stacks):
            x, output_intermediate = self.net[idx](x)
            outputs.append(output_intermediate)
        
        return outputs

class UNet(nn.Module):
    def __init__(self, head_list, channel_list, hidden_channel=256):
        super(UNet, self).__init__()
        
        self.head_list = head_list
        
        self.encoder = Encoder(channel_list)
        self.bottleneck_conv2d = nn.Conv2d(channel_list[-1], channel_list[-1], kernel_size=(1,1), stride=(1,1))
        self.decoder = Decoder(channel_list[::-1])
        self.point_wise_conv2d = nn.Conv2d(channel_list[0], channel_list[0], kernel_size=(1,1), stride=(1,1))
        
        bottleneck_conv2d_intermediate = []
        head_net = []
        
        for head, num_in_features, head_module in head_list:
            bottleneck_conv2d_intermediate.append(nn.Conv2d(channel_list[0], num_in_features, kernel_size=(1,1), stride=(1,1)))
            head_net.append(head_module)
        
        self.bottleneck_conv2d_intermediate = nn.ModuleList(bottleneck_conv2d_intermediate)
        self.head_net = nn.ModuleList(head_net)
        
        # self.head_net =
        
    def forward(self, input):
        residual = input
        x, skips = self.encoder(input)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skips)
        
        _, _, H_original, W_original = residual.size()
        _, _, H, W = x.size()
        H_pad_left, W_pad_left = (H_original-H)//2, (W_original-W)//2
        H_pad_right, W_pad_right = H_original-H-H_pad_left, W_original-W-W_pad_left
        x = F.pad(x, (W_pad_left, W_pad_right, H_pad_left, H_pad_right))
        
        x = x + residual
        x = self.point_wise_conv2d(x)
        output = x
    
        output_intermediate = {}
        
        for idx, (head, _, _) in enumerate(self.head_list):
            x_intermediate = self.bottleneck_conv2d_intermediate[idx](x)
            output_intermediate[head] = self.head_net[idx](x_intermediate)
    
        return output, output_intermediate
        
class Encoder(nn.Module):
    def __init__(self, channel_list):
        super(Encoder, self).__init__()
        
        self.num_blocks = len(channel_list) - 1
        
        net = []
        
        for idx in range(self.num_blocks):
            net.append(EncoderBlock(channel_list[idx], channel_list[idx+1]))
            
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
    def __init__(self, channel_list):
        super(Decoder, self).__init__()
        
        self.num_blocks = len(channel_list) - 1
        
        net = []
        
        for idx in range(self.num_blocks):
            net.append(DecoderBlock(channel_list[idx], channel_list[idx+1]))
            
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
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.channel_wise_conv2d = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=(3,3), stride=(2,2))
        self.point_wise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = F.pad(input, (0,1,0,1))
        x = self.channel_wise_conv2d(x)
        x = self.point_wise_conv2d(x)
        x = self.norm2d(x)
        output = self.relu(x)
        
        return output
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        x = self.point_wise_deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.channel_wise_deconv2d = nn.ConvTranspose2d(out_channels, out_channels, groups=out_channels, kernel_size=(3,3), stride=(2,2))
        self.norm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = self.point_wise_deconv2d(input)
        x = self.channel_wise_deconv2d(x)
        x = self.norm2d(x)
        x = x[:, :, :-1, :-1].contiguous()
        output = self.relu(x)
        
        return output
        
class Preprocess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Preprocess, self).__init__()
        
        self.down_sampling = nn.Sequential(DownConv2d(in_channels, out_channels), DownConv2d(out_channels, out_channels))
        
    def forward(self, input):
        output = self.down_sampling(input)
        
        return output
        
            
class DownConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv2d, self).__init__()
        
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), bias=False)
        
    def forward(self, input):
        x = F.pad(input, (0,1,0,1))
        output = self.net(x)
    
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
    
    unet = StackedUNet(head_list, num_stacks=2, in_channels=C, channel_list=channel_list, hidden_channel=256)
    
    print(unet)
    
    input = torch.randn(batch_size, C, H, W)
    
    outputs = unet(input)
    
    for output in outputs:
        for head in output.keys():
            print(head, output[head].size())
    
    
    
        

