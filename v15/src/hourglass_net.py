import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedHourglassNet(nn.Module):
    def __init__(self, channel_list, num_stacks, num_blocks, num_in_features, in_channels=256, hidden_channels=128):
        super(StackedHourglassNet, self).__init__()
        
        # Hyper parameters
        self.num_stacks = num_stacks
        
        # Network configuration
        net = []
        
        self.down_conv2d = nn.Conv2d(num_in_features, in_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        for idx in range(num_stacks):
            net.append(HourglassNet(channel_list, num_blocks, in_channels, hidden_channels))
            
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        outputs = []
        x = self.down_conv2d(input)
        x = self.max_pool2d(x)
        
        for idx in range(self.num_stacks):
            x, intermediate_idx = self.net[idx](x)
            outputs.append(intermediate_idx)
        
        return outputs

class HourglassNet(nn.Module):
    def __init__(self, channel_list, num_blocks, in_channels=256, hidden_channels=128):
        super(HourglassNet, self).__init__()
        
        self.channel_list = channel_list
        
        self.encoder = Encoder(num_blocks, in_channels, hidden_channels)
        self.decoder = Decoder(num_blocks, in_channels, hidden_channels)
        self.bottleneck_conv2d_out = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=(1,1))
        
        bottleneck_conv2d_intermediate = []
        head_module_intermediate = []
        
        for head, num_in_features, head_module in channel_list:
            bottleneck_conv2d_intermediate.append(nn.Conv2d(in_channels, num_in_features, kernel_size=(1,1), stride=(1,1)))
            head_module_intermediate.append(head_module)
        
        self.bottleneck_conv2d_intermediate = nn.ModuleList(bottleneck_conv2d_intermediate)
        self.head_module_intermediate = nn.ModuleList(head_module_intermediate)
        
    def forward(self, input):
        residual = input
        x, skips = self.encoder(input)
        x = self.decoder(x, skips)
        x = self.bottleneck_conv2d_out(x)
        output = x + residual
        
        intermediate = {}
        
        for idx, (head, _, _) in enumerate(self.channel_list):
            x_intermediate = self.bottleneck_conv2d_intermediate[idx](x)
            intermediate[head] = self.head_module_intermediate[idx](x_intermediate)
        
        return output, intermediate

class Encoder(nn.Module):
    def __init__(self, num_blocks, in_channels=256, hidden_channels=128):
        super(Encoder, self).__init__()
        
        self.num_blocks = num_blocks
        
        # Network configuration
        self.down_sampling = Interpolate(scale_factor=0.5)
        encoder_block_sequence = []

        for idx in range(num_blocks):
            encoder_block_sequence.append(BasicBlock(in_channels, hidden_channels))
            
        self.encoder_block_sequence = nn.Sequential(*encoder_block_sequence)
            
    def forward(self, input):
        skips = []
        
        x = input
        
        for idx in range(self.num_blocks):
            x = self.encoder_block_sequence[idx](x)
            if idx != self.num_blocks-1:
                skips.append(x)
                x = self.down_sampling(x)
        output = x
        
        return output, skips
        
class Decoder(nn.Module):
    def __init__(self, num_blocks, in_channels=256, hidden_channels=128):
        super(Decoder, self).__init__()
        
        self.num_blocks = num_blocks
        
        # Network configuration
        self.up_sampling = Interpolate(scale_factor=2.0)
        decoder_block_sequence = []
        
        for idx in range(num_blocks):
            decoder_block_sequence.append(BasicBlock(in_channels, hidden_channels))
            
        self.decoder_block_sequence = nn.Sequential(*decoder_block_sequence)
            
    def forward(self, input, skips):
        x = input
        
        for idx in range(self.num_blocks):
            if idx != 0:
                x = self.up_sampling(x)
                _, _, H_original, W_original = skips[-idx].size()
                _, _, H, W = x.size()
                
                H_pad_left = (H_original-H)//2
                H_pad_right = H_original-H-H_pad_left
                W_pad_left = (W_original-W)//2
                W_pad_right = W_original-W-W_pad_left
                
                x = F.pad(x, (W_pad_left, W_pad_right, H_pad_left, H_pad_right))
                
                x = x + skips[-idx]

            x = self.decoder_block_sequence[idx](x)

        output = x
        
        return output

"""
In this implementation, Encoder's basic block and Decoder's basic block are same.
"""

class BasicBlock(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128):
        super(BasicBlock, self).__init__()
        
        channel_list = [in_channels, hidden_channels, hidden_channels, in_channels]
        kernel_list = [(1, 1), (3, 3), (1, 1)]
        stride_list = [(1, 1), (1, 1), (1, 1)]
        conv_block_sequence = []
        
        for idx in range(3):
            conv_block_sequence.append(ConvBlock(channel_list[idx], channel_list[idx+1], kernel_list[idx], stride=stride_list[idx]))
            
        self.conv_block_sequence = nn.Sequential(*conv_block_sequence)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        residual = input
        x = self.conv_block_sequence(input)
        x = self.relu(x)
        output = x + residual
        
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1)):
        super(ConvBlock, self).__init__()
        
        paddind_height = (kernel_size[0]//stride[0]-1)//2
        paddind_width = (kernel_size[1]//stride[1]-1)//2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(paddind_height, paddind_width))
        self.norm2d = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv2d(input)
        output = self.norm2d(x)

        return output

class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, input):
        output = F.interpolate(input, scale_factor=self.scale_factor, mode='nearest')
        
        return output
        

if __name__ == '__main__':
    num_stacks, num_blocks = 3, 4
    num_in_features = 3
    batch_size = 1
    height, width = 128, 256
    head_list = [('heatmap', 1), ('object_size', 2)]
    stacked_hourglass_net = StackedHourglassNet(head_list, num_stacks, num_blocks, num_in_features)
    
    print(stacked_hourglass_net)
    
    x = torch.randint(3, (batch_size, num_in_features, height, width), dtype=torch.float32)
    
    outputs = stacked_hourglass_net(x)
    
    print("x: {}".format(x.size()))
    
    estimated_output = {head: None for head, num_out_features in head_list}

    for output in outputs:
        for head in output:
            if estimated_output[head] is None:
                estimated_output[head] = output[head].unsqueeze(dim=0)
            else:
                estimated_output[head] = torch.cat((estimated_output[head], output[head].unsqueeze(dim=0)), dim=0)
                
    for head, num_out_features in head_list:
        print("estimated_output[{}].size(): {}".format(head, estimated_output[head].size()))
