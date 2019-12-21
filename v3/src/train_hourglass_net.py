#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import Dataset
from trainer import Trainer
from hourglass_net import StackedHourglassNet
from head_net import HeatmapNet, ObjectSizeNet, DepthNet
from criterion import StackedHeatmapLoss, StackedDepthLoss

EPS=1e-8

parser = argparse.ArgumentParser("Stacked Hourglass Networks")

parser.add_argument('--train_csv_path', type=str, default=None, help='Path for train.csv')
parser.add_argument('--train_image_dir', type=str, default=None, help='Root directory for train images')
parser.add_argument('--f_x', type=float, default=None, help='Focus of camera')
parser.add_argument('--f_y', type=float, default=None, help='Focus of camera')
parser.add_argument('--c_x', type=float, default=None, help='Offset in image')
parser.add_argument('--c_y', type=float, default=None, help='Offset in image')

parser.add_argument('--S', type=int, default=3, help='Number of hourglass stacks')
parser.add_argument('--B', type=int, default=4, help='Number of encoder (and also decode) blocks')
parser.add_argument('--N', type=int, default=3, help='Number of input features')
parser.add_argument('--P', type=int, default=256, help='Number of output bottleneck channels')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--heatmap', type=float, default=None, help='Weight for heatmap')
parser.add_argument('--local_offset', type=float, default=None, help='Weight for local_offset')
parser.add_argument('--object_size', type=float, default=None, help='Weight for object_size')
parser.add_argument('--depth', type=float, default=None, help='Weight for depth')
parser.add_argument('--optimizer', type=str, default=None, help='Optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--model_dir', type=str, default='tmp', help='Path to model directory')


def main(args):
    R = 4 # resolution
    output_bottleneck_channels = args.P
    
    camera_matrix = np.array([
        [args.f_x, 0, args.c_x],
        [0, args.f_y, args.c_y],
        [0, 0, 1]
    ])
    
    dataset = Dataset(csv_path=args.train_csv_path, image_dir=args.train_image_dir, R=R, camera_matrix=camera_matrix)
    data_loader = {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    data_loader['train'] = train_loader
    # data_loader['valid'] = valid_loader
    
    channel_list = []
    lambdas = {}
    criterions = {}
    
    if args.heatmap is not None:
        num_out_features = 1
        channel_list.append(('heatmap', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
        lambdas['heatmap'] = args.heatmap
        criterions['heatmap'] = StackedHeatmapLoss()
    if args.local_offset is not None:
        num_out_features = 1
        channel_list.append(('local_offset', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
        lambdas['local_offset'] = args.local_offset
        criterions['local_offset'] = StackedLocalOffsetLoss()
    if args.object_size is not None:
        num_out_features = 2
        channel_list.append(('object_size', output_bottleneck_channels, ObjectSizeNet(output_bottleneck_channels, num_out_features)))
        lambdas['object_size'] = args.object_size
    if args.depth is not None:
        num_out_features = 1
        channel_list.append(('depth', output_bottleneck_channels, DepthNet(output_bottleneck_channels, num_out_features)))
        lambdas['depth'] = args.depth
        criterions['depth'] = StackedDepthLoss()

    model = StackedHourglassNet(channel_list, num_stacks=args.S, num_blocks=args.B, num_in_features=args.N)
    print(model)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()
        
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot support optimizer {}".format(args.optimizer))
        
    trainer = Trainer(data_loader, model, optimizer, channel_list, criterions, lambdas, args)
        
    trainer.train()
    
def make_dummy_data(batch_size, H, W, R=4, num_max_objects=10):
    num_objects = torch.randint(num_max_objects, (batch_size, ), dtype=torch.int) + 1

    input = torch.randn(batch_size, args.N, H, W)
    target_pointmap = torch.zeros(batch_size, H//R, W//R)
    batch_target_heatmap = torch.zeros(batch_size, H//R, W//R)
    target_local_offset = torch.zeros(batch_size, H//R, W//R)
    target_depth = torch.zeros(batch_size, H//R, W//R)
    
    for batch_id, num_object in enumerate(num_objects):
        p_h = torch.randint(H, (num_object.item(), ), dtype=torch.int)
        p_w = torch.randint(W, (num_object.item(), ), dtype=torch.int)
            
        h = torch.arange(0, H//R, 1, dtype=torch.float).unsqueeze(dim=1).expand(-1, W//R)
        w = torch.arange(0, W//R, 1, dtype=torch.float).unsqueeze(dim=0).expand(H//R, -1)
            
        target_heatmap = None
        
        for idx in range(num_object.item()):
            target_pointmap[batch_id][p_h[idx]//R, p_w[idx]//R] = 1.0
            target_local_offset[batch_id][p_h[idx]//R, p_w[idx]//R] = torch.randn(1)
            target_depth[batch_id][p_h[idx]//R, p_w[idx]//R] = torch.abs(torch.randn(1))
                
            exponent = - ((h-p_h[idx].float()/R)**2 + (w-p_w[idx].float()/R)**2)/(torch.pow(target_depth[batch_id][p_h[idx]//R, p_w[idx]//R], 2))
            heatmap = torch.exp(exponent)
                
            if target_heatmap is None:
                target_heatmap = heatmap.unsqueeze(dim=0)
            else:
                target_heatmap = torch.cat((target_heatmap, heatmap.unsqueeze(dim=0)), dim=0)
                
        target_heatmap, _ = torch.max(target_heatmap, dim=0)
        
        batch_target_heatmap[batch_id] = target_heatmap
        
    
    if torch.cuda.is_available():
        input = input.cuda()
        target = {'num_objects': num_objects.float().cuda(), 'pointmap': target_pointmap.cuda(), 'heatmap': batch_target_heatmap.cuda(), 'local_offset': target_local_offset.cuda(), 'depth': target_depth.cuda()}
    else:
        target = {'num_objects': num_objects.float(), 'pointmap': target_pointmap, 'heatmap': batch_target_heatmap, 'local_offset': target_local_offset, 'depth': target_depth}
        
        
    # for idx in range(batch_size):
    #     for key in target.keys():
    #         if key != 'num_objects':
    #             plt.imshow(target[key][idx])
    #             plt.savefig('{}_{}.png'.format(idx, key))
        
    return input, target
        
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
