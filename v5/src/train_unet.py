#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import TrainDataset
from driver import Trainer
from unet import StackedUNet
from head_net import HeatmapNet, ObjectSizeNet, DepthNet, YawNet, PitchNet, RollNet
from criterion import StackedHeatmapLoss, StackedDepthLoss, StackedYawLoss, StackedPitchLoss, StackedRollLoss

EPS=1e-8

parser = argparse.ArgumentParser("U-Net")

parser.add_argument('--train_csv_path', type=str, default=None, help='Path for train.csv')
parser.add_argument('--valid_csv_path', type=str, default=None, help='Path for valid.csv')
parser.add_argument('--train_image_dir', type=str, default=None, help='Root directory for train images')
parser.add_argument('--f_x', type=float, default=None, help='Focus of camera')
parser.add_argument('--f_y', type=float, default=None, help='Focus of camera')
parser.add_argument('--c_x', type=float, default=None, help='Offset in image')
parser.add_argument('--c_y', type=float, default=None, help='Offset in image')
parser.add_argument('--height', type=int, default=None, help='Image height (pixel) if known')
parser.add_argument('--width', type=int, default=None, help='Image width (pixel) if known')

parser.add_argument('--S', type=int, default=3, help='Number of hourglass stacks')
parser.add_argument('--C', type=str, default='[64,128,128,256,256,512]', help='Number of channels')
parser.add_argument('--H', type=int, default=256, help='Number of output bottleneck channels')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--heatmap', type=float, default=None, help='Weight for heatmap')
parser.add_argument('--local_offset', type=float, default=None, help='Weight for local_offset')
parser.add_argument('--object_size', type=float, default=None, help='Weight for object_size')
parser.add_argument('--depth', type=float, default=None, help='Weight for depth')
parser.add_argument('--yaw', type=float, default=None, help='Weight for yaw')
parser.add_argument('--pitch', type=float, default=None, help='Weight for pitch')
parser.add_argument('--roll', type=float, default=None, help='Weight for roll')
parser.add_argument('--optimizer', type=str, default=None, help='Optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--model_dir', type=str, default=None, help='Path to model directory')
parser.add_argument('--continue_from', type=str, default=None, help='Path to model')

def main(args):
    R = 4 # resolution
    C = args.C.replace('[', '').replace(']', '').split(',')
    args.C = [int(channel) for channel in C]
    output_bottleneck_channels = args.H
    
    camera_matrix = np.array([
        [args.f_x, 0, args.c_x],
        [0, args.f_y, args.c_y],
        [0, 0, 1]
    ])
    
    train_dataset = TrainDataset(csv_path=args.train_csv_path, image_dir=args.train_image_dir, R=R, camera_matrix=camera_matrix)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TrainDataset(csv_path=args.valid_csv_path, image_dir=args.train_image_dir, R=R, camera_matrix=camera_matrix)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    data_loader = {}
    data_loader['train'] = train_loader
    data_loader['valid'] = valid_loader
    
    head_list = []
    lambdas = {}
    criterions = {}
    
    if args.height is not None and args.width is not None:
        fixed_height, fixed_width = args.height//R, args.width//R
    else:
        fixed_height, fixed_width = None, None
    
    if args.heatmap is not None and args.heatmap > 0:
        num_out_features = 1
        head_list.append(('heatmap', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
        lambdas['heatmap'] = args.heatmap
        criterions['heatmap'] = StackedHeatmapLoss()
    if args.local_offset is not None and args.local_offset > 0:
        num_out_features = 1
        head_list.append(('local_offset', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
        lambdas['local_offset'] = args.local_offset
        criterions['local_offset'] = StackedLocalOffsetLoss()
    if args.object_size is not None and args.object_size > 0:
        num_out_features = 2
        head_list.append(('object_size', output_bottleneck_channels, ObjectSizeNet(output_bottleneck_channels, num_out_features)))
        lambdas['object_size'] = args.object_size
    if args.depth is not None and args.depth > 0:
        num_out_features = 1
        head_list.append(('depth', output_bottleneck_channels, DepthNet(output_bottleneck_channels, num_out_features)))
        lambdas['depth'] = args.depth
        criterions['depth'] = StackedDepthLoss()
    if args.yaw is not None and args.yaw > 0:
        num_out_features = 1
        head_list.append(('yaw', output_bottleneck_channels, YawNet(output_bottleneck_channels, num_out_features)))
        lambdas['yaw'] = args.yaw
        criterions['yaw'] = StackedYawLoss()
    if args.pitch is not None and args.pitch > 0:
        num_out_features = 2
        head_list.append(('pitch', output_bottleneck_channels, PitchNet(output_bottleneck_channels, num_out_features)))
        lambdas['pitch'] = args.pitch
        criterions['pitch'] = StackedPitchLoss()
    if args.roll is not None and args.roll > 0:
        num_out_features = 1
        head_list.append(('roll', output_bottleneck_channels, RollNet(output_bottleneck_channels, num_out_features)))
        lambdas['roll'] = args.roll
        criterions['roll'] = StackedRollLoss()

    model = StackedUNet(head_list, num_stacks=args.S, in_channels=3, channel_list=args.C, hidden_channel=args.H)
    print(model)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()
        
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot support optimizer {}".format(args.optimizer))
        
    trainer = Trainer(data_loader, model, optimizer, head_list, criterions, lambdas, args)
        
    trainer.train()
        
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
