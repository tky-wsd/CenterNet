#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import TestDataset
from driver import Evaluater
from unet import StackedUNet
from head_net import HeatmapNet, ObjectSizeNet, DepthNet
from criterion import StackedHeatmapLoss, StackedDepthLoss

EPS=1e-8

parser = argparse.ArgumentParser("U-Net")

parser.add_argument('--test_csv_path', type=str, default=None, help='Path for test.csv')
parser.add_argument('--test_image_dir', type=str, default=None, help='Root directory for test images')
parser.add_argument('--f_x', type=float, default=None, help='Focus of camera')
parser.add_argument('--f_y', type=float, default=None, help='Focus of camera')
parser.add_argument('--c_x', type=float, default=None, help='Offset in image')
parser.add_argument('--c_y', type=float, default=None, help='Offset in image')

parser.add_argument('--S', type=int, default=3, help='Number of hourglass stacks')
parser.add_argument('--C', type=str, default='[64,128,128,256,256,512]', help='Number of channels')
parser.add_argument('--H', type=int, default=256, help='Number of output bottleneck channels')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--heatmap', type=float, default=None, help='Weight for heatmap')
parser.add_argument('--local_offset', type=float, default=None, help='Weight for local_offset')
parser.add_argument('--object_size', type=float, default=None, help='Weight for object_size')
parser.add_argument('--depth', type=float, default=None, help='Weight for depth')

parser.add_argument('--out_image_dir', type=str, default='tmp', help='Directory to save images')
parser.add_argument('--model_path', type=str, default='tmp', help='Path to model')


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
    
    dataset = TestDataset(csv_path=args.test_csv_path, image_dir=args.test_image_dir, R=R, camera_matrix=camera_matrix)
    data_loader = {}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    data_loader['test'] = test_loader
    
    head_list = []
    
    if args.heatmap is not None and args.heatmap > 0:
        num_out_features = 1
        head_list.append(('heatmap', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
    if args.local_offset is not None and args.local_offset > 0:
        num_out_features = 1
        head_list.append(('local_offset', output_bottleneck_channels, HeatmapNet(output_bottleneck_channels, num_out_features)))
    if args.object_size is not None and args.object_size > 0:
        num_out_features = 2
        head_list.append(('object_size', output_bottleneck_channels, ObjectSizeNet(output_bottleneck_channels, num_out_features)))
    if args.depth is not None and args.depth > 0:
        num_out_features = 1
        head_list.append(('depth', output_bottleneck_channels, DepthNet(output_bottleneck_channels, num_out_features)))
    
    model = StackedUNet(head_list, num_stacks=args.S, in_channels=3, channel_list=args.C, hidden_channel=args.H)
     
    print(model)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    model.load_state_dict(torch.load(args.model_path))
        
    evaluater = Evaluater(data_loader, model, head_list, args)
        
    evaluater.eval()
        
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
