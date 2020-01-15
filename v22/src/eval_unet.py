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
from head_net import HeatmapNet, DepthNet, AngleNet, CategoricalPitchNet, XNet, YNet

EPS=1e-8

parser = argparse.ArgumentParser("U-Net")

parser.add_argument('--test_csv_path', type=str, default=None, help='Path for test.csv')
parser.add_argument('--out_csv_path', type=str, default=None, help='Path for OUTPUT test.csv')
parser.add_argument('--test_image_dir', type=str, default=None, help='Root directory for test images')
parser.add_argument('--f_x', type=float, default=None, help='Focus of camera')
parser.add_argument('--f_y', type=float, default=None, help='Focus of camera')
parser.add_argument('--c_x', type=float, default=None, help='Offset in image')
parser.add_argument('--c_y', type=float, default=None, help='Offset in image')

parser.add_argument('--S', type=int, default=3, help='Number of hourglass stacks')
parser.add_argument('--C', type=str, default='[64,128,128,256,256,512]', help='Number of channels')
parser.add_argument('--H', type=int, default=256, help='Number of output bottleneck channels')

parser.add_argument('--separable', type=int, default=0, help='Depthwise separable convolution')
parser.add_argument('--dilated', type=int, default=0, help='Dilated convolution')
parser.add_argument('--batch_norm', type=int, default=0, help='Whether to use batch normalization')
parser.add_argument('--head_relu', type=int, default=0, help='Whether to use relu before head net')
parser.add_argument('--potential_map', type=int, default=0, help='Whether to use potential map')
parser.add_argument('--cut_upper', type=int, default=0, help='Whether to cut upper part of image')
parser.add_argument('--coeff_sigma', type=float, default=0.99, help='Coefficient of heatmap sigma')
parser.add_argument('--coeff_sigma_decay', type=float, default=0.0, help='Decay coefficient of heamap sigma')
parser.add_argument('--coeff_sigma_min', type=float, default=0.0, help='Mainimum coefficient of heamap sigma')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

parser.add_argument('--heatmap', type=float, default=None, help='Weight for heatmap')
parser.add_argument('--x', type=float, default=None, help='Weight for x (in world coords)')
parser.add_argument('--y', type=float, default=None, help='Weight for y (in world coords)')
parser.add_argument('--local_offset', type=float, default=None, help='Weight for local_offset')
parser.add_argument('--object_size', type=float, default=None, help='Weight for object_size')
parser.add_argument('--depth', type=float, default=None, help='Weight for depth')
parser.add_argument('--yaw', type=float, default=None, help='Weight for yaw')
parser.add_argument('--pitch', type=float, default=None, help='Weight for pitch')
parser.add_argument('--categorical_pitch', type=float, default=None, help='Weight for categorical pitch')
parser.add_argument('--gamma', type=float, default=None, help='Weight for categorical loss in categorical pitch, if 0.0 only consider in-bin offset loss')
parser.add_argument('--roll', type=float, default=None, help='Weight for roll')
parser.add_argument('--shifted_roll', type=float, default=None, help='Weight for shifted roll')

parser.add_argument('--out_image_dir', type=str, default='tmp', help='Directory to save images')
parser.add_argument('--model_path', type=str, default='tmp', help='Path to model')


def main(args):
    R = 16 # resolution
    args.R = R
    C = args.C.replace('[', '').replace(']', '').split(',')
    args.C = [int(channel) for channel in C]
    output_hidden_channels = args.H
    
    camera_matrix = np.array([
        [args.f_x, 0, args.c_x],
        [0, args.f_y, args.c_y],
        [0, 0, 1]
    ])
    args.camera_matrix = camera_matrix
    
    dataset = TestDataset(csv_path=args.test_csv_path, image_dir=args.test_image_dir, R=R, camera_matrix=camera_matrix)
    data_loader = {}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_loader['test'] = test_loader
    
    head_list = []
    
    if args.heatmap is not None and args.heatmap > 0:
        num_out_features = 1
        head_list.append(('heatmap', output_hidden_channels, HeatmapNet(output_hidden_channels, num_out_features)))
    if args.local_offset is not None and args.local_offset > 0:
        num_out_features = 1
        head_list.append(('local_offset',  output_hidden_channels, HeatmapNet(output_hidden_channels, num_out_features)))
    if args.x is not None and args.x > 0:
        num_out_features = 1
        head_list.append(('x', output_hidden_channels, XNet(output_hidden_channels, num_out_features)))
    if args.y is not None and args.y > 0:
        num_out_features = 1
        head_list.append(('y', output_hidden_channels, YNet(output_hidden_channels, num_out_features)))
    if args.depth is not None and args.depth > 0:
        num_out_features = 1
        head_list.append(('depth', output_hidden_channels, DepthNet(output_hidden_channels, num_out_features)))
    if args.yaw is not None and args.yaw > 0:
        num_out_features = 1
        head_list.append(('yaw', output_hidden_channels, AngleNet(output_hidden_channels, num_out_features)))
    if args.pitch is not None and args.pitch > 0:
        num_out_features = 1
        head_list.append(('pitch', output_hidden_channels, AngleNet(output_hidden_channels, num_out_features)))
    if args.roll is not None and args.roll > 0:
        pass
    if args.categorical_pitch is not None and args.categorical_pitch > 0:
        num_out_features = 8
        head_list.append(('categorical_pitch', output_hidden_channels, CategoricalPitchNet(output_hidden_channels, num_out_features)))
    if args.shifted_roll is not None and args.shifted_roll > 0:
        num_out_features = 1
        head_list.append(('shifted_roll', output_hidden_channels, AngleNet(output_hidden_channels, num_out_features)))
    
    model = StackedUNet(head_list, num_stacks=args.S, in_channels=3, channel_list=args.C, separable=args.separable, dilated=args.dilated, batch_norm=args.batch_norm, head_relu=args.head_relu, potential_map=args.potential_map)
    print(model)
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model.cuda()
    
    evaluater = Evaluater(data_loader, model, head_list, args)
        
    evaluater.eval()
        
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
