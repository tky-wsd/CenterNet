#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import random

parser = argparse.ArgumentParser("Make Peking university autonomous driving data.")

parser.add_argument('--original_csv_path', type=str, default=None, help='Path for ORIGINAL train.csv')
parser.add_argument('--train_csv_path', type=str, default=None, help='Path for train.csv')
parser.add_argument('--valid_csv_path', type=str, default=None, help='Path for train.csv')
parser.add_argument('--split', type=float, default=0.1, help='Ratio for splitting, 0.1 means 10% of ORIGINAL train data is used for validatin')

def main(args):
    original_csv_path = args.original_csv_path
    train_csv_path = args.train_csv_path
    valid_csv_path = args.valid_csv_path
    
    original_data_frame = pd.read_csv(original_csv_path)
    original_data_frame = original_data_frame.sample(frac=1).reset_index(drop=True)
    
    n_total = len(original_data_frame)
    n_valid = int(args.split * n_total)
    
    valid_data_frame = original_data_frame[:n_valid]
    train_data_frame = original_data_frame[n_valid:]
    
    print("train: {}".format(len(train_data_frame)))
    print("valid: {}".format(len(valid_data_frame)))
    
    train_data_frame.to_csv(train_csv_path, index=False)
    valid_data_frame.to_csv(valid_csv_path, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    
    print(args)

    main(args)
