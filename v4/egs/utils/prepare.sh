#!/bin/bash

original_train_path="$1"
train_csv_path="$2"
valid_csv_path="$3"

make_data.py \
--original_train_path ${original_train_path} \
--train_csv_path ${train_csv_path} \
--valid_csv_path ${valid_csv_path}
