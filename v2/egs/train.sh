#!/bin/bash

DATA_DIR=$1
out_dir=$2

. ./path.sh

train_csv_path="${DATA_DIR}/train.csv"
train_image_dir="${DATA_DIR}/train_images"

S=1
C=[64,128,128,256,256,512]
H=256
heatmap=1.0
depth=0.0

train_unet.py \
--train_csv_path ${train_csv_path} \
--train_image_dir ${train_image_dir} \
--f_x 2304.5479 \
--f_y 2305.8757 \
--c_x 1686.2379 \
--c_y 1354.9849 \
--S $S \
--C $C \
--H $H \
--batch_size 4 \
--heatmap $heatmap \
--depth $depth \
--optimizer 'Adam' \
--lr 0.001 \
--weight_decay 0 \
--epochs 5 \
--model_dir "${out_dir}/UNet_S${S}_C${C}_H${H}_heatmap${heatmap}_depth${depth}"
