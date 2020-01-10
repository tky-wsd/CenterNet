#!/bin/bash

DATA_DIR=$1
out_dir=$2
epochs=$3
continue_epoch=$4

. ./path.sh

train_csv_path="${DATA_DIR}/train.csv"
valid_csv_path="${DATA_DIR}/valid.csv"
train_image_dir="${DATA_DIR}/train_images"

S=1
C=[64,64,128,128,128,256]
H=64

heatmap=1.0
local_offset=0.0
depth=0.0
yaw=0.0
pitch=0.0
roll=0.0

batch_norm=1
head_relu=1
potential_map=1
batch_size=4

tag="UNet_S${S}_C${C}_H${H}_heatmap${heatmap}_local_offset${local_offset}_depth${depth}_batch_norm${batch_norm}_head_relu${head_relu}_potential_map${potential_map}"

if [ -z ${continue_epoch} ]; then
    argument=model_dir
    model_option="${out_dir}/${tag}"
else
    argument=continue_from
    model_option="${out_dir}/${tag}/epoch${continue_epoch}.pth"
fi

train_unet.py \
--train_csv_path ${train_csv_path} \
--valid_csv_path ${valid_csv_path} \
--train_image_dir ${train_image_dir} \
--f_x 2304.5479 \
--f_y 2305.8757 \
--c_x 1686.2379 \
--c_y 1354.9849 \
--S $S \
--C $C \
--H $H \
--batch_norm ${batch_norm} \
--head_relu ${head_relu} \
--potential_map ${potential_map} \
--batch_size ${batch_size} \
--heatmap $heatmap \
--local_offset ${local_offset} \
--depth $depth \
--yaw ${yaw} \
--pitch ${pitch} \
--roll ${roll} \
--optimizer 'Adam' \
--lr 0.001 \
--weight_decay 0 \
--epochs ${epochs} \
--${argument} "${model_option}"
