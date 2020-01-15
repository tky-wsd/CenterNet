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
C=[64,64,128,128,256,512]
H=64

heatmap=1.0
local_offset=0.0
x=0.0
y=0.0
depth=0.0
yaw=0.0
pitch=0.0
roll=0.0
categorical_pitch=0.0
gamma=0.0
shifted_roll=0.0

cut_upper=0
coeff_sigma=3.0
coeff_sigma_decay=0.0
coeff_sigma_min=3.0

separable=1
dilated=0
batch_norm=1
head_relu=1
potential_map=1
batch_size=8

show_target_heatmap=1

tag="UNet_S${S}_C${C}_H${H}_heatmap${heatmap}_local_offset${local_offset}_x${x}_y${y}_depth${depth}_yaw${yaw}_pitch${pitch}_roll${roll}_categorical_pitch${categorical_pitch}_gamma${gamma}_shifted_roll${shifted_roll}_separable${separable}_dilated${dilated}_batch_norm${batch_norm}_head_relu${head_relu}_potential_map${potential_map}_batch_size${batch_size}_cut_upper${cut_upper}_coeff${coeff_sigma}_decay${coeff_sigma_decay}_min${coeff_sigma_min}"

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
--cut_upper ${cut_upper} \
--coeff_sigma ${coeff_sigma} \
--coeff_sigma_decay ${coeff_sigma_decay} \
--coeff_sigma_min ${coeff_sigma_min} \
--separable ${separable} \
--dilated ${dilated} \
--batch_norm ${batch_norm} \
--head_relu ${head_relu} \
--potential_map ${potential_map} \
--batch_size ${batch_size} \
--heatmap ${heatmap} \
--local_offset ${local_offset} \
--x ${x} \
--y ${y} \
--depth ${depth} \
--yaw ${yaw} \
--pitch ${pitch} \
--roll ${roll} \
--categorical_pitch ${categorical_pitch} \
--gamma ${gamma} \
--shifted_roll ${shifted_roll} \
--optimizer 'Adam' \
--lr 0.001 \
--weight_decay 0 \
--epochs ${epochs} \
--${argument} "${model_option}" \
--show_target_heatmap ${show_target_heatmap}
