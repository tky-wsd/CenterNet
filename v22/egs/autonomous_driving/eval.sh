#!/bin/bash

DATA_DIR=$1
out_dir=$2
end_epoch=$3

. ./path.sh

test_csv_path="${DATA_DIR}/sample_submission.csv"
test_image_dir="${DATA_DIR}/test_images"

# Network configuration
S=1
C=[64,64,128,128,256,512]
H=64

heatmap=1.0
local_offset=0.0
x=0.1
y=0.1
depth=1.0
yaw=0.01
pitch=0.0
roll=0.0
categorical_pitch=1.0
gamma=0.9
shifted_roll=0.01

cut_upper=0
coeff_sigma=1.0
coeff_sigma_decay=0.0
coeff_sigma_min=1.0

separable=1
dilated=0
batch_norm=1
head_relu=1
potential_map=1
batch_size=4

tag="UNet_S${S}_C${C}_H${H}_heatmap${heatmap}_local_offset${local_offset}_x${x}_y${y}_depth${depth}_yaw${yaw}_pitch${pitch}_roll${roll}_categorical_pitch${categorical_pitch}_gamma${gamma}_shifted_roll${shifted_roll}_separable${separable}_dilated${dilated}_batch_norm${batch_norm}_head_relu${head_relu}_potential_map${potential_map}_batch_size${batch_size}_cut_upper${cut_upper}_coeff${coeff_sigma}_decay${coeff_sigma_decay}_min${coeff_sigma_min}"

model_dir="${out_dir}/${tag}"
model_path="${model_dir}/epoch${end_epoch}.pth"
out_image_dir="${model_dir}/epoch${end_epoch}"
out_csv_path="${out_image_dir}/test.csv"

eval_unet.py \
--test_csv_path "${test_csv_path}" \
--out_csv_path "${out_csv_path}" \
--test_image_dir "${test_image_dir}" \
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
--model_path "${model_path}" \
--out_image_dir "${out_image_dir}"
