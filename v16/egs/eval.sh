#!/bin/bash

DATA_DIR=$1
out_dir=$2
end_epoch=$3

. ./path.sh

test_csv_path="${DATA_DIR}/sample_submission.csv"
test_image_dir="${DATA_DIR}/test_images"

# Network configuration
S=1
C=[64,128,128,256,256,512]
H=32

heatmap=1.0
local_offset=0.0
depth=1.0
yaw=1.0
pitch=1.0
roll=1.0

batch_norm=1
potential_map=1
batch_size=4

model_dir="${out_dir}/UNet_S${S}_C${C}_H${H}_heatmap${heatmap}_local_offset${local_offset}_depth${depth}_batch_norm_${batch_norm}_potential_map${potential_map}"
model_path="${model_dir}/epoch${end_epoch}.pth"
out_image_dir="${model_dir}/epoch${end_epoch}"
out_csv_path="${out_image_dir}/test.csv"

eval_unet.py \
--test_csv_path "${test_csv_path}" \
--out_csv_path "${out_csv_path}" \
--test_image_dir ${test_image_dir} \
--f_x 2304.5479 \
--f_y 2305.8757 \
--c_x 1686.2379 \
--c_y 1354.9849 \
--S $S \
--C $C \
--H $H \
--batch_norm ${batch_norm} \
--potential_map ${potential_map} \
--batch_size ${batch_size} \
--heatmap ${heatmap} \
--local_offset ${local_offset} \
--depth ${depth} \
--yaw ${yaw} \
--pitch ${pitch} \
--roll ${roll} \
--model_path "${model_path}" \
--out_image_dir "${out_image_dir}"
