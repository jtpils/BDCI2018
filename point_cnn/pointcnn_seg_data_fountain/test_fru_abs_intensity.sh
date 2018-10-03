#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES = 1

python3 ../test_seg_data_fountain_fru.py \
-i \
/home/leon/Disk/datasets/data_fountain/test/bin_fru_abs \
-o \
/home/leon/Disk/datasets/data_fountain/test_results0 \
-l \
/home/leon/Disk/models/data_fountain/h5/seg/\
pointcnn_seg_data_fountain_data_fountain_x4_2048_xyrgbi_fru+intensity_adjustment0/ckpts/iter-420000 \
-m \
pointcnn_seg_data_fountain \
-x \
./data_fountain_x4_2048_xyrgbi_fru+intensity

