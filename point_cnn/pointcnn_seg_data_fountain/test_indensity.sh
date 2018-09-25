#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

python3 ../test_seg_data_fountain_indensity.py \
-f ../dataset_data_fountain/test/data_fountain_files.txt \
-d ../dataset_data_fountain/download/submit_example/ \
-l ../model_data_fountain/h5/seg/pointcnn_seg_data_fountain_data_fountain_x4_2048_xyrgbi_3162_2018-09-20-08-58-57/ckpts/iter-94000 \
-m pointcnn_seg_data_fountain \
-x data_fountain_x4_2048_xyrgbi

