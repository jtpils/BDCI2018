#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

python3 ../test_seg_data_fountain_intensity.py \
-f ../dataset_data_fountain/test/data_fountain_files.txt \
-d ../dataset_data_fountain/test_results/ \
-l ../model_data_fountain/h5/seg/pointcnn_seg_data_fountain_all_intensity/iter-320000 \
-m pointcnn_seg_data_fountain \
-x data_fountain_x4_2048_xyrgbi_intensity

