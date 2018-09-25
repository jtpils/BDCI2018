#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3


# 接着训练49999
# python3 ../train_val_seg_data_fountain.py \
# -t ../dataset_data_fountain/training/h5_49999/data_fountain_files.txt \
# -s ../model_data_fountain/h5_49999/seg \
# -l ../model_data_fountain/h5_49999/seg/pointcnn_seg_data_fountain_data_fountain_x4_2048_xyrgbi_31881_2018-09-07-12-29-56/ckpts/iter-40000 \
# -m pointcnn_seg_data_fountain \
# -x data_fountain_x4_2048_xyrgbi

# jinming
#nohup 
python3 ../train_val_seg_data_fountain_indensity.py \
-t ../dataset_data_fountain/trainall/h5/data_fountain_files.txt \
-v ../dataset_data_fountain/trainval/val/h5/data_fountain_files.txt \
-s ../model_data_fountain/h5/seg \
-m pointcnn_seg_data_fountain \
-x data_fountain_x4_2048_xyrgbi #&


