#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES = 1

#python3 ../test_seg_data_fountain_fru.py \
python3 ../test_seg_df_fru_process.py \
-i \
/ssd/Datasets/DataFountain/test \
-o \
/ssd/Datasets/DataFountain/test_result \
-l \
/ssd/wyc/models/data_fountain/h5/seg/pointcnn_seg_data_fountain_data_fountain_x4_2048_xyrgbi_fru+intensity_26338_2018-09-26-08-35-48/ckpts/iter-420000 \
-m \
pointcnn_seg_data_fountain \
-x \
./data_fountain_x4_2048_xyrgbi_fru+intensity

