#!/usr/bin/python3

# -i
# /home/leon/Disk/dataset/Downloads/DataFountain/TestSet
# -o
# /home/leon/Disk/dataset/DataFountain/test_results/opensrc_trainval_weights
# -l
# /home/leon/Disk/models/data_fountain/pointcnn_seg_df_x4_2048_fps_2018-10-11-06-30-01_6985/ckpts/iter-75000
# -m
# pointcnn_seg
# -x
# df_x4_2048_fps
# -g
# 0
# -s

# python3 test_df_seg_processes_fru.py -i /home/leon/Disk/datasets/data_fountain/test \
# -o /home/leon/Disk/datasets/data_fountain/test_results \
# -l /home/leon/Disk/models/data_fountain/h5/seg/\
# pointcnn_seg_df_x4_2048_fps_2018-10-12-20-04-12_22301/ckpts/iter-80000 \
# -m pointcnn_seg -x df_x4_2048_fps -g 0,1,2 -s

"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from utils import df_utils, vis_utils
from datetime import datetime
import numpy as np


def main():
    dir_input = "/home/leon/Disk/dataset/Downloads/DataFountain/TestSet_2"
    framenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    test_max_num = -1

    max_point_num = 0
    min_point_num = 5000000
    for id_file, framename in enumerate(framenames):
        print(framename)
        if id_file == test_max_num:
            break

        # Prepare inputs
        print('{}-Preparing datasets...'.format(datetime.now()))
        # load
        pts_ins, _ = df_utils.load_frame(dir_input, framename)

        # clear
        pts_ins_cleared, _, scene_indices = df_utils.clear_data_2(pts_ins, None)

        # split
        frus_pts_ins, _, frus_indices = df_utils.split_frame_to_frus(pts_ins_cleared, None)

        # count
        for fru_pts_ins in frus_pts_ins:
            if max_point_num < len(fru_pts_ins):
                max_point_num = len(fru_pts_ins)
            if min_point_num > len(fru_pts_ins):
                min_point_num = len(fru_pts_ins)

        print(max_point_num, min_point_num)

        # Vis
        dir_vis = os.path.join(dir_input, "pts_colored")
        if not os.path.exists(dir_vis):
            print(dir_vis, "Not Exists! Create", dir_vis)
            os.makedirs(dir_vis)
        path_label_ply = os.path.join(dir_vis, framename[:-4] + '.ply')
        vis_utils.save_ply(path_label_ply, np.array(pts_ins_cleared)[:, 0:3],
                           vis_utils.seg2color(np.zeros(len(pts_ins_cleared), dtype=np.int32)))

    print("max_point_num: ", max_point_num, "min:", min_point_num)


if __name__ == '__main__':
    main()
