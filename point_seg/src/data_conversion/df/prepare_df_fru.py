import math
from multiprocessing import Process

import numpy as np
import os
from utils import df_utils


def rotation_coor(x, y, theta):
    x_rotated = x * math.cos(theta) - y * math.sin(theta)
    y_rotated = x * math.sin(theta) + y * math.cos(theta)
    return x_rotated, y_rotated


def split_to_fru(pts_ins_cates):
    data_front = []
    data_back = []
    data_left = []
    data_right = []
    for p in pts_ins_cates:
        x = p[0]
        y = p[1]
        z = p[2]
        if y >= abs(x):
            data_front.append(p)
        elif y <= -abs(x):
            y = -y
            data_back.append(np.array([x, y, z, p[3], p[4]]))
        elif x < -abs(y):
            x_rotated = y
            y_rotated = -x
            data_left.append(np.array([x_rotated, y_rotated, z, p[3], p[4]]))
        elif x > abs(y):
            x_rotated = -y
            y_rotated = x
            data_right.append(np.array([x_rotated, y_rotated, z, p[3], p[4]]))
    return [np.array(data_front), np.array(data_back),
            np.array(data_left), np.array(data_right)]


def df_to_fru(dir_in, framenames, dir_out):
    test_max_num = -1
    for i, framename in enumerate(framenames):
        if i == test_max_num:
            break
        print(os.getpid(), i, framename)
        pts_ins_cates = df_utils.load_frame_bin(os.path.join(dir_in, framename))
        frus_pts_ins_cates = split_to_fru(pts_ins_cates)

        for id_pos, fru in enumerate(frus_pts_ins_cates):
            df_utils.save_frame_bin(dir_out, framename[:-4] + '_' + str(id_pos) + '.npy', fru, False)


if __name__ == '__main__':
    num_prccess = 2
    dir_input = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin"
    dir_output = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin_fru"
    # dir_vis = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin_vis"

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    # if not os.path.exists(dir_vis):
    #     os.makedirs(dir_vis)

    framenames = sorted(os.listdir(dir_input))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    # prepare_df_box(dir_input, framenames, dir_output)
    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=df_to_fru, args=(dir_input, chunk_filepaths, dir_output))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()
