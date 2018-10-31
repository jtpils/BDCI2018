from multiprocessing import Process

import numpy as np
import os
from utils import df_utils
from utils import vis_utils


def df_to_2d(dir_df, framenames, dir_out, num_height=64, num_width=512):
    test_max_num = -1
    for i, framename in enumerate(framenames):
        if i == test_max_num:
            break
        print(os.getpid(), i, framename)
        path = os.path.join(dir_df, framename)
        pts_ins_cates = df_utils.load_frame_bin(path)

        append_zero = np.zeros((pts_ins_cates.shape[0], 3))
        alig_data = np.hstack((pts_ins_cates, append_zero))
        len_ = alig_data.shape[1]
        for p in alig_data:
            # p: [x, y, z, ins, mask, dis, theta, fie]
            dis = np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
            r = np.sqrt(p[0] * p[0] + p[1] * p[1])
            p[5] = dis
            p[len_ - 2] = np.arcsin(p[2] / dis)
            p[len_ - 1] = np.arcsin(p[1] / r)

        theta = alig_data[:, len_ - 2]
        fie = alig_data[:, len_ - 1]
        t_a = theta.max()
        t_i = theta.min()
        f_a = fie.max()
        f_i = fie.min()
        t_range = (t_a - t_i)
        f_range = (f_a - f_i)

        resolution_h = t_range / num_height
        resolution_w = f_range / num_width

        x_min = f_i / resolution_w
        y_min = t_i / resolution_h

        # print('shift', y_min, x_min)
        # print('angle', theta, fie)
        # print('range', t_range, f_range)
        # print('resl', resolution_h, resolution_w)kjlkjhkhj

        append_64 = np.zeros((num_height, num_width, 6))
        for p in alig_data:
            index_h = (p[len_ - 2] / resolution_h - y_min)
            index_w = (p[len_ - 1] / resolution_w - x_min)
            shitf_h = -round(index_h - num_height)
            shitf_w = -round(index_w - num_width)
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 0] = p[0]     # x
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 1] = p[1]     # y
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 2] = p[2]     # z
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 3] = p[3]     # ins
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 4] = p[5]     # dis
            append_64[int(shitf_h) - 1, int(shitf_w) - 1, 5] = p[4]     # mask

        path_out = os.path.join(dir_out, "df_2d_64_512", framename[:-4] + '.npy')
        np.save(path_out, append_64)

        # vis
        # path_out = os.path.join(dir_out, "vis_2d", framename[:-4] + '.jpg')
        # vis_utils.save_2d(path_out, append_64[:, :, 5])


if __name__ == '__main__':
    num_prccess = 4
    dir_input = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin_fru"
    dir_output = "/home/leon/Disk/dataset/DataFountain/training_cleared"
    dir_vis = "/home/leon/Disk/dataset/DataFountain/training_cleared/vis_2d"

    dir_2d = os.path.join(dir_output, 'df_2d_64_512')
    if not os.path.exists(dir_2d):
        os.makedirs(dir_2d)
    if not os.path.exists(dir_vis):
        os.makedirs(dir_vis)

    framenames = sorted(os.listdir(dir_input))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    # prepare_df_box(dir_input, framenames, dir_output)
    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=df_to_2d, args=(dir_input, chunk_filepaths, dir_output))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()
