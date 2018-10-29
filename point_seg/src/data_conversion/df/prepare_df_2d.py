from multiprocessing import Process

import numpy as np
import os
from utils import df_utils


def split_to_fru():
    return


def split_in_x(pts_ins, categories):
    pts_ins_back = []
    pts_ins_front = []
    categories_back = []
    categories_front = []
    for p, c in zip(pts_ins, categories):
        if p[0] >= 0:
            pts_ins_front.append(p)
            categories_front.append(c)
        else:
            pts_ins_back.append(p)
            categories_back.append(c)

    return [np.array(pts_ins_front), np.array(pts_ins_back)],\
           [np.array(categories_front), np.array(categories_back)]


def df_to_2d(dir_df, framenames, dir_out, num_height=32, num_width=2048//2):
    for i, framename in enumerate(framenames):
        print(os.getpid(), i, framename)
        pts_ins, categories = df_utils.load_frame(dir_df, framename)
        pair_pts_ins, pair_categories = split_in_x(pts_ins, categories)
        # print("computting")

        id_x = 0
        for pts_ins, categories in zip(pair_pts_ins, pair_categories):
            append_zero = np.zeros((pts_ins.shape[0], 3))
            alig_data = np.hstack((pts_ins, append_zero))
            len_ = alig_data.shape[1]
            for p in alig_data:
                # p: [x, y, z, ins, theta, fie]
                dis = np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                r = np.sqrt(p[0] * p[0] + p[1] * p[1])
                p[4] = dis
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
            for p, c in zip(alig_data, categories):
                index_h = (p[len_ - 2] / resolution_h - y_min)
                index_w = (p[len_ - 1] / resolution_w - x_min)
                shitf_h = -round(index_h - num_height)
                shitf_w = -round(index_w - num_width)
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 0] = p[0]     # x
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 1] = p[1]     # y
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 2] = p[2]     # z
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 3] = p[3]     # ins
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 4] = p[4]     # dis
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 5] = c        # mask

            path_out = os.path.join(dir_out, "df_2d_32_1024",
                                    framename[:-4] + '_' + str(id_x) + '.npy')
            np.save(path_out, append_64)
            id_x += 1


if __name__ == '__main__':
    num_prccess = 3
    dir_input = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin"
    dir_output = "/home/leon/Disk/dataset/DataFountain/training_cleared"

    dir_2d = os.path.join(dir_output, 'df_2d_32_512')
    if not os.path.exists(dir_2d):
        os.makedirs(dir_2d)

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
