import os
import numpy as np
from utils import vis_utils


def load_frame(dir_data, filename):
    """
    load a frame of data_fountain dataset
    :param dir_data: dataset dir
    :param filename: frame name
    :return: ((points, intensities), categories) of the frame
    """
    # load points
    path = os.path.join(dir_data, 'pts', filename)
    with open(path, 'r') as file:
        lines = file.readlines()
        data = np.empty((len(lines), 4), dtype=np.float32)
        for i, line in enumerate(lines):
            point_eles = line.strip('\n').split(',')
            data[i, 0:3] = np.array([np.float32(ele) for ele in point_eles])

    # load intensities
    path = os.path.join(dir_data, 'intensity', filename)
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            data[i, 3] = np.float32(line.strip('\n'))

    # load categories, no categories if load test set
    path = os.path.join(dir_data, 'category', filename)
    if os.path.exists(path):
        categories = np.loadtxt(path).astype(np.int32)
    else:
        categories = None

    return data, categories


def save_frame_to_bin(dir_data, filename, pts_ins, categories, vis=False):
    categories = categories.reshape(categories.shape[0], 1)
    pts_ins_cates = np.concatenate((pts_ins, categories), axis=1)
    path = os.path.join(dir_data, 'data_bin', filename[:-3] + 'npy')
    np.save(path, pts_ins_cates)

    if vis:
        path = os.path.join(dir_data, 'ply_colored', filename[:-3] + 'ply')
        vis_utils.save_ply(path, pts_ins[:, 0:3], vis_utils.seg2color(categories))


def save_frame_bin(dir_out, framename, pts_ins_cates, vis=False):
    path = os.path.join(dir_out, framename)
    np.save(path, pts_ins_cates)
    if vis:
        path = os.path.join(dir_out, '../ply_colored', framename[:-3] + 'ply')
        # test = pts_ins_cates[:, -1].astype(np.int32)
        # test_p = pts_ins_cates[:, 0:3]
        vis_utils.save_ply(path, pts_ins_cates[:, 0:3],
                           vis_utils.seg2color(pts_ins_cates[:, -1].astype(np.int32)))


def load_frame_bin(path):
    pts_ins_cates = np.load(path)
    # e = 0.1
    # for p in pts_ins_cates:
    #     p[4] += e
    return pts_ins_cates


