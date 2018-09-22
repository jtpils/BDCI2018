#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib

import shutil

from point_cnn.data_utils import data_utils
import numpy as np
from datetime import datetime


def load_bin(dir):
    print("load bin")
    bin_path = os.path.join(dir, "data_fountain_point.bin")
    points_ele_all = np.fromfile(bin_path, np.float32)

    bin_path = os.path.join(dir, "data_fountain_intensity.bin")
    intensities = np.fromfile(bin_path, np.float32)

    bin_path = os.path.join(dir, "data_fountain_point_num.bin")
    point_nums = np.fromfile(bin_path, np.uint16).astype(int)

    bin_path = os.path.join(dir, "data_fountain_label.bin")
    labels = np.fromfile(bin_path, np.uint8)

    print("create index_length")
    index_length = np.zeros((len(point_nums), 2), int)
    index_sum = 0
    for i in range(len(point_nums)):
        index_length[i][0] = index_sum
        index_length[i][1] = point_nums[i]
        index_sum += point_nums[i]

    return index_length, points_ele_all, intensities, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_train', '-t', help='Path to dir of train set', required=True)
    parser.add_argument('--dir_val', '-v', help='Path to dir of val set', required=False)
    parser.add_argument('--dir_out', '-o', help='', required=True)

    args = parser.parse_args()

    print('PID:', os.getpid())

    print(args)

    num_epochs = 2048
    batch_size = 6
    dir_out = args.dir_out

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))

    index_length_train, points_ele_train, intensities_train, labels_train = load_bin(args.dir_train)
    index_length_val, points_ele_val, intensities_val, labels_val = load_bin(args.dir_val)

    # shuffle
    index_length_train = data_utils.index_shuffle(index_length_train)
    index_length_val = data_utils.index_shuffle(index_length_val)

    num_train = index_length_train.shape[0]
    point_num = max(np.max(index_length_train[:, 1]), np.max(index_length_val[:, 1]))
    num_val = index_length_val.shape[0]

    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))
    batch_num = (num_train * num_epochs + batch_size - 1) // batch_size
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
    batch_num_val = math.ceil(num_val / batch_size)
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    for batch_idx in range(batch_num):
        # Training
        start_idx = (batch_size * batch_idx) % num_train
        end_idx = min(start_idx + batch_size, num_train)
        batch_size_train = end_idx - start_idx

        index_length_train_batch = index_length_train[start_idx:end_idx]
        points_batch = np.zeros((batch_size_train, point_num, 3), np.float32)
        points_num_batch = np.zeros(batch_size_train, np.int32)
        labels_batch = np.zeros((batch_size_train, point_num),np.int32)

        dir_out_point = os.path.join(dir_out, "point")
        if not os.path.exists(dir_out_point):
            os.makedirs(dir_out_point)
        dir_out_label = os.path.join(dir_out, "label")
        if not os.path.exists(dir_out_label):
            os.makedirs(dir_out_label)

        for i, index_length in enumerate(index_length_train_batch):
            points_batch[i, 0:index_length[1], :] = \
                points_ele_train[index_length[0]*3:index_length[0]*3+index_length[1]*3].reshape(index_length[1], 3)
            points_num_batch[i] = index_length[1].astype(np.int32)
            labels_batch[i, 0:index_length[1]] = labels_train[index_length[0]:index_length[0] + index_length[1]]\
                .astype(np.int32)

            f_point = open(os.path.join(dir_out_point, str(i) + ".csv"), 'w')
            f_label = open(os.path.join(dir_out_label, str(i) + ".csv"), 'w')
            for j in range(points_num_batch[i]):
                f_point.write('v %f %f %f\n' % (points_batch[i][j][0], points_batch[i][j][1], points_batch[i][j][2]))
                f_label.write('%d\n' % labels_batch[i][j])
            f_point.close()

        if start_idx + batch_size_train == num_train:
            break

        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
