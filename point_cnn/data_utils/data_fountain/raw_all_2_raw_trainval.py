import argparse
import datetime
import random

import h5py as h5py
import numpy as np
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', '-i', help='Path to data folder', required=True)
    parser.add_argument('--dir_output', '-o', help='Path to h5 folder', required=True)
    parser.add_argument('--rate', '-m', default=0.25, type=float, help='val / train', required=False)
    args = parser.parse_args()
    print(args)

    dir_input = args.dir_input
    dir_output = args.dir_output
    train_occupancy = 100 / (1 + args.rate)
    # print("train_occupancy ", train_occupancy)
    data_ext = '.csv'

    dir_pts = os.path.join(dir_input, 'pts')
    dir_label = os.path.join(dir_input, 'category')
    dir_intensity = os.path.join(dir_input, 'intensity')

    dir_train_pts = os.path.join(dir_output, 'train', 'pts')
    if not os.path.exists(dir_train_pts):
        os.makedirs(dir_train_pts)
    dir_train_label = os.path.join(dir_output, 'train', 'category')
    if not os.path.exists(dir_train_label):
        os.makedirs(dir_train_label)
    dir_train_intensity = os.path.join(dir_output, 'train', 'intensity')
    if not os.path.exists(dir_train_intensity):
        os.makedirs(dir_train_intensity)

    dir_val_pts = os.path.join(dir_output, 'val', 'pts')
    if not os.path.exists(dir_val_pts):
        os.makedirs(dir_val_pts)
    dir_val_label = os.path.join(dir_output, 'val', 'category')
    if not os.path.exists(dir_val_label):
        os.makedirs(dir_val_label)
    dir_val_intensity = os.path.join(dir_output, 'val', 'intensity')
    if not os.path.exists(dir_val_intensity):
        os.makedirs(dir_val_intensity)

    train_list = [path.split(".")[0] for path in sorted(os.listdir(dir_pts))]

    count_t = count_v = 0
    for k, filename in enumerate(train_list):
        path_pts = os.path.join(dir_pts, str(filename) + data_ext)
        path_intensity = os.path.join(dir_intensity, str(filename) + data_ext)
        path_label = os.path.join(dir_label, str(filename) + data_ext)
        if random.randint(1, 100) < train_occupancy:
            path_train_pts = os.path.join(dir_train_pts, str(filename) + data_ext)
            path_train_intensity = os.path.join(dir_train_intensity, str(filename) + data_ext)
            path_train_label = os.path.join(dir_train_label, str(filename) + data_ext)
            shutil.copyfile(path_pts, path_train_pts)
            shutil.copyfile(path_intensity, path_train_intensity)
            shutil.copyfile(path_label, path_train_label)
            print(path_pts, path_train_pts)
            count_t += 1
        else:
            path_val_pts = os.path.join(dir_val_pts, str(filename) + data_ext)
            path_val_intensity = os.path.join(dir_val_intensity, str(filename) + data_ext)
            path_val_label = os.path.join(dir_val_label, str(filename) + data_ext)
            shutil.copyfile(path_pts, path_val_pts)
            shutil.copyfile(path_intensity, path_val_intensity)
            shutil.copyfile(path_label, path_val_label)
            print(path_pts, path_val_pts)
            count_v += 1

    print("count_train: %d" % count_t)
    print("count_val: %d" % count_v)


if __name__ == '__main__':
    main()
