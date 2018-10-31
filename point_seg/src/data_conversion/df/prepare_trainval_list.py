import numpy as np
import os
from random import sample


dir_input = "/home/leon/Disk/dataset/DataFountain/training_cleared/data_bin_fru"
dir_output = "/home/leon/Disk/dataset/DataFountain/training_cleared"


def make_train_val_list():
    framenames = [name.split('.')[0]
                  for name in sorted(os.listdir(dir_input))]
    framenames = np.array(framenames)

    val_num = 40000

    indices = sample(range(framenames.shape[0]), val_num)

    val_framenames = framenames[indices]
    train_framenames = np.delete(framenames, indices)

    path_val = os.path.join(dir_output, 'val.txt')
    path_train = os.path.join(dir_output, 'train.txt')
    f = open(path_val, 'w')
    for name in val_framenames:
        f.write(name + '\n')
    f.close()
    f = open(path_train, 'w')
    for name in train_framenames:
        f.write(name + '\n')
    f.close()


def combine_trainval_list():
    path_val = os.path.join(dir_output, 'val.txt')
    path_train = os.path.join(dir_output, 'train.txt')
    with open(path_train, 'r') as f:
        list_train = f.readlines()
    with open(path_val, 'r') as f:
        list_val = f.readlines()

    path_all = os.path.join(dir_output, 'trainall.txt')
    with open(path_all, 'w') as f:
        f.writelines(list_train)
        f.writelines(list_val)


if __name__ == '__main__':
    combine_trainval_list()




