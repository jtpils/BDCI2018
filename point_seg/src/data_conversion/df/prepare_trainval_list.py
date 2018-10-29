import numpy as np
import os
from random import sample


dir_input = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
dir_output = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"

framenames = [name.split('.')[0]
              for name in sorted(os.listdir(os.path.join(dir_input, 'df_2d_32_1024')))]
framenames = np.array(framenames)

val_num = 10000

indices = sample(range(framenames.shape[0]), val_num)

val_framenames = framenames[indices]
train_framenames = np.delete(framenames, indices)

path_val = os.path.join(dir_output, 'val.txt')
path_train = os.path.join(dir_output, 'train.txt')
f = open(path_val, 'w')
for name in val_framenames:
    f.write(name + '\n')
f = open(path_train, 'w')
for name in train_framenames:
    f.write(name + '\n')





