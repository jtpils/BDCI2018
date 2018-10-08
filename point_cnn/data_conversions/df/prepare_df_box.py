import os
from utils import df_utils

dir_pts = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
name_pts = "ffeeaf44-77d1-4f08-9d3d-b3710445b99f_channelVELO_TOP.csv"
dir_pts_box = "./out"

if not os.path.exists(dir_pts_box):
    os.makedirs(dir_pts_box)

max_distance = 0.01
num_parts = 8

data, categories = df_utils.load_frame(dir_pts, name_pts)
for point, category in zip(data[:, 0:2], categories):
    x, y = point

    print(point, category)