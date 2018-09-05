import os

from data_utils.kitti import kitti_utils

if __name__ == '__main__':
    dir_bin = '/home/leon/Disk/dataset/Downloads/KITTI/data_object_velodyne/training/velodyne'
    dir_pts = '/home/leon/Disk/dataset/KITTI/kitti_seg/train_data'

    files = os.listdir(dir_bin)
    for file in files:
        if not os.path.isdir(file):
            path_in = '/home/leon/Disk/dataset/Downloads/KITTI/data_object_velodyne/training/velodyne/000004.bin'
            path_out = './out/000004.obj'
            pts = kitti_utils.load_bin(path_in)
            kitti_utils.save_pts_as_obj(pts, path_out)
            break
