import argparse
import datetime

import h5py as h5py
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', '-i', help='Path to data folder', required=True)
    parser.add_argument('--dir_output', '-o', help='Path to h5 folder', required=True)
    parser.add_argument('--max_samples', '-m', default='-1', help='The max num of sample', required=True)
    args = parser.parse_args()
    print(args)

    dir_input = args.dir_input
    dir_output = args.dir_output
    max_sample_num = int(args.max_samples)
    dim = 3
    data_ext = '.csv'

    dir_pts = os.path.join(dir_input, 'pts')
    dir_intensity = os.path.join(dir_input, 'intensity')
    train_list = [path.split(".")[0] for path in sorted(os.listdir(dir_pts))]

    f_log = open('/home/leon/Disk/dataset/DataFountain/testset/rawtest2h5.log', 'w')
    point_num_list = []
    for k, filename in enumerate(train_list):
        if 0 < max_sample_num <= k:
            break
        print("parpering:", filename)

        data_file_path = os.path.join(dir_pts, filename + data_ext)
        coordinates = [xyz for xyz in open(data_file_path, 'r') if len(xyz.split(',')) == dim]
        point_num = len(coordinates)
        point_num_list.append(point_num)

        intensity_file_path = os.path.join(dir_intensity, filename + data_ext)
        intensity_this = np.loadtxt(intensity_file_path).astype(np.float32)
        if len(intensity_this) != point_num:
            f_log.write(filename + '\n')
            print("point num don't match to intensity_this num.")

    max_point_num = max(point_num_list)
    f_log.write('max_point_num: %d\n' % max_point_num)
    f_log.write('min_point_num: %d\n' % min(point_num_list))
    print("point_num_max", max(point_num_list))
    print("point_num_min", min(point_num_list))

    f_log.close()

    # h5 file batch
    batch_size = 2048
    data = np.zeros((batch_size, max_point_num, dim), dtype=np.float32)
    data_num = np.zeros(batch_size, dtype=np.int32)
    intensity = np.zeros((batch_size, max_point_num), dtype=np.float32)

    file_num = len(os.listdir(dir_pts))
    idx_h5 = 0
    idx = 0

    save_path = '%s/%s' % (dir_output, "data_fountain")
    filename_txt = '%s_files.txt' % save_path
    file_list = open(filename_txt, 'w')

    for k, filename in enumerate(train_list):
        print("processing:", filename)

        data_file_path = os.path.join(dir_pts, filename + data_ext)
        idx_in_batch = idx % batch_size

        # point cloud and extra features
        coordinates = [[float(value) for value in xyz.split(',')] for xyz in open(data_file_path, 'r') if
                       len(xyz.split(',')) == dim]

        data[idx_in_batch, 0:len(coordinates), ...] = np.array(coordinates)
        data_num[idx_in_batch] = len(coordinates)

        # intensity
        intensity_path = os.path.join(dir_intensity, filename + data_ext)
        intensity_this = np.loadtxt(intensity_path).astype(np.float32)
        if len(coordinates) != intensity_this.shape[0]:
            continue
        intensity[idx_in_batch, 0:len(coordinates)] = intensity_this

        # save h5 file
        if ((idx + 1) % batch_size == 0) or idx == file_num - 1 or (max_sample_num > 0 and k >= max_sample_num - 1):
            item_num = idx_in_batch + 1
            filename_h5 = '%s_%d.h5' % (save_path, idx_h5)
            print('{}-Saving {}...'.format(datetime.datetime.now(), filename_h5))
            file_list.write('%s/%s_%d.h5\n' % (dir_output, os.path.basename(save_path), idx_h5))

            file = h5py.File(filename_h5, 'w')
            file.create_dataset('data', data=data[0:item_num, ...])
            file.create_dataset('data_num', data=data_num[0:item_num, ...])
            file.create_dataset('intensity', data=intensity[0:item_num, ...])

            file.close()

            idx_h5 = idx_h5 + 1

        if max_sample_num > 0 and k >= max_sample_num - 1:
            break
        idx = idx + 1

    file_list.close()


if __name__ == '__main__':
    main()
