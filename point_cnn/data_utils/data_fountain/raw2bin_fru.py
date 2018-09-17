import argparse
import datetime
import numpy as np
import os
import shutil


def prepare(args):
    dir_input = args.dir_input
    dir_output = args.dir_output
    max_sample_num = args.max_samples
    data_ext = '.csv'

    dir_pts = os.path.join(dir_input, 'pts')
    train_list = [path.split(".")[0] for path in sorted(os.listdir(dir_pts))]

    count_point_ele = 0
    count_quadrant = 0

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        print("mkdir:", dir_output)
    f_log = open(os.path.join(dir_output, "raw2bin_fru_check.log"), 'w')

    for k, filename in enumerate(train_list):
        if 0 < max_sample_num <= k:
            break
        filename = str(filename)
        print("checking:", filename)

        points_q1 = []
        points_q2 = []
        points_q3 = []
        points_q4 = []

        data_file_path = os.path.join(dir_pts, filename + data_ext)
        f_data = open(data_file_path, 'r')

        for i, xyz in enumerate(f_data.readlines()):
            ele_xyz = xyz.split(',')
            x = float(ele_xyz[0])
            y = float(ele_xyz[1])
            z = float(ele_xyz[2])
            if x >= 0 and y > 0:
                points_q1.extend([x, y, z])
            elif x < 0 and y >= 0:
                points_q2.extend([x, y, z])
            elif x <= 0 and y < 0:
                points_q3.extend([x, y, z])
            elif x > 0 and y <= 0:
                points_q4.extend([x, y, z])

        if len(points_q1) > 0:
            count_point_ele += len(points_q1)
            count_quadrant += 1
        else:
            print("q1 no points")
            f_log.write("q1 no points:" + filename + '\n')

        if len(points_q2) > 0:
            count_point_ele += len(points_q2)
            count_quadrant += 1
        else:
            print("q2 no points")
            f_log.write("q2 no points:" + filename + '\n')

        if len(points_q3) > 0:
            count_point_ele += len(points_q3)
            count_quadrant += 1
        else:
            print("q3 no points")
            f_log.write("q3 no points:" + filename + '\n')

        if len(points_q4) > 0:
            count_point_ele += len(points_q4)
            count_quadrant += 1
        else:
            print("q4 no points")
            f_log.write("q4 no points:" + filename + '\n')

    f_log.write("count_point_ele: %d\n" % count_point_ele)
    f_log.write("count_quadrant: %d\n" % count_quadrant)

    return count_point_ele, count_quadrant


def main(args):
    all_point_ele_num, all_quadrant_num = prepare(args)
    dir_input = args.dir_input
    dir_output = args.dir_output
    max_sample_num = args.max_samples
    data_ext = '.csv'

    dir_pts = os.path.join(dir_input, 'pts')
    dir_label = os.path.join(dir_input, 'category')
    dir_intensity = os.path.join(dir_input, 'intensity')
    train_list = [path.split(".")[0] for path in sorted(os.listdir(dir_pts))]

    points_ele_all = np.zeros(all_point_ele_num, np.float32)
    intensities = np.zeros(all_point_ele_num // 3, np.float32)
    point_nums = np.zeros(all_quadrant_num, np.uint16)
    labels = np.zeros(all_point_ele_num // 3, np.uint8)

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        print("mkdir:", dir_output)
    f_log = open(os.path.join(dir_output, "raw2bin_fru.log"), 'w')

    id_quadrant = 0
    id_point_ele = 0
    id_point = 0

    for k, filename in enumerate(train_list):
        if 0 < max_sample_num <= k:
            break
        filename = str(filename)
        print("processing:", filename)

        points_q1 = []
        points_q2 = []
        points_q3 = []
        points_q4 = []
        intensities_q1 = []
        intensities_q2 = []
        intensities_q3 = []
        intensities_q4 = []
        labels_q1 = []
        labels_q2 = []
        labels_q3 = []
        labels_q4 = []

        data_file_path = os.path.join(dir_pts, filename + data_ext)
        f_data = open(data_file_path, 'r')
        intensity_path = os.path.join(dir_intensity, filename + data_ext)
        intensity_this = np.loadtxt(intensity_path).astype(np.float32)
        label_file_path = os.path.join(dir_label, filename + data_ext)
        label_seg_this = np.loadtxt(label_file_path).astype(np.int32)
        for i, xyz in enumerate(f_data.readlines()):
            ele_xyz = xyz.split(',')
            x = float(ele_xyz[0])
            y = float(ele_xyz[1])
            z = float(ele_xyz[2])
            if x >= 0 and y > 0:
                points_q1.extend([x, y, z])
                intensities_q1.append(intensity_this[i])
                labels_q1.append(label_seg_this[i])
            elif x < 0 and y >= 0:
                points_q2.extend([x, y, z])
                intensities_q2.append(intensity_this[i])
                labels_q2.append(label_seg_this[i])
            elif x <= 0 and y < 0:
                points_q3.extend([x, y, z])
                intensities_q3.append(intensity_this[i])
                labels_q3.append(label_seg_this[i])
            elif x > 0 and y <= 0:
                points_q4.extend([x, y, z])
                intensities_q4.append(intensity_this[i])
                labels_q4.append(label_seg_this[i])

        if len(points_q1) > 0:
            point_num_this = len(points_q1) // 3
            points_ele_all[id_point_ele:id_point_ele + len(points_q1)] = np.array(points_q1).astype(np.float32)
            intensities[id_point:id_point + point_num_this] = np.array(intensities_q1).astype(np.float32)
            labels[id_point:id_point + point_num_this] = np.array(labels_q1).astype(np.uint8)
            point_nums[id_quadrant] = np.array(point_num_this).astype(np.uint16)

            id_point_ele += len(points_q1)
            id_point += point_num_this
            id_quadrant += 1
        else:
            print("q1 no points")
            f_log.write("q1 no points:" + filename + '\n')

        if len(points_q2) > 0:
            point_num_this = len(points_q2) // 3
            points_ele_all[id_point_ele:id_point_ele + len(points_q2)] = np.array(points_q2).astype(np.float32)
            intensities[id_point:id_point + point_num_this] = np.array(intensities_q2).astype(np.float32)
            labels[id_point:id_point + point_num_this] = np.array(labels_q2).astype(np.uint8)
            point_nums[id_quadrant] = point_num_this

            id_point_ele += len(points_q2)
            id_point += point_num_this
            id_quadrant += 1
        else:
            print("q2 no points")
            f_log.write("q2 no points:" + filename + '\n')

        if len(points_q3) > 0:
            point_num_this = len(points_q3) // 3
            points_ele_all[id_point_ele:id_point_ele + len(points_q3)] = np.array(points_q3).astype(np.float32)
            intensities[id_point:id_point + point_num_this] = np.array(intensities_q3).astype(np.float32)
            labels[id_point:id_point + point_num_this] = np.array(labels_q3).astype(np.uint8)
            point_nums[id_quadrant] = point_num_this

            id_point_ele += len(points_q3)
            id_point += point_num_this
            id_quadrant += 1
        else:
            print("q3 no points")
            f_log.write("q3 no points:" + filename + '\n')

        if len(points_q4) > 0:
            point_num_this = len(points_q4) // 3
            points_ele_all[id_point_ele:id_point_ele + len(points_q4)] = np.array(points_q4).astype(np.float32)
            intensities[id_point:id_point + point_num_this] = np.array(intensities_q4).astype(np.float32)
            labels[id_point:id_point + point_num_this] = np.array(labels_q4).astype(np.uint8)
            point_nums[id_quadrant] = point_num_this

            id_point_ele += len(points_q4)
            id_point += point_num_this
            id_quadrant += 1
        else:
            print("q4 no points")
            f_log.write("q4 no points:" + filename + '\n')

    f_log.write("max point num: %d\n" % np.max(point_nums))
    f_log.write("min point num: %d\n" % np.min(point_nums))
    f_log.close()

    bin_point_path = os.path.join(dir_output, "data_fountain_point.bin")
    print("save to: %s\n" % bin_point_path)
    points_ele_all.tofile(bin_point_path)

    bin_point_num_path = os.path.join(dir_output, "data_fountain_point_num.bin")
    print("save to: %s\n" % bin_point_num_path)
    point_nums.tofile(bin_point_num_path)

    bin_intensity_path = os.path.join(dir_output, "data_fountain_intensity.bin")
    print("save to: %s\n" % bin_intensity_path)
    intensities.tofile(bin_intensity_path)

    bin_label_path = os.path.join(dir_output, "data_fountain_label.bin")
    print("save to: %s\n" % bin_label_path)
    labels.tofile(bin_label_path)


def test(args):
    print("test")
    dir_output = args.dir_output

    bin_path = os.path.join(dir_output, "data_fountain_point.bin")
    points_all = np.fromfile(bin_path, np.float32)
    print(points_all[0])
    print(points_all[1])

    bin_points_nums_path = os.path.join(dir_output, "data_fountain_point_num.bin")
    points_nums = np.fromfile(bin_points_nums_path, np.uint16)
    print(points_nums.dtype, points_nums[0])
    print(points_nums.dtype, points_nums[1])

    bin_path = os.path.join(dir_output, "data_fountain_intensity.bin")
    points_all = np.fromfile(bin_path, np.float32)
    print(points_all[0])
    print(points_all[1])

    bin_path = os.path.join(dir_output, "data_fountain_label.bin")
    points_all = np.fromfile(bin_path, np.uint8)
    print(points_all[0])
    print(points_all[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', '-i', help='Path to data folder', required=True)
    parser.add_argument('--dir_output', '-o', help='Path to h5 folder', required=True)
    parser.add_argument('--max_samples', '-m', default=-1, type=int, help='The max num of sample', required=False)
    parser.add_argument('--min_label_seg', '-l', default=-1, type=int, help='The min num of label_seg', required=True)
    args = parser.parse_args()
    print(args)

    main(args)
    test(args)
