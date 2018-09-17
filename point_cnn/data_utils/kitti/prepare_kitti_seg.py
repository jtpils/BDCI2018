import math
import os
from data_utils.kitti import kitti_utils


def save_bbox_obj(path, pts, bboxs):
    with open(path, "w") as f:
        for i, bbox in enumerate(bboxs):
            x_min = bbox[0]
            y_min = bbox[1]
            z_min = bbox[2]
            x_max = bbox[3]
            y_max = bbox[4]
            z_max = bbox[5]
            l = x_max - x_min
            w = y_max - y_min

            box_points = [[x_min, y_min, z_min],
                          [x_min + l, y_min, z_min],
                          [x_min + l, y_min + w, z_min],
                          [x_min, y_min + w, z_min],
                          [x_max, y_max, z_max],
                          [x_max - l, y_max, z_max],
                          [x_max - l, y_max - w, z_max],
                          [x_max, y_max - w, z_max]]

            for p in box_points:
                f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
            f.writelines('l ' + str(i * 8 + 1) + ' ' + str(i * 8 + 2) + '\n')
            f.writelines('l ' + str(i * 8 + 2) + ' ' + str(i * 8 + 3) + '\n')
            f.writelines('l ' + str(i * 8 + 3) + ' ' + str(i * 8 + 4) + '\n')
            f.writelines('l ' + str(i * 8 + 4) + ' ' + str(i * 8 + 1) + '\n')
            f.writelines('l ' + str(i * 8 + 5) + ' ' + str(i * 8 + 6) + '\n')
            f.writelines('l ' + str(i * 8 + 6) + ' ' + str(i * 8 + 7) + '\n')
            f.writelines('l ' + str(i * 8 + 7) + ' ' + str(i * 8 + 8) + '\n')
            f.writelines('l ' + str(i * 8 + 8) + ' ' + str(i * 8 + 5) + '\n')
            f.writelines('l ' + str(i * 8 + 1) + ' ' + str(i * 8 + 7) + '\n')
            f.writelines('l ' + str(i * 8 + 2) + ' ' + str(i * 8 + 8) + '\n')
            f.writelines('l ' + str(i * 8 + 3) + ' ' + str(i * 8 + 5) + '\n')
            f.writelines('l ' + str(i * 8 + 4) + ' ' + str(i * 8 + 6) + '\n')
            for p in pts:
                f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")


def clockwiserot(pt,pt_c,deg):

    rx = pt[0] - pt_c[0]
    ry = pt[1] - pt_c[1]
    deg = -deg

    rx_n = math.cos(deg)*rx-math.sin(deg)*ry
    ry_n = math.cos(deg)*ry+math.sin(deg)*rx

    return [rx_n + pt_c[0],ry_n+pt_c[1]]


if __name__ == '__main__':
    dir_bin = '/home/leon/Disk/dataset/Downloads/KITTI/data_object_velodyne/training/velodyne'
    dir_label = '/home/leon/Disk/dataset/Downloads/KITTI/data_object_label_2/training/label_2'

    files = sorted(os.listdir(dir_bin))
    for i, file in enumerate(files):
        if not os.path.isdir(file):
            print(file)
            file_name = file.split('.')[0]
            path_bin = os.path.join(dir_bin, file)
            pts = kitti_utils.load_bin(path_bin)

            path_label = os.path.join(dir_label, file_name + '.txt')
            labels = kitti_utils.load_labels(path_label)

            bboxs = []
            for label in labels:
                x, y, z = label.location
                height, width, length = label.dimensions
                ry = label.rotation_y
                # print(x, y, z, h, w, l)
                y_min = y - height - 0.8
                y_max = y - 0.8

                pt_c = [x, z]
                box_bottom_p_max = clockwiserot([x + length / 2, z + width / 2], pt_c, ry)
                box_bottom_p_min = clockwiserot([x - length / 2, z - width / 2], pt_c, ry)

                l_cx = x
                l_cz = z
                l_w = width
                l_l = length
                bbp1 = clockwiserot([l_cx - l_l / 2, l_cz + l_w / 2], pt_c, ry)
                bbp2 = clockwiserot([l_cx + l_l / 2, l_cz + l_w / 2], pt_c, ry)
                bbp3 = clockwiserot([l_cx + l_l / 2, l_cz - l_w / 2], pt_c, ry)
                bbp4 = clockwiserot([l_cx - l_l / 2, l_cz - l_w / 2], pt_c, ry)
                # bboxs.append((bbp1[0], y_min, bbp1[1],
                #               bbp3[0], y_max, bbp3[1]))
                # x_min, y_min, z_min = label.location
                # z_min = 0
                # x_max = x_min + width
                # y_max = y_min + width
                # z_max = z_min + width

                bboxs.append((box_bottom_p_min[1], -box_bottom_p_min[0], y_min,
                              box_bottom_p_max[1], -box_bottom_p_max[0], y_max))
            save_bbox_obj("./out/" + file_name + ".obj", pts, bboxs)
            if i == 10:
                break
