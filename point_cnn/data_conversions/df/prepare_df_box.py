import os
import sys
import numpy as np
from utils import df_utils
from utils import vis_utils
from multiprocessing import Process


def bound_point(dir_df, name_frame, max_distance, padding):
    """

    :param dir_df:
    :param name_frame:
    :param max_distance:
    :param padding:
    :return:
    """

    data, categories = df_utils.load_frame(dir_df, name_frame)

    points_instance = []
    # categories_instace = []
    for point, category in zip(data[:, 0:3].tolist(), categories):
        if category != 0:
            points_instance.append(point)
            # categories_instace.append(category)

    i = 0
    boxes_points = []
    box_points = []
    boxes = []
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 0, 0, 0

    # Divide points to boxes
    while len(points_instance) > 0:
        if len(box_points) == 0:
            point = points_instance.pop(0)
            x_min, y_min, z_min = point
            x_max, y_max, z_max = point
            box_points.append(point)
        flag_matched = False
        # print(len(points_instance))
        for point in points_instance[:]:
            for box_point in box_points:
                diff_x = box_point[0] - point[0]
                diff_y = box_point[1] - point[1]
                dis = diff_x * diff_x + diff_y * diff_y
                if dis < max_distance * max_distance:
                    box_points.append(point)
                    points_instance.remove(point)
                    x_min = min(point[0], x_min)
                    y_min = min(point[1], y_min)
                    z_min = min(point[2], z_min)
                    x_max = max(point[0], x_max)
                    y_max = max(point[1], y_max)
                    z_max = max(point[2], z_max)
                    flag_matched = True
                    break
            if flag_matched:
                break
        if not flag_matched or len(points_instance) <= 0:
            i += len(box_points)
            # print(i)
            boxes_points.append(box_points[:])
            box_points.clear()
            x_min -= padding
            y_min -= padding
            z_min -= padding
            x_max += padding
            y_max += padding
            z_max += padding
            boxes.append(((x_min, y_min, z_min), (x_max, y_max, z_max)))
    return boxes


def is_intersecting_rects(rect0, rect1):
    point01, point02 = rect0
    point11, point12 = rect1
    x01, y01 = point01
    x02, y02 = point02
    x11, y11 = point11
    x12, y12 = point12
    zx = abs(x01 + x02 - x11 - x12)     # 两个矩形重心在x轴上的距离的两倍
    x = abs(x01 - x02) + abs(x11 - x12)     # 两矩形在x方向的边长的和
    zy = abs(y01 + y02 - y11 - y12)     # 重心在y轴上距离的两倍
    y = abs(y01 - y02) + abs(y11 - y12)     # y方向边长的和
    if zx <= x and zy <= y:
        return True
    else:
        return False


# Merge intersecting cuboid(rectangle)
def merge_intersecting_boxes(boxes):
    while True:
        flag_intersect = False
        for point0_min, point0_max in boxes:
            x0_min, y0_min, z0_min = point0_min
            x0_max, y0_max, z0_max = point0_max
            rect0 = (x0_min, y0_min), (x0_max, y0_max)
            for point1_min, point1_max in boxes:
                if point0_min == point1_min and point0_max == point1_max:
                    continue
                x1_min, y1_min, z1_min = point1_min
                x1_max, y1_max, z1_max = point1_max
                rect1 = (x1_min, y1_min), (x1_max, y1_max)
                if is_intersecting_rects(rect0, rect1):
                    flag_intersect = True
                    boxes.remove((point0_min, point0_max))
                    boxes.remove((point1_min, point1_max))
                    # merge box
                    x_min = min(x0_min, x1_min)
                    y_min = min(y0_min, y1_min)
                    z_min = min(z0_min, z1_min)
                    x_max = max(x0_max, x1_max)
                    y_max = max(y0_max, y1_max)
                    z_max = max(z0_max, z1_max)
                    boxes.append(((x_min, y_min, z_min), (x_max, y_max, z_max)))
                    break
            if flag_intersect:
                break
        if not flag_intersect:
            break
    return boxes


def vis_box(path_out, boxes):
    """
    Visualization
    :param path_out:
    :return:
    """

    vertexes = []
    faces = []
    for id_box, point_min_max in enumerate(boxes):
        x_min, y_min, z_min = point_min_max[0]
        x_max, y_max, z_max = point_min_max[1]
        w_box = y_max - y_min
        l_box = x_max - x_min
        vertexes.extend([[x_min, y_min, z_min],
                         [x_min + l_box, y_min, z_min],
                         [x_min + l_box, y_min + w_box, z_min],
                         [x_min, y_min + w_box, z_min],
                         [x_max - l_box, y_max - w_box, z_max],
                         [x_max, y_max - w_box, z_max],
                         [x_max, y_max, z_max],
                         [x_max - l_box, y_max, z_max]])
        id_vertex = id_box * 8
        faces.extend([[id_vertex + 0, id_vertex + 4, id_vertex + 5], [id_vertex + 0, id_vertex + 1, id_vertex + 5],
                      [id_vertex + 1, id_vertex + 5, id_vertex + 6], [id_vertex + 1, id_vertex + 2, id_vertex + 6],
                      [id_vertex + 2, id_vertex + 6, id_vertex + 7], [id_vertex + 2, id_vertex + 3, id_vertex + 7],
                      [id_vertex + 3, id_vertex + 7, id_vertex + 4], [id_vertex + 3, id_vertex + 0, id_vertex + 4],
                      [id_vertex + 0, id_vertex + 1, id_vertex + 2], [id_vertex + 0, id_vertex + 3, id_vertex + 2],
                      [id_vertex + 4, id_vertex + 5, id_vertex + 6], [id_vertex + 4, id_vertex + 7, id_vertex + 6]])

    vis_utils.save_ply(path_out, vertexes, vis_utils.seg2color(np.zeros(len(vertexes), dtype=int)), faces)


def save_bbox(path, bboxes):
    f = open(path, 'w')
    for point_min, point_max in bboxes:
        x_min, y_min, z_min = point_min
        x_max, y_max, z_max = point_max
        f.write("%f %f %f %f %f %f\n" % (x_min, y_min, z_min, x_max, y_max, z_max))


def prepare_df_box(dir_df, framenames, dir_out):
    i = 0
    for framename in framenames:
        i += 1
        print(os.getpid(), i, framename)
        boxes = bound_point(dir_df, framename, max_distance, padding)
        boxes_merged = merge_intersecting_boxes(boxes)

        path_bbox = os.path.join(dir_out, "bbox", framename)
        save_bbox(path_bbox, boxes_merged)

        # path_vis = os.path.join(dir_out, "vis_bbox", framename[:-3] + 'ply')
        # vis_box(path_vis, boxes_merged)


if __name__ == '__main__':
    # print(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
    # os.chdir(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
    num_prccess = 16
    dir_input = "/home/leon/Disk/datasets/data_fountain/trainval/train"
    dir_output = "/home/leon/Disk/datasets/data_fountain/trainval/train"
    max_distance = 1.2
    padding = 0.3

    dir_bbox = os.path.join(dir_output, 'bbox')
    dir_vis_bbox = os.path.join(dir_output, 'vis_bbox')
    if not os.path.exists(dir_bbox):
        os.makedirs(dir_bbox)
    if not os.path.exists(dir_vis_bbox):
        os.makedirs(dir_vis_bbox)

    framenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    # prepare_df_box(dir_input, framenames, dir_output)
    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=prepare_df_box, args=(dir_input, chunk_filepaths, dir_output))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()






