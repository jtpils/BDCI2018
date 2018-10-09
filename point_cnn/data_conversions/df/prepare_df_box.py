import os
import numpy as np
from utils import df_utils
from utils import vis_utils

dir_pts = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
name_pts = "000f3ebc-50da-4b20-b3f1-e2d9759a0fa6_channelVELO_TOP.csv"
dir_pts_box = "./out"

if not os.path.exists(dir_pts_box):
    os.makedirs(dir_pts_box)

max_distance = 1
# padding = 0.1
num_parts = 8

data, categories = df_utils.load_frame(dir_pts, name_pts)

points_instance = []
# categories_instace = []
for point, category in zip(data[:, 0:3].tolist(), categories):
    if category != 0:
        points_instance.append(point)
        # categories_instace.append(category)

print(len(points_instance))
i = 0
boxes_points = []
box_points = []

while len(points_instance) > 0:
    if len(box_points) == 0:
        # i += 1
        # print(i)
        box_points.append(points_instance.pop(0))
    flag_matched = False
    # print(len(points_instance))
    for point in points_instance[:]:
        for box_point in box_points:
            diff_x = box_point[0] - point[0]
            diff_y = box_point[1] - point[1]
            dis = diff_x * diff_x + diff_y * diff_y
            if dis < max_distance * max_distance:
                # i += 1
                # print(i)
                box_points.append(point)
                points_instance.remove(point)
                flag_matched = True
                break
        if flag_matched:
            break
    if not flag_matched:
        i += len(box_points)
        print(i)
        boxes_points.append(box_points[:])
        box_points.clear()
boxes_points.append(box_points[:])

x_min, y_min, z_min = 0, 0, 0
x_max, y_max, z_max = 0, 0, 0
vertexes = []
faces = []
count_point = 0
for id_box, box_points in enumerate(boxes_points):
    for id_point, box_point in enumerate(box_points):
        count_point += 1
        if id_point == 0:
            x_min, y_min, z_min = box_point
            x_max, y_max, z_max = box_point
        x_min = min(box_point[0], x_min)
        y_min = min(box_point[1], y_min)
        z_min = min(box_point[2], z_min)
        x_max = max(box_point[0], x_max)
        y_max = max(box_point[1], y_max)
        z_max = max(box_point[2], z_max)
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

print(count_point)
path_out = os.path.join(dir_pts_box, name_pts[:-3] + 'ply')
vis_utils.save_ply(path_out, vertexes, vis_utils.seg2color(np.zeros(len(vertexes), dtype=int)), faces)