import os
import numpy as np
import datetime
from data_utils.data_fountain import df_thread


def load_frame(dir_data, filename):
    """
    load a frame of data_fountain dataset
    :param dir_data: dataset dir
    :param filename: frame name
    :return: (points, intensities, categories) of the frame
    """
    # load points
    path = os.path.join(dir_data, 'pts', filename)
    file = open(path, 'r')
    lines = file.readlines()
    points = np.empty((len(lines), 3), dtype=np.float32)
    for i, line in enumerate(lines):
        point_eles = line.strip('\n').split(',')
        points[i] = np.array([np.float32(ele) for ele in point_eles])

    # load intensities
    path = os.path.join(dir_data, 'intensity', filename)
    intensities = np.loadtxt(path).astype(np.float32)

    # load categories, no categories if load test set
    path = os.path.join(dir_data, 'category', filename)
    if os.path.exists(path):
        categories = np.loadtxt(path).astype(np.int32)
    else:
        categories = None

    return points, intensities, categories


def split_frame_to_quadrants(points, intensities, categories):
    """
    split a frame to 4 quadrants
    :param points: point array
    :param intensities: intensity array
    :param categories: categories array
    :return: 4 quadrants' point,intensity and categories
    """
    quadrants_points = []
    quadrants_intensities = []
    if categories is None:
        quadrants_categories = None
    else:
        quadrants_categories = []
    quadrants_indices = []  # index for restoration from quadrant to frame

    for i in range(4):
        quadrants_points.append([])
        quadrants_intensities.append([])
        if categories is not None:
            quadrants_categories.append([])
        quadrants_indices.append([])

    for i, point in enumerate(points):
        x, y, z = point
        if x >= 0 and y > 0:
            id_quadrant = 0
            quadrants_points[id_quadrant].append([abs(x), abs(y), z])
            quadrants_intensities[id_quadrant].append(intensities[i])
            if categories is not None:
                quadrants_categories[id_quadrant].append(categories[i])
            quadrants_indices[id_quadrant].append(i)
        elif x < 0 and y >= 0:
            id_quadrant = 1
            quadrants_points[id_quadrant].append([abs(x), abs(y), z])
            quadrants_intensities[id_quadrant].append(intensities[i])
            if categories is not None:
                quadrants_categories[id_quadrant].append(categories[i])
            quadrants_indices[id_quadrant].append(i)
        elif x <= 0 and y < 0:
            id_quadrant = 2
            quadrants_points[id_quadrant].append([abs(x), abs(y), z])
            quadrants_intensities[id_quadrant].append(intensities[i])
            if categories is not None:
                quadrants_categories[id_quadrant].append(categories[i])
            quadrants_indices[id_quadrant].append(i)
        elif x > 0 and y <= 0:
            id_quadrant = 3
            quadrants_points[id_quadrant].append([abs(x), abs(y), z)])
            quadrants_intensities[id_quadrant].append(intensities[i])
            if categories is not None:
                quadrants_categories[id_quadrant].append(categories[i])
            quadrants_indices[id_quadrant].append(i)

    return quadrants_points, quadrants_intensities, quadrants_categories, quadrants_indices


def compute_frames_quadrants_max_point_num(filepaths):
    max_point_num = 0
    for filepath in filepaths:
        starttime = datetime.datetime.now()
        print(os.path.basename(filepath))
        frame_quadrants_point_num = []
        for i in range(4):
            frame_quadrants_point_num.append(0)
        starttime_open = datetime.datetime.now()
        file = open(filepath, 'r')
        lines = file.readlines()
        endtime_open = datetime.datetime.now()
        print("open time: %f" % (endtime_open - starttime_open).seconds)
        for line in lines:
            point_eles = line.strip('\n').split(',')
            x, y, z = [float(ele) for ele in point_eles]
            if x >= 0 and y > 0:
                id_quadrant = 0
                frame_quadrants_point_num[id_quadrant] += 1
            elif x < 0 and y >= 0:
                id_quadrant = 1
                frame_quadrants_point_num[id_quadrant] += 1
            elif x <= 0 and y < 0:
                id_quadrant = 2
                frame_quadrants_point_num[id_quadrant] += 1
            elif x > 0 and y <= 0:
                id_quadrant = 3
                frame_quadrants_point_num[id_quadrant] += 1

        max_point_num = max(max_point_num, max(frame_quadrants_point_num))
        endtime = datetime.datetime.now()
        print("compute_frames_quadrants_max_point_num time: %f" % (endtime - starttime).seconds)

    return max_point_num


def compute_frames_quadrants_max_point_num_multi_thread(dir_data, thread_num=8):
    filenames = sorted(os.listdir(os.path.join(dir_data, 'pts')))
    filepaths = [os.path.join(dir_data, 'pts', filename) for filename in filenames]
    chunk_size = len(filepaths) // thread_num
    chunks_filepaths = [filepaths[i:i + chunk_size] for i in range(0, len(filepaths), chunk_size)]

    tasks = []
    for chunk_filepaths in chunks_filepaths:
        task = df_thread.DfThread(compute_frames_quadrants_max_point_num, chunk_filepaths)
        tasks.append(task)
        task.start()

    # get max result
    max_point_num = 0
    for task in tasks:
        rul = task.get_result()
        max_point_num = max(rul, max_point_num)

    return max_point_num
