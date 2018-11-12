import os
import numpy as np
import datetime


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
    data = np.empty((len(lines), 4), dtype=np.float32)
    for i, line in enumerate(lines):
        point_eles = line.strip('\n').split(',')
        data[i, 0:3] = np.array([np.float32(ele) for ele in point_eles])

    # load intensities
    path = os.path.join(dir_data, 'intensity', filename)
    file = open(path, 'r')
    lines = file.readlines()
    for i, line in enumerate(lines):
        data[i, 3] = np.float32(line.strip('\n'))

    # load categories, no categories if load test set
    path = os.path.join(dir_data, 'category', filename)
    if os.path.exists(path):
        categories = np.loadtxt(path).astype(np.int32)
    else:
        categories = None

    return data, categories


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
            quadrants_points[id_quadrant].append([abs(x), abs(y), z])
            quadrants_intensities[id_quadrant].append(intensities[i])
            if categories is not None:
                quadrants_categories[id_quadrant].append(categories[i])
            quadrants_indices[id_quadrant].append(i)

    return quadrants_points, quadrants_intensities, quadrants_categories, quadrants_indices


def split_frame_to_frus(pts_ins, categories):
    """
    split a frame to 4 frustums
    :param frame_data: [[x, y, z, ins, mask], ...]
    :return: 4 frustums' points,intensities and categories
    """
    frus_pts_ins = []
    frus_indices = []  # index for restoration from quadrant to frame
    if categories is None:
        frus_categories = None
    else:
        frus_categories = []

    for i in range(4):
        frus_pts_ins.append([])
        frus_indices.append([])
        if categories is not None:
            frus_categories.append([])

    for i, p in enumerate(pts_ins):
        x, y, z, iny = p

        if y > abs(x):
            id_fru = 0
            x_rotated = y
            y_rotated = -x
            frus_pts_ins[id_fru].append(np.array([x_rotated, y_rotated, z, iny]))
            frus_indices[id_fru].append(i)
            if categories is not None:
                frus_categories[id_fru].append(categories[i])

        elif y < -abs(x):
            id_fru = 1
            x_rotated = -y
            y_rotated = x
            frus_pts_ins[id_fru].append(np.array([x_rotated, y_rotated, z, iny]))
            frus_indices[id_fru].append(i)
            if categories is not None:
                frus_categories[id_fru].append(categories[i])

        elif x <= -abs(y):
            id_fru = 2
            x_rotated = -x
            frus_pts_ins[id_fru].append(np.array([x_rotated, y, z, iny]))
            frus_indices[id_fru].append(i)
            if categories is not None:
                frus_categories[id_fru].append(categories[i])

        elif x >= abs(y):
            id_fru = 3
            frus_pts_ins[id_fru].append(np.array([x, y, z, iny]))
            frus_indices[id_fru].append(i)
            if categories is not None:
                frus_categories[id_fru].append(categories[i])

    return frus_pts_ins, frus_categories, frus_indices


def clear_data(pts_ins, categories):
    pts_ins_cleared = []
    categories_cleared = []
    scene_indices = []
    if categories is None:
        for i, p in enumerate(pts_ins):
            if not (abs(p[0]) < 0.5 and abs(p[1]) < 0.5 and abs(p[2]) < 0.5):
                pts_ins_cleared.append(p)
                scene_indices.append(i)
    else:
        i = 0
        for p, c in zip(pts_ins, categories):
            if not (abs(p[0]) < 0.5 and abs(p[1]) < 0.5 and abs(p[2]) < 0.5):
                pts_ins_cleared.append(p)
                categories_cleared.append(c)
                scene_indices.append(i)
                i += 1

    return pts_ins_cleared, categories_cleared, scene_indices


def clear_data_2(pts_ins, categories):
    pts_ins_cleared = []
    categories_cleared = []
    scene_indices = []
    if categories is None:
        for i, p in enumerate(pts_ins):
            # if abs(p[0]) < 0.5 and abs(p[1]) < 0.5 and abs(p[2]) < 0.5:
            #     print("ins", p[3])
            if not (abs(p[0]) < 2 and abs(p[1]) < 2 and abs(p[2]) < 2 and p[3] < 0.2):
                pts_ins_cleared.append(p)
                scene_indices.append(i)
    else:
        i = 0
        for p, c in zip(pts_ins, categories):
            if not (abs(p[0]) < 2 and abs(p[1]) < 2 and abs(p[2]) < 2 and p[3] < 0.2):
                pts_ins_cleared.append(p)
                categories_cleared.append(c)
                scene_indices.append(i)
                i += 1

    return pts_ins_cleared, categories_cleared, scene_indices


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


def group_sampling(pts_fts, labels, label_weights, sample_num, pts_nums):
    batch_size = pts_fts.shape[0]
    pts_fts_sampled = []
    labels_sampled = []
    label_weights_sampled = []

    for i in range(batch_size):
        pt_num = pts_nums[i]
        pool_size = pt_num

        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]

        pts_fts_sampled.append(pts_fts[i][choices, ...])
        labels_sampled.append(labels[i][choices, ...])
        label_weights_sampled.append(label_weights[i][choices, ...])

    return np.array(pts_fts_sampled), np.array(labels_sampled), np.array(label_weights_sampled)


def group_sampling_fru(fru_batch, sample_num, label_weights_list=None):
    batch_size = len(fru_batch)
    pts_fts_sampled = []
    labels_sampled = []
    label_weights_sampled = []

    for i in range(batch_size):
        pt_num = fru_batch[i].shape[0]
        pool_size = pt_num

        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]

        pts_fts_sampled.append(fru_batch[i][choices, 0:4])
        label_sampled = fru_batch[i][choices, 4].astype(np.int32)
        labels_sampled.append(label_sampled)
        if label_weights_list is not None:
            label_weights_sampled.append(np.array(label_weights_list)[label_sampled])

    return np.array(pts_fts_sampled), np.array(labels_sampled), np.array(label_weights_sampled)


def sampling_infer(fru_pts_ins, sample_num):
    pool_size = len(fru_pts_ins)

    if pool_size > sample_num:
        choices = np.random.choice(pool_size, sample_num, replace=False)
    else:
        choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                  np.random.choice(pool_size, sample_num - pool_size, replace=True)))

    return np.array(fru_pts_ins)[choices]


def index_shuffle(index_length):
    shuffle_indices = np.arange(index_length.shape[0])
    np.random.shuffle(shuffle_indices)
    return index_length[shuffle_indices, ...]
