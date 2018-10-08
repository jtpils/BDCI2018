import numpy as np


def load_bin(path):
    point_list = []
    np_pts = np.fromfile(path, dtype=np.float32)

    for i in np_pts.reshape((-1, 4)):
        point_list.append(i)

    return point_list


def load_pts(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split(' ')
            pts.append(p)
    return pts


def save_pts(pts, path):
    with open(path, "w") as f:
        for p in pts:
            f.writelines(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")


def save_pts_as_obj(pts, path):
    with open(path, "w") as f:
        for p in pts:
            f.writelines("v " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")


def load_labels(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            label = KittiLabel(line)
            labels.append(label)
            label.print()

    return labels


class KittiLabel:
    types = {'DontCare': 0, 'Cyclist': 1, 'Tram': 2, 'Car': 3, 'Van': 4, 'Pedestrian': 5, 'Person_sitting': 6,
             'Misc': 7, 'Truck': 8}

    def __init__(self, label_str):
        label_ele = label_str.strip().split(' ')
        self.type_str = label_ele[0]
        self.type_num = self.types[self.type_str]
        self.truncated = float(label_ele[1])
        self.occluded = float(label_ele[2])
        self.alpha = float(label_ele[3])
        self.bbox = list(map(eval, label_ele[4:8]))
        self.dimensions = list(map(eval, label_ele[8:11]))
        self.location = list(map(eval, label_ele[11:14]))
        self.rotation_y = float(label_ele[14])

    def print(self):
        print(self.__dict__)
