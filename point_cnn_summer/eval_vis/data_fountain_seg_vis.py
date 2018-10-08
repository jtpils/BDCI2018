import os
import plyfile
import numpy as np


def dir(root, type='f', addroot=True):
    dirList = []
    fileList = []

    files = os.listdir(root)

    for f in files:

        if os.path.isdir(os.path.join(root, f)):

            if addroot:
                dirList.append(os.path.join(root, f))
            else:
                dirList.append(f)

        if os.path.isfile(os.path.join(root, f)):

            if addroot:
                fileList.append(os.path.join(root, f))
            else:
                fileList.append(f)

    if type == "f":
        return fileList

    elif type == "d":
        return dirList

    else:
        print("ERROR: TMC.dir(root,type) type must be [f] for file or [d] for dir")

        return 0


def save_ply(points, colors, filename):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_color = np.array([tuple(c) for c in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, dtype=vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write(filename)


def seg2color(seg):
    color_list = [(192, 153, 110), (188, 199, 253), (214, 255, 0), (159, 0, 142), (153, 255, 85), (118, 79, 2),
                  (123, 72, 131), (2, 176, 127), (1, 126, 184), (0, 144, 161), (106, 107, 128), (254, 230, 0),
                  (0, 255, 255), (255, 167, 254), (233, 93, 189), (0, 100, 0), (132, 169, 1), (150, 0, 61),
                  (188, 136, 0), (0, 0, 255)]
    abn_color = (0, 0, 0)
    color = []

    for s in seg:
        color.append(color_list[s])
    return color


pts_file_root = "/home/leon/Disk/dataset/Downloads/DataFountain/TestSet/pts"
seg_file_root = "/home/leon/Disk/dataset/DataFountain/test_results/trainval_2048_840k"
out_ply_root = "/home/leon/Disk/dataset/DataFountain/test_results/vis_trainval_2048_840k"

if not os.path.exists(out_ply_root):
    print(out_ply_root, "Not Exists! Create", out_ply_root)
    os.makedirs(out_ply_root)

seg_file_list = dir(seg_file_root, 'f', False)

seg_files = []
pts_files = []
out_segs = []
out_plys = []

for pred_seg in seg_file_list:
    seg_files.append(seg_file_root + "/" + pred_seg)
    pts_files.append(pts_file_root + "/" + pred_seg)

    out_plys.append(out_ply_root + "/" + pred_seg.replace(".csv", ".ply"))

for k, seg_f in enumerate(seg_files):

    print("Process:", k, "/", len(seg_files))

    seg = []
    ori_pts = []

    # read pts
    print("Read pts:", pts_files[k])
    with open(pts_files[k], 'r') as pts_f:

        for line in pts_f:
            line_s = line.strip().split(",")

            ori_pts.append((float(line_s[0]), float(line_s[1]), float(line_s[2])))

    # read seg
    print("Read seg:", seg_files[k])
    with open(seg_files[k], 'r') as seg_f:

        for line in seg_f:
            line_s = int(line.strip())

            seg.append(line_s)

    print("Save ply:", out_plys[k])
    save_ply(ori_pts, seg2color(seg), out_plys[k])
