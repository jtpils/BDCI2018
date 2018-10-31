import plyfile
import numpy as np
from matplotlib import pyplot as plt


def save_ply(path_out, points, colors, faces=None):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_color = np.array([tuple(c) for c in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, dtype=vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    if faces is not None:
        face = np.array([tuple([f]) for f in faces], dtype=[('vertex_indices', 'i4', (3,))])
        ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex'),
                               plyfile.PlyElement.describe(face, 'face')], text=False)
    else:
        ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write(path_out)


# {‘DontCare’: black, ‘cyclist’: red, ‘tricycle’: green, ‘smallMot’: blue,
# ‘bigMot’ light blue: 4, ‘pink’: 5, ‘crowds’: yellow, ‘unknown’: OrangeRed}
color_list = [(100, 100, 100), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (	255, 255, 0), (255, 69, 0), (255, 0, 0), (0, 144, 161), (106, 107, 128), (254, 230, 0),
                  (0, 255, 255), (255, 167, 254), (233, 93, 189), (0, 100, 0), (132, 169, 1), (150, 0, 61),
                  (188, 136, 0), (0, 0, 255)]
def seg2color(seg):
    abn_color = (0, 0, 0)
    color = []

    for s in seg:
        color.append(color_list[s])
    return color


def save_2d(path_out, mask_img):
    img = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), np.int32)
    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):
            r, g, b = color_list[np.int32(mask_img[i][j])]
            img[i][j] = np.array([r, g, b]).astype(np.int32)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path_out)
