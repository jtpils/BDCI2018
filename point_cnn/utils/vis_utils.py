import plyfile
import numpy as np


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


# {‘DontCare’: black, ‘cyclist’: red, ‘tricycle’: green, ‘smallMot’: blue,
# ‘bigMot’ light blue: 4, ‘pink’: 5, ‘crowds’: yellow, ‘unknown’: OrangeRed}
def seg2color(seg):
    color_list = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (	255, 255, 0), (255, 69, 0), (255, 0, 0), (0, 144, 161), (106, 107, 128), (254, 230, 0),
                  (0, 255, 255), (255, 167, 254), (233, 93, 189), (0, 100, 0), (132, 169, 1), (150, 0, 61),
                  (188, 136, 0), (0, 0, 255)]
    abn_color = (0, 0, 0)
    color = []

    for s in seg:
        color.append(color_list[s])
    return color