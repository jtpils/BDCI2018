''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
knn_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_knn.so'))


def knn(k, queries, points):
    """

    :param queries: (N, P_queries, C)
    :param points:  (N, P_points, C)
    :param k:   int
    :return:    (N, P_queries, k, dis),  (N, P_queries, k, indices)
    """

    return knn_module.my_knn(k=k, queries=queries, points=points)

ops.NoGradient('MyKnn')


def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return knn_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')

if __name__ == '__main__':
    batch_size = 8
    pts_num = 2048
    qrs_num = 768
    dim = 64
    k = 12

    points = tf.random_normal([batch_size, pts_num, dim], dtype=tf.float32)
    queries = tf.random_normal([batch_size, qrs_num, dim], dtype=tf.float32)

    # test = farthest_point_sample(1024, points)
    dis, ids = knn(k, queries, points)

    rul = tf.gather_nd(points, ids)
    printer = tf.Print(rul, [rul],first_n=10, summarize=20)

    with tf.Session() as sess:
        sess.run(printer)

