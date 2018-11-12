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


def knn(queries, points, k):
    """

    :param queries: (N, P_queries, C)
    :param points:  (N, P_points, C)
    :param k:   int
    :return:    (N, P_queries, k, dis),  (N, P_queries, k, indices)
    """

    return knn_module.KNN(k, queries, points)

ops.NoGradient('KNN')


if __name__ == '__main__':
    batch_size = 8
    pts_num = 2048
    qrs_num = 768
    dim = 64
    k = 12

    points = tf.random_normal([batch_size, pts_num, dim], dtype=tf.float32)
    queries = tf.random_normal([batch_size, qrs_num, dim], dtype=tf.float32)

    rul = knn(queries, points, k)

    # printer = tf.Print(rul, [rul])

    with tf.Session() as sess:
        sess.run(rul)

