#!/usr/bin/python3

# -i
# /home/leon/DataFountain/test
# -o
# /home/leon/DataFountain/test_result
# -l
# /home/leon/Disk/models/data_fountain/h5/seg/pointcnn_seg_data_fountain_data_fountain_x4_2048_xyrgbi_fru+intensity_1777_2018-09-26-20-33-18/ckpts/iter-810000
# -m
# pointcnn_seg_data_fountain
# -x
# ./pointcnn_seg_data_fountain/data_fountain_x4_2048_xyrgbi_intensity

"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib
from data_utils.data_fountain import df_utils
import numpy as np
import tensorflow as tf
from datetime import datetime

df_test_max_point_num = 44792


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', '-i', help='Path to input points files', required=True)
    parser.add_argument('--dir_output', '-o', help='Path to save inference results', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--sample_num', help='Point sample num', type=int, default=2048)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)

    args = parser.parse_args()
    print(args)

    model = importlib.import_module(args.model)
    sys.path.append(os.path.dirname(args.setting))
    setting = importlib.import_module(os.path.basename(args.setting))

    sample_num = setting.sample_num
    num_parts = setting.num_parts

    dir_input = args.dir_input
    dir_output = os.path.join(args.dir_output, 'pred_' + str(args.repeat_num))

    # check the path
    if not os.path.exists(dir_output):
        print(dir_output, "Not Exists! Create", dir_output)
        os.makedirs(dir_output)

    filenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    max_point_num = df_test_max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.point_dim), name='pts')
    fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.extra_dim), name='fts')

    # Sample
    features_sampled = None

    if setting.extra_dim > 0:

        points = pts
        features = fts

        if setting.use_extra_features:
            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')

    else:
        points = pts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')

    # define net
    net = model.Net(points_sampled, features_sampled, None, None, num_parts, is_training, setting)

    probs_op = net.probs

    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        for id_file, filename in enumerate(filenames):
            # Prepare inputs
            print('{}-Preparing datasets...'.format(datetime.now()))
            frame_points, frame_intensities, _ = df_utils.load_frame(dir_input, filename)

            # split frame to quadrant
            quadrants_points, quadrants_intensities, _, quadrants_indices = \
                df_utils.split_frame_to_quadrants(frame_points, frame_intensities, None)

            batch_num = len(quadrants_points)

            frame_categories = []

            for batch_idx in range(batch_num):
                quadrant_points = np.zeros((1, max_point_num, 3), np.float32)
                quadrant_intensities = np.zeros((1, max_point_num, 1), np.float32)
                if len(quadrants_points[batch_idx]) == 0:
                    continue
                quadrant_points[0, 0:len(quadrants_points[batch_idx]), ...] = quadrants_points[batch_idx]
                quadrant_intensities[0, 0:len(quadrants_intensities[batch_idx]), ...] = \
                    np.array(quadrants_intensities[batch_idx]).reshape((1, len(quadrants_intensities[batch_idx]), 1))

                points_batch = quadrant_points[[0] * batch_size, ...]
                intensity_batch = quadrant_intensities[[0] * batch_size, ...]
                point_num = len(quadrants_points[batch_idx])

                tile_num = math.ceil((sample_num * batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

                sess_op_list = [probs_op]

                sess_feed_dict = {pts: points_batch,
                                  fts: intensity_batch,
                                  indices: indices_batch,
                                  is_training: False}

                # sess run
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                probs = sess.run(sess_op_list, feed_dict=sess_feed_dict)

                # output seg probs
                probs_2d = np.reshape(probs, (sample_num * batch_size, -1))
                predictions = [(-1, 0.0, [])] * point_num

                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    point_probs = probs_2d[idx, :]
                    prob = np.amax(point_probs)
                    seg_idx = np.argmax(point_probs)
                    if prob > predictions[point_idx][1]:
                        predictions[point_idx] = [seg_idx, prob, point_probs]
                for seg_idx, prob, probs in predictions:
                    frame_categories.append(seg_idx)

            results = np.zeros(len(frame_categories), int)
            i = 0
            for quadrant_indices in quadrants_indices:
                for index in quadrant_indices:
                    results[index] = int(frame_categories[i])
                    i += 1

            path_output = os.path.join(dir_output, filename)
            with open(path_output, 'w') as file_seg:
                for result in results:
                    file_seg.write(str(result) + "\n")

            frame_categories.clear()

            print('{}-[Testing]-Iter: {:06d} \nseg  saved to {}'.format(datetime.now(), id_file, filename))
            sys.stdout.flush()

    print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
