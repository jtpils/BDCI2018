#!/usr/bin/python3

# -i
# /home/leon/Disk/dataset/Downloads/DataFountain/TestSet
# -o
# /home/leon/Disk/dataset/DataFountain/test_results/opensrc_trainval_weights
# -l
# /home/leon/Disk/models/data_fountain/pointcnn_seg_df_x4_2048_fps_2018-10-11-06-30-01_6985/ckpts/iter-75000
# -m
# pointcnn_seg
# -x
# df_x4_2048_fps
# -g
# 0
# -s

# python3 test_df_seg_processes.py -i /home/leon/Disk/datasets/data_fountain/test \
# -o /home/leon/Disk/datasets/data_fountain/test_results \
# -l /home/leon/Disk/models/data_fountain/h5/seg/\
# pointcnn_seg_df_x4_2048_fps_2018-10-12-20-04-12_22301/ckpts/iter-80000 \
# -m pointcnn_seg -x df_x4_2048_fps -g 0,1,2 -s

"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib
from utils import df_utils
from utils import vis_utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from multiprocessing import Process


def inference_frames_frus(args, framenames, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num

    dir_input = args.dir_input
    dir_output = os.path.join(args.dir_output, 'pred_' + str(args.repeat_num))
    dir_vis = os.path.join(args.dir_output, 'vis_' + str(args.repeat_num))

    max_point_num = args.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')

    # Sample
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    # define net
    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        for id_file, framename in enumerate(framenames):
            # Prepare inputs
            print('{}-Preparing datasets...'.format(datetime.now()))
            frame_data, _ = df_utils.load_frame(dir_input, framename)

            point_num = frame_data.shape[0]
            batch_data = np.zeros((1, max_point_num, setting.data_dim), np.float32)
            batch_data[0, 0:point_num, ...] = frame_data
            batch_data = batch_data[[0] * batch_size, ...]

            tile_num = math.ceil((sample_num * batch_size) / point_num)
            indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
            np.random.shuffle(indices_shuffle)
            indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
            indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

            seg_probs = sess.run([seg_probs_op],
                                 feed_dict={
                                     pts_fts: batch_data,
                                     indices: indices_batch,
                                     is_training: False,
                                 })
            probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

            # output seg probs
            predictions = [(-1, 0.0)] * point_num
            for idx in range(sample_num * batch_size):
                point_idx = indices_shuffle[idx]
                probs = probs_2d[idx, :]
                confidence = np.amax(probs)
                label = np.argmax(probs)
                if confidence > predictions[point_idx][1]:
                    predictions[point_idx] = [label, confidence]
            labels_pred = np.full(point_num, -1, dtype=np.int32)
            labels_pred[0:point_num] = np.array([label for label, _ in predictions])

            path_output = os.path.join(dir_output, framename)
            with open(path_output, 'w') as file_seg:
                for result in labels_pred:
                    file_seg.write(str(result) + "\n")

            if args.save_ply:
                print('{}-Saving ply of {}...'.format(datetime.now(), framename))
                path_label_ply = os.path.join(dir_vis, framename[:-3] + 'ply')
                vis_utils.save_ply(path_label_ply, frame_data[:, 0:3], vis_utils.seg2color(labels_pred))

            print('PID:{}-{}-[Testing]-Iter: {:06d} \nseg  saved to {}'.format(os.getpid(),
                                                                               datetime.now(), id_file, framename))

    print('{}-Done! PID = {}'.format(datetime.now(), os.getpid()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', '-i', help='Path to input points files', required=True)
    parser.add_argument('--dir_output', '-o', help='Path to save inference results', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--max_point_num', '-p', help='Max point number of each sample', type=int, default=58016)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    parser.add_argument('--gpu_available', '-g', help='Gpus to use', type=str, default='0,1,2')
    args = parser.parse_args()
    print(args)

    # check the path
    dir_output = os.path.join(args.dir_output, 'pred_' + str(args.repeat_num))
    if not os.path.exists(dir_output):
        print(dir_output, "Not Exists! Create", dir_output)
        os.makedirs(dir_output)
    if args.save_ply:
        dir_vis = os.path.join(args.dir_output, 'vis_' + str(args.repeat_num))
        if not os.path.exists(dir_vis):
            print(dir_vis, "Not Exists! Create", dir_vis)
            os.makedirs(dir_vis)

    gpu_available = [int(gpu_id.strip()) for gpu_id in args.gpu_available.split(',')]
    framenames = sorted(os.listdir(os.path.join(args.dir_input, 'pts')))

    chunk_size = len(framenames) // len(gpu_available)
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > len(gpu_available):
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=inference_frames_frus, args=(args, chunk_filepaths, gpu_available[i]))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()


if __name__ == '__main__':
    main()
