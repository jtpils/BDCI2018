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

# python3 test_df_seg_processes_fru.py -i /home/leon/Disk/datasets/data_fountain/test \
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
import time


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
    # batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)
    batch_size = 8

    # Placeholders
    is_training = tf.placeholder(tf.bool, name='is_training')
    points_sampled = tf.placeholder(tf.float32, shape=(batch_size, sample_num, setting.data_dim - 1), name='points')
    features_sampled = tf.placeholder(tf.float32, shape=(batch_size, sample_num, setting.data_dim - 3), name='fts')

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

        test_max_num = 5

        for id_file, framename in enumerate(framenames):
            if id_file == test_max_num:
                break

            # Prepare inputs
            print('{}-Preparing datasets...'.format(datetime.now()))
            # load
            time_before_loading = time.time()
            pts_ins, _ = df_utils.load_frame(dir_input, framename)

            # clear
            time_before_clearing = time.time()
            pts_ins_cleared, _, scene_indices = df_utils.clear_data(pts_ins, None)

            # split
            time_before_spliting = time.time()
            frus_pts_ins, _, frus_indices = df_utils.split_frame_to_frus(pts_ins_cleared, None)

            # inference
            time_before_inference = time.time()
            batch_num = len(frus_pts_ins)
            frame_categories = []

            for batch_idx in range(batch_num):
                if len(frus_pts_ins[batch_idx]) == 0:
                    continue

                time_before_prepare_batch = time.time()

                batch_pts_ins = df_utils.sampling_infer(frus_pts_ins[batch_idx], sample_num)
                batch_pts_ins = np.array([batch_pts_ins] * batch_size)

                time_before_run = time.time()
                seg_probs = sess.run([seg_probs_op],
                                     feed_dict={
                                         points_sampled: batch_pts_ins[:, :, 0:3],
                                         features_sampled: batch_pts_ins[:, :, 3].reshape((batch_size, sample_num, 1)),
                                         is_training: False,
                                     })

                point_num = 2048
                # output seg probs
                time_before_pred = time.time()
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))
                predictions = [(-1, 0.0, [])] * point_num

                for idx in range(sample_num * batch_size):
                    # point_idx = indices_shuffle[idx]
                    point_idx = 0
                    point_probs = probs_2d[idx, :]
                    prob = np.amax(point_probs)
                    seg_idx = np.argmax(point_probs)
                    if prob > predictions[point_idx][1]:
                        predictions[point_idx] = [seg_idx, prob, point_probs]

                # print(predictions[point_num - 1])
                for seg_idx, prob, probs in predictions:
                    frame_categories.append(seg_idx)

                time_end_batch_infer = time.time()
                print("prepare data time:%fs, run time:%fs, get pred time:%fs\n" %
                      (time_before_run - time_before_prepare_batch,
                       time_before_pred - time_before_run,
                       time_end_batch_infer - time_before_pred))

            # # Recovery order
            # time_before_recovering_cleared_rul = time.time()
            # results_cleared = np.zeros(len(frame_categories), int)
            # i = 0
            # for fru_indices in frus_indices:
            #     for index in fru_indices:
            #         results_cleared[index] = int(frame_categories[i])
            #         i += 1
            #
            # time_before_recovering_ori_rul = time.time()
            # results = np.zeros(pts_ins.shape[0], int)
            # for i, cate in enumerate(results_cleared):
            #     results[scene_indices[i]] = cate
            #
            # # Save rul
            # time_before_saving = time.time()
            # path_output = os.path.join(dir_output, framename)
            # with open(path_output, 'w') as file_seg:
            #     for result in results:
            #         file_seg.write(str(result) + "\n")

            frame_categories.clear()

            # Vis
            time_before_vis = time.time()
            if args.save_ply:
                print('{}-Saving ply of {}...'.format(datetime.now(), framename))
                path_label_ply = os.path.join(dir_vis, framename[:-4] + '_colored.ply')
                vis_utils.save_ply(path_label_ply, pts_ins[:, 0:3], vis_utils.seg2color(results))

            # print
            # print("Load time:%fs, clear time:%fs, split time:%fs, infer time:%fs, re_clear time:%fs, "
            #       "re_truth time:%fs, save time:%fs\n" % ((time_before_clearing - time_before_loading),
            #                                               (time_before_spliting - time_before_clearing),
            #                                               (time_before_inference - time_before_spliting),
            #                                               (time_before_recovering_cleared_rul - time_before_inference),
            #                                               (time_before_recovering_ori_rul - time_before_recovering_cleared_rul),
            #                                               (time_before_saving - time_before_recovering_ori_rul),
            #                                               (time_before_vis - time_before_saving)))
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
