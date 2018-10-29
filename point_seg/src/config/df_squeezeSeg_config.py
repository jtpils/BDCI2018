# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified : Wang Yuan
"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config


def df_squeezeSeg_config():
    """Specify the parameters to tune below."""
    mc = base_model_config('DF')

    mc.BATCH_SIZE = 16
    mc.AZIMUTH_LEVEL = 1024
    mc.ZENITH_LEVEL = 32

    mc.LCN_HEIGHT = 3
    mc.LCN_WIDTH = 5
    mc.RCRF_ITER = 3
    mc.BILATERAL_THETA_A = np.array([.9, .9, .6, .6])
    mc.BILATERAL_THETA_R = np.array([.015, .015, .01, .01])
    mc.BI_FILTER_COEF = 0.1
    mc.ANG_THETA_A = np.array([.9, .9, .6, .6])
    mc.ANG_FILTER_COEF = 0.02

    mc.CLS_LOSS_COEF = 15.0
    mc.WEIGHT_DECAY = 0.0001
    mc.LEARNING_RATE = 0.01
    mc.DECAY_STEPS = 8000
    mc.MAX_GRAD_NORM = 1.0
    mc.MOMENTUM = 0.9
    mc.LR_DECAY_FACTOR = 0.5

    mc.DATA_AUGMENTATION = True
    mc.RANDOM_FLIPPING = True

    # x, y, z, intensity, distance
    mc.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
    mc.INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

    return mc
