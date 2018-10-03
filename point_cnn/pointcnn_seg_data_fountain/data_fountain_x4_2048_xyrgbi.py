#!/usr/bin/python3
import math

# switch

num_parts = 8

sample_num = 2048

batch_size = 6

num_epochs = 1024

label_weights = []

for c in range(num_parts):
    if c == 0:
        label_weights.append(0.4)
    else:
        label_weights.append(1.0)

learning_rate_base = 0.005
decay_steps = 20000
decay_rate = 0.8
learning_rate_min = 1e-6
step_val = 30000
weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0, 0, 0, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 4

# K, D, P, C
xconv_params = [(8, 1, -1, 32 * x),
                (12, 2, 768, 64 * x),
                (16, 2, 384, 128 * x),
                (16, 6, 128, 156 * x)]

# K, D, pts_layer_idx, qrs_layer_idx
xdconv_params = [(16, 6, 3, 2),
                 (12, 4, 2, 1),
                 (8, 4, 1, 0)]

# C, dropout_rate
fc_params = [(32 * x, 0.0), (32 * x, 0.5)]

# sampleing
with_fps = False

optimizer = 'adam'
epsilon = 1e-3

# imput data
data_dim = 3

use_extra_features = True
data_format = {"pts_xyz": 3}

with_X_transformation = True

sorting_method = None
keep_remainder = True
