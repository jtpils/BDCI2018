#!/usr/bin/python3
import math
num_class = 8

sample_num = 2048

batch_size = 8

num_epochs = 256

label_weights = [0.4, 0.95, 0.97, 0.7, 0.8,  0.99, 1.0, 0.9]
# label_weights = [0.1, 0.7, 0.8, 0.4, 0.5, 0.9, 1.0, 0.6]
# for c in range(num_class):
#     label_weights.append(1.0)

learning_rate_base = 0.005
decay_steps = 60000
decay_rate = 0.8
learning_rate_min = 1e-6

step_val = 1000

weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.0, 0.0, 0.0, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 4

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(4, 1, -1, 16 * x, []),
                 (6, 2, 768, 32 * x, []),
                 (8, 2, 384, 64 * x, []),
                 (8, 2, 128, 96 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(8, 2, 3, 2),
                  (6, 2, 2, 1),
                  (4, 1, 1, 0)]]


fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(16 * x, 0.0),
              (16 * x, 0.7)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-3

data_dim = 4
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True