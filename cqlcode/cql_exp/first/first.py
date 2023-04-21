import os
import sys

# with new logger
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
import d4rl

import absl.app
import absl.flags

from SimpleSAC.utils import get_user_flags, define_flags_with_default
from exp_scripts.grid_utils import *
from SimpleSAC.run_cql import run_single_exp

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def main(argv):
    FLAGS = absl.flags.FLAGS

    ###########################################################
    exp_prefix = 'zzz'
    settings = [
        'env', '', MUJOCO_3_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'seed', '', [42, 666, 1024],
    ]
    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, FLAGS.setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)

    print("YOOOOOO")
    print(FLAG_DEF)
    print(FLAGS.setting)
    print(actual_setting)
    print(exp_name_full)
    print(FLAGS)
    quit()

    ###########################################################

    variant = get_user_flags(FLAGS, FLAGS_DEF) # variant is a dict
    # new logger
    data_dir = '/checkpoints'
    exp_prefix = FLAGS.exp_prefix
    exp_suffix = '_%s_%s' % (FLAGS.env, FLAGS.dataset)
    exp_name_full = exp_prefix + exp_suffix
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    run_single_exp(variant, FLAGS)

if __name__ == '__main__':
    absl.app.run(main)
