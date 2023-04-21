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

from SimpleSAC.conservative_sac import ConservativeSAC
from SimpleSAC.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch, index_batch
from SimpleSAC.model import TanhGaussianPolicy, SamplerPolicy, FullyConnectedQFunctionPretrain
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from SimpleSAC.utils import WandBLogger
# from viskit.logging import logger_other, setup_logger
from exp_scripts.grid_utils import *
from redq.utils.logx import EpochLogger
from SimpleSAC.run_cql import run_single_exp

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah',
    dataset='medium',
    max_traj_length=1000,
    seed=42,
    device=DEVICE,
    save_model=True,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=200,
    bc_epochs=0,
    n_pretrain_epochs=200,
    pretrain_mode='none', #
    n_train_step_per_epoch=5000,
    eval_period=1,
    eval_n_trajs=10,
    exp_prefix='cqltest',
    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
    do_pretrain_only=False,
    setting=0,
)


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
