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
from SimpleSAC.conservative_sac import ConservativeSAC
from SimpleSAC.utils import WandBLogger

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
    qf_hidden_layer=2,
    qf_hidden_unit=256,
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
    rl_data_ratio=1, #TODO run cpu experiments on this?

    setting=0,
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    ###########################################################
    exp_prefix = 'cql'
    settings = [
        'env', '', MUJOCO_3_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'pretrain_mode', 'pre', ['q_sprime', 'none'], # q_sprime
        'n_pretrain_epochs', 'pe', [200,],
        'seed', '', [42,],
    ] #

    #'qf_hidden_layer', 'layer', [4, 6, 8],

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, FLAGS.setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)
    # replace the values in FLAGS.sth with actual_setting['sth']

    exp_name_full = exp_name_full + '_layer%d' % FLAGS.qf_hidden_layer

    """replace values"""
    for key, value in actual_setting.items():
        setattr(FLAGS, key, value)

    # qf_hidden_layer and qf_hidden_unit will override qf_arch
    FLAGS.qf_arch = '-'.join([str(FLAGS.qf_hidden_unit) for _ in range(FLAGS.qf_hidden_layer)])

    ###########################################################
    variant = get_user_flags(FLAGS, FLAGS_DEF) # variant is a dict
    # new logger
    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    run_single_exp(variant, FLAGS)

if __name__ == '__main__':
    absl.app.run(main)
