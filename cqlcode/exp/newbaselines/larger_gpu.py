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
from SimpleSAC.utils import get_user_flags, define_flags_with_default
from exp_scripts.grid_utils import *
from SimpleSAC.run_cql_watcher import run_single_exp, get_default_variant_dict
from SimpleSAC.conservative_sac import ConservativeSAC
from SimpleSAC.utils import WandBLogger
import argparse
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    setting = args.setting

    variant = get_default_variant_dict() # this is a dictionary
    ###########################################################
    exp_prefix = 'cqlnew'
    settings = [
        'env', '', MUJOCO_3_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'pretrain_mode', 'pre', ['none', 'q_sprime', ],
        'qf_hidden_layer', 'l', [8, 6, 4],
    ] #

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)
    # replace default values with grid values

    # GPU job only, either set seeds in this file, or set seed in gpu .sh script
    # but not both
    if 'seed' not in actual_setting:
        if args.seed != -1:
            actual_setting['seed'] = args.seed
        else:
            print("Make sure the seed part is correct!!!!!!")
            quit()

    """replace values"""
    for key, value in actual_setting.items():
        variant[key] = value

    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    run_single_exp(variant)


if __name__ == '__main__':
    main()
