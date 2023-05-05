import os
import sys
# TODO still working on this

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

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
