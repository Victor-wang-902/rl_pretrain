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
    args = parser.parse_args()
    setting = args.setting

    variant = get_default_variant_dict() # this is a dictionary
    ###########################################################
    exp_prefix = 'cql_cross_pretrain' #TODO specify the experimnent name prefix
    settings = [
        'env', '', MUJOCO_4_ENVS, # this contains 4 mujoco envs: ant, halfcheetah, hopper, walker2d
        'dataset', '', MUJOCO_3_DATASETS, # this contains 3 mujoco datasets: medium, medium-expert, medium-replay
        'pretrain_mode', 'pre', ['q_sprime_text', # pretrain on the same dataset without projections using next state prediction (MDP)
                                'q_noact_sprime_text', # pretrain on the same dataset without projections using next state prediction without conditioning on the action
                                'proj0_q_sprime_text', # pretrain on the same dataset but still use projections, using next state prediction (MDP)
                                "proj0_q_noact_sprime_text", # pretrain on the same dataset but still use projections, using next state prediction without conditioning on the action (MC)
                                'TD', # pretrain on the same dataset without projections using CQL 
                                'proj0_TD', # pretrain on the same dataset with projections using CQL
                                'proj3_q_sprime_text', # pretrain on the current dataset of the next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_1', # pretrain on the current dataset of the second next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_2',  # pretrain on the current dataset of the third next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_all', # pretrain on all envs (the current dataset for all envs) with projections, using next state prediction (MDP)
                                'proj3_q_sprime_text_allbut', # pretrain on all but current env (the current dataset for each env) with projections, using next state prediction (MDP)
                                'proj3_q_noact_sprime_text', # pretrain on the current dataset of the next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_1', # pretrain on the current dataset of the second next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_2', # pretrain on the current dataset of the third next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_all', # pretrain on all envs (the current dataset for all envs) with projections, using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_allbut', # pretrain on all but current env (the current dataset for each env) with projections, using next state prediction without conditioning on the action (MC)
                                'proj3_TD_text', # pretrain on the current dataset of the next env (with projections) using CQL
                                'proj3_TD_text_1', # pretrain on the current dataset of the second next env (with projections) using CQL
                                'proj3_TD_text_2', # pretrain on the current dataset of the third next env (with projections) using CQL
                                'proj3_TD_text_all', # pretrain on all envs (the current dataset for all envs) with projections, using CQL
                                'proj3_TD_text_allbut', # pretrain on all but current env (the current dataset for each env) with projections, using CQL
                                'proj3_q_sprime_text_3x', # pretrain on the all datasets of the next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_1_3x', # pretrain on all datasets of the second next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_2_3x', # pretrain on all datasets of the third next env (with projections) using next state prediction (MDP)
                                'proj3_q_sprime_text_all_3x', # pretrain on all datasets of all envs (10 datasets in total, EXCLUDING the two other datasets for the current env) with projections, using next state prediction (MDP)
                                'proj3_q_sprime_text_allbut_3x', # pretrain on all datasets of all envs EXCEPT for the current env (9 datasets in total) with projections, using next state prediction (MDP)
                                'proj3_q_noact_sprime_text_3x', # pretrain on the all datasets of the next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_1_3x', # pretrain on all datasets of the second next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_2_3x', # pretrain on all datasets of the third next env (with projections) using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_all_3x', # pretrain on all datasets of all envs (10 datasets in total, EXCLUDING the two other datasets for the current env) with projections, using next state prediction without conditioning on the action (MC)
                                'proj3_q_noact_sprime_text_allbut_3x', # pretrain on all datasets of all envs EXCEPT for the current env (9 datasets in total) with projections, using next state prediction without conditioning on the action (MC)
                                'proj3_TD_text_3x', # pretrain on the all datasets of the next env (with projections) using CQL
                                'proj3_TD_text_1_3x', # pretrain on all datasets of the second next env (with projections) using CQL
                                'proj3_TD_text_2_3x', # pretrain on all datasets of the third next env (with projections) using CQL
                                'proj3_TD_text_all_3x', # pretrain on all datasets of all envs (10 datasets in total, EXCLUDING the two other datasets for the current env) with projections, using CQL
                                'proj3_TD_text_allbut_3x', # pretrain on all datasets of all envs EXCEPT for the current env (9 datasets in total) with projections, using CQL
                                ],
        "text_encoder", 'te', [
                                ("sentence-transformers/all-MiniLM-L12-v2", # Sentence encoder to use
                                    "allMiniLML12"), # shorter name for path to save
                                (None, # None for disabling text encoder
                                    "None"),
                                ],

        "policy_with_text", 'pwt', [True, False], # whether to use text discription in policy networks

        "offline_data_ratio", 'fr', [1.0], # data ratio for offline learning
        "pretrain_data_ratio", 'pr', [1.0], # data ratio for pretraining

        'qf_hidden_layer', 'ql', [3], # hidden layers for q networks
        "policy_hidden_layer", 'pl', [3], # hidden layers for policy networks

        'n_pretrain_step_per_epoch', 'preUps', [5000], # pretrain updates per epoch
        'n_pretrain_epochs', 'preEp', [100], # pretrain epochs
        'n_epochs', 'e', [200], # offline epochs
        'n_train_step_per_epoch', 's', [5000],  # offline updates per epoch
        'pretrain_lr', 'plr', [3e-4], # pretrain learning rate for q networks and policy networks (if applicable)
        'finetune_qf_lr', 'fqlr', [3e-4], # offline learning rate for q networks
        'finetune_policy_lr', 'fplr', [3e-4], # offline learning rate for policy networks
        'seed', '', [42, 666, 1024], # random seeds
    ]
    
    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)
    # replace default values with grid values

    """replace values"""
    for key, value in actual_setting.items():
        if isinstance(value, tuple):
            variant[key] = value[0]
        else:
            variant[key] = value

    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    # TODO for now we set this to 3 for faster experiments
    variant['cql'].cql_n_actions = 3
    run_single_exp(variant)


if __name__ == '__main__':
    main()
