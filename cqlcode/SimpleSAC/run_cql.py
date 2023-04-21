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

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def define_default_flags_and_get_flags_def():
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
    )
    return FLAGS_DEF

def get_convergence_index(ret_list, threshold_gap=2):
    best_value = max(ret_list)
    convergence_threshold = best_value - threshold_gap
    k_conv = len(ret_list) - 1
    for k in range(len(ret_list) - 1, -1, -1):
        if ret_list[k] >= convergence_threshold:
            k_conv = k
    return k_conv

def concatenate_weights_of_model_list(model_list, weight_only=True):
    concatenated_weights = []
    for model in model_list:
        for name, param in model.named_parameters():
            if not weight_only or 'weight' in name:
                concatenated_weights.append(param.view(-1))
    return torch.cat(concatenated_weights)

# when compute weight diff, just provide list of important layers...from
def get_weight_diff(agent1, agent2):
    # weight diff, agent class should have layers_for_weight_diff() func
    weights1 = concatenate_weights_of_model_list(agent1.layers_for_weight_diff())
    weights2 = concatenate_weights_of_model_list(agent2.layers_for_weight_diff())
    weight_diff = torch.norm(weights1-weights2, p=2).item()
    return weight_diff

def get_feature_diff(agent1, agent2, dataset, device, ratio=0.1, seed=0):
    # feature diff: for each data point, get difference of feature from old and new network
    # compute l2 norm of this diff, average over a number of data points.
    # agent class should have features_from_batch() func
    n_total_data = dataset['observations'].shape[0]
    average_feature_l2_norm_list = []
    num_feature_timesteps = int(n_total_data * ratio)
    if num_feature_timesteps % 2 == 1: # avoid potential sampling issue
        num_feature_timesteps = num_feature_timesteps + 1
    np.random.seed(seed)
    idxs_all = np.random.choice(np.arange(0, n_total_data), size=num_feature_timesteps, replace=False)
    batch_size = 1000
    n_done = 0
    i = 0
    while True:
        if n_done >= num_feature_timesteps:
            break
        idxs = idxs_all[i*batch_size:min((i+1)*batch_size, num_feature_timesteps)]

        batch = index_batch(dataset, idxs)
        batch = batch_to_torch(batch, device)

        old_feature = agent1.features_from_batch(batch)
        new_feature = agent2.features_from_batch(batch)
        feature_diff = old_feature - new_feature

        feature_l2_norm = torch.norm(feature_diff, p=2, dim=1, keepdim=True)
        average_feature_l2_norm_list.append(feature_l2_norm.mean().item())
        i += 1
        n_done += 1000

    return np.mean(average_feature_l2_norm_list), num_feature_timesteps

def main(argv):
    FLAGS = absl.flags.FLAGS

    FLAGS_DEF = define_default_flags_and_get_flags_def()
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

def run_single_exp(variant, FLAGS):
    logger = EpochLogger(variant["outdir"], 'progress.csv', variant["exp_name"])
    logger.save_config(variant)
    pretrain_logger = EpochLogger(variant["outdir"], 'pretrain_progress.csv', variant["exp_name"])

    set_random_seed(FLAGS.seed)

    env_full = '%s-%s-v2' % (FLAGS.env, FLAGS.dataset)
    eval_sampler = TrajSampler(gym.make(env_full).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    print("D4RL dataset loaded for", env_full)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunctionPretrain(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunctionPretrain(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    agent = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    agent.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    """pretrain stage"""
    print("============================ PRETRAIN STAGE STARTED! ============================")
    st = time.time()
    if FLAGS.pretrain_mode != 'none':
        # TODO check if can load pretrained model
        pretrain_model_folder_path = '/cqlcode/pretrained_cql_models/'
        pretrain_model_name = '%s_%s_%s_%s_%d_%d_%s.pth' % ('cql', FLAGS.env, FLAGS.dataset, FLAGS.pretrain_mode,
                                                            FLAGS.qf_hidden_layer, 256, FLAGS.n_pretrain_epochs)
        pretrain_full_path = os.path.join(pretrain_model_folder_path, pretrain_model_name)
        try:
            pretrain_dict = torch.load(pretrain_full_path)
            agent.qf1.load_state_dict(pretrain_dict['agent'].qf1.state_dict())
            agent.qf2.load_state_dict(pretrain_dict['agent'].qf2.state_dict())
            loaded = True
            print("Pretrained model loaded from:", pretrain_full_path)
        except Exception as e:
            print(e, "No pretrained model, start pretraining.")
            loaded = False

        if not loaded:
            for epoch in range(FLAGS.n_pretrain_epochs):
                metrics = {'pretrain_epoch': epoch+1}
                for i_pretrain in range(FLAGS.n_train_step_per_epoch):
                    batch = subsample_batch(dataset, FLAGS.batch_size)
                    batch = batch_to_torch(batch, FLAGS.device)
                    metrics.update(agent.pretrain(batch, FLAGS.pretrain_mode))

                pretrain_logger.log_tabular("PretrainIteration", epoch + 1)
                pretrain_logger.log_tabular("PretrainSteps", (epoch + 1) * FLAGS.n_train_step_per_epoch)
                pretrain_logger.log_tabular("pretrain_loss", metrics['pretrain_loss'])
                pretrain_logger.log_tabular("total_pretrain_steps", metrics['total_pretrain_steps'])
                pretrain_logger.log_tabular("current_hours", (time.time() - st) / 3600)
                pretrain_logger.log_tabular("est_total_hours", (FLAGS.n_pretrain_epochs / (epoch + 1) * (time.time() - st)) / 3600)
                pretrain_logger.dump_tabular()
                sys.stdout.flush()

                if (epoch+1) in (2, 20):
                    pretrain_model_name_mid = '%s_%s_%s_%s_%d_%d_%s.pth' % (
                    'cql', FLAGS.env, FLAGS.dataset, FLAGS.pretrain_mode,
                    FLAGS.qf_hidden_layer, 256, epoch+1)
                    pretrain_full_path_mid = os.path.join(pretrain_model_folder_path, pretrain_model_name_mid)
                    pretrain_dict_mid = {'agent': agent,
                                     'algorithm': 'cql',
                                     'env': FLAGS.env,
                                     'dataset': FLAGS.dataset,
                                     'pretrain_mode': FLAGS.pretrain_mode,
                                     'hidden_layer': FLAGS.qf_hidden_layer,
                                     'hidden_size': 256,  # TODO allow generalize to other architectures
                                     'n_pretrain_epochs': epoch+1,
                                     }
                    if not os.path.exists(pretrain_full_path_mid):
                        torch.save(pretrain_dict_mid, pretrain_full_path_mid)
                        print("Saved intermediate pretrained model to:", pretrain_full_path_mid)
                    else:
                        print("Intermediate pretrained model not saved. Already exist:", pretrain_full_path_mid)

            pretrain_dict = {'agent':agent,
                             'algorithm':'cql',
                             'env':FLAGS.env,
                             'dataset':FLAGS.dataset,
                             'pretrain_mode':FLAGS.pretrain_mode,
                             'hidden_layer':FLAGS.qf_hidden_layer,
                             'hidden_size':256, #TODO allow generalize to other architectures
                             'n_pretrain_epochs':FLAGS.n_pretrain_epochs,
                             }
            if not os.path.exists(pretrain_full_path):
                torch.save(pretrain_dict, pretrain_full_path)
                print("Saved pretrained model to:", pretrain_full_path)
            else:
                print("Pretrained model not saved. Already exist:", pretrain_full_path)

    if FLAGS.do_pretrain_only:
        return
    agent_after_pretrain = deepcopy(agent)

    """offline stage"""
    print("============================ OFFLINE STAGE STARTED! ============================")

    best_agent = deepcopy(agent)
    agent_100k = None
    best_step, best_iter = 0, 0
    iter_list, step_list, ret_list, ret_normalized_list = [],[],[],[]
    best_return, best_return_normalized = -np.inf, -np.inf
    viskit_metrics = {}
    st = time.time()
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(agent.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac', connector_string='_'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])*100) for t in trajs]
                )

                # record best return and other things
                iter_list.append(epoch+1)
                step_list.append((epoch+1) * FLAGS.n_train_step_per_epoch)
                ret_normalized_list.append(metrics['average_normalizd_return'])
                ret_list.append(metrics['average_return'])
                if metrics['average_normalizd_return'] > best_return_normalized:
                    best_return = metrics['average_return']
                    best_return_normalized = metrics['average_normalizd_return']
                    best_agent = deepcopy(agent)
                    best_iter = epoch + 1
                    best_step = (epoch+1) * FLAGS.n_train_step_per_epoch
            if (epoch + 1) == 20:
                agent_100k = deepcopy(agent)
            if FLAGS.save_model and (epoch + 1) in (1, 20, 200):
                save_dict = {'agent': agent, 'variant': variant, 'epoch': epoch+1}
                logger.save_dict(save_dict, 'agent_e%d.pth' % (epoch + 1))
                # wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        # wandb_logger.log(metrics)
        viskit_metrics.update(metrics)


        logger.log_tabular("Iteration", epoch + 1)
        logger.log_tabular("Steps", (epoch + 1) * FLAGS.n_train_step_per_epoch)
        logger.log_tabular("TestEpRet", viskit_metrics['average_return'])
        logger.log_tabular("TestEpNormRet", viskit_metrics['average_normalizd_return'])

        things_to_log = ['sac_log_pi', 'sac_policy_loss', 'sac_qf1_loss', 'sac_qf2_loss', 'sac_alpha_loss', 'sac_alpha',
                         'sac_average_qf1', 'sac_average_qf2', 'average_traj_length']
        for m in things_to_log:
            logger.log_tabular(m, viskit_metrics[m])

        logger.log_tabular("total_time", time.time()-st)
        logger.log_tabular("train_time", viskit_metrics["train_time"])
        logger.log_tabular("eval_time", viskit_metrics["eval_time"])
        logger.log_tabular("current_hours", (time.time()-st)/3600)
        logger.log_tabular("est_total_hours", (FLAGS.n_epochs/(epoch + 1) * (time.time()-st))/3600)

        logger.dump_tabular()
        sys.stdout.flush()
        # logger_other.record_dict(viskit_metrics)
        # logger_other.dump_tabular(with_prefix=False, with_timestamp=False)

    """get extra dict"""
    # get convergence steps
    conv_k = get_convergence_index(ret_list)
    convergence_iter, convergence_step = iter_list[conv_k], step_list[conv_k]
    # get weight and feature diff
    if agent_100k is not None:
        weight_diff_100k = get_weight_diff(agent_100k, agent_after_pretrain)
        feature_diff_100k = get_feature_diff(agent_100k, agent_after_pretrain, dataset, FLAGS.device)
    else:
        weight_diff_100k = -1
        feature_diff_100k = -1
    final_weight_diff = get_weight_diff(agent, agent_after_pretrain)
    final_feature_diff, _ = get_feature_diff(agent, agent_after_pretrain, dataset, FLAGS.device)
    best_weight_diff = get_weight_diff(best_agent, agent_after_pretrain)
    best_feature_diff, num_feature_timesteps = get_feature_diff(best_agent, agent_after_pretrain, dataset, FLAGS.device)
    # save extra dict
    extra_dict = {
        'final_weight_diff':final_weight_diff,
        'final_feature_diff':final_feature_diff,
        'best_weight_diff': best_weight_diff,
        'best_feature_diff': best_feature_diff,
        'num_feature_timesteps':num_feature_timesteps,
        'final_test_returns':float(ret_list[-1]),
        'final_test_normalized_returns': float(ret_normalized_list[-1]),
        'best_return': float(best_return),
        'best_return_normalized':float(best_return_normalized),
        'convergence_step':convergence_step,
        'convergence_iter':convergence_iter,
        'best_step':best_step,
        'best_iter':best_iter,
        'weight_diff_100k':weight_diff_100k, # unique to cql due to more training updates
        'feature_diff_100k': feature_diff_100k,
    }
    logger.save_extra_dict_as_json(extra_dict, 'extra.json')

    if FLAGS.save_model:
        save_data = {'sac': agent, 'variant': variant, 'epoch': epoch}
        # wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
