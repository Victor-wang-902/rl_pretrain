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
from SimpleSAC.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from SimpleSAC.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from SimpleSAC.utils import WandBLogger
# from viskit.logging import logger_other, setup_logger
from exp_scripts.grid_utils import *
from logx import EpochLogger

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
    save_model=False,
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
    n_train_step_per_epoch=5000,
    eval_period=1,
    eval_n_trajs=10,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF) # variant is a dict
    # wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    # setup_logger(
    #     variant=variant,
    #     exp_id=wandb_logger.experiment_id,
    #     seed=FLAGS.seed,
    #     base_log_dir=FLAGS.logging.output_dir,
    #     include_exp_prefix_sub_dir=False
    # )

    # new logger
    data_dir = '/checkpoints'
    exp_prefix = 'cqlnew'
    exp_suffix = '_%s-%s' % (FLAGS.env, FLAGS.dataset)
    exp_name_full = exp_prefix + exp_suffix
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    logger = EpochLogger(variant["outdir"], 'progress.csv', variant["exp_name"])
    logger.save_config(variant)

    set_random_seed(FLAGS.seed)

    env_full = '%s-%s-v2' % (FLAGS.env, FLAGS.dataset)
    eval_sampler = TrajSampler(gym.make(env_full).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
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

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    st = time.time()
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

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
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    # wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        # wandb_logger.log(metrics)
        viskit_metrics.update(metrics)

        things_to_log = ['sac/log_pi', 'sac/policy_loss', 'sac/qf1_loss', 'sac/qf2_loss', 'sac/alpha_loss', 'sac/alpha',
                         'sac/average_qf1', 'sac/average_qf2', 'average_traj_length']
        for m in things_to_log:
            logger.log_tabular(m, viskit_metrics[m])

        logger.log_tabular("TestEpRet", viskit_metrics['average_return'])
        logger.log_tabular("TestEpNormRet", viskit_metrics['average_normalizd_return'])
        logger.log_tabular("Iteration", epoch + 1)
        logger.log_tabular("Steps", (epoch + 1) * FLAGS.n_train_step_per_epoch)
        logger.log_tabular("total_time", time.time()-st)
        logger.log_tabular("train_time", viskit_metrics["train_time"])
        logger.log_tabular("eval_time", viskit_metrics["eval_time"])

        logger.log_tabular("est_total_hours", (FLAGS.n_epochs/(epoch + 1) * (time.time()-st))/3600)

        logger.dump_tabular()
        sys.stdout.flush()
        # logger_other.record_dict(viskit_metrics)
        # logger_other.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        # wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)