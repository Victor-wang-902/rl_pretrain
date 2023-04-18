import os
import os.path
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

import numpy as np
import torch
import gym
import d4rl
import time
import sys
import copy
from redq.algos.cql import CQLAgent
from redq.algos.il import ILAgent
from redq.algos.core import mbpo_epoches, test_agent_d4rl, get_weight_diff, get_feature_diff
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger
from exp_scripts.grid_utils import *

def copy_agent_without_buffer(agent):
    buffer_temp = agent.replay_buffer
    agent.replay_buffer = None
    agent_copy = copy.deepcopy(agent)
    agent.replay_buffer = buffer_temp
    return agent_copy

def save_dict(logger, dictionary, save_name, verbose=1):
    # save_name can be e.g. sth.pt
    save_path = os.path.join(logger.output_dir, save_name)
    torch.save(dictionary, save_path)
    if verbose > 0:
        print("Model saved to", save_path)

def train_d4rl(env_name, dataset, seed=0, epochs=20, steps_per_epoch=5000,
               max_ep_len=1000, n_evals_per_epoch=10,
               logger_kwargs=dict(), debug=False,
               # following are agent related hyperparameters
               hidden_layer=2, hidden_unit=256,
               replay_size=int(2e6), batch_size=256,
               lr=3e-4, gamma=0.99, polyak=0.995,
               alpha=0.2, auto_alpha=True, target_entropy='mbpo',
               start_steps=5000, delay_update_steps='auto',
               utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
               policy_update_delay=20,
               # following are bias evaluation related
               evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
               # new experiments
               ensemble_decay_n_data=20000, safe_q_target_factor=0.5,
               do_pretrain=False, pretrain_epochs=20, pretrain_mode='pi_sprime',
               save_agent=True, offline_data_ratio=1, agent_type='il',
               cql_weight=1, cql_n_random=10, cql_temp=1, std=0.1,
               # pretrain_mode:
               # 1. pi_sprime
               # 2. pi_mc
               # 3. q_sprime
               # 4. q_mc
               # e.g. we might save the pretrained networks in a folder called pi_predict_next_state_h2_256_e1000
               ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_layer: number of hidden layers
    :param hidden_unit: hidden layer number of units
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param q_target_mode: 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture
    :param policy_update_delay: how many updates until we update policy network
    """
    if debug: # use --debug for very quick debugging
        for _ in range(3):
            print("!!!!USING DEBUG SETTINGS!!!!")
        hidden_layer = 2
        hidden_unit = 3
        batch_size = 4
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100
        epochs = 5
        pretrain_epochs = 5
        n_evals_per_epoch = 1
        offline_data_ratio = 0.4

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set number of epoch
    n_offline_updates = steps_per_epoch * epochs + 1
    n_pretrain_updates = steps_per_epoch * pretrain_epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    #     logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
    #                          exp_name=exp_name)
    logger_kwargs['output_fname'] = 'pretrain_progress.txt'
    pretrain_logger = EpochLogger(**logger_kwargs)

    """set up environment and seeding"""
    env_fn = lambda: gym.make(env_name)

    if env_name == "hopper":
        env_fn = lambda: gym.make("Hopper-v3")
    elif env_name == "halfcheetah":
        env_fn = lambda: gym.make("HalfCheetah-v3")
    elif env_name == "walker2d":
        env_fn = lambda: gym.make("Walker2d-v3")
    elif env_name == "reacher2d":
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env_fn = lambda: Reacher2dEnv()
    else:
        raise NotImplementedError

    # env_fn = lambda: gym.make(env_name)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################
    quit()

    """load data here"""
    dataset = d4rl.qlearning_dataset(env)
    print("Env: %s, number of data loaded: %d." % (env_name, dataset['actions'].shape[0]))

    """init agent and load data into buffer"""
    if agent_type == 'il':
        agent = ILAgent(env_name, obs_dim, act_dim, act_limit, device,
                         hidden_layer, hidden_unit, replay_size, batch_size,
                         lr, gamma, polyak,
                         alpha, auto_alpha, target_entropy,
                         start_steps, delay_update_steps,
                         utd_ratio, num_Q, num_min, q_target_mode,
                         policy_update_delay, ensemble_decay_n_data, safe_q_target_factor)
    if agent_type == 'cql':
        agent = CQLAgent(env_name, obs_dim, act_dim, act_limit, device,
                         hidden_layer, hidden_unit, replay_size, batch_size,
                         lr, gamma, polyak,
                         alpha, auto_alpha, target_entropy,
                         start_steps, delay_update_steps,
                         utd_ratio, num_Q, num_min, q_target_mode,
                         policy_update_delay, ensemble_decay_n_data, safe_q_target_factor,
                         cql_weight, cql_n_random, cql_temp, std,
                         )

    agent.load_data(dataset, offline_data_ratio)


    """========================================== pretrain stage =========================================="""
    # TODO here add check on whether pretrained model already exist
    seed_all(epoch=0)
    pretrain_stage_start_time = time.time()
    if do_pretrain and pretrain_mode is not None:
        print("Pretraining start, mode:",pretrain_mode)
        # check if pretrain
        pretrain_model_folder_path = '/code/pretrain'
        try:
            agent.load_pretrained_model(pretrain_model_folder_path, pretrain_mode, pretrain_epochs)
            pretrain_loaded = True
        except Exception as e:
            pretrain_loaded = False
            print("Load pretrained model failed. Start pretraining.")

        if not pretrain_loaded:
            for t in range(n_pretrain_updates):
                agent.pretrain_update(pretrain_logger, pretrain_mode)

                # End of epoch wrap-up
                if (t+1) % steps_per_epoch == 0:
                    epoch = t // steps_per_epoch
                    """logging"""
                    # Log info about epoch
                    time_used = time.time()-pretrain_stage_start_time
                    time_hrs = int(time_used / 3600 * 100)/100
                    time_total_est_hrs = (n_pretrain_updates/t) * time_hrs
                    pretrain_logger.log_tabular('Epoch', epoch)
                    pretrain_logger.log_tabular('TotalEnvInteracts', t)
                    pretrain_logger.log_tabular('Time', time_used)
                    pretrain_logger.log_tabular('LossPretrain', with_min_and_max=True)
                    pretrain_logger.log_tabular('Hours', time_hrs)
                    pretrain_logger.log_tabular('TotalHoursEst', time_total_est_hrs)
                    pretrain_logger.dump_tabular()

                    # flush logged information to disk
                    sys.stdout.flush()

            time_used = time.time() - pretrain_stage_start_time
            time_hrs = int(time_used / 3600 * 100) / 100
            print('Pretraining finished in %.2f hours.' % time_hrs)
            print('Log saved to %s' % pretrain_logger.output_file.name)
            agent.save_pretrained_model(pretrain_model_folder_path, pretrain_mode, pretrain_epochs)

    agent_after_pretrain = copy_agent_without_buffer(agent)
    """========================================== offline stage =========================================="""
    best_agent = agent
    best_step = 0
    best_iter = 0
    iter_list, step_list, ret_normalized_list = [],[],[]
    best_return, best_return_normalized = 0, 0
    seed_all(100000)
    # keep track of run time
    offline_stage_start_time = time.time()
    print("Offline stage start!")
    for t in range(n_offline_updates):
        agent.update(logger)

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
        # if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            rets, rets_normalized = test_agent_d4rl(agent, test_env, max_ep_len, logger, n_eval=n_evals_per_epoch) # add logging here
            iter_list.append(epoch)
            step_list.append(t)
            ret_normalized_list.append(np.mean(rets_normalized))
            if np.mean(rets_normalized) > best_return_normalized:
                best_return = np.mean(rets)
                best_return_normalized = np.mean(rets_normalized)
                best_agent = copy_agent_without_buffer(agent)
                best_step = t
                best_iter = epoch
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            time_used = time.time()-offline_stage_start_time
            time_hrs = int(time_used / 3600 * 100)/100
            time_total_est_hrs = (n_offline_updates/t) * time_hrs
            logger.log_tabular('Iteration', epoch)
            logger.log_tabular('Steps', t)
            logger.log_tabular('total_time', time_used)
            logger.log_tabular('TestEpRet', average_only=True)
            logger.log_tabular('TestEpNormRet',average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('BestRet', best_return)
            logger.log_tabular('BestNormRet', best_return_normalized)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('Alpha', with_min_and_max=True)
            # logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            # if evaluate_bias:
            #     logger.log_tabular("MCDisRet", with_min_and_max=True)
            #     logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
            #     logger.log_tabular("QPred", with_min_and_max=True)
            #     logger.log_tabular("QBias", with_min_and_max=True)
            #     logger.log_tabular("QBiasAbs", with_min_and_max=True)
            #     logger.log_tabular("NormQBias", with_min_and_max=True)
            #     logger.log_tabular("QBiasSqr", with_min_and_max=True)
            #     logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.log_tabular('Hours', time_hrs)
            logger.log_tabular('TotalHoursEst', time_total_est_hrs)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()
    time_used = time.time() - offline_stage_start_time
    time_hrs = int(time_used / 3600 * 100) / 100
    print('Offline stage finished in %.2f hours.' % time_hrs)
    print('Log saved to %s' % logger.output_file.name)
    if save_agent:
        save_dict(logger, {'agent':copy_agent_without_buffer(agent)}, 'agent.pt')

    """extra info"""
    seed_all(200000)
    final_test_returns, final_test_normalized_returns = np.mean(rets) , np.mean(rets_normalized)
    # TODO find convergence step
    convergence_threshold = best_return_normalized - 2
    k_conv = len(ret_normalized_list) - 1
    for k in range(len(ret_normalized_list)-1,-1,-1):
        if ret_normalized_list[k] >= convergence_threshold:
            k_conv = k
    convergence_iter, convergence_step = iter_list[k_conv], step_list[k_conv]

    """get weight difference and feature difference"""
    final_weight_diff = get_weight_diff(agent, agent_after_pretrain)
    final_feature_diff, _ = get_feature_diff(agent, agent_after_pretrain, agent.replay_buffer)
    best_weight_diff = get_weight_diff(agent, best_agent)
    best_feature_diff, num_feature_timesteps = get_feature_diff(agent, best_agent, agent.replay_buffer)
    extra_dict = {
        'final_weight_diff':final_weight_diff,
        'final_feature_diff':final_feature_diff,
        'best_weight_diff': best_weight_diff,
        'best_feature_diff': best_feature_diff,
        'num_feature_timesteps':num_feature_timesteps,
        'final_test_returns':final_test_returns,
        'final_test_normalized_returns': final_test_normalized_returns,
        'best_return': best_return,
        'best_return_normalized':best_return_normalized,
        'convergence_step':convergence_step,
        'convergence_iter':convergence_iter,
        'best_step':best_step,
        'best_iter':best_iter,
    }
    logger.save_extra_dict_as_json(extra_dict, 'extra.json')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='cql')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--agent', type=str, default='cql')

    args = parser.parse_args()

    ######
    data_dir = '/checkpoints'
    exp_prefix = 'IL'
    exp_suffix = "_%s_%s" % (args.env, args.dataset)
    exp_name_full = exp_prefix + exp_suffix
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, args.seed, data_dir)
    #####

    train_d4rl(args.env, args.dataset, seed=args.seed, epochs=args.epochs,
               logger_kwargs=logger_kwargs, debug=args.debug, do_pretrain=args.pretrain,
               agent_type=args.agent)
