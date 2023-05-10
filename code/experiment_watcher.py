import os
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

import gym
import numpy as np
import torch
import wandb
from logx import EpochLogger
import argparse
import pickle
import random
import sys
from utils import get_normalized_score
from utils import calculate_feature_diff, calculate_statistics, calculate_weight_diff
import copy
from exp_scripts.grid_utils import *
import time
import json
from serialization_utils import convert_json

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer_new import ActTrainer
from decision_transformer.training.seq_trainer_new import SequenceTrainer

from utils import get_optimizer, discount_cumsum
import os


def experiment(
    exp_prefix,
    variant,
):
    if variant["calculate_extra"]:

        torch.manual_seed(variant["seed"])
        os.makedirs(variant["outdir"], exist_ok=True)
        device = variant.get("device", "cuda")
        log_to_wandb = variant.get("log_to_wandb", False)
        seed = variant["seed"]
        env_name, dataset = variant["env"], variant["dataset"]
        model_type = variant["model_type"]
        group_name = f"{exp_prefix}-{env_name}-{dataset}"
        exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

        if env_name == "hopper":
            env = gym.make("Hopper-v3")
            max_ep_len = 1000
            env_targets = [3600, 1800]  # evaluation conditioning targets
            scale = 1000.0  # normalization for rewards/returns
            final_target = 3600
        elif env_name == "halfcheetah":
            env = gym.make("HalfCheetah-v3")
            max_ep_len = 1000
            env_targets = [12000, 6000]
            scale = 1000.0
            final_target = 6000
        elif env_name == "walker2d":
            env = gym.make("Walker2d-v3")
            max_ep_len = 1000
            env_targets = [5000, 2500]
            scale = 1000.0
            final_target = 5000
        elif env_name == "reacher2d":
            from decision_transformer.envs.reacher_2d import Reacher2dEnv

            env = Reacher2dEnv()
            max_ep_len = 100
            env_targets = [76, 40]
            scale = 10.0
        else:
            raise NotImplementedError

        if model_type == "bc":
            env_targets = env_targets[
                :1
            ]  # since BC ignores target, no need for different evaluations

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        dataset_path = f"data/{env_name}-{dataset}-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        np.random.seed(0) # for data size
        num_traj = len(trajectories)
        if variant["data_size"] != 1.0:
            idx = np.random.choice(np.arange(num_traj, dtype=int), size=int(num_traj * variant["data_size"]), replace=False)
            new_trajectories = []
            for t_id in idx:
                new_trajectories.append(trajectories[t_id])
            trajectories = new_trajectories[:]
            del new_trajectories

        # save all path information into separate lists
        mode = variant.get("mode", "normal")
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            if mode == "delayed":  # delayed: all rewards moved to end of trajectory
                path["rewards"][-1] = path["rewards"].sum()
                path["rewards"][:-1] = 0.0
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name} {dataset}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print("=" * 50)

        K = variant["K"]
        batch_size = variant["batch_size"]
        num_eval_episodes = variant["num_eval_episodes"]
        pct_traj = variant.get("pct_traj", 1.0)

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        def get_batch(batch_size=256, max_len=K):
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
                si = random.randint(0, traj["rewards"].shape[0] - 1)
                # get sequences from dataset
                s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
                a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
                r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
                if "terminals" in traj:
                    d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
                else:
                    d.append(traj["dones"][si : si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = (
                    max_ep_len - 1
                )  # padding cutoff
                rtg.append(
                    discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                        : s[-1].shape[1] + 1
                    ].reshape(1, -1, 1)
                )
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
                )
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate(
                    [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
                )
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = (
                    np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                    / scale
                )
                timesteps[-1] = np.concatenate(
                    [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
                )
                mask.append(
                    np.concatenate(
                        [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                    )
                )

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(
                dtype=torch.float32, device=device
            )
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(
                dtype=torch.float32, device=device
            )
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(
                dtype=torch.float32, device=device
            )
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(
                dtype=torch.long, device=device
            )
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
                dtype=torch.float32, device=device
            )
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
                dtype=torch.long, device=device
            )
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

            return s, a, r, d, rtg, timesteps, mask


        if model_type == "dt":
            model = DecisionTransformer(
                args=variant,
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
                n_head=variant["n_head"],
                n_inner=4 * variant["embed_dim"],
                activation_function=variant["activation_function"],
                n_positions=1024,
                resid_pdrop=variant["dropout"],
                attn_pdrop=0.1,
            )
            if variant["load_checkpoint"]:
                state_dict = torch.load(variant["load_checkpoint"])
                model.load_state_dict(state_dict)
                print(f"Loaded from {variant['load_checkpoint']}")
        elif model_type == "bc":
            model = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
            )
        else:
            raise NotImplementedError
        init_model = copy.deepcopy(model)

        model = model.to(device=device)


        (
            final_test_returns,
            final_test_normalized_returns,
            best_return,
            best_return_normalized,
            convergence_step,
            convergence_iter,
            best_step,
            best_iter
        ) = calculate_statistics(variant["outdir"])

        (
            final_weight_diff, 
            final_weight_sim, 
            final_block_weight_diff, 
            final_block_weight_sim 
            ) = calculate_weight_diff(variant["outdir"], variant["max_iters"], init_model)
        (
            final_feature_diff, 
            final_feature_sim,
            num_feature_traj, 
            num_feature_timesteps
            ) = calculate_feature_diff(
                variant,
                variant["outdir"],
                variant["max_iters"],
                init_model,
                trajectories,
                state_mean,
                state_std,
                max_ep_len,
                scale,
                state_dim,
                act_dim
                )
        (
            best_weight_diff, 
            best_weight_sim, 
            best_block_weight_diff, 
            best_block_weight_sim 
            ) = calculate_weight_diff(variant["outdir"], best_iter, init_model)
        (
            best_feature_diff,
            best_feature_sim, 
            num_feature_traj, 
            num_feature_timesteps
            ) = calculate_feature_diff(
                variant,
                variant["outdir"],
                best_iter,
                init_model,
                trajectories,
                state_mean,
                state_std,
                max_ep_len,
                scale,
                state_dim,
                act_dim
                )
        new_final_block_weight_diff = dict()
        for item in final_block_weight_diff:
            new_final_block_weight_diff["final_"+ item+"_weight_diff"] = final_block_weight_diff[item]
        new_final_block_weight_sim = dict()
        for item in final_block_weight_sim:
            new_final_block_weight_sim["final_" + item+"_weight_sim"] = final_block_weight_sim[item]
        new_best_block_weight_diff = dict()
        for item in best_block_weight_diff:
            new_best_block_weight_diff["best_" + item+"_weight_diff"] = best_block_weight_diff[item]
        new_best_block_weight_sim = dict()
        for item in best_block_weight_sim:
            new_best_block_weight_sim["best_" + item+"_weight_sim"] = best_block_weight_sim[item]
        extra_dict = {
            "final_weight_diff": final_weight_diff,
            "final_weight_sim": final_weight_sim,
            "final_feature_diff": final_feature_diff,
            "final_feature_sim": final_feature_sim,
            **new_final_block_weight_diff,
            **new_final_block_weight_sim,
            "best_weight_diff": best_weight_diff,
            "best_weight_sim": best_weight_sim,
            "best_feature_diff": best_feature_diff,
            "best_feature_sim": best_feature_sim,
            **new_best_block_weight_diff,
            **new_best_block_weight_sim,
            "num_feature_traj": num_feature_traj,
            "num_feature_timesteps": num_feature_timesteps,
            "final_test_returns": final_test_returns,
            "final_test_normalized_returns": final_test_normalized_returns,
            "best_return": best_return,
            "best_return_normalized": best_return_normalized,
            "convergence_step": convergence_step,
            "convergence_iter": convergence_iter,
            "best_step": best_step,
            "best_iter": best_iter
            }

        with open(os.path.join(variant["outdir"], "extra_new.json"), "w") as f:
            extra_dict = convert_json(extra_dict)
            json.dump(extra_dict, f, indent=4)
        return
    logger = EpochLogger(variant["outdir"], variant["output_fname"], variant["exp_name"])
    logger.save_config(locals())

    torch.manual_seed(variant["seed"])
    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)
    seed = variant["seed"]
    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
        final_target = 3600
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
        final_target = 6000
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.0
        final_target = 5000
    elif env_name == "reacher2d":
        from decision_transformer.envs.reacher_2d import Reacher2dEnv

        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.0
    else:
        raise NotImplementedError

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations
        final_target = 1

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f"data/{env_name}-{dataset}-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    np.random.seed(0) # for data size
    num_traj = len(trajectories)
    if variant["data_size"] != 1.0:
        idx = np.random.choice(np.arange(num_traj, dtype=int), size=int(num_traj * variant["data_size"]), replace=False)
        new_trajectories = []
        for t_id in idx:
            new_trajectories.append(trajectories[t_id])
        trajectories = new_trajectories[:]
        del new_trajectories
        
    def seed_numpy(epoch=None):
        if epoch is not None:
            seed_shift = epoch * 9999
            mod_value = 9999999
            env_seed = (seed + seed_shift) % mod_value
            np.random.seed(env_seed)
            random.seed(env_seed)

    def seed_env(epoch=None, env=None):
        if env is not None and epoch is not None:
            seed_shift = epoch * 9999
            mod_value = 999999
            env_seed = (seed + seed_shift) % mod_value
            env.seed(env_seed)
            env.action_space.np_random.seed(env_seed)

    # save all path information into separate lists
    mode = variant.get("mode", "normal")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)
            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    if model_type == "dt":
        model = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
        )
        if variant["load_checkpoint"]:
            state_dict = torch.load(variant["load_checkpoint"])
            model.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    init_model = copy.deepcopy(model)

    if variant["perturb"]:
        if variant["perturb_per_layer"] and variant["perturb_absolute"]:
            raise Exception("only one std value mode can be set at a time.")
        elif not variant["perturb_per_layer"] and not variant["perturb_absolute"]:
            raise Exception("set a std value")
        
        if variant["perturb_per_layer"]:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if variant["perturb_transformer_only"]:
                        if "transformer" not in n:
                            continue
                    if variant["perturb_attn_only"]:
                        if "attn" not in n:
                            continue
                    if variant["perturb_mlp_only"]:
                        if "mlp" not in n:
                            continue
                    if variant["perturb_ln_only"]:
                        if "ln" not in n:
                            continue
                    if variant["not_perturb_attn"]:
                        if "attn" in n:
                            continue
                    if variant["not_perturb_mlp"]:
                        if "mlp" in n:
                            continue
                    if variant["not_perturb_ln"]:
                        if "ln" in n:
                            continue
                    orig_std = torch.std(p)
                    p.add_(torch.normal(torch.zeros_like(p), orig_std * variant["perturb_per_layer"]))
        elif variant["perturb_absolute"]:
            with torch.no_grad():
                for p in model.parameters():
                    if variant["perturb_transformer_only"]:
                        if "transformer" not in p:
                            continue
                    p.add_(torch.normal(torch.zeros_like(p), 1. * variant["perturb_absolute"]))

    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant["max_iters"]):
        print("HI!")

        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, seeder=seed_numpy, print_logs=True
        )
        print("HI2!")
        logger.log_tabular("current_itr_train_time", outputs["time/training"])
        logger.log_tabular("current_itr_train_loss_mean", outputs["training/train_loss_mean"])
        logger.log_tabular("current_itr_train_loss_std", outputs["training/train_loss_std"])
        logger.log_tabular("current_itr_eval_" + str(env_targets[0]) + "_return_mean", outputs["evaluation/target_" + str(env_targets[0]) + "_return_mean"])
        logger.log_tabular("current_itr_eval_" + str(env_targets[0]) + "_return_std", outputs["evaluation/target_" + str(env_targets[0]) + "_return_std"])
        logger.log_tabular("current_itr_eval_" + str(env_targets[0]) + "_length_mean", outputs["evaluation/target_" + str(env_targets[0]) + "_length_mean"])
        logger.log_tabular("current_itr_eval_" + str(env_targets[0]) + "_length_std", outputs["evaluation/target_" + str(env_targets[0]) + "_length_std"])
        if len(env_targets) > 1:
            logger.log_tabular("current_itr_eval_" + str(env_targets[1]) + "_return_mean", outputs["evaluation/target_" + str(env_targets[1]) + "_return_mean"])
            logger.log_tabular("current_itr_eval_" + str(env_targets[1]) + "_return_std", outputs["evaluation/target_" + str(env_targets[1]) + "_return_std"])
            logger.log_tabular("current_itr_eval_" + str(env_targets[1]) + "_length_mean", outputs["evaluation/target_" + str(env_targets[1]) + "_length_mean"])
            logger.log_tabular("current_itr_eval_" + str(env_targets[1]) + "_length_std", outputs["evaluation/target_" + str(env_targets[1]) + "_length_std"])
        logger.log_tabular("TestEpRet", outputs["evaluation/target_" + str(final_target) + "_return_mean"])
        logger.log_tabular("TestEpNormRet", get_normalized_score(env_name, outputs["evaluation/target_" + str(final_target) + "_return_mean"]))
        logger.log_tabular("Iteration", iter + 1)
        logger.log_tabular("Steps", (iter + 1) * variant["num_steps_per_iter"])
        logger.log_tabular("total_time", outputs["time/total"])
        logger.log_tabular("current_eval_time", outputs["time/evaluation"])
        logger.dump_tabular()
        if log_to_wandb:
            wandb.log(outputs)

    (
        final_test_returns,
        final_test_normalized_returns,
        best_return,
        best_return_normalized,
        convergence_step,
        convergence_iter,
        best_step,
        best_iter
    ) = calculate_statistics(variant["outdir"])

    (
        final_weight_diff, 
        final_weight_sim, 
        final_block_weight_diff, 
        final_block_weight_sim 
        ) = calculate_weight_diff(variant["outdir"], variant["max_iters"], init_model)
    (
        final_feature_diff, 
        final_feature_sim,
        num_feature_traj, 
        num_feature_timesteps
        ) = calculate_feature_diff(
            variant,
            variant["outdir"],
            variant["max_iters"],
            init_model,
            trajectories,
            state_mean,
            state_std,
            max_ep_len,
            scale,
            state_dim,
            act_dim
            )
    (
        best_weight_diff, 
        best_weight_sim, 
        best_block_weight_diff, 
        best_block_weight_sim 
        ) = calculate_weight_diff(variant["outdir"], best_iter, init_model)
    (
        best_feature_diff,
        best_feature_sim, 
        num_feature_traj, 
        num_feature_timesteps
        ) = calculate_feature_diff(
            variant,
            variant["outdir"],
            best_iter,
            init_model,
            trajectories,
            state_mean,
            state_std,
            max_ep_len,
            scale,
            state_dim,
            act_dim
            )
    new_final_block_weight_diff = dict()
    for item in final_block_weight_diff:
        new_final_block_weight_diff["final_"+ item+"_weight_diff"] = final_block_weight_diff[item]
    new_final_block_weight_sim = dict()
    for item in final_block_weight_sim:
        new_final_block_weight_sim["final_" + item+"_weight_sim"] = final_block_weight_sim[item]
    new_best_block_weight_diff = dict()
    for item in best_block_weight_diff:
        new_best_block_weight_diff["best_" + item+"_weight_diff"] = best_block_weight_diff[item]
    new_best_block_weight_sim = dict()
    for item in best_block_weight_sim:
        new_best_block_weight_sim["best_" + item+"_weight_sim"] = best_block_weight_sim[item]
    extra_dict = {
        "final_weight_diff": final_weight_diff,
        "final_weight_sim": final_weight_sim,
        "final_feature_diff": final_feature_diff,
        "final_feature_sim": final_feature_sim,
        **new_final_block_weight_diff,
        **new_final_block_weight_sim,
        "best_weight_diff": best_weight_diff,
        "best_weight_sim": best_weight_sim,
        "best_feature_diff": best_feature_diff,
        "best_feature_sim": best_feature_sim,
        **new_best_block_weight_diff,
        **new_best_block_weight_sim,
        "num_feature_traj": num_feature_traj,
        "num_feature_timesteps": num_feature_timesteps,
        "final_test_returns": final_test_returns,
        "final_test_normalized_returns": final_test_normalized_returns,
        "best_return": best_return,
        "best_return_normalized": best_return_normalized,
        "convergence_step": convergence_step,
        "convergence_iter": convergence_iter,
        "best_step": best_step,
        "best_iter": best_iter
        }

    with open(os.path.join(variant["outdir"], "extra.json"), "w") as f:
        extra_dict = convert_json(extra_dict)
        json.dump(extra_dict, f, indent=4)

def set_dt_args(args_to_parse=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)

    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--num_steps_per_iter", type=int, default=5000)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_fname", type=str, default="progress.csv")

    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--gpt_kmeans", type=int, default=None)
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--gpt_kmeans_const", type=float, default=None)
    parser.add_argument("--kmeans_cache", type=str, default=None)

    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    parser.add_argument("--kmeans_mean", action="store_true", default=False)

    parser.add_argument("--perturb", action="store_true", default=False)
    parser.add_argument("--perturb_transformer_only", action="store_true", default=False)
    parser.add_argument("--perturb_attn_only", action="store_true", default=False)
    parser.add_argument("--perturb_mlp_only", action="store_true", default=False)
    parser.add_argument("--perturb_ln_only", action="store_true", default=False)
    parser.add_argument("--not_perturb_attn", action="store_true", default=False)
    parser.add_argument("--not_perturb_mlp", action="store_true", default=False)
    parser.add_argument("--not_perturb_ln", action="store_true", default=False)
    parser.add_argument("--perturb_per_layer", type=float, default=None)
    parser.add_argument("--perturb_absolute", type=float, default=None)

    parser.add_argument("--data_size", type=float, default=1.0)
    parser.add_argument("--calculate_extra", action="store_true", default=False)

    if args_to_parse is not None:
        args = parser.parse_args(args_to_parse)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    start_time = time.time()
    args = set_dt_args()
    '''
    new_args = sys.argv[10:-2]
    args_dict = dict()
    num_args = len(new_args) // 2
    for i in range(0, new_args, 2):
        args_dict[new_args[i][2:]] = new_args[i+1]
    '''
    data_dir = '/checkpoints'
    exp_prefix = args.outdir
    exp_suffix = "_%s_%s" % (args.env, args.dataset)
    exp_name_full = exp_prefix + exp_suffix
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, args.seed, data_dir)
    args.outdir = logger_kwargs["output_dir"]
    args.exp_name = logger_kwargs["exp_name"]

    experiment("gym-experiment", variant=vars(args))
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

