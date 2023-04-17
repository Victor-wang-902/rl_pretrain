from torch import optim
from itertools import chain
from infos import REF_MAX_SCORE, REF_MIN_SCORE
import pandas as pd
import numpy as np
import torch
import os
import copy

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def get_optimizer(args, model):
    if args["pretrained_lm"]:
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" not in str(type(module)))
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
    return optimizer

def get_normalized_score(env_name, score):
    name = env_name + "-medium-v0"
    return 100 * (score - REF_MIN_SCORE[name]) / (REF_MAX_SCORE[name] - REF_MIN_SCORE[name])

def calculate_statistics(dir, fname="progress.csv"):
    with open(os.path.join(dir, fname), "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        final_test_returns = df["TestEpRet"].iloc[-1]
        final_test_normalized_returns = df["TestEpNormRet"].iloc[-1]
        best_return = max(df["TestEpRet"])
        best_return_normalized = max(df["TestEpNormRet"])
        convergence_step = df["Steps"].iloc[df["TestEpRet"].ge(best_return_normalized - 2.0).idxmax()]
        convergence_iter = df["Iteration"].iloc[df["TestEpRet"].ge(best_return_normalized - 2.0).idxmax()]
        best_step = df["Steps"][df["TestEpRet"] == best_return].iat[0]
        best_iter = df["Iteration"][df["TestEpRet"] == best_return].iat[0]

    return final_test_returns, final_test_normalized_returns, best_return, best_return_normalized, convergence_step, convergence_iter, best_step, best_iter

@torch.no_grad()
def calculate_weight_diff(dir, iter, init_model, weight_only=True):
    model_state_dict = torch.load(os.path.join(dir, "model_" + str(iter) + ".pt"), map_location=torch.device("cpu"))
    layers = []
    for name, layer in model_state_dict.items():
        if "transformer" in name:
            if not weight_only or "weight" in name:
                layers.append(layer.view(-1))
    weights = torch.cat(layers)
    init_model.to("cpu")
    init_state_dict = init_model.state_dict()
    layers = []
    for name, layer in init_state_dict.items():
        if "transformer" in name:
            if not weight_only or "weight" in name:
                layers.append(layer.view(-1))
    init_weights = torch.cat(layers)
    weight_diff = torch.norm(weights - init_weights, p=2).item()
    return weight_diff

@torch.no_grad()
def calculate_feature_diff(
    variant, 
    dir, 
    iter, 
    init_model, 
    data, 
    state_mean, 
    state_std, 
    max_ep_len, 
    scale, 
    state_dim, 
    act_dim, 
    ratio=0.1
    ):
    init_model.to("cpu")
    cur_state_dict = torch.load(os.path.join(dir, "model_" + str(iter) + ".pt"), map_location=torch.device("cpu"))
    cur_model = copy.deepcopy(init_model)
    cur_model.load_state_dict(cur_state_dict)
    num_traj = len(data)
    np.random.seed(0)
    idx = np.random.choice(np.arange(num_traj, dtype=int), size=int(num_traj * ratio), replace=False)
    new_trajectories = []
    for t_id in idx:
        new_trajectories.append(data[t_id])
    trajectories = new_trajectories[:]
    del new_trajectories

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)

    num_timesteps = sum(traj_lens)
    print("=" * 50)
    print(f"Feature difference calculations.")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print("=" * 50)
    batch_size = variant["batch_size"]
    K = variant["K"]
    cur_model.eval()
    init_model.eval()
    def yield_batch():
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        batch_id = 0
        for traj in trajectories:
            start_pos = 0
            traj_l = len(traj["observations"])
            while True:
                s.append(traj["observations"][start_pos : start_pos + K].reshape(1, -1, state_dim))
                a.append(traj["actions"][start_pos : start_pos + K].reshape(1, -1, act_dim))
                r.append(traj["rewards"][start_pos : start_pos + K].reshape(1, -1, 1))
                if "terminals" in traj:
                    d.append(traj["terminals"][start_pos : start_pos + K].reshape(1, -1))
                else:
                    d.append(traj["dones"][start_pos : start_pos + K].reshape(1, -1))
                timesteps.append(np.arange(start_pos, start_pos + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = (
                    max_ep_len - 1
                )  # padding cutoff
                rtg.append(
                    discount_cumsum(traj["rewards"][start_pos:], gamma=1.0)[
                        : s[-1].shape[1] + 1
                    ].reshape(1, -1, 1)
                )
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate(
                    [np.zeros((1, K - tlen, state_dim)), s[-1]], axis=1
                )
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate(
                    [np.ones((1, K - tlen, act_dim)) * -10.0, a[-1]], axis=1
                )
                r[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, K - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = (
                    np.concatenate([np.zeros((1, K - tlen, 1)), rtg[-1]], axis=1)
                    / scale
                )
                timesteps[-1] = np.concatenate(
                    [np.zeros((1, K - tlen)), timesteps[-1]], axis=1
                )
                mask.append(
                    np.concatenate(
                        [np.zeros((1, K - tlen)), np.ones((1, tlen))], axis=1
                    )
                )

                if batch_id >= batch_size - 1:
                    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
                        dtype=torch.float32, device=variant["device"]
                    )
                    a = torch.from_numpy(np.concatenate(a, axis=0)).to(
                        dtype=torch.float32, device=variant["device"]
                    )
                    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
                        dtype=torch.float32, device=variant["device"]
                    )
                    d = torch.from_numpy(np.concatenate(d, axis=0)).to(
                        dtype=torch.long, device=variant["device"]
                    )
                    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
                        dtype=torch.float32, device=variant["device"]
                    )
                    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
                        dtype=torch.long, device=variant["device"]
                    )
                    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=variant["device"])
                    yield s, a, r, d, rtg, timesteps, mask
                    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
                    batch_id = 0
                else:
                    batch_id += 1

                if start_pos + K >= traj_l - 1:
                    start_pos = 0
                    break
                else:
                    start_pos += K
    average_feature_norm_list = []
    cur_model.to(variant["device"])
    init_model.to(variant["device"])
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in yield_batch():        
        cur_feature = cur_model.get_feature(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        init_feature = init_model.get_feature(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )
        feature_diff = (init_feature - cur_feature).cpu()
        temp = attention_mask[:,:,None].to(torch.device("cpu"))
        masked_feature_diff = feature_diff * temp
        masked_feature_diff = masked_feature_diff.reshape(batch_size * K, -1)
        feature_norm = torch.norm(masked_feature_diff, p=2, dim=1, keepdim=True)
        average_feature_norm_list.append(feature_norm.mean().item())
        #except Exception as err:
        #    raise Exception(err)
        #    break

    return np.mean(average_feature_norm_list), len(traj_lens), num_timesteps