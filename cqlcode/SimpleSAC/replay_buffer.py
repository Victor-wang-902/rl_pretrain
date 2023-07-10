from copy import copy, deepcopy
from queue import Queue
import threading

import d4rl

import numpy as np
import torch
import joblib


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )

def get_d4rl_dataset_with_ratio(env, ratio=1, seed=0):
    dataset = d4rl.qlearning_dataset(env)
    n_data = dataset['observations'].shape[0]
    use_size = int(n_data * ratio)
    np.random.seed(seed)
    idxs = np.random.choice(n_data, use_size, replace=False)

    return dict(
        observations=dataset['observations'][idxs],
        actions=dataset['actions'][idxs],
        next_observations=dataset['next_observations'][idxs],
        rewards=dataset['rewards'][idxs],
        dones=dataset['terminals'][idxs].astype(np.float32),
    )

def get_d4rl_dataset_from_multiple_envs(envs):
    n_data = 0
    d = None
    for env in envs:
        dataset = d4rl.qlearning_dataset(env)
        dataset['dones'] = dataset['terminals']
        del dataset['terminals']
        n_data += dataset['observations'].shape[0]
        if not d:
            d = dict(
                observations = dataset['observations'],
                actions = dataset['actions'],
                next_observations = dataset['next_observations'],
                rewards = dataset['rewards'],
                dones = dataset['dones'].astype(np.float32),
            )
        else:
            for key in d:
                d[key] = np.concatenate((d[key], dataset[key]), axis=0)
    return d

def get_mdp_dataset_with_ratio(n_traj, n_state, n_action, policy_temperature, transition_temperature,
                               ratio=1, seed=0, verbose=True):
    data_name = 'mdp_traj%d_ns%d_na%d_pt%s_tt%s.pkl' % (n_traj, n_state, n_action,
                                                        str(policy_temperature), str(transition_temperature))
    save_name = '/cqlcode/mdpdata/%s' % data_name

    dataset = joblib.load(save_name)
    if verbose:
        print("MDP pretrain data loaded from:", save_name)

    n_data = dataset['observations'].shape[0]
    use_size = int(n_data * ratio)
    np.random.seed(seed)
    idxs = np.random.choice(n_data, use_size, replace=False)

    return dict(
        observations=dataset['observations'][idxs],
        actions=dataset['actions'][idxs],
        next_observations=dataset['next_observations'][idxs],
    )



def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
