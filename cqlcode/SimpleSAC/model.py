import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)


        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, orthogonal_init
        )

    @multiple_action_q_function
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)

class FullyConnectedQFunctionPretrain(nn.Module):

    def __init__(self, obs_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        self.hidden_layers = nn.ModuleList()
        self.hidden_activation = F.relu
        d = obs_dim + action_dim

        hidden_sizes = [int(h) for h in arch.split('-')]
        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            self.hidden_layers.append(fc)
            d = hidden_size

        self.last_fc_layer = nn.Linear(d, self.output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(self.last_fc_layer.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(self.last_fc_layer.weight, gain=1e-2)

        nn.init.constant_(self.last_fc_layer.bias, 0.0)

        # pretrain mode: q_sprime
        self.hidden_to_next_obs = nn.Linear(d, obs_dim)
        # pretrain mode: q_mc
        self.hidden_to_value = nn.Linear(d, 1)
        # pretrain mode: q_mle
        self.hidden_to_dist = nn.Linear(d, 2 * obs_dim)

        if orthogonal_init:
            nn.init.orthogonal_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.orthogonal_(self.hidden_to_value.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.xavier_uniform_(self.hidden_to_value.weight, gain=1e-2)

        nn.init.constant_(self.hidden_to_next_obs.bias, 0.0)
        nn.init.constant_(self.hidden_to_value.bias, 0.0)

    def get_feature(self, observations, actions):
        h = torch.cat([observations, actions], dim=-1)
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def predict_next_obs(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_next_obs(h)

    def predict_value(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_value(h)

    def predict_next_dist(self, obervations, actions):
        h = self.get_feature(obervations, actions)
        stats = self.hidden_to_dist(h)
        mean, log_std = torch.split(stats, self.obs_dim, dim=-1)
        log_std = torch.clamp(log_std, min=-2.0, max=7.5)
        std = torch.exp(log_std)
        obs_distribution = Normal(mean, std)
        return obs_distribution

    @multiple_action_q_function
    def forward(self, observations, actions):
        h = self.get_feature(observations, actions)
        return torch.squeeze(self.last_fc_layer(h), dim=-1)

    # def forward(self, observations, actions):
    #     multiple_actions = False
    #     batch_size = observations.shape[0]
    #     if actions.ndim == 3 and observations.ndim == 2:
    #         multiple_actions = True
    #         observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
    #         actions = actions.reshape(-1, actions.shape[-1])
    #
    #     h = self.get_feature(observations, actions)
    #     q_values = self.last_fc_layer(h)
    #     if multiple_actions:
    #         q_values = q_values.reshape(batch_size, -1)
    #     return q_values


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant
