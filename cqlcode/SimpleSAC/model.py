import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F
from text_encoding.utils import get_text_embedding, get_text_embedding_tensor
from text_encoding.info import TEXT_DESCRIPTIONS



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

#########################################################################################
class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, pretrain_obs_dim=None, pretrain_act_dim=None, encoder_dim=0, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False, pretrain_env_name=None, offline_env_name=None, encoder=None, tokenizer=None):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim + encoder_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)
        if pretrain_obs_dim is not None:
            self.has_proj = True
            if isinstance(pretrain_obs_dim, list):
                pretrain_env_unique = list(dict.fromkeys([name.split("-")[0] for name in pretrain_env_name]))
                pretrain_obs_unique = list(dict.fromkeys(pretrain_obs_dim))
                pretrain_act_unique = list(dict.fromkeys(pretrain_act_dim))
                #print("debug", pretrain_env_unique)
                #print("debug", pretrain_obs_unique)
                #print("debug", pretrain_act_unique)
                #assert len(pretrain_env_unique) == len(pretrain_obs_unique) and len(pretrain_obs_unique) == len(pretrain_act_unique)
                if len(pretrain_env_unique) == 1:
                    self.proj = nn.Linear(pretrain_obs_unique[0] + encoder_dim, observation_dim)
                    self.out = nn.Linear(2 * action_dim, 2 * pretrain_act_unique[0])
                else:
                    self.proj = nn.ModuleDict({name.split("-")[0]: nn.Linear(obs + encoder_dim, observation_dim) for (name, obs) in zip(pretrain_env_name, pretrain_obs_dim)})
                    self.out = nn.ModuleDict({name.split("-")[0]: nn.Linear(2 * action_dim, 2 * act) for (name, act) in zip(pretrain_env_name, pretrain_act_dim)})
                    #self.proj = nn.ModuleDict({name: nn.Linear(obs + encoder_dim, observation_dim) for (name, obs) in zip(pretrain_env_unique, pretrain_obs_unique)})
                    #self.out = nn.ModuleDict({name: nn.Linear(2 * action_dim, 2 * act) for (name, act) in zip(pretrain_env_unique, pretrain_act_unique)})
            else:
                self.proj = nn.Linear(pretrain_obs_dim + encoder_dim, observation_dim)
                self.out = nn.Linear(2 * action_dim, 2 * pretrain_act_dim)
            ##DEBUG
            #print(f"self proj: {self.proj}")
            #print(f"self out: {self.out}")
            ##

            if orthogonal_init:
                if isinstance(self.proj, nn.ModuleDict):
                    for name in self.proj.keys():
                        nn.init.orthogonal_(self.proj[name].weight, gain=1e-2)
                else:
                    nn.init.orthogonal_(self.proj.weight, gain=1e-2)

                if isinstance(self.out, nn.ModuleDict):
                    for name in self.out.keys():
                        nn.init.orthogonal_(self.out[name].weight, gain=1e-2)
                else:
                    nn.init.orthogonal_(self.out.weight, gain=1e-2)
            else:
                if isinstance(self.proj, nn.ModuleDict):
                    for name in self.proj.keys():
                        nn.init.xavier_uniform_(self.proj[name].weight, gain=1e-2)
                else:
                    nn.init.xavier_uniform_(self.proj.weight, gain=1e-2)

                if isinstance(self.out, nn.ModuleDict):
                    for name in self.out.keys():
                        nn.init.xavier_uniform_(self.out[name].weight, gain=1e-2)
                else:
                    nn.init.xavier_uniform_(self.out.weight, gain=1e-2)
            
            if isinstance(self.proj, nn.ModuleDict):
                for name in self.proj.keys():
                    nn.init.constant_(self.proj[name].bias, 0.0)
            else:
                nn.init.constant_(self.proj.bias, 0.0)

            if isinstance(self.out, nn.ModuleDict):
                for name in self.out.keys():
                    nn.init.constant_(self.out[name].bias, 0.0)
            else:
                nn.init.constant_(self.out.bias, 0.0)

        else:
            self.has_proj = False
        if encoder_dim != 0:
            self.observations_text, _ = get_text_embedding_tensor(encoder, tokenizer, TEXT_DESCRIPTIONS[offline_env_name])
        else:
            self.observations_text = None
        ##DEBUG
        #print(f"self observation text: {self.observations_text}")
        ##

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, observations_text=None, deterministic=False, repeat=None, chosen=None, proj=None):


        if observations_text is not None:
            #print("debug", observations_text)
            observations = torch.concat([observations, observations_text.expand(observations.shape[0], -1)], dim=1)
        '''
        if not deterministic:
            ##DEBUG
            #print("policy forward")
            #print(f"policy observations dim: {observations.shape}")
            #print(f"policy observations text dim: {observations_text}")
            #print(f"policy chosen: {chosen}")
            #print(f"policy proj: {proj}")
            #print(f"policy observations after concat: {observations.shape}")
            ##
        '''
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        '''
        if not deterministic:
            ##DEBUG
            #print(f"policy observations after repeat: {observations.shape}")
            ##
        '''
        if proj:
            if chosen is not None:
                projection = self.proj[chosen]
                out = self.out[chosen]
            else:
                projection = self.proj
                out = self.out
            observations = projection(observations)
            #print("debug", observations.shape)
            if observations_text is not None:
                if len(observations.shape) == 3:
                    observations = torch.concat([observations, self.observations_text.expand(observations.shape[0], observations.shape[1], -1)], dim=-1)
                else:
                    observations = torch.concat([observations, self.observations_text.expand(observations.shape[0], -1)], dim=1)
            '''
            if not deterministic:
                ##DEBUG
                #print(f"policy observations after projection and concat: {observations.shape}")
                ##
            '''
        base_network_output = self.base_network(observations)
        
        ##DEBUG
        '''
        if not deterministic:
            print(f"base network output dim: {base_network_output.shape}")
        '''
        ##
        if proj:
            base_network_output = out(base_network_output)
            ##DEBUG
            '''
            if not deterministic:
                print(f"base network output after output layer: {base_network_output.shape}")
            '''
            ##
            mean, log_std = torch.split(base_network_output, base_network_output.shape[-1] // 2, dim=-1)

        else:
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
            actions, _ = self.policy(observations, observations_text=self.policy.observations_text, deterministic=deterministic)
            actions = actions.cpu().numpy()
        return actions

################################################################################################
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


class FullyConnectedQFunctionPretrain2(nn.Module):
    # this version can support pretraining on data from a different task, with a projection layer...

    def __init__(self, obs_dim, action_dim, pre_obs_dim, pre_act_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        self.pre_obs_dim = pre_obs_dim
        self.pre_act_dim = pre_act_dim

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

        # pretrain mode: proj_q_sprime
        # use a linear layer to project whatever pretraining task data dim into downstream task input dim.
        # this proj layer is not used in downstream task
        self.proj = nn.Linear(pre_obs_dim + pre_act_dim, obs_dim + action_dim)

        # pretrain mode: q_sprime
        self.hidden_to_next_obs = nn.Linear(d, pre_obs_dim)
        # pretrain mode: q_mc
        self.hidden_to_value = nn.Linear(d, 1)

        if orthogonal_init:
            nn.init.orthogonal_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.orthogonal_(self.hidden_to_value.weight, gain=1e-2)
            nn.init.orthogonal_(self.proj.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.xavier_uniform_(self.hidden_to_value.weight, gain=1e-2)
            nn.init.xavier_uniform_(self.proj.weight, gain=1e-2)

        nn.init.constant_(self.hidden_to_next_obs.bias, 0.0)
        nn.init.constant_(self.hidden_to_value.bias, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def get_feature(self, observations, actions):
        h = torch.cat([observations, actions], dim=-1)
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def get_pretrain_next_obs(self, observations, actions):
        h = torch.cat([observations, actions], dim=-1)
        h = self.proj(h)
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return self.hidden_to_next_obs(h)

    def predict_next_obs(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_next_obs(h)

    def predict_value(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_value(h)

    @multiple_action_q_function
    def forward(self, observations, actions):
        h = self.get_feature(observations, actions)
        return torch.squeeze(self.last_fc_layer(h), dim=-1)


############################################################################
class FullyConnectedQFunctionPretrain3(nn.Module):
    # this version can support pretraining on data from a different task, with a projection layer...

    def __init__(self, obs_dim, action_dim, pre_obs_dim, pre_act_dim, encoder_dim=0, arch='256-256', orthogonal_init=False, pretrain_env_name=None, offline_env_name=None, encoder=None, tokenizer=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.pre_obs_dim = pre_obs_dim
        self.pre_act_dim = pre_act_dim

        self.hidden_layers = nn.ModuleList()
        self.hidden_activation = F.relu
        d = obs_dim + action_dim + 2 * encoder_dim
        if encoder_dim != 0:
            self.observations_text, self.actions_text = get_text_embedding_tensor(encoder, tokenizer, TEXT_DESCRIPTIONS[offline_env_name])
        else:
            self.observations_text = None
            self.actions_text = None
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

        # pretrain mode: proj_q_sprime
        # use a linear layer to project whatever pretraining task data dim into downstream task input dim.
        # this proj layer is not used in downstream task
        if isinstance(pre_obs_dim, list):
            #print("debug", pre_obs_dim)
            pretrain_env_unique = list(dict.fromkeys([name.split("-")[0] for name in pretrain_env_name]))
            pretrain_obs_unique = list(dict.fromkeys(pre_obs_dim))
            pretrain_act_unique = list(dict.fromkeys(pre_act_dim))
            #print("debug", pretrain_env_unique)
            #print("debug", pretrain_obs_unique)
            #assert len(pretrain_env_unique) == len(pretrain_obs_unique) and len(pretrain_obs_unique) == len(pretrain_act_unique)
            if len(pretrain_env_unique) == 1:
                self.proj = nn.Linear(pretrain_obs_unique[0] + pretrain_act_unique[0] + 2 * encoder_dim, obs_dim + action_dim)
            else:
                #self.proj = nn.ModuleDict({name: nn.Linear(obs + act + 2 * encoder_dim, obs_dim + action_dim) for (name, obs, act) in zip(pretrain_env_unique, pretrain_obs_unique, pretrain_act_unique)})
                self.proj = nn.ModuleDict({name.split("-")[0]: nn.Linear(obs + act + 2 * encoder_dim, obs_dim + action_dim) for (name, obs, act) in zip(pretrain_env_name, pre_obs_dim, pre_act_dim)})
        else:
            self.proj = nn.Linear(pre_obs_dim + pre_act_dim + 2 * encoder_dim, obs_dim + action_dim)

        # pretrain mode: q_sprime
        if isinstance(pre_obs_dim, list):
            if len(pretrain_env_unique) == 1:
                self.hidden_to_next_obs = nn.Linear(d, pretrain_obs_unique[0])
            else:
                #self.hidden_to_next_obs = nn.ModuleDict({name: nn.Linear(d, obs) for (name, obs) in zip(pretrain_env_name, pretrain_obs_unique)})
                self.hidden_to_next_obs = nn.ModuleDict({name.split("-")[0]: nn.Linear(d, obs) for (name, obs) in zip(pretrain_env_name, pre_obs_dim)})
        else:
            self.hidden_to_next_obs = nn.Linear(d, pre_obs_dim)
        # pretrain mode: q_mc
        self.hidden_to_value = nn.Linear(d, 1)

        if orthogonal_init:
            if isinstance(self.hidden_to_next_obs, nn.ModuleDict):
                for name in self.hidden_to_next_obs.keys():
                    nn.init.orthogonal_(self.hidden_to_next_obs[name].weight, gain=1e-2)
            else:
                nn.init.orthogonal_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.orthogonal_(self.hidden_to_value.weight, gain=1e-2)
            if isinstance(self.proj, nn.ModuleDict):
                for name in self.proj.keys():
                    nn.init.orthogonal_(self.proj[name].weight, gain=1e-2)
            else:
                nn.init.orthogonal_(self.proj.weight, gain=1e-2)
        else:
            if isinstance(self.hidden_to_next_obs, nn.ModuleDict):
                for name in self.hidden_to_next_obs.keys():
                    nn.init.xavier_uniform_(self.hidden_to_next_obs[name].weight, gain=1e-2)
            else:
                nn.init.xavier_uniform_(self.hidden_to_next_obs.weight, gain=1e-2)
            nn.init.xavier_uniform_(self.hidden_to_value.weight, gain=1e-2)
            if isinstance(self.proj, nn.ModuleDict):
                for name in self.proj.keys():
                    nn.init.xavier_uniform_(self.proj[name].weight, gain=1e-2)
            else:
                nn.init.xavier_uniform_(self.proj.weight, gain=1e-2)
        
        if isinstance(self.hidden_to_next_obs, nn.ModuleDict):
            for name in self.hidden_to_next_obs.keys():
                nn.init.constant_(self.hidden_to_next_obs[name].bias, 0.0)
        else:
            nn.init.constant_(self.hidden_to_next_obs.bias, 0.0)
        nn.init.constant_(self.hidden_to_value.bias, 0.0)
        if isinstance(self.proj, nn.ModuleDict):
            for name in self.proj.keys():
                nn.init.constant_(self.proj[name].bias, 0.0)
        else:
            nn.init.constant_(self.proj.bias, 0.0)
        ##DEBUG
        '''
        print(f"q proj: {self.proj}")
        print(f"q hidden to next obs: {self.hidden_to_next_obs}")
        if self.observations_text is not None:
            print(f"q observations text: {self.observations_text.shape}")
            print(f"q actions text: {self.actions_text.shape}")
        else:
            print(f"q observations text: {self.observations_text}")
            print(f"q actions text: {self.actions_text}")
        '''
        ##

    def get_feature(self, observations, actions): # has to be used after concat
        h = torch.cat([observations, actions], dim=-1)
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h
    ########################################

    def get_feature_extra(self, observations, actions, observations_text=None, actions_text=None): # get feature forward style for offline
        if observations_text is not None:
            observations = torch.concat([observations, observations_text.expand(observations.shape[0], -1)], dim=1)
            actions = torch.concat([actions, actions_text.expand(actions.shape[0], -1)], dim=1)
        h = self.get_feature(observations, actions)
        return h

    def get_pretrain_feature(self, observations, actions, env=None): # get feature assuming projection and after concat
        if env is not None:
            projection = self.proj[env]
        else:
            projection = self.proj
        ##DEBUG
        #print(f"projection used: {projection}")
        ##
        h = torch.cat([observations, actions], dim=-1)
        h = projection(h)
        ##DEBUG
        #print(f"hidden after projection: {h.shape}")
        ##
        if self.observations_text is not None:
            observations, actions = torch.split(h, [self.obs_dim, self.action_dim], dim=1)
            ##DEBUG
            #print(f"observations after split: {observations.shape}")
            #print(f"actions after split: {actions.shape}")
            ##
            observations = torch.concat([observations, self.observations_text.expand(observations.shape[0], -1)], dim=1)
            actions = torch.concat([actions, self.actions_text.expand(actions.shape[0], -1)], dim=1)
            ##DEBUG
            #print(f"observations after concat: {observations.shape}")
            #print(f"actions after concat: {actions.shape}")
            ##
            return self.get_feature(observations, actions)

        else:
            for fc_layer in self.hidden_layers:
                h = self.hidden_activation(fc_layer(h))
            return h
        #if self.
        #for fc_layer in self.hidden_layers:
        #    h = self.hidden_activation(fc_layer(h))
        #return h
    '''
    @multiple_action_q_function
    def forward_pretrain(self, observations, actions, env=None):
        h = self.get_pretrain_feature(observations, actions, env)
        return torch.squeeze(self.last_fc_layer(h), dim=-1)
    '''
    @multiple_action_q_function
    def forward(self, observations, actions, observations_text=None, actions_text=None, chosen=None, proj=False):
        ##DEBUG
        '''
        print("q forward")
        print(f"observation dim: {observations.shape}")
        print(f"actions dim: {actions.shape}")
        if observations_text is not None:
            print(f"observations text: {observations_text.shape}")
            print(f"actions text: {actions_text.shape}")
        else:
            print(f"observations text: {observations_text}")
            print(f"actions text: {actions_text}")
        print(f"chosen: {chosen}")
        print(f"proj: {proj}")
        '''
        ##
        if observations_text is not None:
            observations = torch.concat([observations, observations_text.expand(observations.shape[0], -1)], dim=1)
            actions = torch.concat([actions, actions_text.expand(actions.shape[0], -1)], dim=1)
        if not proj:
            h = self.get_feature(observations, actions)
        else:
            h = self.get_pretrain_feature(observations, actions, chosen)
        ##DEBUG
        #print(f"feature after: {h.shape}")
        ##
        return torch.squeeze(self.last_fc_layer(h), dim=-1)

    def get_pretrain_next_obs(self, observations, actions, env=None):
        ##DEBUG
        #print(f"env: {env}")
        #print("proj used")
        ##
        h = self.get_pretrain_feature(observations, actions, env)
        if env is not None:
            hidden_to_next_obs = self.hidden_to_next_obs[env]
        else:
            hidden_to_next_obs =  self.hidden_to_next_obs
        ##DEBUG
        #print(f"hidden to next obs: {hidden_to_next_obs}")
        ##
        return hidden_to_next_obs(h)

    
    ##########################################
    def predict_next_obs(self, observations, actions):
        ##DEBUG
        #print("no proj used")
        ##
        h = self.get_feature(observations, actions)
        return self.hidden_to_next_obs(h)

    def predict_value(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_value(h)

    ###############################################
    '''
    @multiple_action_q_function
    def forward(self, observations, actions):
        h = self.get_feature(observations, actions)
        return torch.squeeze(self.last_fc_layer(h), dim=-1)
    '''
    ############################################################################################
class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant
