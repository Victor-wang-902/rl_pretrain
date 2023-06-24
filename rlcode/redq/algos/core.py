import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torch.distributions import Distribution, Normal
# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# these numbers are from the MBPO paper
mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2,
                            'Hopper-v3':-1, 'HalfCheetah-v3':-3, 'Walker2d-v3':-3, 'Ant-v3':-4, 'Humanoid-v3':-2,
                            'Hopper-v4':-1, 'HalfCheetah-v4':-3, 'Walker2d-v4':-3, 'Ant-v4':-4, 'Humanoid-v4':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300,
                'Hopper-v3':125, 'Walker2d-v3':300, 'Ant-v3':300, 'HalfCheetah-v3':400, 'Humanoid-v3':300,
                'Hopper-v4':125, 'Walker2d-v4':300, 'Ant-v4':300, 'HalfCheetah-v4':400, 'Humanoid-v4':300}

def get_d4rl_target_entropy(env_name):
    if 'hopper' in env_name:
        return -1
    if 'halfcheetah' in env_name:
        return -3
    if 'walker' in env_name:
        return -3
    raise ValueError("Unknown env name for target entropy.")

def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer
    """
    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        ## init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        data will get stored in the pointer's location
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        ## move the pointer to store in next location in buffer
        self.ptr = (self.ptr+1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class QNetPretrain(nn.Module):
    def __init__(
            self,
            obs_size,
            act_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = obs_size + act_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.predict_next_layer = nn.Linear(in_size, obs_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

    def predict_next_state(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        return self.predict_next_layer(h)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        z here is mu+sigma+eksee
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( ## this part is eksee~N(0,1)
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        ## this is the layer that gives log_std, init this layer with small weight and bias
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        ## action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        self.apply(weights_init_)

    def get_feature(self, obs):
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, take determinisitc (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )

class PolicyNetworkPretrain(nn.Module):
    """
    A Gaussian policy network with Tanh to enforce action limits
    with state independent std
    and can be pretrained
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0,
    ):
        super().__init__()
        self.input_size = obs_dim
        self.output_size = action_dim
        self.hidden_activation = hidden_activation
        self.action_limit = action_limit
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        # hidden to action
        self.last_fc_layer = nn.Linear(in_size, self.output_size)

        # pretrain mode: pi_sprime
        self.hidden_to_next_obs = nn.Linear(hidden_sizes[-1], obs_dim)
        # pretrain mode: pi_mc
        self.hidden_to_value = nn.Linear(hidden_sizes[-1], 1)

        # init networks
        self.apply(weights_init_)

    def get_feature(self, obs):
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def predict_next_obs(self, obs):
        h = self.get_feature(obs)
        return self.hidden_to_next_obs(h)

    def predict_value(self, obs):
        h = self.get_feature(obs)
        return self.hidden_to_value(h)

    def forward(
            self,
            obs,
            std,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, take determinisitc (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = self.get_feature(obs)
        mean = self.last_fc_layer(h)

        std = max(std, 1e-9)
        normal = Normal(mean, torch.ones_like(mean)*std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, None, log_prob, std, pre_tanh_value,
        )

class QNetworkPretrain(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)

        # pretrain mode: q_sprime
        self.hidden_to_next_obs = nn.Linear(hidden_sizes[-1], input_size)
        # pretrain mode: q_mc
        self.hidden_to_value = nn.Linear(hidden_sizes[-1], 1)

        self.apply(weights_init_)

    def get_feature(self, input):
        h = input
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def predict_next_obs(self, input):
        h = self.get_feature(input)
        return self.hidden_to_next_obs(h)

    def predict_value(self, input):
        h = self.get_feature(input)
        return self.hidden_to_value(h)

    def forward(self, input):
        h = self.get_feature(input)
        output = self.last_fc_layer(h)
        return output


def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

def test_agent(agent, test_env, max_ep_len, logger, n_eval=1, return_list=False):
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    if return_list:
        ep_return_list = list(ep_return_list)
    return ep_return_list

def test_agent_d4rl(agent, test_env, max_ep_len, logger, n_eval=1, return_list=False, data_env=None):
    # save returns and normalized returns
    ep_return_list = np.zeros(n_eval)
    ep_normalized_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if data_env is not None:
            ep_norm_ret = data_env.get_normalized_score(ep_ret) * 100
        else:
            ep_norm_ret = test_env.get_normalized_score(ep_ret) * 100
        ep_normalized_return_list[j] = ep_norm_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,
                         TestEpNormRet=ep_norm_ret)
    if return_list:
        ep_return_list = list(ep_return_list)
        ep_normalized_return_list = list(ep_normalized_return_list)
    return ep_return_list, ep_normalized_return_list

def concatenate_weights(model, weight_only=True):
    concatenated_weights = []
    for name, param in model.named_parameters():
        if not weight_only or 'weight' in name:
            concatenated_weights.append(param.view(-1))
    return torch.cat(concatenated_weights)

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

def get_feature_diff(agent1, agent2, replay_buffer, ratio=0.1, seed=0):
    # feature diff: for each data point, get difference of feature from old and new network
    # compute l2 norm of this diff, average over a number of data points.
    # agent class should have features_from_batch() func
    average_feature_l2_norm_list = []
    num_feature_timesteps = int(replay_buffer.size * ratio)
    if num_feature_timesteps % 2 == 1: # avoid potential sampling issue
        num_feature_timesteps = num_feature_timesteps + 1
    np.random.seed(seed)
    idxs_all = np.random.choice(np.arange(0, replay_buffer.size), size=num_feature_timesteps, replace=False)
    batch_size = 1000
    n_done = 0
    i = 0
    while True:
        if n_done >= num_feature_timesteps:
            break
        idxs = idxs_all[i*1000:min((i+1)*1000, num_feature_timesteps)]

        batch = replay_buffer.sample_batch(batch_size, idxs)
        old_feature = agent1.features_from_batch_no_grad(batch)
        new_feature = agent2.features_from_batch_no_grad(batch)
        feature_diff = old_feature - new_feature

        feature_l2_norm = torch.norm(feature_diff, p=2, dim=1, keepdim=True)
        average_feature_l2_norm_list.append(feature_l2_norm.mean().item())
        i += 1
        n_done += 1000

    return np.mean(average_feature_l2_norm_list), num_feature_timesteps

