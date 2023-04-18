import os.path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from redq.algos.core import soft_update_model1_with_model2, ReplayBuffer,\
    get_d4rl_target_entropy, PolicyNetworkPretrain, concatenate_weights_of_model_list

class ILAgent(object):
    """
    imitation learning baseline
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_layer=2, hidden_unit=256, replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
                 policy_update_delay=20,
                 ensemble_decay_n_data=20000, # wait for how many data to reduce number of ensemble (e.g. 20,000 data)
                 safe_q_target_factor=0.5,
                 ):
        # set up networks
        hidden_sizes = [hidden_unit for _ in range(hidden_layer)]
        self.policy_net = PolicyNetworkPretrain(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # set up replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.hidden_layer = hidden_layer
        self.hidden_unit = hidden_unit
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.init_num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.device = device

        self.ensemble_decay_n_data = ensemble_decay_n_data
        self.max_reward_per_step = 0.1
        self.safe_q_threshold = 100
        self.safe_q_target_factor = safe_q_target_factor

    def __get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.size

    def get_test_action(self, obs):
        # given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, std=0, deterministic=True,
                                         return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def store_data(self, o, a, r, o2, d):
        # store one transition to the buffer
        self.replay_buffer.store(o, a, r, o2, d)

    def load_data(self, dataset, data_ratio, seed=0): # load a d4rl q learning dataset
        # e.g. if ratio is 0.4, we use 40% of the data (depends on how many we have in the dataset)
        assert self.replay_buffer.size == 0
        n_data = dataset['actions'].shape[0]
        n_data = min(n_data, self.replay_buffer.max_size)

        n_data_to_use = int(n_data * data_ratio)
        np.random.seed(seed)
        idxs = np.random.choice(n_data, n_data_to_use, replace=False)

        self.replay_buffer.obs1_buf[0:n_data_to_use] = dataset['observations'][idxs]
        self.replay_buffer.obs2_buf[0:n_data_to_use] = dataset['next_observations'][idxs]
        self.replay_buffer.acts_buf[0:n_data_to_use] = dataset['actions'][idxs]
        self.replay_buffer.rews_buf[0:n_data_to_use] = dataset['rewards'][idxs]
        self.replay_buffer.done_buf[0:n_data_to_use] = dataset['terminals'][idxs]
        self.replay_buffer.ptr, self.replay_buffer.size = n_data_to_use, n_data_to_use

    def sample_data(self, batch_size, idxs=None):
        # sample data from replay buffer
        batch = self.replay_buffer.sample_batch(batch_size, idxs)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def update(self, logger):
        # this function is called after each datapoint collected.
        for i_update in range(1):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """BC loss"""
            # get policy loss
            a_tilda, mean_a_tilda, _, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor, std=0)
            policy_loss = F.mse_loss(mean_a_tilda, acts_tensor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            """update networks"""
            self.policy_optimizer.step()

            # by default only log for the last update out of <num_update> updates
            logger.store(LossPi=policy_loss.item(), LogPi=log_prob_a_tilda.mean().item(),
                         PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

    def pretrain_update(self, logger, pretrain_mode):
        # predict next obs with current obs
        for i_update in range(1):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            if pretrain_mode == 'pi_sprime':
                """mse loss for predicting next obs"""
                obs_next_pred = self.policy_net.predict_next_obs(obs_tensor)
                pretrain_loss = F.mse_loss(obs_next_pred, obs_next_tensor)
                self.policy_optimizer.zero_grad()
                pretrain_loss.backward()

                """update networks"""
                self.policy_optimizer.step()

                # by default only log for the last update out of <num_update> updates
                logger.store(LossPretrain=pretrain_loss.cpu().item())
            else:
                raise NotImplementedError("Pretrain mode not implemented: %s" % pretrain_mode)

    def layers_for_weight_diff(self):
        return self.policy_net.hidden_layers
    def features_from_batch(self, batch):
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return self.policy_net.get_feature(obs_tensor)
    def load_pretrained_model(self, folder_path, pretrain_mode, pretrain_epochs):
        pretrain_model_file_name = '%s_h%s_%s_e%s.pth' % (pretrain_mode, self.hidden_layer, self.hidden_unit, pretrain_epochs)
        pretrain_full_path = os.path.join(folder_path, pretrain_model_file_name)
        self.policy_net.load_state_dict(torch.load(pretrain_full_path))
    def save_pretrained_model(self, folder_path, pretrain_mode, pretrain_epochs):
        pretrain_model_file_name = '%s_h%s_%s_e%s.pth' % (pretrain_mode, self.hidden_layer, self.hidden_unit, pretrain_epochs)
        pretrain_full_path = os.path.join(folder_path, pretrain_model_file_name)
        if not os.path.exists(pretrain_full_path):
            torch.save(self.policy_net.state_dict(), pretrain_full_path)
            print("Saved pretrained model to:", pretrain_full_path)
        else:
            print("Pretrained model not saved. Already exist:", pretrain_full_path)
