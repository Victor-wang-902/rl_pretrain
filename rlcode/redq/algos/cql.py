import os.path
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from redq.algos.core import Mlp, soft_update_model1_with_model2, ReplayBuffer, PolicyNetworkPretrain

def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins

class CQLAgent(object):
    """
    Naive SAC: num_Q = 2, num_min = 2
    REDQ: num_Q > 2, num_min = 2
    MaxMin: num_mins = num_Qs
    for above three variants, set q_target_mode to 'min' (default)
    Ensemble Average: set q_target_mode to 'ave'
    REM: set q_target_mode to 'rem'
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
                 cql_weight=1, cql_n_random=10, cql_temp=1, std=0.1,
                 ):
        # set up networks
        hidden_sizes = [hidden_unit for _ in range(hidden_layer)]
        self.policy_net = PolicyNetworkPretrain(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = []
        for q_i in range(num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=lr))
        # set up replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.std = std
        self.cql_weight = cql_weight
        self.cql_n_random = cql_n_random
        self.cql_temp = cql_temp
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_layer = hidden_layer
        self.hidden_unit = hidden_unit
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
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
            action_tensor = self.policy_net.forward(obs_tensor, std=self.std,
                                                    deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs): #TODO modify the readme here
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor, std=self.std,
                                                                                   deterministic=False, return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        # given obs_tensor and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

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

    def get_cql_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        with torch.no_grad():
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor, std=self.std)
            q_prediction_next_list = []
            for sample_idx in range(2):
                q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            y_q = rews_tensor + self.gamma * (1 - done_tensor) * min_q
        return y_q

    def get_q1_q2(self, obs, act):
        q1 = self.q_net_list[0](torch.cat([obs, act], 1))
        q2 = self.q_net_list[1](torch.cat([obs, act], 1))
        return q1, q2
    def get_q1_q2_target(self, obs, act):
        q1 = self.q_target_net_list[0](torch.cat([obs, act], 1))
        q2 = self.q_target_net_list[1](torch.cat([obs, act], 1))
        return q1, q2
    def get_act(self, obs):
        a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs, std=self.std)
        return a_tilda

    def update(self, logger):
        # cql offline update
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        num_update = 1
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """standard Q loss"""
            # q target value
            y_q = self.get_cql_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            # q pred
            Q1, Q2 = self.get_q1_q2(obs_tensor, acts_tensor)
            q_prediction_list = [Q1, Q2]
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            critic_loss = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            """conservative q loss"""
            if self.cql_weight > 0:
                random_actions = (torch.rand((self.batch_size * self.cql_n_random, self.act_dim),
                                             device=self.device) - 0.5) * 2

                current_actions = self.get_act(obs_tensor)
                next_current_actions = self.get_act(obs_next_tensor)

                # now get Q values for all these actions (for both Q networks)
                obs_repeat = obs_tensor.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(obs_tensor.shape[0] * self.cql_n_random,
                                                                                   obs_tensor.shape[1])

                Q1_rand, Q2_rand = self.get_q1_q2(obs_repeat, random_actions)
                Q1_rand = Q1_rand.view(obs_tensor.shape[0], self.cql_n_random)
                Q2_rand = Q2_rand.view(obs_tensor.shape[0], self.cql_n_random)

                Q1_curr, Q2_curr = self.get_q1_q2(obs_tensor, current_actions)
                Q1_curr_next, Q2_curr_next = self.get_q1_q2(obs_tensor, next_current_actions)

                # now concat all these Q values together
                Q1_cat = torch.cat([Q1_rand, Q1, Q1_curr, Q1_curr_next], 1)
                Q2_cat = torch.cat([Q2_rand, Q2, Q2_curr, Q2_curr_next], 1)

                cql_min_q1_loss = torch.logsumexp(Q1_cat / self.cql_temp,
                                                  dim=1, ).mean() * self.cql_weight * self.cql_temp
                cql_min_q2_loss = torch.logsumexp(Q2_cat / self.cql_temp,
                                                  dim=1, ).mean() * self.cql_weight * self.cql_temp

                """Subtract the log likelihood of data"""
                conservative_q_loss = cql_min_q1_loss + cql_min_q2_loss - (
                            Q1.mean() + Q2.mean()) * self.cql_weight
            else:
                conservative_q_loss = 0
            q_loss_all = critic_loss + conservative_q_loss

            """q network update"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            """policy loss"""
            # get policy loss
            current_action, mean_a_tilda, _, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor, std=self.std)
            Q1, Q2 = self.get_q1_q2(obs_tensor, acts_tensor)
            Q = torch.min(Q1, Q2)
            policy_loss = -Q.mean()

            """policy update"""
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            for sample_idx in range(self.num_Q):
                self.q_net_list[sample_idx].requires_grad_(True)

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            # by default only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                with torch.no_grad():
                    logger.store(LossPi=policy_loss.item(), LossQ=q_loss_all.item() / self.num_Q,
                             Q1Vals=Q1.mean().item(), LogPi=log_prob_a_tilda.mean().item(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

    def pretrain_update(self, logger, pretrain_mode): # TODO fix
        # pretrain mode example: q_sprime
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
        layer_list = []
        for i in range(2):
            for layer in self.q_net_list[i].hidden_layers:
                layer_list.append(layer)
        return layer_list
    def features_from_batch(self, batch):
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)

        features1 = self.q_net_list[0].get_feature(torch.cat([obs_next_tensor, acts_tensor], 1))
        features2 = self.q_net_list[1].get_feature(torch.cat([obs_next_tensor, acts_tensor], 1))
        return torch.cat([features1, features2], 1)

    def load_pretrained_model(self, folder_path, pretrain_mode, pretrain_epochs):
        pretrain_model_file_name = '%s_h%s_%s_e%s.pth' % (pretrain_mode, self.hidden_layer, self.hidden_unit, pretrain_epochs)
        pretrain_full_path = os.path.join(folder_path, pretrain_model_file_name)
        d = torch.load(pretrain_full_path)
        self.q_net_list[0].load_state_dict(d['q1'])
        self.q_net_list[1].load_state_dict(d['q2'])
        self.q_target_net_list[0].load_state_dict(d['q1'])
        self.q_target_net_list[1].load_state_dict(d['q2'])
    def save_pretrained_model(self, folder_path, pretrain_mode, pretrain_epochs):
        pretrain_model_file_name = '%s_h%s_%s_e%s.pth' % (pretrain_mode, self.hidden_layer, self.hidden_unit, pretrain_epochs)
        pretrain_full_path = os.path.join(folder_path, pretrain_model_file_name)
        d = {
            'q1': self.q_net_list[0].state_dict(),
            'q2': self.q_net_list[1].state_dict()
        }
        if not os.path.exists(pretrain_full_path):
            torch.save(d, pretrain_full_path)
            print("Saved pretrained model to:", pretrain_full_path)
        else:
            print("Pretrained model not saved. Already exist:", pretrain_full_path)
