from collections import OrderedDict
from copy import deepcopy

from ml_collections import ConfigDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from SimpleSAC.model import Scalar, soft_target_update
from SimpleSAC.utils import prefix_metrics


class ConservativeSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 5.0
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, variant):
        self.config = ConservativeSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )

        self.qf_optimizer = optimizer_class(
                list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
            )

        # if variant['q_network_feature_lr_scale'] == 1:
        #     self.qf_optimizer = optimizer_class(
        #         list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        #     )
        # else:
        #     reduced_lr = variant['q_network_feature_lr_scale'] * self.config.qf_lr
        #     self.qf_optimizer = optimizer_class([
        #         {"params": self.qf1.hidden_layers.parameters(), "lr": reduced_lr},
        #         {"params": self.qf2.hidden_layers.parameters(), "lr": reduced_lr},
        #         {"params": self.qf1.last_fc_layer.parameters(),},
        #         {"params": self.qf2.last_fc_layer.parameters(),},
        #     ], lr=self.config.qf_lr)


        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0
        self._total_pretrain_steps = 0

    ####################################
    def set_qf_optimizer_lr(self, lr):
        if self.config.qf_lr != lr:
            self.config.qf_lr = lr
            self.qf_optimizer = optimizer_class(
                    list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
                )
            if self.config.cql_lagrange:
                self.log_alpha_prime = Scalar(1.0)
                self.alpha_prime_optimizer = optimizer_class(
                    self.log_alpha_prime.parameters(),
                    lr=self.config.qf_lr,
                )

    def set_policy_optimizer_lr(self, lr):
        if self.config.policy_lr != lr:
            self.config.policy_lr = lr
            self.policy_optimizer = optimizer_class(
                self.policy.parameters(), self.config.policy_lr,
            if self.config.use_automatic_entropy_tuning:
                self.alpha_optimizer = optimizer_class(
                    self.log_alpha.parameters(),
                    lr=self.config.policy_lr,
                )
    ####################################
    def update_qf_feature_lr(self, scale):
        if scale != 1:
            if scale == 0:
                self.qf_optimizer = {
                    'adam': torch.optim.Adam,
                    'sgd': torch.optim.SGD,
                }[self.config.optimizer_type]([
                    {"params": self.qf1.hidden_layers.parameters(), "lr": 0.},
                    {"params": self.qf2.hidden_layers.parameters(), "lr": 0.},
                    {"params": self.qf1.last_fc_layer.parameters(), },
                    {"params": self.qf2.last_fc_layer.parameters(), },
                ], lr=self.config.qf_lr)
            else:
                new_lr = self.config.qf_lr * scale
                for p in self.qf_optimizer.param_groups:
                    p['lr'] = new_lr

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def q_distill_only(self, batch, ready_agent, q_distill_weight):
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)
        with torch.no_grad():
            q1_ready = ready_agent.qf1(observations, actions)
            q2_ready = ready_agent.qf2(observations, actions)
        qf1_distill_loss = F.mse_loss(q1_pred, q1_ready) * q_distill_weight
        qf2_distill_loss = F.mse_loss(q2_pred, q2_ready) * q_distill_weight
        qf_loss = qf1_distill_loss + qf2_distill_loss

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

    def train(self, batch, bc=False, ready_agent=None, q_distill_weight=0, distill_only=False,
              safe_q_max=Nonem chosen=None):
        self._total_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        if distill_only:
            # distill Q
            q1_pred = self.qf1(observations, actions)
            q2_pred = self.qf2(observations, actions)

            with torch.no_grad():
                q1_ready = ready_agent.qf1(observations, actions)
                q2_ready = ready_agent.qf2(observations, actions)
            qf1_distill_loss = F.mse_loss(q1_pred, q1_ready)
            qf2_distill_loss = F.mse_loss(q2_pred, q2_ready)

            qf_loss = qf1_distill_loss + qf2_distill_loss

            # distill policy
            new_actions, log_pi = self.policy(observations)
            with torch.no_grad():
                actions_ready, log_pi_ready = ready_agent.policy(observations)
            policy_loss = F.mse_loss(new_actions, actions_ready)

            # optimizer
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            metrics = dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                qf1_loss=qf1_distill_loss.item(),
                qf2_loss=qf2_distill_loss.item(),
                average_qf1=q1_pred.mean().item(),
                average_qf2=q2_pred.mean().item(),
                total_steps=self.total_steps,
                qf1_distill_loss=qf1_distill_loss.item(),
                qf2_distill_loss=qf2_distill_loss.item(),
            )

            return metrics

        new_actions, log_pi = self.policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        if bc:
            log_probs = self.policy.log_prob(observations, actions)
            policy_loss = (alpha*log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.qf1(observations, new_actions),
                self.qf2(observations, new_actions),
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        if self.config.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_qf1(next_observations, new_next_actions),
                    self.target_qf2(next_observations, new_next_actions),
                ),
                dim=-1
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.policy(next_observations)
            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )
            if safe_q_max is not None:
                target_q_values[target_q_values > safe_q_max] = safe_q_max

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards + (1. - dones) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, td_target.detach())
        qf2_loss = F.mse_loss(q2_pred, td_target.detach())


        ### CQL
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            cql_random_actions = actions.new_empty((batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            cql_q1_rand = self.qf1(observations, cql_random_actions)
            cql_q2_rand = self.qf2(observations, cql_random_actions)
            cql_q1_current_actions = self.qf1(observations, cql_current_actions)
            cql_q2_current_actions = self.qf2(observations, cql_current_actions)
            cql_q1_next_actions = self.qf1(observations, cql_next_actions)
            cql_q2_next_actions = self.qf2(observations, cql_next_actions)

            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            if self.config.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_q1 = torch.cat(
                    [cql_q1_rand - random_density,
                     cql_q1_next_actions - cql_next_log_pis.detach(),
                     cql_q1_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand - random_density,
                     cql_q2_next_actions - cql_next_log_pis.detach(),
                     cql_q2_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )

            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp

            """Subtract the log likelihood of data"""
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - q1_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - q2_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()

            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)

            if q_distill_weight > 0:
                with torch.no_grad():
                    q1_ready = ready_agent.qf1(observations, actions)
                    q2_ready = ready_agent.qf2(observations, actions)
                qf1_distill_loss = F.mse_loss(q1_pred, q1_ready) * q_distill_weight
                qf2_distill_loss = F.mse_loss(q2_pred, q2_ready) * q_distill_weight
            else:
                device = q1_pred.device
                qf1_distill_loss, qf2_distill_loss = torch.zeros(1).to(device), torch.zeros(1).to(device)
            conservative_loss = cql_min_qf1_loss + cql_min_qf2_loss + qf1_distill_loss + qf2_distill_loss
            qf_loss = qf1_loss + qf2_loss + conservative_loss


        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )


        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            qf_average_loss=(qf1_loss.item() + qf2_loss.item())/2,
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
            qf1_distill_loss=qf1_distill_loss.item(),
            qf2_distill_loss=qf2_distill_loss.item(),
            combined_loss=qf_loss.item()
        )

        if self.config.use_cql:
            metrics.update(prefix_metrics(dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                conservative_loss=conservative_loss.item(),
            ), 'cql'))

        return metrics

    def pretrain(self, batch, pretrain_mode, mdppre_n_state, chosen=None):
        # pretrain mode:  q_sprime, 4. q_mc
        self._total_pretrain_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        # rewards = batch['rewards']
        next_observations = batch['next_observations']
        # dones = batch['dones']

        if pretrain_mode in ['q_sprime', 'mdp_same_noproj', 'q_sprime_3x', 'random_fd_1000_state']:
            # here use both q networks to predict next obs
            if int(mdppre_n_state) != 42:
                obs_next_q1 = self.qf1.predict_next_obs(observations, actions)
                obs_next_q2 = self.qf2.predict_next_obs(observations, actions)
                pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
                pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
                pretrain_loss = pretrain_loss1 + pretrain_loss2
            else:
                observations = 2 * torch.rand(observations.shape, device=observations.device) - 1
                actions = 2 * torch.rand(actions.shape, device=observations.device) - 1
                next_observations = 2 * torch.rand(next_observations.shape, device=observations.device) - 1
                obs_next_q1 = self.qf1.predict_next_obs(observations, actions)
                obs_next_q2 = self.qf2.predict_next_obs(observations, actions)
                pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
                pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
                pretrain_loss = pretrain_loss1 + pretrain_loss2
        elif pretrain_mode in ['q_noact_sprime']:
            actions = torch.zeros(actions.shape, device=actions.device)
            obs_next_q1 = self.qf1.predict_next_obs(observations, actions)
            obs_next_q2 = self.qf2.predict_next_obs(observations, actions)
            pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
            pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
            pretrain_loss = pretrain_loss1 + pretrain_loss2
        elif pretrain_mode in ['proj0_q_sprime', 'proj1_q_sprime', 'proj2_q_sprime', 'mdp_q_sprime', 'mdp_same_proj',
                               'proj0_q_sprime_3x', 'proj1_q_sprime_3x', 'proj2_q_sprime_3x']:
            obs_next_q1 = self.qf1.get_pretrain_next_obs(observations, actions)
            obs_next_q2 = self.qf2.get_pretrain_next_obs(observations, actions)
            pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
            pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
            pretrain_loss = pretrain_loss1 + pretrain_loss2
        elif pretrain_mode == 'rand_q_sprime':
            observations= 2*torch.rand(observations.shape, device=observations.device) - 1
            actions = 2*torch.rand(actions.shape, device=observations.device) - 1
            next_observations = 2*torch.rand(next_observations.shape, device=observations.device) - 1
            obs_next_q1 = self.qf1.predict_next_obs(observations, actions)
            obs_next_q2 = self.qf2.predict_next_obs(observations, actions)
            pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
            pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
            pretrain_loss = pretrain_loss1 + pretrain_loss2
        elif pretrain_mode in [""]
        else:
            raise NotImplementedError

        self.qf_optimizer.zero_grad()
        pretrain_loss.backward()
        self.qf_optimizer.step()

        # TODO when pretrain finished need to update target networks
        metrics = dict(
            pretrain_loss=pretrain_loss.item(),
            total_pretrain_steps=self.total_pretrain_steps,
        )

        return metrics

    def layers_for_weight_diff(self):
        layer_list = []
        for layer in self.qf1.hidden_layers:
            layer_list.append(layer)
        for layer in self.qf2.hidden_layers:
            layer_list.append(layer)
        return layer_list

    def layers_for_weight_diff_extra(self):
        layer1 = [self.qf1.hidden_layers[0], self.qf2.hidden_layers[0]]
        layer2 = [self.qf1.hidden_layers[1], self.qf2.hidden_layers[1]]
        layer_fc = [self.qf1.last_fc_layer, self.qf2.last_fc_layer]
        return layer1, layer2, layer_fc

    def features_from_batch_no_grad(self, batch):
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        with torch.no_grad():
            feature_q1 = self.qf1.get_feature(observations, actions)
            feature_q2 = self.qf2.get_feature(observations, actions)
            return torch.cat([feature_q1, feature_q2], 1)

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def total_pretrain_steps(self):
        return self._total_pretrain_steps