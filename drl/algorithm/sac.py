import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAC = lambda model, **kwargs: SAC1(model, **kwargs) if 'alpha' in kwargs.keys() else SAC2(model, **kwargs)

class SAC1(BasePolicy): # pg_net + q_net + v_net
    def __init__(
        self, 
        model,
        buffer_size=1e6,
        batch_size=256,
        policy_freq=2,
        tau=0.005,
        discount=0.99,
        policy_lr=3e-4,
        value_lr=3e-4,
        learn_iteration=1,
        verbose=False,
        act_dim=None,
        alpha=1.0,
        ):
         super().__init__()
        self.tau = tau
        self.gamma = discount
        self.policy_freq = policy_freq
        self.learn_iteration = learn_iteration
        self.verbose = verbose
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size) # off-policy

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()
        self.value_eval = model.v_net.to(device).train()

        self.value_target = self.copy_net(self.value_eval)
        
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=policy_lr)
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=value_lr)
        self.value_eval_optim = torch.optim.Adam(self.value_eval.parameters(), lr=value_lr)

        self.criterion = nn.SmoothL1Loss()
        
        self.alpha = alpha
        self.eps = np.finfo(np.float32).eps.item()
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

    def learn(self):
        pg_loss, q_loss, v_loss = 0, 0, 0
        for _ in range(self.learn_iteration):
            batch = self.buffer.split_batch(self.batch_size)
            if self.act_dim is None:
                self.act_dim = np.array(batch['a']).shape[-1]

            S = torch.tensor(batch['s'], dtype=torch.float32, device=device)
            A = torch.tensor(batch['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)
            M = torch.tensor(batch['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch['s_'], dtype=torch.float32, device=device)

            new_A, log_prob = self.actor_eval.evaluate(S)
            
            # V_value loss
            with torch.no_grad():
                new_q1_value, new_q2_value = self.critic_eval(S, new_A)
                next_value = torch.min(new_q1_value, new_q2_value) - self.alpha * log_prob
            value = self.value_eval(S)
            value_loss = self.criterion(value, next_value)

            # Soft q loss
            with torch.no_grad():
                target_value = self.value_target(S_)
                target_q_value = R + M * self._gamma * target_value.cpu()
                target_q_value = target_q_value.to(device)
            q1_value, q2_value = self.critic_eval(S, A)
            loss1 = self.criterion(q1_value, target_q_value)
            loss2 = self.criterion(q2_value, target_q_value)
            critic_loss = 0.5 * (loss1 + loss2)

            # update V
            self.value_eval_optim.zero_grad()
            value_loss.backward()
            self.value_eval_optim.step()

            # update soft Q
            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1
            
            actor_loss = torch.tensor(0)
            # update policy
            if self._learn_critic_cnt % self.policy_freq == 0:
                # policy loss
                actor_loss = (self.alpha * log_prob - torch.min(new_q1_value, new_q2_value)).mean()
                # actor_loss = (log_prob - torch.min(new_q1_value, new_q2_value).detach()).mean()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1

                self.soft_sync_weight(self.value_target, self.value_eval, self.tau)

            pg_loss += actor_loss.item()
            q_loss += critic_loss.item()
            v_loss += value_loss.item()
                
            return pg_loss, q_loss, v_loss


class SAC2(BasePolicy): # pg_net + q_net + alpha
    def __init__(
        self, 
        model,
        buffer_size=1e6,
        batch_size=256,
        policy_freq=2,
        tau=0.005,
        discount=0.99,
        policy_lr=3e-4,
        value_lr=3e-4,
        learn_iteration=1,
        verbose=False,
        act_dim=None,
        ):
        super().__init__()
        self.tau = tau
        self.gamma = discount
        self.policy_freq = policy_freq
        self.learn_iteration = learn_iteration
        self.verbose = verbose
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size) # off-policy

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)
        
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=policy_lr)
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=value_lr)

        self.criterion = nn.SmoothL1Loss()
        self.target_entropy = -torch.tensor(1).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=policy_lr)
        self.alpha = self.log_alpha.exp()

        self.eps = np.finfo(np.float32).eps.item()
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

    def learn(self):
        pg_loss, q_loss, a_loss = 0, 0, 0
        for _ in range(self.learn_iteration):
            batch = self.buffer.split_batch(self.batch_size)

            if self.act_dim is None:
                self.act_dim = np.array(batch['a']).shape[-1]
                self.target_entropy = -torch.tensor(self.act_dim).to(device)

            S = torch.tensor(batch['s'], dtype=torch.float32, device=device)
            A = torch.tensor(batch['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)
            M = torch.tensor(batch['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch['s_'], dtype=torch.float32, device=device)

            if self.verbose:
                print(f'shape S:{S.size()}, A:{A.size()}, M:{M.size()}, R:{R.size()}, S_:{S_.size()}, W:{W.size()}')

            with torch.no_grad():
                next_A, next_log = self.actor_target.evaluate(S_)
                q1_next, q2_next = self.critic_target(S_, next_A)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log
                q_target = R + M * self.gamma * q_next.cpu()
                q_target = q_target.to(device)
            # q_loss
            q1_eval, q2_eval = self.critic_eval(S, A)
            loss1 = self.criterion(q1_eval, q_target)
            loss2 = self.criterion(q2_eval, q_target)
            critic_loss = 0.5 * (loss1 + loss2)

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            actor_loss = alpha_loss = torch.tensor(0)
            if self._learn_critic_cnt % self.policy_freq == 0:
                curr_A, curr_log = self.actor_eval.evaluate(S)
                q1_next, q2_next = self.critic_eval(S, curr_A)
                q_next = torch.min(q1_next, q2_next)

                # pg_loss
                actor_loss = (self.alpha * curr_log - q_next).mean()
                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1

                # alpha loss
                alpha_loss = -(self.log_alpha * (curr_log + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

            q_loss += critic_loss.item()
            pg_loss += actor_loss.item()
            a_loss += alpha_loss.item()
        
        return pg_loss, q_loss, a_loss
