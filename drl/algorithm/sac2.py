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

class SAC2(BasePolicy): # no value network
    def __init__(
        self, 
        model,
        buffer_size=1000,
        batch_size=100,
        actor_learn_freq=1,
        target_update_freq=5,
        target_update_tau=1e-2,
        learning_rate=1e-3,
        discount_factor=0.99,
        verbose=False,
        update_iteration=10,
        act_dim=None,
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._update_iteration = update_iteration
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size) # off-policy

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)
        
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        self.act_dim = act_dim
        self.target_entropy = -torch.tensor(1).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = self.log_alpha.exp()

    def learn(self):
        pg_loss, q_loss, a_loss = 0, 0, 0
        for _ in range(self._update_iteration):
            batch = self.buffer.split_batch(self._batch_size)
            if self.act_dim is None:
                self.act_dim = np.array(batch['a']).shape[-1]
                self.target_entropy = -torch.tensor(self.act_dim).to(device)

            S = torch.tensor(batch['s'], dtype=torch.float32, device=device)
            A = torch.tensor(batch['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)
            M = torch.tensor(batch['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch['s_'], dtype=torch.float32, device=device)
            if self._verbose:
                print(f'shape S:{S.size()}, A:{A.size()}, M:{M.size()}, R:{R.size()}, S_:{S_.size()}, W:{W.size()}')

            with torch.no_grad():
                next_A, next_log = self.actor_target.evaluate(S_)
                q1_next, q2_next = self.critic_target(S_, next_A)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(device)
            # q_loss
            q1_eval, q2_eval = self.critic_eval(S, A)
            critic_loss = self.criterion(q1_eval, q_target) + self.criterion(q2_eval, q_target)

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            actor_loss = torch.tensor(0)
            alpha_loss = torch.tensor(0)
            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                curr_A, curr_log = self.actor_eval.evaluate(S)
                q1_next, q2_next = self.critic_eval(S, curr_A)
                q_next = torch.min(q1_next, q2_next)

                # pg_loss
                actor_loss = (self.alpha * curr_log - q_next).mean()
                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()

                # alpha loss
                alpha_loss = -(self.log_alpha * (curr_log + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

            q_loss += critic_loss.item() * 0.5
            pg_loss += actor_loss.item()
            a_loss += alpha_loss.item()
        
        return pg_loss, q_loss, a_loss
