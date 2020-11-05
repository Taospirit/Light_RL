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

class SAC1(BasePolicy):
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
        self.value_eval = model.v_net.to(device).train()

        self.value_target = self.copy_net(self.value_eval)
        
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.value_eval_optim = optim.Adam(self.value_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        self.act_dim = act_dim

    def learn(self):
        pg_loss, v_loss, q_loss = 0, 0, 0
        for _ in range(self._update_iteration):
            batch = self.buffer.split_batch(self._batch_size)
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
                next_value = torch.min(new_q1_value, new_q2_value) - log_prob
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
            q_value_loss = 0.5 * (loss1 + loss2)

            # policy loss
            policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()
            # policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value).detach()).mean()

            # update V
            self.value_eval_optim.zero_grad()
            value_loss.backward()
            self.value_eval_optim.step()

            # update soft Q
            self.critic_eval_optim.zero_grad()
            q_value_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            # update policy
            self.actor_eval_optim.zero_grad()
            policy_loss.backward()
            self.actor_eval_optim.step()

            pg_loss += policy_loss.item()
            v_loss += value_loss.item()
            q_loss += q_value_loss.item()

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.value_target, self.value_eval, self.tau)
                
            return pg_loss, q_loss, v_loss
