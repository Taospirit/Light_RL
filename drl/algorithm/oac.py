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

class OAC(BasePolicy): # no value network
    def __init__(
        self, 
        model,
        buffer_size=1000,
        batch_size=100,
        actor_learn_freq=1,
        target_update_freq=5,
        target_update_tau=0.01,
        learning_rate=1e-3,
        discount_factor=0.99,
        verbose=False,
        update_iteration=10,
        act_dim=None
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

    def choose_action(self, state, test=False, beta_UB=1.0, delta=1.0):
        # paper: Better Exploration with Optimistic Actor-Critic, NeurIPS 2019
        # pdf: https://arxiv.org/pdf/1910.12807.pdf
        # ref: https://github.com/microsoft/oac-explore/blob/master/optimistic_exploration.py
        # paper param: beta_UB=4.66 delta=23.53, env_name=humanoid
        state = torch.tensor(state, dtype=torch.float32, device=device)
        if test:
            self.actor_eval.eval()
            mean, log_std = self.actor_eval(state)
            return mean.detach().cpu().numpy()
            
        assert len(list(state.shape)) == 1 # not batch
        mu_T, log_std = self.actor_eval(state)
        std = torch.exp(log_std)
        # assert len(list(mu_T.shape)) == 1, mu_T
        # assert len(list(std.shape)) == 1
        mu_T.requires_grad_()
        curr_act = torch.tanh(mu_T).unsqueeze(0) # action
        state = state.unsqueeze(0)

        q1, q2 = self.critic_target(state, curr_act)
        mu_q = (q1 + q2) / 2.0
        sigma_q = torch.abs(q1 - q2) / 2.0
        Q_UB = mu_q + beta_UB * sigma_q

        grad = torch.autograd.grad(Q_UB, mu_T)
        grad = grad[0]

        assert grad is not None
        assert mu_T.shape == grad.shape

        sigma_T = torch.pow(std, 2)
        denom = torch.sqrt(
            torch.sum(
                torch.mul(torch.pow(grad, 2), sigma_T)
            )
        ) + 10e-6

        mu_C = np.sqrt(2.0 * delta) * torch.mul(sigma_T, grad) / denom
        assert mu_C.shape == mu_T.shape
        mu_E = mu_T + mu_C
        assert mu_E.shape == std.shape

        normal = Normal(mu_E, std)
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    def learn(self):
        pg_loss, q_loss, a_loss = 0, 0, 0
        for _ in range(self._update_iteration):
            batch = self.buffer.split_batch(self._batch_size)
            if self.act_dim is None:
                self.act_dim = np.array(batch['a']).shape[-1]
                self.target_entropy = -torch.tensor(self.act_dim).to(device)

            S = torch.tensor(batch['s'], dtype=torch.float32, device=device)
            A = torch.tensor(batch['a'], dtype=torch.float32, device=device).view(-1, 1)
            M = torch.tensor(batch['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch['s_'], dtype=torch.float32, device=device)

            # print (f'size S:{S.size()}, A:{A.size()}, M:{M.size()}, R:{R.size()}, S_:{S_.size()}, W:{W.size()}')
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

            q_loss += critic_loss.item() * 0.5
            pg_loss += actor_loss.item()
            a_loss += alpha_loss.item()

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        
        return pg_loss, q_loss, a_loss
