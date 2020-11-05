import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG(BasePolicy):
    def __init__(
        self,
        model,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=1,
        target_update_tau=0.005,
        learning_rate=1e-4,
        discount_factor=0.99,
        batch_size=100,
        update_iteration=10,
        verbose=False,
        act_dim=None,
        num_episodes=1000,
    ):
        super().__init__()
        self.lr = learning_rate
        self.end_lr = learning_rate * 0.1
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._update_iteration = update_iteration
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = False
        self._batch_size = batch_size
        self.schedule_adam = True
        self.buffer = ReplayBuffer(buffer_size)

        self.actor_eval = model.policy_net.to(device).train()  # pi(s)
        self.critic_eval = model.value_net.to(device).train()  # Q(s, a)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.MSELoss()  # why mse?
        self.act_dim = act_dim
        self.num_episodes = num_episodes

    def learn(self):
        loss_actor_avg, loss_critic_avg = 0, 0

        for _ in range(self._update_iteration):
            batch = self.buffer.split_batch(self._batch_size)
            if self.act_dim is None:
                self.act_dim = np.array(batch['a']).shape[-1]

            S = torch.tensor(batch['s'], dtype=torch.float32, device=device)  # [batch_size, state_dim]
            # print(batch['a'])
            A = torch.tensor(batch['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)  # [batch_size, act_dim]
            M = torch.tensor(batch['m'], dtype=torch.float32).view(-1, 1) # [batch_size, 1]
            R = torch.tensor(batch['r'], dtype=torch.float32).view(-1, 1) # [batch_size, 1]
            S_ = torch.tensor(batch['s_'], dtype=torch.float32, device=device) # [batch_size, state_dim]
            if self._verbose:
                print(f'Shape S:{S.shape}, A:{A.shape}, M:{M.shape}, R:{R.shape}, S_:{S_.shape}')

            with torch.no_grad():
                q_next = self.critic_target(S_, self.actor_target(S_))
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(device)
            q_eval = self.critic_eval(S, A)  # [batch_size, q_value_size]
            critic_loss = self.criterion(q_eval, q_target)

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()

            loss_critic_avg += critic_loss.item()
            self._learn_critic_cnt += 1
            if self._verbose:
                print(f'=======Learn_Critic_Net, cnt:{self._learn_critic_cnt}=======')

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                actor_loss = -self.critic_eval(S, self.actor_eval(S)).mean()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()

                loss_actor_avg += actor_loss.item()
                self._learn_actor_cnt += 1
                if self._verbose:
                    print(f'=======Learn_Actort_Net, cnt:{self._learn_actor_cnt}=======')

            if self._learn_critic_cnt % self.target_update_freq == 0:
                if self._verbose:
                    print(f'=======Soft_sync_weight of DDPG, tau:{self.tau}=======')
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

        if self.schedule_adam:

            new_lr = self.lr + (self.end_lr - self.lr) / self.num_episodes * self._learn_critic_cnt / self._update_iteration
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration
        return loss_actor_avg, loss_critic_avg