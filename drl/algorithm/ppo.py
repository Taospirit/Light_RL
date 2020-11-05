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

class PPO(BasePolicy):  # option: double
    def __init__(
        self,
        model,
        buffer_size=1000,
        actor_learn_freq=1,
        learning_rate=1e-4,
        discount_factor=0.99,
        ratio_clip=0.2,
        lam_entropy=0.01,
        gae_lamda=0.995,  # td
        batch_size=100,
        verbose=False,
        act_dim=None,
    ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.ratio_clip = ratio_clip
        self.lam_entropy = lam_entropy
        self.adv_norm = False  # normalize advantage, defalut=False
        self.rew_norm = False  # normalize reward, default=False
        self.schedule_clip = False
        self.schedule_adam = False

        self.actor_learn_freq = actor_learn_freq
        self._gamma = discount_factor
        self._gae_lam = gae_lamda
        self._update_iteration = 10
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

        self._verbose = verbose
        self._batch_size = batch_size
        self._normalized = lambda x, e: (x - x.mean()) / (x.std() + e)
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        self.act_dim = act_dim

    def learn(self, i_episode=0, num_episode=100):
        if not self.buffer.is_full():
            print(f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0

        loss_actor_avg, loss_critic_avg = 0, 0

        mem = self.buffer.split(self.buffer.all_memory())
        if self.act_dim is None:
            self.act_dim = mem['a'].shape[-1]
        S = torch.tensor(mem['s'], dtype=torch.float32, device=device)
        A = torch.tensor(mem['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)
        S_ = torch.tensor(mem['s_'], dtype=torch.float32, device=device)
        R = torch.tensor(mem['r'], dtype=torch.float32).view(-1, 1)
        Log = torch.tensor(mem['l'], dtype=torch.float32, device=device).view(-1, 1)
        if self._verbose:
            print(f'Shape S:{S.shape}, A:{A.shape}, R:{R.shape}, S_:{S_.shape}, Log:{Log.shape}')

        with torch.no_grad():
            v_evals = self.critic_eval(S).cpu().numpy()
            end_v_eval = self.critic_eval(S_[-1]).cpu().numpy()

        rewards = self._normalized(R, self.eps).numpy() if self.rew_norm else R.numpy()
        adv_gae = self.GAE(rewards, v_evals, next_v_eval=end_v_eval,
                              gamma=self._gamma, lam=self._gae_lam)
        advantage = torch.from_numpy(adv_gae).to(device).unsqueeze(-1)
        advantage = self._normalized(advantage, 1e-10) if self.adv_norm else advantage

        for _ in range(self._update_iteration):
            v_eval = self.critic_eval(S)
            v_target = advantage + v_eval.detach()

            critic_loss = self.criterion(v_eval, v_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1
            if self._verbose:
                print(f'=======Learn_Critic_Net, cnt{self._learn_critic_cnt}=======')

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                # actor_core
                mu, sigma = self.actor_eval(S)
                dist = Normal(mu, sigma)
                new_log_prob = dist.log_prob(A)

                pg_ratio = torch.exp(new_log_prob - Log)  # size = [batch_size, 1]
                clipped_pg_ratio = torch.clamp(pg_ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                surrogate_loss = -torch.min(pg_ratio * advantage, clipped_pg_ratio * advantage).mean()

                # policy entropy
                entropy_loss = -torch.mean(torch.exp(new_log_prob) * new_log_prob)

                actor_loss = surrogate_loss - self.lam_entropy * entropy_loss

                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1
                if self._verbose:
                    print(f'=======Learn_Actort_Net, cnt{self._learn_actor_cnt}=======')

        self.buffer.clear()
        assert self.buffer.is_empty()

        # update param
        ep_ratio = 1 - (i_episode / num_episode)
        if self.schedule_clip:
            self.ratio_clip = 0.2 * ep_ratio

        if self.schedule_adam:
            new_lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg
