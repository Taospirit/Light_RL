import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C(BasePolicy): #option: double
    def __init__(
        self, 
        model, 
        buffer_size=1000,
        learning_rate=1e-3,
        discount_factor=0.99,
        gae_lamda=1, # mc
        verbose = False,
        num_episodes=1000,
        ):
        super().__init__()
        self.lr = learning_rate
        self.end_lr = self.lr * 0.1
        self.eps = np.finfo(np.float32).eps.item()

        self._gamma = discount_factor
        self._gae_lamda = gae_lamda # default: 1, MC
        self._learn_cnt = 0
        self._verbose = verbose
        self.schedule_adam = True
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        self.num_episodes = num_episodes

    def learn(self):
        pg_loss, v_loss = 0, 0
        mem = self.buffer.split(self.buffer.all_memory()) # s, r, l, m
        S = torch.tensor(mem['s'], dtype=torch.float32, device=device)
        R = torch.tensor(mem['r'], dtype=torch.float32).view(-1, 1)
        M = torch.tensor(mem['m'], dtype=torch.float32).view(-1, 1)
        # Log = torch.stack(list(mem['l'])).view(-1, 1)
        Log = torch.stack(mem['l']).view(-1, 1)

        v_eval = self.critic_eval(S)

        v_evals = v_eval.detach().cpu().numpy()
        rewards = R.numpy()
        masks = M.numpy()
        adv_gae_mc = self.GAE(rewards, v_evals, next_v_eval=0, masks=masks, gamma=self._gamma, lam=self._gae_lamda) # MC adv
        advantage = torch.from_numpy(adv_gae_mc).to(device).reshape(-1, 1)

        # critic_core
        v_target = advantage + v_eval.detach()
        critic_loss = self.criterion(v_eval, v_target)
        # actor_core
        actor_loss = (-Log * advantage).sum()

        self.critic_eval_optim.zero_grad()
        critic_loss.backward()
        self.critic_eval_optim.step()

        self.actor_eval_optim.zero_grad()
        actor_loss.backward()
        self.actor_eval_optim.step()

        v_loss += critic_loss.item()
        pg_loss += actor_loss.item()
        self._learn_cnt += 1

        self.buffer.clear()
        assert self.buffer.is_empty()

        if self.schedule_adam:
            new_lr = self.lr + (self.end_lr - self.lr) / self.num_episodes * self._learn_cnt
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr
        return pg_loss, v_loss
        