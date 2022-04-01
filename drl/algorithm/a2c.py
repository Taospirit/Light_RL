import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from collections import namedtuple
# ac_model = namedtuple('model', ['policy_net', 'value_net'])

class A2C(BasePolicy):
    def __init__(
        self, 
        a_model,
        c_model,
        buffer_size=1000,
        learning_rate=1e-3,
        discount_factor=0.99,
        gae_lamda=1, # mc
        num_episodes=1000,
        schedule_adam = False,
        ):
        super().__init__()
        self.lr = learning_rate
        self.end_lr = self.lr * 0.1
        self.eps = np.finfo(np.float32).eps.item()

        self._gamma = discount_factor
        self._gae_lamda = gae_lamda # default: 1, MC
        self._schedule_adam = schedule_adam
        self._learn_cnt = 0

        self.actor_eval = a_model.to(device).train()
        self.critic_eval = c_model.to(device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.criterion = nn.SmoothL1Loss()
        self.num_episodes = num_episodes

    def sample(self, env, max_len=None, train=1, render=0, avg=0):
        rews = 0
        state = env.reset()

        if not max_len:
            max_len = self.buffer.capacity() if train else int(1e6)
        for i in range(max_len):
            act, log_prob = self.action(state, train)

            next_state, rew, done, info = env.step(act)
            if train:
                mask = 0 if done else 1
                self.process(s=state, r=rew, l=log_prob, m=mask)
            if render:
                env.render() # for self define env, you must define env.render() for visual
            rews += rew
            if done:
                break
            state = next_state
        rews = rews / (i + 1) if avg else rews
        return rews

    def action(self, state, train=1):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        if train:
            self.actor_eval.train()
        else:
            self.actor_eval.eval()

        act_source = self.actor_eval(state)
        dist = F.softmax(act_source, dim=-1)
        m = Categorical(dist)
        act = m.sample()
        log_prob = m.log_prob(act)

        return act.item(), log_prob

    def learn(self):
        pg_loss, v_loss = 0, 0
        mem = self.buffer.split(self.buffer.all_memory()) # s, r, l, m
        S = torch.tensor(mem['s'], dtype=torch.float32, device=device)
        R = torch.tensor(mem['r'], dtype=torch.float32).view(-1, 1)
        M = torch.tensor(mem['m'], dtype=torch.float32).view(-1, 1)
        # Log = torch.stateck(list(mem['l'])).view(-1, 1)
        Log = torch.stack(mem['l']).view(-1, 1)

        v_eval = self.critic_eval(S)
        # v_eval = self.critic_eval.net(S)

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

        if self._schedule_adam:
            new_lr = self.lr + (self.end_lr - self.lr) / self.num_episodes * self._learn_cnt
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr
        return pg_loss, v_loss