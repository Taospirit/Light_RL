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
from drl.utils import PriorityReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSAC(BasePolicy):
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
        n_step=1,
        use_munchausen=False,
        use_priority=False,
        use_dist_q=False,
        use_PAL=False,
        ):
        super().__init__()
        self.tau = tau
        self.gamma = discount
        self.policy_freq = policy_freq
        self.learn_iteration = learn_iteration
        self.verbose = verbose
        self.act_dim = act_dim
        self.batch_size = batch_size

        self.use_dist_q = use_dist_q
        self.use_priority = use_priority
        self.use_munchausen = use_munchausen
        self.use_PAL = use_PAL

        assert not (self.use_priority and self.use_PAL)

        self.buffer = ReplayBuffer(buffer_size)
        if self.use_priority:
            self.buffer = PriorityReplayBuffer(buffer_size, gamma=discount, n_step=n_step)

        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)
        
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=policy_lr)
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=value_lr)

        self.criterion = nn.SmoothL1Loss(reduction='none') # keep batch dim
        self.target_entropy = -torch.tensor(1).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=policy_lr)
        self.alpha = self.log_alpha.exp()

        self.eps = np.finfo(np.float32).eps.item()
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

    def learn_dist(self, obs, act, rew, next_obs, mask):
        with torch.no_grad():
            next_act, next_log_pi = self.actor_target(next_obs)
            # q(s, a) change to z(s, a) to discribe a distributional
            z1_next, z2_next = self.critic_target.get_probs(next_obs, next_act) # [batch_size, num_atoms]
            p_next = torch.stack([torch.where(z1.sum() < z2.sum(), z1, z2) for z1, z2 in zip(z1_next, z2_next)])
            p_next -= (self.alpha * next_log_pi)
            Tz = rew.unsqueeze(1) + mask * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            b = (Tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            m = obs.new_zeros(self._batch_size, self.num_atoms).cpu()
            p_next = p_next.cpu()
            # print (f'm device: {m.device}')
            # print (f'p_next device: {p_next.device}')
            offset = torch.linspace(0, ((self._batch_size - 1) * self.num_atoms), self._batch_size).unsqueeze(1).expand(self._batch_size, self.num_atoms).to(l)
            m.view(-1).index_add_(0, (l + offset).view(-1), (p_next * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (p_next * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        m = m.to(device)
        log_z1, log_z2 = self.critic_eval.get_probs(obs, act, log=True)
        loss1 = -(m * log_z1).sum(dim=1)
        loss2 = -(m * log_z2).sum(dim=1)
        batch_loss = 0.5 * (loss1 + loss2)
        return batch_loss

    def get_munchausen_rew(self, obs, act, rew):
        self.m_alpha = 0.9
        self.m_tau = 0.03
        self.lo = -1
        mu, log_std = self.actor_eval(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_pi_a = self.m_tau * dist.log_prob(act).mean(1).unsqueeze(1).cpu()
        m_rew = rew + self.m_alpha * torch.clamp(log_pi_a, min=self.lo, max=0)
        return m_rew

    def LAP_loss(self, q_eval, q_target):
        self.min_priority = 1.0
        self.alpha = 0.4
        def huber(x):
            return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

        td_loss1 = (q_eval[0] - q_target).abs()
        td_loss2 = (q_eval[1] - q_target).abs()
        critic_loss = huber(td_loss1) + huber(td_loss2)
        priority = torch.max(td_loss1, td_loss2).clamp(min=self.min_priority).pow(self.alpha).cpu().data.numpy().flatten()

        return critic_loss, priority

    def PAL_loss(self, q_eval, q_target):
        self.min_priority = 1.0
        self.alpha = 0.4
        def PAL(x):
            return torch.where(
                x.abs() < self.min_priority, 
                (self.min_priority ** self.alpha) * 0.5 * x.pow(2), 
                self.min_priority * x.abs().pow(1. + self.alpha)/(1. + self.alpha)
            ).mean()

        td_loss1 = (q_eval[0] - q_target)
        td_loss2 = (q_eval[1] - q_target)
        critic_loss = PAL(td_loss1) + PAL(td_loss2)
        critic_loss /= torch.max(td_loss1.abs(), td_loss2.abs()).clamp(min=self.min_priority).pow(self.alpha).mean().detach()

        return critic_loss

    def learn(self):
        pg_loss, q_loss, a_loss = 0, 0, 0
        for _ in range(self.learn_iteration):
            if self.use_priority:
                S, A, R, S_, M, indices, weights = self.buffer.sample(self.batch_size)
                W = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
            else:
                batch_split = self.buffer.split_batch(self.batch_size)
                S, A, R, S_, M = batch_split['s'], batch_split['a'], batch_split['r'],  batch_split['s_'], batch_split['m']

            if self.act_dim is None:
                self.act_dim = np.array(A).shape[-1]
                self.target_entropy = -torch.tensor(self.act_dim).to(device)

            S = torch.tensor(S, dtype=torch.float32, device=device)
            A = torch.tensor(A, dtype=torch.float32, device=device).view(-1, self.act_dim)
            M = torch.tensor(M, dtype=torch.float32).view(-1, 1)
            R = torch.tensor(R, dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(S_, dtype=torch.float32, device=device)

            if self.verbose:
                print(f'shape S:{S.shape}, A:{A.shape}, M:{M.shape}, R:{R.shape}, S_:{S_.shape}, W:{W.shape}')

            if self.use_munchausen:
                R = self.get_munchausen_rew(S, A, R)

            if self.use_dist_q:
                batch_q_loss = self.learn_dist(S, A, R, S_, M)
            else:
                with torch.no_grad():
                    next_A, next_log = self.actor_target.evaluate(S_)
                    q1_next, q2_next = self.critic_target(S_, next_A)
                    q_next = torch.min(q1_next, q2_next) - self.alpha * next_log
                    q_target = R + M * self.gamma * q_next.cpu()
                    q_target = q_target.to(device)
                q1_eval, q2_eval = self.critic_eval(S, A)

                if self.use_PAL:
                    critic_loss = self.PAL_loss((q1_eval, q2_eval), q_target)
                else:
                    loss1 = self.criterion(q1_eval, q_target)
                    loss2 = self.criterion(q2_eval, q_target)
                    batch_q_loss = 0.5 * (loss1 + loss2)

            if self.use_priority:
                critic_loss = (W * batch_q_loss).mean()
                td_errors = batch_q_loss.detach().cpu().numpy().sum(1)
                self.buffer.update_priorities(indices, np.abs(td_errors) + 1e-6)
            if self.use_PAL:
                pass
            else:
                critic_loss = batch_q_loss.mean()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            actor_loss = alpha_loss = torch.tensor(0)
            if self._learn_critic_cnt % self.policy_freq == 0:
                curr_A, curr_log = self.actor_eval.evaluate(S)

                if self.use_dist_q:
                    z1_next, z2_next = self.critic_eval.get_probs(S, curr_A)
                    p_next = torch.stack([torch.where(z1.sum() < z2.sum(), z1, z2) for z1, z2 in zip(z1_next, z2_next)])
                    num_atoms = torch.tensor(self.num_atoms, dtype=torch.float32, device=device)
                    actor_loss = (self.alpha * curr_log - p_next)
                    actor_loss = torch.sum(actor_loss, dim=1)
                    actor_loss = actor_loss.mean()
                else:
                    q1_next, q2_next = self.critic_eval(S, curr_A)
                    q_next = torch.min(q1_next, q2_next)
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

                self._learn_actor_cnt += 1

                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

            q_loss += critic_loss.item()
            pg_loss += actor_loss.item()
            a_loss += alpha_loss.item()
        
        return pg_loss, q_loss, a_loss