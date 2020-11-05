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

def _l2_project(next_distr_v, rewards_v, dones_mask_t, gamma, n_atoms, v_min, v_max):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    rewards = np.squeeze(rewards, axis=1)

    for atom in range(n_atoms):
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        # print (eq_mask.shape, l.shape)
        # print (eq_mask)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones_mask]))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


class MSAC(BasePolicy):
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
        update_iteration=10,
        verbose=False,
        use_priority=False,
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
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size

        self.use_priority = use_priority
        self.use_dist = model.value_net.use_dist

        if self.use_priority:
            self.buffer = PriorityReplayBuffer(buffer_size)
        else:
            self.buffer = ReplayBuffer(buffer_size) # off-policy

        if self.use_dist:
            assert model.value_net.num_atoms > 1
            # assert isinstance(model.value_net, CriticModelDist)
            self.v_min = model.value_net.v_min
            self.v_max = model.value_net.v_max
            self.num_atoms = model.value_net.num_atoms
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)


        self.actor_eval = model.policy_net.to(device).train()
        self.critic_eval = model.value_net.to(device).train()

        self.actor_target = self.copy_net(self.actor_eval)
        self.critic_target = self.copy_net(self.critic_eval)
        
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss(reduction='none') # keep batch dim
        self.act_dim = act_dim

        self.target_entropy = -torch.tensor(1).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = self.log_alpha.exp()


    def _tensor(self, data, use_cuda=False):
        if np.array(data).ndim == 1:
            data = torch.tensor(data, dtype=torch.float32).view(-1, 1)
        else:
            data = torch.tensor(data, dtype=torch.float32)
        if use_cuda:
            data = data.to(device)
        return data

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

    def learn(self):
        pg_loss, q_loss, a_loss = 0, 0, 0
        for _ in range(self._update_iteration):
            if self.use_priority:
                # s_{t}, n-step_rewards, s_{t+n}
                tree_idxs, S, A, R, S_, M, weights = self.buffer.sample(self._batch_size)
                W = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
            else:
                batch_split = self.buffer.split_batch(self._batch_size)
                S, A, M, R, S_ = batch_split['s'], batch_split['a'], batch_split['m'], batch_split['r'],  batch_split['s_']
            # print ('after sampling from buffer!')
            if self.act_dim is None:
                self.act_dim = A.shape[-1]
                self.target_entropy = -torch.tensor(self.act_dim).to(device)
                print(self.target_entropy)
                assert 0

            R = torch.tensor(R, dtype=torch.float32).view(-1, 1)
            S = torch.tensor(S, dtype=torch.float32, device=device)
            # A = torch.tensor(A, dtype=torch.float32, device=device).view(-1, 1)
            A = torch.tensor(A, dtype=torch.float32, device=device).view(-1, self.act_dim)
            # print (f'A shape {A.shape}')
            M = torch.tensor(M, dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(S_, dtype=torch.float32, device=device)
            
            # print (f'size S:{S.size()}, A:{A.size()}, M:{M.size()}, R:{R.size()}, S_:{S_.size()}')
            if self.use_dist:
                # print (M[0].size())
                # print (M[0])
                # print (M[0].item())
                # assert 0
                # D = torch.from_numpy(np.array([1^int(mask.item()) for mask in M])).view(-1, 1)
                # print (f'size S:{S.shape}, A:{A.size()}, M:{M.size()}, R:{R.size()}, S_:{S_.size()}, D:{D.size()}')
                # assert 0
                batch_loss = self.learn_dist(S, A, R, S_, M)

            else:
                with torch.no_grad():
                    next_A, next_log = self.actor_target.evaluate(S_)
                    q1_next, q2_next = self.critic_target(S_, next_A)
                    q_next = torch.min(q1_next, q2_next) - self.alpha * next_log
                    q_target = R + M * self._gamma * q_next.cpu()
                    q_target = q_target.to(device)
                # q_loss
                q1_eval, q2_eval = self.critic_eval(S, A)
                loss1 = self.criterion(q1_eval, q_target)
                loss2 = self.criterion(q2_eval, q_target)
                # print(f'q_eval {q1_eval.shape}, q_target {q_target.shape}')
                batch_loss = 0.5 * (loss1 + loss2)

            if self.use_priority:
                critic_loss = (W * batch_loss).mean()
                self.buffer.update_priorities(tree_idxs, np.abs(batch_loss.detach().cpu().numpy()) + 1e-6)
            else:
                critic_loss = batch_loss.mean()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            actor_loss = torch.tensor(0)
            alpha_loss = torch.tensor(0)
            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                curr_A, curr_log = self.actor_eval.evaluate(S)
                if self.use_dist:
                    z1_next, z2_next = self.critic_eval.get_probs(S, curr_A)
                    p_next = torch.stack([torch.where(z1.sum() < z2.sum(), z1, z2) for z1, z2 in zip(z1_next, z2_next)])
                    num_atoms = torch.tensor(self.num_atoms, dtype=torch.float32, device=device)
                    # actor_loss = p_next * num_atoms
                    # actor_loss = torch.sum(actor_loss, dim=1)
                    # actor_loss = -(actor_loss + self.alpha * curr_log).mean()
                    actor_loss = (self.alpha * curr_log - p_next)
                    actor_loss = torch.sum(actor_loss, dim=1)
                    actor_loss = actor_loss.mean()
                else:
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

            q_loss += critic_loss.item()
            pg_loss += actor_loss.item()
            a_loss += alpha_loss.item()

            if self._learn_critic_cnt % self.target_update_freq:
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        

        return pg_loss, q_loss, a_loss
