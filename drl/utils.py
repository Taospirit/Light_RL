import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot(steps, y_label, model_save_dir, step_interval):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Reward')
    ax.plot(steps)
    RunTime = len(steps)

    path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % step_interval == 0:
        plt.savefig(path)
        print(f'save fig in {path}')
    plt.pause(0.0000001)

class Buffer(object):
    def __init__(self, size, **kwargs):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def append(self, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs):
        raise NotImplementedError

    def capacity(self):
        return self.memory.maxlen

    def is_full(self):
        return len(self) == self.capacity()

    def is_empty(self):
        return len(self) == 0

class ReplayBuffer(Buffer):
    def __init__(self, size, replay=True):
        super().__init__(size)
        self.allow_replay = replay

    def append(self, **kwargs):
        if not self.allow_replay and self.is_full():
            return
        self.memory.append(kwargs)
        
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def clear(self): # manual clear
        self.memory.clear()

    def split_batch(self, batch_size):
        return self.split(self.sample(batch_size))

    def split(self, batchs):
        split_res = {}
        for key in batchs[-1].keys():
            split_res[key] = [item[key] for item in batchs]
          
        return split_res
    
    # @property
    def all_memory(self):
        return self.memory

class PriorityReplayBuffer(Buffer):
    def __init__(self, size, gamma=0.99, alpha=0.5, beta=0.4, beta_increment=6e-4, n_step=3):
        super().__init__(size)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.n_step = n_step
      
        self.priorities = deque(maxlen=size)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def append(self, **kwargs):
        max_prior = np.max(self.priorities) if self.memory else 1.0
        obs, act, rew, next_obs, mask = kwargs['s'], kwargs['a'], kwargs['r'], kwargs['s_'], kwargs['m']

        self.n_step_buffer.append([obs, act, rew, next_obs, mask])
        if len(self.n_step_buffer) < self.n_step:
            return
        rew, next_obs, mask = self._get_n_step_info()
        obs, act = self.n_step_buffer[0][:2]

        self.memory.append([obs, act, rew, next_obs, mask])
        self.priorities.append(max_prior)

    def sample(self, batch_size):
        memory_size = len(self)

        batch_size = min(batch_size, memory_size)
        probs = np.array(self.priorities) ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(memory_size, batch_size, replace=False, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (memory_size * probs[indices]) ** (- self.beta)
        self.beta = min(1, self.beta + self.beta_increment)
       
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        obs, act, rew, next_obs, mask = zip(* samples)
        return obs, act, rew, next_obs, mask, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def _get_n_step_info(self):
        n_step_rew, n_step_obs, mask = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, m in reversed(list(self.n_step_buffer)[:-1]):
            n_step_rew = self.gamma * n_step_rew * m + rew
            n_step_obs, mask = (next_obs, m) if not m else (n_step_obs, mask)
        return n_step_rew, n_step_obs, mask

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape