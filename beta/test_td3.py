#region
import gym
import os
from os.path import abspath, dirname
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import numpy as np

from drl.algorithm import TD3
from utils.plot import plot
from utils.config import config

#config
config = config['td3']

env_name = config['env_name']
buffer_size = config['buffer_size']
actor_learn_freq = config['actor_learn_freq']
update_iteration = config['update_iteration']
target_update_freq = config['target_update_freq']
batch_size = config['batch_size']
hidden_dim = config['hidden_dim']
episodes = config['episodes'] + 10
max_step = config['max_step']
target_update_tau = config['target_update_tau']
lr = config['lr']

LOG_DIR = config['LOG_DIR']
SAVE_DIR = config['SAVE_DIR'] + env_name
POLT_NAME = config['POLT_NAME'] + env_name
PKL_DIR = config['PKL_DIR'] + env_name

file_path = abspath(dirname(__file__))
pkl_dir = file_path + PKL_DIR
model_save_dir = file_path + SAVE_DIR
save_file = model_save_dir.split('/')[-1]
writer_path = model_save_dir + LOG_DIR

try:
    os.makedirs(model_save_dir)
except FileExistsError:
    import shutil
    shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)

env = gym.make(env_name)
env = env.unwrapped

#Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2
#endregion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ActorModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        layer_norm(self.fc1, std=1.0)
        layer_norm(self.fc2, std=1.0)
        layer_norm(self.out, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.tanh(self.out(x))
        return out

    def action(self, state, noise_std=0, noise_clip=0.5):
        action = self.forward(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(device)
            action += noise_norm.clamp(-noise_clip, noise_clip)
        action = action.clamp(-action_max, action_max)
        return action

class CriticModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        self.net2 = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q = self.net1(x)
        return q

    def twinQ(self, state, action):
        x = torch.cat((state, action), dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2

TRAIN = True
PLOT = True
WRITER = False

def map_action(action):
    if isinstance(action, torch.Tensor):
        action = action.item()
    return action * action_scale + action_bias

def sample(env, policy, max_step, warm_up=False):
    rewards = []
    state = env.reset()
    for step in range(max_step):
        #==============choose_action==============
        action = policy.choose_action(state) if not warm_up else env.action_space.sample()
        next_state, reward, done, info = env.step(map_action(action))
        if TRAIN:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
        else:
            env.render()
        rewards.append(reward)
        if done:
            break
        state = next_state
    return rewards

def train():
    model = namedtuple('model', ['policy_net', 'value_net'])
    actor = ActorModel(state_space, hidden_dim, action_space)
    critic = CriticModel(state_space, hidden_dim, action_space)
    rl_agent = model(actor, critic)
    policy = TD3(rl_agent, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq,
        update_iteration=update_iteration, target_update_freq=target_update_freq, 
        target_update_tau=target_update_tau, batch_size=batch_size, learning_rate=lr)
    writer = SummaryWriter(writer_path)

    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    mean, std = [], []
    live_time = []

    while policy.warm_up():
        sample(env, policy, max_step, warm_up=True)
        print (f'Warm up for buffer {policy.buffer.size()}', end='\r')

    for i_eps in range(episodes):
        rewards = sample(env, policy, max_step)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        mean.append(reward_mean)
        std.append(reward_std)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{round(reward_mean, 3)}')
        else:
            #==============learn==============
            pg_loss, q_loss = policy.learn()
            if PLOT:
                live_time.append(reward_mean)
                plot(live_time, POLT_NAME, model_save_dir, 100)
            if WRITER:
                writer.add_scalar('reward', reward_mean, global_step=i_eps)
                writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
                writer.add_scalar('loss/q_loss', q_loss, global_step=i_eps)
            if i_eps % 5 == 0:
                print (f'EPS:{i_eps}, reward_mean:{round(reward_mean, 3)}, pg_loss:{round(pg_loss, 3)}, q_loss:{round(q_loss, 3)}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    writer.close()
    env.close()
    return mean, std

if __name__ == '__main__':
    env.seed(1)
    torch.manual_seed(1)
    #only for logistic test
    train()