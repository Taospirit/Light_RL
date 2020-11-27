# from test_tool import policy_test
# region
from utils.config import config
from utils.plot import plot
import gym
import os
import numpy as np
import argparse
from os.path import abspath, dirname
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

import sys
sys.path.append('..')

from drl.algorithm import MSAC

mujuco_env = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2']
env_name = mujuco_env[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# region Network
def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ActorModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        layer_norm(self.fc1, std=1.0)
        layer_norm(self.fc2, std=1.0)
        layer_norm(self.mean, std=1.0)
        layer_norm(self.log_std, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def action(self, state, test=False):
        mean, log_std = self.forward(state)
        if test:
            return torch.tanh(mean).detach().cpu().numpy()

        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std*z.to(device))
        log_prob = normal.log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


class CriticModelDist(nn.Module):
    def __init__(self, obs_dim, mid_dim, act_dim, use_dist=False, v_min=-10, v_max=0, num_atoms=51):
        super().__init__()
        self.use_dist = use_dist
        if use_dist:
            self.v_min = v_min
            self.v_max = v_max
            self.num_atoms = num_atoms
            self.net1 = self.build_network(obs_dim, mid_dim, act_dim, num_atoms)
            self.net2 = self.build_network(obs_dim, mid_dim, act_dim, num_atoms)
        else:
            self.net1 = self.build_network(obs_dim, mid_dim, act_dim)
            self.net2 = self.build_network(obs_dim, mid_dim, act_dim)

        # self.fc1 = nn.Linear(obs_dim + act_dim, mid_dim)
        # self.fc2 = nn.Linear(mid_dim, mid_dim)
        # self.fc3 = nn.Linear(mid_dim, num_atoms)

        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def build_network(self, obs_dim, mid_dim, act_dim, num_atoms=1):
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, act_dim * num_atoms), )
        return self.net

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2

    def get_probs(self, obs, act, log=False):
        z1, z2 = self.forward(obs, act)
        if log:
            z1 = torch.log_softmax(z1, dim=1)
            z2 = torch.log_softmax(z2, dim=1)
        else:
            z1 = torch.softmax(z1, dim=1)
            z2 = torch.softmax(z2, dim=1)
        return z1, z2
# endregion


def map_action(env, action):
    if isinstance(action, torch.Tensor):
        action = action.item()
    action_scale = (env.action_space.high - env.action_space.low) / 2
    action_bias = (env.action_space.high + env.action_space.low) / 2
    return action * action_scale + action_bias

def eval_policy(policy, env, seed, eval_episodes=10):
    eval_env = gym.make(env)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.choose_action(state, test=True)
            action = map_action(eval_env, action) 
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main(seed):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    kwargs = {
        'buffer_size': int(1e6),
        'batch_size': 256,
        'policy_freq': 2,
        'tau': 0.005,
        'discount': 0.99,
        'policy_lr': 3e-4,
        'value_lr': 3e-4,
        'learn_iteration': 1,
        'verbose': False,
        'act_dim': action_dim,
        'use_priority': False,
        'use_munchausen': False,
        'use_PAL': False,
        'n_step': 1,
    }
    
    file_name = f"MSAC_{env_name}_{seed}_{kwargs['use_priority']}_{kwargs['use_munchausen']}_{kwargs['use_PAL']}"
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    model = namedtuple('model', ['policy_net', 'value_net'])
    actor = ActorModel(state_dim, hidden_dim, action_dim)
    critic = CriticModelDist(state_dim, hidden_dim, action_dim, use_dist=False)
    rl_agent = model(actor, critic)
    policy = MSAC(rl_agent, **kwargs)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_name, seed)]

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_timesteps = 3e6
    start_timesteps = 25e3
    eval_freq = 5e3

    state = env.reset()
    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.choose_action(state)
            action = map_action(env, action)
        # Perform action
        next_state, reward, done, _ = env.step(action)
        # env.render()

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        mask = 0 if done_bool else 1

        policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            pg_loss, q_loss, a_loss = policy.learn()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, env_name, seed))
            np.save("./results/%s" % (file_name), evaluations)


if __name__ == "__main__":
    for seed in [0, 10, 20]:
        main(seed)