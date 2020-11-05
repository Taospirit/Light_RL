import gym
import os
from os.path import abspath, dirname
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple
import sys
sys.path.append('../..')
# sys.path.appedn('..')

from drl.algorithm import DDPG
from drl.utils import plot
from tqdm import tqdm

env_name = 'Pendulum-v0'
buffer_size = 10000
actor_learn_freq = 1
target_update_freq = 10
target_update_tau = 0.1
batch_size = 128
hidden_dim = 128
episodes = 1000
max_step = 300
lr = 3e-3

POLT_NAME = 'DDPG_' + env_name
SAVE_DIR = '/save/ddpg_' + env_name
# LOG_DIR = '/logs'

model_save_dir = abspath(dirname(__file__)) + SAVE_DIR
save_file = model_save_dir.split('/')[-1]
# writer_path = model_save_dir + LOG_DIR


env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2

class ActorModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        action = self.net(state)
        return action

    def action(self, state, noise_std=0, noise_clip=0.5):
        action = self.net(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(self.device)
            action += noise_norm.clamp(-noise_clip, noise_clip)
        action = action.clamp(-action_max, action_max)
        return action.detach().cpu().numpy()

class CriticModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # inpur_dim = state_dim + action_dim, 
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net(x)
        return q_value

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorModel(state_space, hidden_dim, action_space)
critic = CriticModel(state_space, hidden_dim, action_space)
model = model(actor, critic)
policy = DDPG(model, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq,
        target_update_freq=target_update_freq, target_update_tau=target_update_tau, 
        batch_size=batch_size, learning_rate=lr, num_episodes=episodes)
# writer = SummaryWriter(writer_path)

TRAIN = True
PLOT = True
# WRITER = False

def map_action(action):
    if isinstance(action, torch.Tensor):
        action = action.item()
    return action * action_scale + action_bias

def sample(env, policy, max_step, train=True, render=False):
    reward_avg = 0
    state = env.reset()
    for step in range(max_step):
        #==============choose_action==============
        action = policy.choose_action(state)
        next_state, reward, done, info = env.step(map_action(action))
        if train:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
        if render:
            env.render()
        reward_avg += reward
        if done:
            break
        state = next_state
    reward_avg /= (step + 1)
    return reward_avg

def eval():
    policy.load_model(model_save_dir, save_file, load_actor=True)
    for i_eps in range(100):
        reward_avg = sample(env, policy, max_step, train=False, render=True)
        print (f'EPS:{i_eps + 1}, reward:{round(reward_avg, 3)}')
    env.close()

def train():
    try:
        os.makedirs(model_save_dir)
    except FileExistsError:
        import shutil
        shutil.rmtree(model_save_dir)
        os.makedirs(model_save_dir)
    live_time = []

    while policy.warm_up(int(1e3)):
        sample(env, policy, max_step)
        print (f'Warm up for buffer {len(policy.buffer)}', end='\r')

    for i_eps in tqdm(range(episodes)):
        reward_avg = sample(env, policy, max_step, render=False)
        #==============learn==============
        pg_loss, v_loss = policy.learn()
        if PLOT:
            live_time.append(reward_avg)
            plot(live_time, 'DDPG_'+env_name, model_save_dir, 100)
        # if WRITER:
        #     writer.add_scalar('reward', reward_avg, global_step=i_eps)
        #     writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
        #     writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
        if i_eps % 50 == 0:
            # pass
            print (f'EPS:{i_eps}, reward_avg:{round(reward_avg, 3)}, pg_loss:{round(pg_loss, 3)}, v_loss:{round(v_loss, 3)}', end='\r')
        if i_eps % 200 == 0:
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    # writer.close()

if __name__ == '__main__':
    train()
    eval()
