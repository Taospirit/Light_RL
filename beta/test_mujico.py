# from test_tool import policy_test
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
import sys
sys.path.append('..')

from drl.algorithm import SACV as SAC
from utils.plot import plot
from utils.config import config

mujuco_env = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2']

#config
config = config['sacv']

# env_name = config['env_name']
env_name = mujuco_env[3]
buffer_size = int(config['buffer_size'])
actor_learn_freq = config['actor_learn_freq']
update_iteration = config['update_iteration']
target_update_freq = config['target_update_freq']
batch_size = int(config['batch_size'])
hidden_dim = config['hidden_dim']
episodes = config['episodes'] + 10
# max_step = config['max_step']
max_step = 1000
lr = config['lr']        
tau = config['tau']


LOG_DIR = config['LOG_DIR']
SAVE_DIR = config['SAVE_DIR']
POLT_NAME = config['POLT_NAME']
PKL_DIR = config['PKL_DIR']

use_priority = False
if use_priority:
    p = 'per_'
    SAVE_DIR += p
    POLT_NAME += p
    PKL_DIR += p

POLT_NAME += env_name
SAVE_DIR += env_name
PKL_DIR += env_name

file_path = abspath(dirname(__file__))
pkl_dir = file_path + PKL_DIR
model_save_dir = file_path + SAVE_DIR
save_file = model_save_dir.split('/')[-1]
writer_path = model_save_dir + LOG_DIR

# try:
#     os.makedirs(model_save_dir)
# except FileExistsError:
#     import shutil
#     shutil.rmtree(model_save_dir)
#     os.makedirs(model_save_dir)

env = gym.make(env_name)
env = env.unwrapped

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2
#endregion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#region Network
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
        noise = Normal(0,1)
        
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
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim , mid_dim), nn.ReLU(),
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
#endregion

PLOT = True
WRITER = False

def map_action(action):
    if isinstance(action, torch.Tensor):
        action = action.item()
    return action * action_scale + action_bias

def sample(env, policy, max_step, warm_up=False):
    rewards = []
    state = env.reset()
    cnt = 0
    pg_loss, q_loss, a_loss = 0, 0, 0
    max_step = 10000 if not TRAIN else max_step
    for step in range(max_step):
    # while True:
        cnt += 1
        #==============choose_action==============
        if TRAIN:
            action = policy.choose_action(state) if not warm_up else env.action_space.sample()
        # print ('get action')
        else:
            action = policy.choose_action(state, test=True)
        next_state, reward, done, info = env.step(map_action(action))
        if TRAIN:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
            if len(policy.buffer) > int(1e4):
                pg_loss, q_loss, a_loss = policy.learn()
        else:
            env.render()
            pass
    
        rewards.append(reward)
        if done:
            break
        # if warm_up and len(policy.buffer) > int(1e4):
        #     break
        state = next_state
    return rewards, pg_loss, q_loss, a_loss

def train():
    model = namedtuple('model', ['policy_net', 'value_net'])
    actor = ActorModel(state_space, hidden_dim, action_space)
    # critic = CriticModel(state_space, hidden_dim, action_space)
    critic = CriticModelDist(state_space, hidden_dim, action_space, use_dist=False)
    rl_agent = model(actor, critic)
    policy = SAC(rl_agent, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq, act_dim=action_space,
            update_iteration=update_iteration, target_update_freq=target_update_freq, 
            target_update_tau=tau, batch_size=batch_size, learning_rate=lr, use_priority=use_priority)
    writer = SummaryWriter(writer_path)

    live_time = []
    mean, std = [], []
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    # else:
    while policy.warm_up(int(1e4)):
        sample(env, policy, max_step, warm_up=True)
        print (f'Warm up for buffer {policy.buffer.size()}', end='\r')

    for i_eps in range(episodes):
        rewards, pg_loss, q_loss, a_loss = sample(env, policy, max_step)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        reward_sum = np.sum(rewards)

        mean.append(reward_mean)
        std.append(reward_std)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{reward_mean:.3f}')
        else:
            #==============learn==============
            # pg_loss, q_loss, a_loss = policy.learn()
            if PLOT:
                live_time.append(reward_sum)
                plot(live_time, POLT_NAME, model_save_dir, 100)
            if WRITER:
                writer.add_scalar('reward', reward_mean, global_step=i_eps)
                writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
                writer.add_scalar('loss/q_loss', q_loss, global_step=i_eps)
                writer.add_scalar('loss/alpha_loss', a_loss, global_step=i_eps)

            if i_eps % 5 == 0:
                print (f'EPS:{i_eps}, reward_mean:{reward_mean:.3f}, pg_loss:{pg_loss:.3f}, q_loss:{q_loss:.3f}, alpha_loss:{a_loss:.3f}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    writer.close()
    env.close()
    return mean, std

if __name__ == '__main__':
    TRAIN = False
    env.seed(1)
    torch.manual_seed(1)
    #only for logistic test 
    train()
    #for pkl 
    # size_list = [128, 256, 512, 1024, 2048]
    # for size in size_list:
    #     means, stds = [], []
    #     for seed in range(5):
    #         env.seed(seed  * 10)
    #         torch.manual_seed(seed * 10)
    #         mean, std = train()
    #         means.append(mean)
    #         stds.append(std)
    #     d = {'mean': means, 'std': stds}
    #     import pickle
    #     with open(pkl_dir + f'_batch_size_{size}.pkl', 'wb') as f:
    #         pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    #     # print (f'finish learning at actor_learn_freq:{learn_freq}')
    # print ('finish all learning!')