import gym
import os
from os.path import abspath, dirname
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import drl.utils as Tools
from drl.algorithm import DDPG

class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        # self.net1 = nn.Sequential(nn.Conv2d(3, ), nn.ReLU(),
        #                         )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, s):
        return self.net(s)

class CriticNet(nn.Module): # Q(s,a)
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net(x)
        return q_value

# env
env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = env.unwrapped
env.reset(seed=1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]
action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2
hidden_dim = 128

kwargs = {
        'buffer_size': 10000,
        'actor_learn_freq': 1,
        'target_update_freq': 10,
        'target_update_tau': 0.1,
        'batch_size': 128,
        'learning_rate': 3e-3,
        'num_episodes': 1000,
        'act_max': action_max,
        'act_scale': action_scale,
        'act_bias': action_bias,
    }

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticNet(state_space, hidden_dim, action_space)
policy = DDPG(actor, critic, **kwargs)

# model save setting
save_dir = 'save/ddpg_' + env_name
save_dir = os.path.join(os.path.dirname(__file__), save_dir)
save_file = save_dir.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(os.path.dirname(save_dir)+'/logs/ddpg_')

PLOT = 1
# WRITER = False

def eval():
    policy.load_model(save_dir, save_file, load_actor=1)
    for i_eps in range(1000):
        rewards = policy.sample(env, train=0, render=1)
        print (f'EPS:{i_eps + 1}, reward:{round(rewards, 3)}')
    env.close()

def train():
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        import shutil
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    #======warm up========
    warm_up_size = int(1e4)
    while policy.warm_up(warm_up_size):
        policy.sample(env, max_len=1000)
        print (f'Warm up for buffer {len(policy.buffer)}/{warm_up_size}')
    else:
        print (f'Warm up over! buffer {len(policy.buffer)}')
    
    live_time = []
    from tqdm import tqdm
    for i_eps in tqdm(range(policy.num_episodes)):
        rewards = policy.sample(env, max_len=300)
        #==============learn==============
        pg_loss, v_loss = policy.learn()
        if PLOT:
            live_time.append(rewards)
            Tools.plot(live_time, 'DDPG_'+env_name, save_dir, 100)
        # if WRITER:
        #     writer.add_scalar('reward', rewards, global_step=i_eps)
        #     writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
        #     writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
        if (i_eps + 1) % 50 == 0:
            print (f'EPS:{i_eps}, rewards:{round(rewards, 3)}, pg_loss:{round(pg_loss, 3)}, v_loss:{round(v_loss, 3)}', end='\r')
        if (i_eps + 1) % 200 == 0 or i_eps == (policy.num_episodes - 1):
            policy.save_model(save_dir, save_file, str(i_eps + 1), save_actor=1, save_critic=1)
    # writer.close()

if __name__ == '__main__':
    train()
    eval()
