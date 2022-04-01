import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import drl.utils as Tools
from drl.algorithm import A2C

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                # nn.Dropout(p=0.5), 
                                nn.Linear(hidden_dim, output_dim), )

    def forward(self, s):
        return self.net(s)

class CriticNet(nn.Module): # V(s)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), )

    def forward(self, s):
        return self.net(s)

# env
env_name = 'CartPole-v1'
'''
'CartPole-v1'
'buffer_size': 1000,
'learning_rate': 1e-2,
'num_episodes': 300

'Acrobot-v1'
'buffer_size': 3000,
'learning_rate': 1e-3,
'num_episodes': 1000
'''
# env_name = 'Acrobot-v1'
env = gym.make(env_name)
env = env.unwrapped
env.reset(seed=1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 32

kwargs = {
    'buffer_size': 1000,
    'learning_rate': 1e-2,
    'num_episodes': 300
    }

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticNet(state_space, hidden_dim, 1)
policy = A2C(actor, critic, **kwargs)

# model save setting
save_dir = 'save/a2c_' + env_name
save_dir = os.path.join(os.path.dirname(__file__), save_dir)
save_file = save_dir.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(os.path.dirname(save_dir)+'/logs/a2c_1')

PLOT = 1
# WRITER = 0

def eval():
    policy.load_model(save_dir, load_actor=1)
    for i_eps in range(1000):
        rewards = policy.sample(env, train=0, render=1)
        print (f'EPS:{i_eps + 1}, reward:{round(rewards, 3)}')
    env.close()

def train():
    # rm dir exist
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        import shutil
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    live_time = []
    from tqdm import tqdm
    for i_eps in tqdm(range(policy.num_episodes)):
        rewards = policy.sample(env)
        #==============learn==============
        pg_loss, v_loss = policy.learn()
        if PLOT:
            live_time.append(rewards)
            Tools.plot(live_time, 'A2C_'+env_name, save_dir, 100)
        # if WRITER:
        #     writer.add_scalar('reward', reward_avg, global_step=i_eps)
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