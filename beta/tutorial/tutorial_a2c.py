import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import sys
sys.path.append('../..')

from drl.algorithm import A2C
from drl.utils import plot
from tqdm import tqdm

# env
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), )

    def forward(self, x):
        action_sorce = self.net(x)
        dist = F.softmax(action_sorce, dim=-1)
        return dist

    def action(self, state, test=False):        
        dist = self.forward(state)
        m = Categorical(dist)
        act = m.sample()
        log_prob = m.log_prob(act)

        return act.item(), log_prob

class CriticV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), )

    def forward(self, x):
        return self.net(x)

episodes = 200
max_step = 300
hidden_dim = 32

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, 1)
model = model(actor, critic)
policy = A2C(model, buffer_size=max_step, learning_rate=1e-2, num_episodes=episodes)

model_save_dir = 'save/a2c'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

writer = SummaryWriter(os.path.dirname(model_save_dir)+'/logs/a2c_1')

TRAIN = True
PLOT = True
WRITER = False

def sample(env, policy, max_step):
    rewards = 0
    state = env.reset()
    for step in range(max_step):
        #==============choose_action==============
        action, log_prob = policy.choose_action(state)
        next_state, reward, done, info = env.step(action)
        if TRAIN:
            mask = 0 if done else 1
            #==============process==============
            policy.process(s=state, r=reward, l=log_prob, m=mask)
        else:
            env.render()
        rewards += reward
        if done:
            break
        state = next_state
    return rewards

def main():
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)
    live_time = []
    for i_eps in range(episodes):
        rewards = sample(env, policy, max_step)
        if not TRAIN:
            print (f'EPS:{i_eps + 1}, reward:{round(rewards, 3)}')
        else:
            #==============learn==============
            pg_loss, v_loss = policy.learn()
            if PLOT:
                live_time.append(rewards)
                plot(live_time, 'A2C_'+env_name, model_save_dir, 100)
            if WRITER:
                writer.add_scalar('reward', rewards, global_step=i_eps)
                writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
                writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
            if i_eps % 5 == 0:
                print (f'EPS:{i_eps}, reward:{round(rewards, 3)}, pg_loss:{round(pg_loss, 3)}, v_loss:{round(v_loss, 3)}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    writer.close()
    env.close()

if __name__ == '__main__':
    main()
