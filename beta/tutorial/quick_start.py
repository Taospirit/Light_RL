import gym
import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import sys
sys.path.append('../..')
from drl.algorithm import A2C

# env
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped # 还原env的原始设置，env外包了一层防作弊层

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.action_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_linear(x)
        dist = F.softmax(action_score, dim=-1)
        return dist

    def action(self, state, test=False):        
        dist = self.forward(state)
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob

class CriticV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value_linear(x)
        return state_value

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorNet(state_space, 32, action_space)
critic = CriticV(state_space, 32, 1)
model = model(actor, critic)
policy = A2C(model, buffer_size=1000, learning_rate=1e-2)

def sample(env, policy, max_step):
    # reward_avg = 0
    rewards = 0
    state = env.reset()
    for _ in range(max_step):
        # step1: policy choose action base state
        action, log_prob = policy.choose_action(state)
        
        # feed action into env to step, get next env infomation
        next_state, reward, done, info = env.step(action)
        # show gym env when trained reward over threhold
        env.render()

        # step2: policy store transition infomation in Buffer
        mask = 0 if done else 1
        policy.process(s=state, r=reward, l=log_prob, m=mask)

        # record rewards
        rewards += reward
        if done:
            break
        state = next_state
    return rewards

def main():
    for i_eps in range(200):
        rewards = sample(env, policy, 1000)

        # step3: policy learn from stored transition in Buffer, to update network
        pg_loss, v_loss = policy.learn()
        reward, pg_loss, v_loss = round(rewards, 3), round(pg_loss, 3), round(v_loss, 3)
        print (f'EPS:{i_eps}, reward:{rewards}, pg_loss:{pg_loss}, v_loss:{v_loss}')    
    env.close()

if __name__ == '__main__':
    main()
