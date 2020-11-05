import gym, os, time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        dist = F.softmax(action_score, dim=-1)
        return dist, state_value

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        dist = F.softmax(action_score, dim=-1)
        return dist

class ActorGaussian(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

# DDPG & TD3
class ActorDPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.net(state)
        return action

    def predict(self, state, action_max, noise_std=0, noise_clip=0.5):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.net(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(self.device)
            action += noise_norm.clamp(-noise_clip, noise_clip)

        action = action.clamp(-action_max, action_max)
        return action

# PPO
class ActorPPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        if layer_norm:
            layer_norm(self.fc1, std=1.0)
            layer_norm(self.mu_head, std=1.0)
            layer_norm(self.sigma_head, std=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        # x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.mu_head(x)) # test for gym_env: 'Pendulum-v0'
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma
    
    # @staticmethod
    # def layer_norm(layer, std=1.0, bias_const=0.0):
    #     torch.nn.init.orthogonal_(layer.weight, std)
    #     torch.nn.init.constant_(layer.bias, bias_const)

class CriticV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, output_dim)
        if layer_norm:
            layer_norm(self.fc1, std=1.0)
            layer_norm(self.value_head, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value_head(x)
        return state_value

    # @staticmethod
    # def layer_norm(layer, std=1.0, bias_const=0.0):
    #     torch.nn.init.orthogonal_(layer.weight, std)
    #     torch.nn.init.constant_(layer.bias, bias_const)

class CriticDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, atoms=51, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, action_dim)
        self.v_value = nn.Linear(hidden_dim, 1)
        self.use_dueling = False
        self.use_distributional = False

        if layer_norm:
            layer_norm(self.fc1, std=1.0)
            layer_norm(self.q_value, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.use_dueling:
            v_value = self.v_value(x)
            adv = self.q_value(x)
            return v_value + adv - adv.mean()
        
        if self.use_distributional:
            self.q_value = nn.Linear(hidden_dim, action_dim*atoms)
            x = self.q_value(x)
            return F.softmax(x.view(-1, action_dim, atoms), dim=2)

        q_value = self.q_value(x)
        return q_value

    # @staticmethod
    # def layer_norm(layer, std=1.0, bias_const=0.0):
    #     torch.nn.init.orthogonal_(layer.weight, std)
    #     torch.nn.init.constant_(layer.bias, bias_const)

class CriticQ(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # inpur_dim = state_dim + action_dim, 
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        # self.net = build_critic_network(state_dim, hidden_dim, action_dim)

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        q_value = self.net(input)
        return q_value

class CriticQTwin(CriticQ):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__(state_dim, hidden_dim, action_dim)
        self.net_copy = deepcopy(self.net)
        # self.net_copy = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
        #                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #                          nn.Linear(hidden_dim, 1), )
        # self.net1 = build_critic_network(state_dim, hidden_dim, action_dim)
        # self.net2 = build_critic_network(state_dim, hidden_dim, action_dim)

    # def forward(self, state, action):
    #     x = torch.cat((state, action), dim=1)
    #     q_value = self.net(x)
    #     return q_value

    def twinQ(self, state, action):
        x = torch.cat((state, action), dim=1)
        q1_value = self.net(x)
        q2_value = self.net_copy(x)
        return q1_value, q2_value

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

def build_critic_network(state_dim, hidden_dim, action_dim, norm=False):
    nn_list = []
    nn_list.extend([nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),])
    nn_list.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 1), ])
    net = nn.Sequential(*nn_list)
    return net


class DenseNet(nn.Module):
    def __init__(self, mid_dim):
        super(DenseNet, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1),
            HardSwish(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(mid_dim * 2, mid_dim * 2),
            HardSwish(),
        )

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.cat((x, self.dense1(x)), dim=1)
        x = torch.cat((x, self.dense2(x)), dim=1)
        # self.dropout.p = rd.uniform(0.0, 0.1)
        # return self.dropout(x)
        return x

class HardSwish(nn.Module): # 一种激活函数
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x

