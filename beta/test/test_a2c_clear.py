# from test_tool import policy_test
# import sys
# sys.path.append('/Users/liulintao/drl_repo/DRL_beta/beta')
# sys.path.append('..')
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

# from drl.model import ActorNet, CriticV
from drl.algorithm import A2C
from drl.utils import ReplayBuffer as Buffer

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

# env
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
episodes = 2000
max_step = 300
hidden_dim = 32

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, 1)
model = model(actor, critic)
policy = A2C(model, buffer_size=max_step, actor_learn_freq=1, target_update_freq=0)

model_save_dir = 'save/a2c'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

writer = SummaryWriter(os.path.dirname(model_save_dir)+'/logs/a2c_1')
TRAIN = True
WRITER = False

def main():
    if not TRAIN:
        policy.load_model(model_save_dir, save_file, load_actor=True)

    for i_eps in range(episodes):
        state = env.reset()
        rewards = 0

        for step in range(max_step):
            action, log_prob = policy.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if TRAIN:
                mask = 0 if done else 1
                policy.process(s=state, r=reward, l=log_prob, m=mask)
            else:
                env.render()
            rewards += reward
            if done:
                break
            state = next_state

        if TRAIN:
            if i_eps % 5 == 0:
                print (f'EPS {i_eps}, reward {rewards}')
            if i_eps % 200 == 0:
                policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
            pg_loss, v_loss = policy.learn()

        if WRITER:
            writer.add_scalar('reward', rewards, global_step=i_eps)
            writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
            writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)

    writer.close()
    env.close()

if __name__ == '__main__':
    main()
