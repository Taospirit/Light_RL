from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# from drl.model import ActorNet, CriticDQN
from drl.algorithm import DQN
# from drl.algorithm import DoubleDQN

env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
action_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

hidden_dim = 32
episodes = 1000
buffer_size = 5000
batch_size = 100
target_update_freq = 100
max_step = 1000

model_save_dir = 'save/test_dqn_double_'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

class CriticDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, atoms=51, layer_norm=False):
        super().__init__()
        self.epsilon = 0.5
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

    def action(self, state, test=False):
        q_values = self.forward(state)
        action = q_values.argmax(dim=1).cpu().data.numpy()
        action = action[0] if action_shape == 0 else action.reshape(action_shape)  # return the argmax index

        if test:
            self.epsilon = 1.0
        if np.random.randn() >= self.epsilon:  # epsilon-greedy
            action = np.random.randint(0, q_values.size()[-1])
            action = action if action_shape == 0 else action.reshape(action_shape)

        return action

model = namedtuple('model', ['value_net'])
critic = CriticDQN(state_space, hidden_dim, action_space)
model = model(critic)
policy = DQN(model, buffer_size=buffer_size, batch_size=batch_size, target_update_freq=100)

def sample(env, policy, max_step, test=False):
    assert env, 'You must set env for sample'
    rewards = 0
    state = env.reset()

    for step in range(max_step):
        action = policy.choose_action(state, test)
        next_state, reward, done, info = env.step(action)
        # env.render()
        # process env callback
        if not test:
            mask = 0 if done else 1
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)

        rewards += reward
        if done:
            break
        state = next_state

    if not test:
        policy.learn()
    return rewards, step


run_type = ['train', 'eval', 'retrain']
run = run_type[0]
plot_name = 'DQN_Double'


def main():
    test = False
    if run == 'eval':
        global episodes
        episodes = 100
        test = True
        print('Loading model...')
        policy.load_model(model_save_dir, save_file, load_actor=True)

    elif run == 'train':
        print('Saving model setting...')
        # save_setting()
        policy_test.save_setting(env_name, state_space, action_space, episodes,
                                 max_step, policy, model_save_dir, save_file)
    elif run == 'retrain':
        print('Loading model...')
        policy.load_model(model_save_dir, save_file, load_actor=True, load_critic=True)
    else:
        print('Setting your run type!')
        return 0

    live_time = []
    for i_eps in range(episodes):
        rewards, step = sample(env, policy, max_step, test=test)
        if run == 'eval':
            print(f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)

        if i_eps > 0 and i_eps % 100 == 0:
            print(f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_critic=True)
    env.close()


if __name__ == '__main__':
    main()
