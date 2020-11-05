from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from drl.model import ActorPPO, CriticV
from drl.algorithm import PPO
from drl.utils import ZFilter

env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_max = env.action_space.high[0]

hidden_dim = 100
episodes = 2000
max_step = 200
buffer_size = 32
actor_learn_freq = 1
target_update_freq = 0
batch_size = 100

model_save_dir = 'save/test_ppo_adv_reward_tanh'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

class ActorPPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, layer_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        # if layer_norm:
        #     layer_norm(self.fc1, std=1.0)
        #     layer_norm(self.mu_head, std=1.0)
        #     layer_norm(self.sigma_head, std=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        # x = F.relu(self.fc1(state))
        mu = 2.0 * torch.tanh(self.mu_head(x)) # test for gym_env: 'Pendulum-v0'
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

    def action(self, state, test=False):
        with torch.no_grad():
            mu, sigma = self.actor_eval(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        # print (f'mu:{mu}, sigma:{sigma}, dist: {dist}, action sample before clamp: {action}')
        action = action.clamp(-action_max, action_max)
        # print (f'action after clamp {action}')
        log_prob = dist.log_prob(action)
        assert abs(action.item()) <= action_max, f'ERROR: action out of {action}'

        return action.item(), log_prob.item()

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

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorPPO(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, action_space)
model = model(actor, critic)
policy = PPO(model, buffer_size=32, actor_learn_freq=1, target_update_freq=0)


def sample(env, policy, max_step, i_episode=0, num_episode=100, test=False):
    assert env, 'You must set env for sample'
    reward_avg = 0
    running_state = ZFilter((state_space,), clip=10.0)

    state = env.reset()
    # test state norm
    # state = running_state(state)

    for step in range(max_step):
        action, log_prob = policy.choose_action(state, test)
        # action = action.clip(-1, 1) * env.action_space.high[0]
        next_state, reward, done, info = env.step([action])
        env.render()

        # test state norm
        # next_state = running_state(next_state)

        # process env callback
        if not test:
            policy.process(s=state, a=action, s_=next_state, r=(reward + 8) / 8, l=log_prob)
            policy.learn(i_episode, num_episode)

        reward_avg += reward
        if done:
            break
        state = next_state

    return reward_avg/(step+1), step


run_type = ['train', 'eval', 'retrain']
run = run_type[1]
plot_name = 'Training_PPO_TwoNet_'+model_save_dir.split('/')[-1]


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
        rewards, step = sample(env, policy, max_step, i_episode=i_eps, num_episode=episodes, test=test)
        if run == 'eval':
            print(f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)

        if i_eps > 0 and i_eps % 100 == 0:
            print(f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    env.close()


if __name__ == '__main__':
    main()
