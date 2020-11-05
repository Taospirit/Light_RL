from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple

# from drl.model import ActorDPG, CriticQ
from drl.algorithm import DDPG

env_name = 'Pendulum-v0'
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

episodes = 5000
buffer_size = 50000
hidden_dim = 32
# max_step = 200
actor_learn_freq = 1
target_update_freq = 10
# batch_size = 1000

# hidden_dim = 256 # mush slow, even cannt shoulian
max_step = 300
# actor_learn_freq = 5 # 非常震荡、不稳定
# target_update_freq = 20 # 收敛非常慢
batch_size = 128

model_save_dir = 'save/test_ddpg_6'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

action_scale = (env.action_space.high - env.action_space.low) / 2
action_bias = (env.action_space.high + env.action_space.low) / 2

class ActorDPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh(), )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        action = self.net(state)
        return action

    def action(self, state, noise_std=0.2, noise_clip=0.5):
        action = self.net(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(self.device)
            action += noise_norm.clamp(-noise_clip, noise_clip)
        action = action.clamp(-action_max, action_max)
        return action.item()


class CriticQ(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # inpur_dim = state_dim + action_dim, 
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim , hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), )
        # self.net = build_critic_network(state_dim, hidden_dim, action_dim)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net(x)
        return q_value

model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorDPG(state_space, hidden_dim, action_space)
critic = CriticQ(state_space, hidden_dim, action_space)
model = model(actor, critic)
policy = DDPG(model, buffer_size=buffer_size, actor_learn_freq=actor_learn_freq, target_update_freq=target_update_freq, batch_size=batch_size)

def map_action(action):
    return action * action_scale + action_bias

def sample(env, policy, max_step, test=False):
    assert env, 'You must set env for sample'
    reward_avg = 0
    state = env.reset()

    for step in range(max_step):
        action = policy.choose_action(state, test)
        # action *= env.action_space.high[0]
        next_state, reward, done, info = env.step(map_action(action))
        if test:
            env.render()
        # process env callback
        if not test:
            mask = 0 if done else 1
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)

        reward_avg += reward
        if done:
            break
        state = next_state

    if not test:
        pg_loss, v_loss = policy.learn()
        return reward_avg/(step+1), step, pg_loss, v_loss
    return reward_avg/(step+1), step, 0, 0


run_type = ['train', 'eval']
run = run_type[0]
plot_name = 'DDPG_TwoNet_Double'

writer = SummaryWriter('./logs/ddpg')
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
        # woc, 这句写的太丑了
        policy_test.save_setting(env_name, state_space, action_space, episodes,
                                 max_step, policy, model_save_dir, save_file)
    else:
        print('Setting your run type!')
        return 0

    # while policy.warm_up():
    #     sample(env, policy, max_step)
    #     print (f'Warm up for buffer {policy.buffer.size()}', end='\r')

    live_time = []
    for i_eps in range(episodes):
        rewards, step, pg_loss, v_loss = sample(env, policy, max_step)
        if run == 'eval':
            print(f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)

        writer.add_scalar('reward', rewards, global_step=i_eps)
        writer.add_scalar('loss/pg', pg_loss, global_step=i_eps)
        writer.add_scalar('loss/v', v_loss, global_step=i_eps)

        if i_eps > 0 and i_eps % 100 == 0:
            print(f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    env.close()


if __name__ == '__main__':
    main()