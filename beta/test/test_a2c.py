from test_tool import policy_test
import gym
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from drl.model import ActorNet, CriticV
from drl.algorithm import A2C
from drl.utils import ReplayBuffer as Buffer
from collections import namedtuple

model = namedtuple('model', ['policy_net', 'value_net'])
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

hidden_dim = 32
episodes = 300
max_step = 1000
actor_learn_freq = 1
target_update_freq = 0

model_save_dir = 'save/test_a2c_buffer'
model_save_dir = os.path.join(os.path.dirname(__file__), model_save_dir)
save_file = model_save_dir.split('/')[-1]
os.makedirs(model_save_dir, exist_ok=True)

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
    
    def action(self, state, test=False):
        if test:
            self.actor_eval.eval()
            return Categorical(self.actor_eval(state)).sample().item(), 0
        
        dist = self.forward(state)
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob


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

from collections import namedtuple
model = namedtuple('model', ['policy_net', 'value_net'])
actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticV(state_space, hidden_dim, 1)
model = model(actor, critic)
policy = A2C(model, buffer_size=max_step, actor_learn_freq=actor_learn_freq,
             target_update_freq=target_update_freq)

def sample(env, policy, max_step, test=False):
    assert env, 'You must set env for sample'
    rewards = 0
    state = env.reset()

    for step in range(max_step):
        action, log_prob = policy.choose_action(state, test)
        # print (action, type(action))
        next_state, reward, done, info = env.step(action)
        # env.render() 
        # process env callback
        if not test:
            # policy.process(s=state, a=action, s_=next_state, r=reward, d=done)
            mask = 0 if done else 1
            policy.process(s=state, r=reward, l=log_prob, m=mask)

        rewards += reward
        if done:
            break
        state = next_state

    if not test:
        policy.learn()
    return rewards, step


run_type = ['train', 'eval', 'retrain']
run = run_type[0]
plot_name = 'A2C_TwoNet_no_Double'


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

    writer = SummaryWriter('./logs/1/')
    live_time = []
    for i_eps in range(episodes):
        rewards, step = sample(env, policy, max_step, test=test)
        if run == 'eval':
            print(f'Eval eps:{i_eps+1}, Rewards:{rewards}, Steps:{step+1}')
            continue
        live_time.append(rewards)
        policy_test.plot(live_time, plot_name, model_save_dir)
        
        writer.add_scalar('rewards_per_epi', rewards, global_step=i_eps)
        
        if i_eps > 0 and i_eps % 100 == 0:
            print(f'i_eps is {i_eps}')
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
    env.close()
    writer.close()


if __name__ == '__main__':
    main()
