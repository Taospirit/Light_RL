import gym
import time
import mujoco
import gym_rocketlander


envids = [spec.id for spec in gym.envs.registry.all()]
for envid in sorted(envids):
    print(envid)
# import gym_rocketlander
# env_list = ['Pendulum-v1', 'MountainCarContinuous-v0', 'Hopper-v3', ]
# mujuco_env = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
env = gym.make("CartPole-v1")
# env = gym.make("CarRacing-v1")
# env = gym.make("rocketlander-v0")

# env = gym.make("rocketlander-v0")
observation = env.reset()

begin = time.time()
max_step = 3000000
tmp = 'obs:{}, res:{}, done:{}, info:{}'
for _ in range(max_step):
    action = env.action_space.sample() # your agent here (this takes random actions)
    print(action)
    observation, reward, done, info = env.step(action)
    env.render()
    # print(f'step is {_}', end='\r')
    # print()
    # print(f'======step {_}======')
    # print(f'shape: obs: {len(observation), len(observation[0]),  len(observation[0][0])}')
    # # print(tmp.format(observation, reward, done, info))

    # state_space = env.observation_space.shape[0]
    # action_space = env.action_space.shape[0]
    # action_max = env.action_space.high[0]

    # action_scale = (env.action_space.high - env.action_space.low) / 2
    # action_bias = (env.action_space.high + env.action_space.low) / 2
    # print(state_space, action_space, action_max, action_scale, action_bias)

    if done:
        observation = env.reset() 

env.close()
