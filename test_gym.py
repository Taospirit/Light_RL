import gym
import time
env_list = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Hopper-v3', ]
mujuco_env = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
env_name = mujuco_env[3]
env = gym.make(env_name)
observation = env.reset()

begin = time.time()
max_step = 3000000
for _ in range(max_step):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    # print(f'step is {_}', end='\r')
    # print()
    if done:
        print(done)
        observation = env.reset()

env.close()