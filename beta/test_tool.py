import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class policy_test():
    def __init__(self):
        pass
        
    @abstractmethod
    def plot(steps, y_label, model_save_dir):
        ax = plt.subplot(111)
        ax.cla()
        ax.grid()
        ax.set_title(y_label)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Run Reward')
        ax.plot(steps)
        RunTime = len(steps)

        path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
        if len(steps) % 20 == 0:
            plt.savefig(path)
        plt.pause(0.0000001)
    
    @abstractmethod
    def save_setting(env_name, state_space, action_space, episodes, max_step, policy, model_save_dir, save_file):
        line = '===============================\n'
        env_info = f'env: {env_name} \nstate_space: {state_space}, action_space: {action_space}\n' 
        env_info += f'episodes: {episodes}, max_step: {max_step}\n'
        policy_dict = vars(policy)
        policy_info = ''
        for item in policy_dict.keys():
            policy_info += f'{item}: {policy_dict[item]} \n'

        data = line.join([env_info, policy_info])
        path = model_save_dir + '/' + save_file + '.txt'

        with open(path, 'w+') as f:
            f.write(data)
        print (f'Save train setting in {path}!')