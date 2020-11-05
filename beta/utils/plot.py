import os
import matplotlib.pyplot as plt

# class plot():
#     def __init__(self, nums):
#        pass

#     def plot(self, steps, y_label, model_save_dir):
#             ax = plt.subplot(111)
#             ax.cla()
#             ax.grid()
#             ax.set_title(y_label)
#             ax.set_xlabel('Episode')
#             ax.set_ylabel('Run Reward')
#             ax.plot(steps)
#             RunTime = len(steps)

#             path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
#             if len(steps) % 20 == 0:
#                 plt.savefig(path)
#             plt.pause(0.0000001)

def plot(steps, y_label, model_save_dir, step_interval):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title(y_label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Reward')
    ax.plot(steps)
    RunTime = len(steps)

    path = model_save_dir + '/RunTime' + str(RunTime) + '.jpg'
    if len(steps) % step_interval == 0:
        plt.savefig(path)
    plt.pause(0.0000001)