import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dir_path = '/home/lintao/drl_repo/Light_RL/beta/results'
path1 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_False_False_False.npy') # SAC
path2 = np.load(dir_path+'/MSAC_HalfCheetah-v2_10_False_False_False.npy') # SAC
path3 = np.load(dir_path+'/MSAC_HalfCheetah-v2_20_False_False_False.npy') # SAC

data_list = [path1, path2, path3]

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    else: 
        smooth_data = data
    return smooth_data

# linestyle = ['-', '--', ':', '-.']
# color = ['r', 'g', 'b', 'k']
# label = ['TD3', 'SAC', 'algo3', 'algo4']

linestyle = ['-', '--', ':', '-.', 'solid']
color = ['r', 'g', 'b', 'k', 'gray']
label = ['SAC', 'SAC_PER', 'SAC_M', 'SAC_PAL', 'SAC_MPAL']

data_list = smooth(data_list, sm=10)
time = range(data_list[0].shape[-1])
time = np.array(time)/200

fig = plt.figure()
sns.set(style="darkgrid", font_scale=1)
sns.tsplot(time=time, data=data_list, color=color[0], linestyle=linestyle[0], condition=label[0])

plt.ylabel("Evaluation Reward")
plt.xlabel("Time steps(1e6)")
plt.title("HalfCheetah")
# plt.savefig('./test/'+label[i]+'.jpg')
plt.show()
