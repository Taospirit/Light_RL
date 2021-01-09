import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# np_path = '/home/lintao/drl_repo/Light_RL/beta/results/LAP_TD3_HalfCheetah-v3_0.npy'
dir_path = '/home/lintao/drl_repo/Light_RL/beta/results'
f1 = '/MSAC_HalfCheetah-v2_0_False_False_False.npy'
f2 = '/MSAC_HalfCheetah-v2_0_True_False_False.npy'
f3 = '/SAC_Humanoid-v2_0.npy'

data1 = np.load(dir_path + f3)
# data2 = np.load(dir_path + f2)[:300]

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

data = [data1]

linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['TD3', 'SAC', 'algo3', 'algo4']

fig = plt.figure()
linestyle = ['-', '--', ':', '-.', 'solid']
color = ['r', 'g', 'b', 'k', 'gray']
label = ['SAC', 'SAC_PER', 'SAC_M', 'SAC_PAL', 'SAC_MPAL']

sns.set(style="darkgrid", font_scale=1)

# data = smooth(data, sm=10)
time = range(data[0].shape[-1])
time = np.array(time)/200

sns.tsplot(time=time, data=data[0], color='r', linestyle='-', condition='SAC')
# sns.tsplot(time=time, data=data[1], color='b', linestyle='-', condition='SAC_PER')

plt.ylabel("Evaluation Reward")
plt.xlabel("Time steps(1e6)")
plt.title("HalfCheetah")
# plt.savefig('./test/'+label[i]+'.jpg')
plt.show()
