import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# np_path = '/home/lintao/drl_repo/Light_RL/beta/results/LAP_TD3_HalfCheetah-v3_0.npy'
dir_path = '/home/lintao/drl_repo/Light_RL/beta/results/1'
data_list = []
files = os.listdir(dir_path)

path1 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_False_False_False.npy') # SAC
path2 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_True_False_False.npy') # SAC_PER
path3 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_False_True_False.npy') # SAC_M
path4 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_False_False_True.npy') # SAC_PAL
path5 = np.load(dir_path+'/MSAC_HalfCheetah-v2_0_False_True_True.npy') # SAC_MPAL


# m = min([len(item) for item in data_list])
# for f in files:
#     if 'MSAC' in f:
#         data = np.load(dir_path+'/'+f)
        # data_list.append(data[:25])

# len(path2)
# path2[:m]
# m = min(len(path1), len(path5))
# data_list = [path1[:m], path5[:m]]

 = min(len(path1), len(path2), len(path3), len(path4), len(path5))
m = 45
data_list = [path1[:m], path2[:m], path3[:m], path4[:m], path5[:m]]
# m = min(len(path2), len(path4))
# m = min(len(path4), 1000000)
# data_list = [path4[:m]]
# data_list = [path1]


# data_list = [path1, path2, path3, path4, path5]
# print(type(data), data.shape)
# print(data[-1])

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
linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['TD3', 'SAC', 'algo3', 'algo4']

fig = plt.figure()
# xdata = np.array([0, 1, 2, 3, 4, 5, 6])/5
linestyle = ['-', '--', ':', '-.', 'solid']
color = ['r', 'g', 'b', 'k', 'gray']
label = ['SAC', 'SAC_PER', 'SAC_M', 'SAC_PAL', 'SAC_MPAL']
# label = ['SAC_PER', 'SAC_PAL']
# 'SAC_PER',
sns.set(style="darkgrid", font_scale=1)
# # for i in range(4):    
# #     sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])


# base = np.array(data2[0])
# def ran():
#     return np.random.uniform(-1, 1, size=(7,))
# print(ran())
# data3= [base,
#         base + ran(), 
#         base + ran()]
i = 1
time = range(data_list[0].shape[-1])
time = np.array(time)/200
for i in range(len(data_list)):
    sns.tsplot(time=time, data=data_list[i], color=color[i], linestyle=linestyle[i], condition=label[i])

plt.ylabel("Evaluation Reward")
plt.xlabel("Time steps(1e6)")
plt.title("HalfCheetah")
# plt.savefig('./test/'+label[i]+'.jpg')
plt.show()
