import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def getdata():    
    basecond = [[18, 20, 19, 18, 13, 4, 1],                
                [20, 17, 12, 9, 3, 0, 0],               
                [20, 20, 20, 12, 5, 3, 0]]    
    
    cond1 = [[18, 19, 18, 19, 20, 15, 14],             
             [19, 20, 18, 16, 20, 15, 9],             
             [19, 20, 20, 20, 17, 10, 0],             
             [20, 20, 20, 20, 7, 9, 1]]   
    
    cond2 = [[20, 20, 20, 20, 19, 17, 4],            
             [20, 20, 20, 20, 20, 19, 7],            
             [19, 20, 20, 19, 19, 15, 2]]   
    
    cond3 = [[20, 20, 20, 20, 19, 17, 12],           
             [18, 20, 19, 18, 13, 4, 1],            
             [20, 19, 18, 17, 13, 2, 0],            
             [19, 18, 20, 20, 15, 6, 0]]    
    
    return basecond, cond1, cond2, cond3


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

data = getdata()
fig = plt.figure()
xdata = np.array([0, 1, 2, 3, 4, 5, 6])/5
linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['algo1', 'algo2', 'algo3', 'algo4']

sns.set(style="darkgrid", font_scale=1)
# for i in range(4):    
#     sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])
data1 = [[18, 20, 19, 18, 13, 4, 1],                
        [20, 17, 12, 9, 3, 0, 0],               
        [20, 20, 20, 12, 5, 3, 0]]

data2 = [[17, 20, 19, 18, 13, 4, 1],                
        [17, 17, 12, 9, 3, 0, 0],               
        [17, 20, 20, 12, 5, 3, 0]]


base = np.array(data2[0])
def ran():
    return np.random.uniform(-1, 1, size=(7,))
print(ran())
data3= [base,
        base + ran(), 
        base + ran()]

sns.tsplot(time=xdata, data=data1, color=color[0], linestyle=linestyle[0], condition=label[0])
data2 = smooth(data1, sm=5)
sns.tsplot(time=xdata, data=data2, color=color[1], linestyle=linestyle[0], condition=label[1])
# sns.tsplot(time=xdata, data=data3, color=color[2], linestyle=linestyle[0], condition=label[2])

# plt.ylabel("Success Rate", fontsize=25)
# plt.xlabel("Iteration Number", fontsize=25)
# plt.title("Awesome Robot Performance", fontsize=30)

plt.ylabel("Success Rate")
plt.xlabel("Iteration Number")
plt.title("Awesome Robot Performance")
plt.show()