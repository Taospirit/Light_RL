import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set_theme()


def get_data():
    """获取数据"""
    basecond = np.array([
        [18, 20, 19, 18, 13, 4, 1],                
        [20, 17, 12, 9, 3, 0, 0],
        [20, 20, 20, 12, 5, 3, 0]
    ])

    cond1 = np.array([
        [18, 19, 18, 19, 20, 15, 14],             
        [19, 20, 18, 16, 20, 15, 9],
        [19, 20, 20, 20, 17, 10, 0],
        [20, 20, 20, 20, 7, 9, 1]
    ])

    cond2 = np.array([
        [20, 20, 20, 20, 19, 17, 4],            
        [20, 20, 20, 20, 20, 19, 7],            
        [19, 20, 20, 19, 19, 15, 2]
    ])

    cond3 = np.array([
        [20, 20, 20, 20, 19, 17, 12],           
        [18, 20, 19, 18, 13, 4, 1],            
        [20, 19, 18, 17, 13, 2, 0],          
        [19, 18, 20, 20, 15, 6, 0]
    ])

    return basecond, cond1, cond2, cond3


def main():
    # 获取数据
    data = get_data()
    label = ['algo1', 'algo2', 'algo3', 'algo4']
    
    # 数据处理
    df = []
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='loss'))
        df[i]['algo'] = label[i] 

    # 合并数据框
    df = pd.concat(df)
    
    # 绘图
    # plt.figure(figsize=(10, 6))
    sns.lineplot(x="episode", y="loss", hue="algo", style="algo", data=df)
    plt.title("Some Loss")
    
    # 保存图片
    plt.savefig('loss.png')
    plt.show()


if __name__ == "__main__":
    main()
