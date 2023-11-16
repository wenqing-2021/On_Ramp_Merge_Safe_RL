'''
Author: wenqing-hnu 1140349586@qq.com
Date: 2022-10-19 10:01:33
LastEditors: wenqing-hnu
LastEditTime: 2022-11-20
FilePath: /code/post_data4.py
Description: decision making module for 51simone competition

Copyright (c) 2022 by wenqing-hnu 1140349586@qq.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import os
import argparse
import math

from wandb import agent


def plot_data(data_set, y_value=None, sub_axis=False, save_name=None):
    all_data = pd.concat(data_set, axis=0, join='inner', ignore_index=True)
    all_data['Crashed'] = all_data['Crashed'].astype(float)
    all_data['Costs'] = all_data['Costs'].astype(float)
    all_data['Success'] = all_data['Success'].astype(float)
    all_data['Length'] = all_data['Length'].astype(float)

    max_cost = all_data['Costs'].max()
    max_cost = math.ceil(max_cost)
    
    # agent_name = all_data['Agent'].unique()
    # all_data = all_data.loc[(all_data['Costs'] < 1 )]
    
    show_label = False
    labels = None
    # labels = ['SACD-$\lambda$', 'SACD-$\lambda$-M', 'SACD-$\lambda$-TM']
    # labels = ['Dueling DQN','SACD','SACD-$\lambda$']
    # labels = ['$\sigma=1$', '$\sigma=5$', '$\sigma=10$']
    # labels = ['Dueling DQN', 'SACD', 'SACD-$\lambda$', 'SACD-$\lambda$-M','SACD-$\lambda$-TM']
    labels = ['SACD-$\lambda$-TM', 'SACD-$\lambda$-M', 'SACD-$\lambda$','PPO','SACD','Dueling DQN']
    label_size = 50
    ticks_size = 48
    legend_size = 30
    line_size = 4
    titlesize = 55
    # title_order = ['(a) The Cost of SACD-$\lambda$-TM','(b) The Reward of SACD-$\lambda$-TM','(c)']
    # for crash ratio
    # ylimit = (-0.005, 0.5)
    ylabel = 'Percentage (%)'
    fig = plt.figure()
    fig.set_size_inches(16, 12)

    sns.set(style="white", font='Times New Roman')
    sns.set_context(rc={"lines.linewidth": line_size})

    hue_order = ['Dueling_DQN','SACD','PPO','SACD_lambda','SACD_lambda_M','SACD_lambda_TM']

    # plot_data = all_data[all_data['Agent'] == 'Original_DuelingDQN_1']
    # plot_data2 = all_data[all_data['Agent'] == 'Original_SACD_original_1']

    # ax = sns.histplot(data=plot_data, x='Costs', multiple='dodge',stat='percent', legend=True)
    
    ax = sns.histplot(data=all_data, x='Costs', hue='Agent',multiple='dodge',hue_order=hue_order,legend=True,bins=3, shrink=0.8, binrange=(0,1.5))
    ax.grid(True)
    # plt.title(label=title_order[k-1], fontdict={'fontsize': title_size}, loc='left')
    # plt.xlabel('TotalEnvInteracts(Million)', fontsize=label_size, labelpad=20)
    # plt.title(title_order[index], fontsize = titlesize, y=-0.3)
    plt.xlabel('Cost', fontsize=label_size,labelpad=20)
    plt.ylabel(ylabel, fontsize=label_size, labelpad=20)
    plt.xticks(np.linspace(0.2, 1.3, 3), labels=['[0,0.5)','[0.5,1.0)','[1.0,1.5]'] ,fontsize=ticks_size)
    plt.yticks(np.linspace(0, 1200, 6), labels=['0','20','40','60','80','100'] ,fontsize=ticks_size)
    # plt.ylim(ylimit)
    if not show_label:
        plt.legend(labels=labels, fontsize=legend_size)
        show_label = True

def get_data(args):
    all_data = []
    for i in args.algo:
        path = args.file_path + i
        for root, dirs, files in os.walk(path):
            if 'process.txt' in files:
                data_path = os.path.join(root, 'process.txt')
                exp_data = pd.read_table(data_path)[3:]
                # corect agent name
                agent_name = exp_data['Agent'][3]
                if 'Original_DuelingDQN' in agent_name:
                    exp_data['Agent'] = 'Dueling_DQN'
                elif 'Original_SACD' in agent_name:
                    exp_data['Agent'] = 'SACD'
                elif 'sac_baseline' in agent_name:
                    exp_data['Agent'] = 'SACD_lambda'
                elif 'sac_mpc' in agent_name and 'nstep' not in agent_name:
                    exp_data['Agent'] = 'SACD_lambda_M'
                elif 'sac_mpc_nstep' in agent_name:
                    exp_data['Agent'] = 'SACD_lambda_TM'
                elif 'PPO' in agent_name:
                    exp_data['Agent'] = 'PPO'
                all_data.append(exp_data)

    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='eval_result/')
    parser.add_argument("--algo", type=list, default=['original'])
    # parser.add_argument("--y_value", type=list, default=['Crash_ratio',
    #                                                      'AverageEpCost',
    #                                                      'AverageEpRet'])
    parser.add_argument("--y_value", type=list, default=['AverageEpCost','AverageEpRet'])
    parser.add_argument('--add_subaxis', type=bool, default=False)
    parser.add_argument("--save_name", type=str, default='cost_distribution')
    args = parser.parse_args()

    data_set = get_data(args)

    plot_data(data_set,
            y_value=args.y_value,
            sub_axis=args.add_subaxis,
            save_name=args.save_name)
    
    save_dir = './pictures/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = args.save_name
    plt.savefig(os.path.join(save_dir, save_name), dpi=600, format='pdf', bbox_inches='tight')

    plt.show()