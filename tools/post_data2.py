'''
Author: wenqing-hnu 1140349586@qq.com
Date: 2022-04-26 12:25:09
LastEditors: wenqing-hnu
LastEditTime: 2022-11-20
FilePath: /code/post_data2.py
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
import json

label_size = 50
ticks_size = 48
bar_size = 35
legend_size = 30
titlesize = 55
line_size = 4
traw_legend = False

def plot_data(order,
              all_data,
              file_path):

    all_data = pd.concat(all_data, axis=0, join='inner', ignore_index=True)
    all_data['SuccessRatio'] = all_data['SuccessRatio'].astype(float)
    all_data['SuccessRatio'] = (all_data['SuccessRatio'] * 100).round(1)
    all_data = all_data.loc[(all_data['Methods'] == 'SACD-$\lambda$-TM')]
    ax = sns.barplot(data=all_data, x='Density', y='SuccessRatio',
                     order=['low','mixed','high'],width=0.5)

    for i in ax.containers:
        plt.bar_label(i, fontsize=bar_size, padding=5)

    if order == 1:
        plt.ylabel('Success Ratio (%)', fontsize=label_size, labelpad=20)
        plt.yticks([0,20,40,60,80,100,110], labels=['0','20','40','60','80','100',''], fontsize=ticks_size)
    else:
        plt.yticks([0,20,40,60,80,100,110], labels=['','','','','','',''])
        plt.ylabel('')
    
    plt.grid()
    plt.ylim(0, 110)
    plt.xticks(fontsize=ticks_size)
    # plt.legend(fontsize=legend_size)

    if file_path == 'eval_result/generalization/low':
        # plt.title('Trained in Low Density', loc='center', fontsize=titlesize, pad=15)
        plt.title('(a) Low Density', fontsize = titlesize, y=-0.28)
        plt.xlabel('Test Traffic Density', fontsize=label_size, labelpad=10)
    elif file_path == 'eval_result/generalization/mixed':
        # plt.title('Trained in Medium Density', loc='center', fontsize=titlesize, pad=15)
        plt.title('(b) Medium Density', fontsize = titlesize, y=-0.28)
        plt.xlabel('Test Traffic Density', fontsize=label_size, labelpad=10)
        # plt.legend([],[], frameon=False)
    elif file_path == 'eval_result/generalization/high':
        # plt.title('Trained in High Density', loc='center', fontsize=titlesize, pad=15)
        plt.title('(c) High Density', fontsize = titlesize, y=-0.28)
        plt.xlabel('Test Traffic Density', fontsize=label_size, labelpad=10)

def get_data(file_path):
    all_data = []
    for root, dirs, files in os.walk(file_path):
        if 'progress.txt' in files:
            data_path = os.path.join(root, 'progress.txt')
            Name = os.path.basename(root)
            exp_name = Name.split(sep='_')[0]
            density = Name.split(sep='_')[3]
            if file_path == 'eval_result/generalization/low':
                exp_data = pd.read_table(data_path)
            else:
                exp_data = pd.read_table(data_path, sep=' ')

            if exp_name == 'baseline':
                legend_name = 'SACD-$\lambda$'
            elif exp_name == 'nsteps':
                legend_name = 'SACD-$\lambda$-TM'
            exp_data.insert(len(exp_data.columns), 'Methods', legend_name)
            exp_data.insert(len(exp_data.columns), 'Density', density)
            all_data.append(exp_data)

    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=list, default=['eval_result/generalization/low',
                                                           'eval_result/generalization/mixed',
                                                           'eval_result/generalization/high'])
    # parser.add_argument("--algo", type=list, default=['PPO-sigma', 'SAC-sigma'])
    parser.add_argument("--y_value", type=list, default=['SuccessRatio'])
    # parser.add_argument('--add_subaxis', type=bool, default=False)
    parser.add_argument("--save_name", type=str, default='generalization_ability_barplot_total_test')
    args = parser.parse_args()

    fig = plt.figure(1)
    fig.set_size_inches(16*2,12)
    sns.set(style="white", font='Times New Roman')
    # sns.set_context(rc={"lines.linewidth": line_size})
    for k in range(1, 4):
        fig.add_subplot(1, 3, k)
        data_set = get_data(args.file_path[k-1])
        plot_data(k, data_set, args.file_path[k-1])
    
    save_dir = './pictures/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, args.save_name), dpi=600, format='pdf', bbox_inches='tight')
    plt.show()