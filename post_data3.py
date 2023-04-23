import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import os
import argparse
import json


def plot_data(data_set, y_value=None, sub_axis=False, save_name=None):
    all_data = pd.concat(data_set, axis=0, join='inner', ignore_index=True)
    fig = plt.figure()
    fig.set_size_inches(16*len(y_value), 12*len(save_name))
    k = 1
    agent_name = ['sac_m', 'sac_tm']
    for name in save_name:
        if name == 'sacd_tm_sigma':
            show_label = False
            name_index = 1
        else:
            show_label = False
            name_index = 0
        labels = None
        # labels = ['SACD-$\lambda$', 'SACD-$\lambda$-M', 'SACD-$\lambda$-TM']
        # labels = ['Dueling DQN','SACD','SACD-$\lambda$']
        labels = ['$\sigma=1$', '$\sigma=5$', '$\sigma=10$']
        # labels = ['$\eta=0.1$', '$\eta=0.05$', '$\eta=0.01$']
        label_size = 50
        ticks_size = 48
        legend_size = 30
        line_size = 4
        titlesize = 55
        if name == 'sacd_tm_sigma':
            title_order = ['(a)','(b)','(c)']
        else:
            title_order = ['(a)','(b)','(c)']
        # for crash ratio
        ylimit = (-0.005, 0.5)
        ylabel = 'Crash Ratio'
        for index, i in enumerate(y_value):
            if i == 'AverageEpCost':
                ylimit = (-0.005, 0.8)
                ylabel = 'Average Cost'
            elif i == 'AverageEpRet':
                ylimit = (2, 17)
                ylabel = 'Average Reward'

            # fig = plt.figure()
            sns.set(style="white", font='Times New Roman')
            sns.set_context(rc={"lines.linewidth": line_size})
            hue_order = None
            # fig.set_size_inches(16*len(y_value), 12)
            if name == 'ppo_sigma':
                hue_order = ['PPO_mpc_0th', 'PPO_mpc', 'PPO_mpc_10th']
            elif name == 'sacd_tm_sigma':
                hue_order = ['SAC_SACD-TDn-MPC-0th', 'SAC_nsteps_mpc', 'SAC_SACD-TDn-MPC-10th']
            elif name == 'sacd_m_sigma':
                hue_order = ['sac_mpc-1th', 'sac_mpc-5th', 'sac_mpc-10th']
            elif name == 'sacd_lambda':
                hue_order = ['sac_mpc_nstep_0.1', 'sac_mpc_nstep_0.05', 'SAC_nsteps_mpc']
            elif name == 'sacd_compare':
                hue_order = ['SAC_baseline', 'SAC_mpc', 'SAC_nsteps_mpc']
            elif name == 'original_compare':
                hue_order = ['DuelingDQN', 'SACD_original', 'SAC_baseline']
            fig.add_subplot(len(save_name), len(y_value), k)
            ax = sns.lineplot(data=all_data.loc[(all_data['Agent_name'] == agent_name[name_index])], x='Epoch', y=i, hue='Exp_name', hue_order=hue_order, style='Exp_name', style_order=hue_order, legend=False)
            ax.grid(True)
            # plt.title(label=title_order[k-1], fontdict={'fontsize': title_size}, loc='left')
            plt.xlabel('TotalEnvInteracts(Million)', fontsize=label_size, labelpad=20)
            plt.title(title_order[index], fontsize = titlesize, y=-0.3)
            # plt.xlabel('TotalEnvInteracts(Million)', fontsize=label_size,labelpad=20)
            plt.ylabel(ylabel, fontsize=label_size, labelpad=20)
            plt.xticks(np.linspace(1, 125, 6), ['0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=ticks_size)
            plt.yticks(fontsize=ticks_size)
            plt.ylim(ylimit)
            if not show_label:
                plt.legend(labels=labels, fontsize=legend_size)
                show_label = True

            # add subaxis
            if sub_axis:
                if i == 'Crash_ratio':
                    axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                    bbox_to_anchor=(2.5 / 5, 1.5 / 5, 0.8, 0.7),
                                    bbox_transform=ax.transAxes)

                    subax = sns.lineplot(data=all_data, x='Epoch', y='Crash_ratio', hue='Exp_name', ax=axins,
                                        legend=False)
                    subax.axes.yaxis.set_visible(False)
                    subax.axes.xaxis.set_visible(False)

                    subax.set_xlim(100, 125)
                    subax.set_ylim(0, 0.04)

                    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)

                elif i == 'AverageEpCost':
                    axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                    bbox_to_anchor=(3 / 5, 4.5 / 8, 0.8, 0.7),
                                    bbox_transform=ax.transAxes)
                    subax = sns.lineplot(data=all_data, x='Epoch', y='AverageEpCost', hue='Exp_name', ax=axins,
                                        legend=False)
                    subax.axes.yaxis.set_visible(False)
                    subax.axes.xaxis.set_visible(False)

                    subax.set_xlim(100, 125)
                    subax.set_ylim(0, 0.06)

                    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)

                elif i == 'AverageEpRet':
                    axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                    bbox_to_anchor=(3 / 5, 9 / 17, 0.8, 0.7),
                                    bbox_transform=ax.transAxes)
                    subax = sns.lineplot(data=all_data, x='Epoch', y='AverageEpRet', hue='Exp_name', ax=axins,
                                        legend=False)
                    subax.axes.yaxis.set_visible(False)
                    subax.axes.xaxis.set_visible(False)

                    subax.set_xlim(100, 125)
                    subax.set_ylim(14.5, 15.8)

                    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)

            k += 1

    # plt.savefig(os.path.join(save_dir, save_name), dpi=600, format='pdf', bbox_inches='tight')

    # plt.show()


def get_data(args):
    all_data = []
    for i in args.algo:
        path = args.file_path + i
        for root, dirs, files in os.walk(path):
            if 'progress.txt' in files:
                data_path = os.path.join(root, 'progress.txt')
                exp_data = pd.read_table(data_path)
                # make sure epoch starts from 1
                if exp_data['Epoch'][0] == 0:
                    exp_data['Epoch'] += 1

                exp_name = 'NAN'

                try:
                    config_path = open(os.path.join(root, 'config.json'))
                    config = json.load(config_path)
                    exp_name = config['exp_name']
                    if 'sac_mpc' in exp_name:
                        agent_name = 'sac_m'
                    else:
                        agent_name = 'sac_tm'
                except:
                    print('No file named config.json')

                exp_data.insert(len(exp_data.columns), 'Exp_name', exp_name)
                exp_data.insert(len(exp_data.columns), 'Agent_name', agent_name)
                all_data.append(exp_data)

    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='data/')
    parser.add_argument("--algo", type=list, default=['SAC-sigma'])
    # parser.add_argument("--y_value", type=list, default=['Crash_ratio',
    #                                                      'AverageEpCost',
    #                                                      'AverageEpRet'])
    parser.add_argument("--y_value", type=list, default=['AverageEpCost','AverageEpRet'])
    parser.add_argument('--add_subaxis', type=bool, default=False)
    # parser.add_argument("--save_name", type=str, default='sacd_tm_sigma')
    args = parser.parse_args()

    data_set = get_data(args)

    plot_data(data_set,
            y_value=args.y_value,
            sub_axis=args.add_subaxis,
            save_name=['sacd_tm_sigma'])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.53)
    save_dir = './pictures/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = 'sacd_tm_sigma'
    plt.savefig(os.path.join(save_dir, save_name), dpi=600, format='pdf', bbox_inches='tight')

    plt.show()