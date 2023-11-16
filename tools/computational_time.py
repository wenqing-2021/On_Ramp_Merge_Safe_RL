import gym
import highway_env
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import random
import time

label_size = 50
ticks_size = 48
legend_size = 30
line_size = 4
title_size = 55

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': label_size,}

def initial_env(env_name):
    merge_env = gym.make(env_name)
    # merge_env = merge_env.unwrapped
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 3200,
        "screen_height": 800,
        "scaling": 12,
        "manual_control":False, # TRUE for human-driven
        "mpc_control": True,
        "show_mpc_trajectory": False,
        "show_other_vehicles_predict": False,
        'show_trajectories': False,
        "only_show_ego_history": False,
        'show_history_frequency': 8,
        "show_history_duration": 0.8
    })
    merge_env.reset()

    return merge_env

def initial_agent(agent_name):
    agent_path = "trained_models/" + agent_name + ".pt"
    agent = torch.load(agent_path)
    return agent

def plot(data):
    sns.set(style="white", font='Times New Roman')
    sns.set_context(rc={"lines.linewidth": line_size})
    ax = sns.boxplot(data=data, y = 'env_name', x = 'decision_time', width=0.6)
    ax.grid(True)

    plt.xlim((0,0.05))

    plt.xticks(fontsize=ticks_size, fontproperties='Times New Roman')
    plt.yticks([0,1,2],['Low','Medium','High'], fontsize = ticks_size)
    
    plt.ylabel('Traffic Density', font, labelpad=20)
    plt.xlabel('Decision Time (s)',font, labelpad=20)

if __name__ == "__main__":
    import argparse
    perser = argparse.ArgumentParser()
    perser.add_argument('--total_episodes', type=int, default=3)
    perser.add_argument('--seed', type=int, default=4)
    perser.add_argument('--render', type=bool, default=False)
    perser.add_argument('--save_pic', type=bool, default=True)
    perser.add_argument('--re_action', type=bool, default=True)
    perser.add_argument('--plot_comp_time', type=bool, default=True)
    args = perser.parse_args()
    agent_name = 'SAC_SACD-TDn-MPC-5th_2'

    multiplier = 1
    total_episodes = args.total_episodes * multiplier
    seed = args.seed
    render = args.render
    save = args.save_pic
    re_action = args.re_action
    plot_computation_time = args.plot_comp_time
    
    env_name = ['merge_eval_low_density-v0',
                'merge_game_env-v0',
                'merge_eval_high_density-v0']

    # initialization
    decision_time = []
    env_name_list = []

    for j in range(total_episodes):
        print(j)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        current_env_name = env_name[j // multiplier]
        env = initial_env(current_env_name)
        env = env.unwrapped
        env.action_space.seed(seed)
        agent = initial_agent(agent_name)
        
        obs = env.reset()

        for k in range(1000):
            # print('------')
            obs = obs[1:].flatten()
            time_1 = time.time()
            a, _, _, _ = agent.step(torch.from_numpy(obs).float())

            if re_action:
                a = env.check_action(int(a))

            time_2 = time.time()
            # print('delta time', time_2 - time_1)

            obs_, r, d, info = env.step(int(a))

            decision_time.append(time_2 - time_1)
            env_name_list.append(current_env_name)
            # print(env_name[j])

            obs = obs_
            if render:
                env.render()
            if d:
                break
    
    if plot_computation_time:
        dict_data = {'env_name':env_name_list,
                     'decision_time':decision_time}
        
        data = pd.DataFrame(dict_data)

        fig = plt.figure(1)
        fig.set_size_inches(16, 12)
            
        plot(data)
        
        if save:
            save_dir = './pictures/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'Decision_Time'
            plt.savefig(os.path.join(save_dir, save_name), dpi=600, format='pdf', bbox_inches='tight')

    plt.show()