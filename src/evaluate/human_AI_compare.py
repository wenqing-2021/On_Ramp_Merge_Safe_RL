import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import highway_env
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import time

label_size = 50
ticks_size = 48
legend_size = 30
line_size = 4
title_size = 55
title_order = ['(a)','(b)','(c)', '(d)']

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': label_size,}

font_title = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': title_size,}

def initial_env():
    merge_env = gym.make('merge_eval_high_density-v0')
    # merge_env = merge_env.unwrapped
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 3200,
        "screen_height": 800,
        "scaling": 12,
        "manual_control":False, # TRUE for human-driven
        "mpc_control": True,
        "show_mpc_trajectory": True,
        "show_other_vehicles_predict": False,
        'show_trajectories': True,
        "only_show_ego_history": True,
        'show_history_frequency': 8,
        "show_history_duration": 0.8
    })
    merge_env.reset()
    return merge_env

def initial_agent(agent_name):
    agent_path = "trained_models/" + agent_name + ".pt"
    agent = torch.load(agent_path)
    return agent

def plot(i, data_x, data_y, label_name):
    num = 221 + i
    plt.subplot(num)
    plt.plot(data_x[0], data_y[0], color = 'blue', linestyle='dashed', linewidth=line_size)
    plt.plot(data_x[1], data_y[1], color = 'red', linewidth=line_size)
    plt.yticks(fontsize=ticks_size, fontproperties='Times New Roman')
    plt.xticks(fontsize=ticks_size, fontproperties='Times New Roman')
    plt.title(title_order[i], font_title, y=-0.3)
    plt.xlabel('Time (s)', font)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace = 0.4) 
    if i == 0:
        plt.legend(['SACD-$\lambda$-TM','Human'], loc='upper right',fontsize = legend_size)
    if label_name == 'speed':
        plt.ylabel('Velocity (m/s)',font, labelpad=20)
    if label_name == 'acc':
        plt.ylabel('Acceleration (m/s$^2$)',font, labelpad=20)
    if label_name == 'st_angle':
        plt.ylabel('Steering Angle (rad)',font, labelpad=20)
    if label_name == 'heading':
        plt.ylabel('Heading Angle (rad)', font, labelpad=20)

if __name__ == "__main__":
    import argparse
    perser = argparse.ArgumentParser()
    perser.add_argument('--total_episodes', type=int, default=2) # first by AI, second by Human
    perser.add_argument('--seed', type=int, default=333)
    perser.add_argument('--render', type=bool, default=True)
    perser.add_argument('--safe_pic', type=bool, default=True)
    perser.add_argument('--re_action', type=bool, default=True)
    perser.add_argument('--plot_state', type=bool, default=True)
    perser.add_argument('--save_freq', type=int, default=1)
    args = perser.parse_args()
    agent_name = 'SAC_SACD-TDn-MPC-5th_2'

    all_speed = []
    all_acc = []
    all_heading = []
    all_st_angle = []
    all_time = []

    total_episodes = args.total_episodes
    seed = args.seed
    render = args.render
    save = args.safe_pic
    re_action = args.re_action
    save_freq = args.save_freq
    plot_state = args.plot_state

    # initialization
    speed = []
    acc = []
    st_angle = []

    for j in range(total_episodes):
        np.random.seed(seed)
        env = initial_env()
        env = env.unwrapped
        agent = initial_agent(agent_name)

        # HUMAN-DRIVEN
        if j == 1:
            env.configure({
                "manual_control":True, # TRUE for human-driven
            })
            re_action = False
        
        obs = env.reset()
        speed = []
        heading = []
        acc = []
        st_angle = []
        time_ = []
        for k in range(1000):
            print('------')
            print(k)
            obs = obs[1:].flatten()
            time_1 = time.time()
            a, _, _, _ = agent.step(torch.from_numpy(obs).float())

            if re_action:
                a = env.check_action(int(a))

            time_2 = time.time()
            print('delta time', time_2 - time_1)

            obs_, r, d, info = env.step(int(a))

            speed.append(env.vehicle.speed)
            heading.append(env.vehicle.heading)
            st_angle.append(env.vehicle.action['steering'])
            acc.append(env.vehicle.action['acceleration'])
            time_.append(0.1 * env.time) # because simulation frequency is 10Hz

            obs = obs_
            if render:
                env.render()
            if d:
                break

        all_speed.append(speed)
        all_acc.append(acc)
        all_st_angle.append(st_angle)
        all_heading.append(heading)
        all_time.append(time_)
    
    if plot_state:
        data_y = [all_speed, all_acc, all_st_angle, all_heading]
        name = ['speed', 'acc', 'st_angle', 'heading']
        fig = plt.figure(1)
        fig.set_size_inches(16 * 2, 12 * 2)
        for k in range(len(data_y)):
            plot(k, all_time, data_y[k], label_name=name[k])
            # plt.title(name[k])
        
        if save:
            save_dir = './pictures/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'human_AI_compare'
            plt.savefig(os.path.join(save_dir, save_name), dpi=600, format='pdf', bbox_inches='tight')

    plt.show()