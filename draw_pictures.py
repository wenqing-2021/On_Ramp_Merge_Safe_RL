import gym
import highway_env
import torch
from matplotlib import pyplot as plt
import os
import numpy as np
import random
import time

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': 16,}

def save_picture(picture, id, agent_name):
    save_dir = './draw_picture/' + agent_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, id)
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.imshow(picture)
    plt.axis('off')
    plt.savefig(save_path, dpi=800, format='svg')

def initial_env():
    merge_env = gym.make('merge_eval_high_density-v0')
    # merge_env = merge_env.unwrapped
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 3200,
        "screen_height": 800,
        "scaling": 12,
        "manual_control":False,
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
    plt.figure(i)
    plt.plot(data_x, data_y)
    plt.xlabel('Time(s)', font)
    if label_name == 'speed':
        plt.ylabel('Velocity(m/s)',font)
    if label_name == 'acc':
        plt.ylabel('Acceleration(m/s$^2$)',font)
    if label_name == 'st_angle':
        plt.ylabel('Steering Angle(rad)',font)

if __name__ == "__main__":
    import argparse
    perser = argparse.ArgumentParser()
    perser.add_argument('--total_episodes', type=int, default=1)
    perser.add_argument('--seed', type=int, default=123)
    perser.add_argument('--render', type=bool, default=True)
    perser.add_argument('--safe_pic', type=bool, default=False)
    perser.add_argument('--re_action', type=bool, default=True)
    perser.add_argument('--plot_state', type=bool, default=True)
    perser.add_argument('--save_freq', type=int, default=1)
    args = perser.parse_args()
    agent_name = ['/SAC_SACD-TDn-MPC-5th_2']

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

    for count, i in enumerate(agent_name):
        np.random.seed(seed)
        speed = []
        acc = []
        st_angle = []
        env = initial_env()
        env = env.unwrapped
        agent = initial_agent(i)

        for j in range(total_episodes):
            obs = env.reset()
            speed = []
            heading = []
            acc = []
            st_angle = []
            time_ = []
            for k in range(1000):
                print(k)
                obs = obs[1:].flatten()
                time_1 = time.time()
                a, _, _, _ = agent.step(torch.from_numpy(obs).float())

                if re_action:
                    a = env.check_action(int(a))
                time_2 = time.time()
                print('------')
                print('delta time', time_2 - time_1)
                obs_, r, d, info = env.step(int(a))

                speed.append(env.vehicle.speed)
                heading.append(env.vehicle.heading)
                st_angle.append(env.vehicle.action['steering'])
                acc.append(env.vehicle.action['acceleration'])
                time_.append(env.time)

                obs = obs_
                if render:
                    env.render()
                if save:
                    if k % save_freq == 0 and k > 40:
                        # episode num_steps_decision
                        id = str(j) + '_' + str(k) + '_' + str(a)
                        save_picture(env.render(mode="rgb_array"), id, i)
                if d:
                    break

        all_speed.append(speed)
        all_acc.append(acc)
        all_st_angle.append(st_angle)
        all_heading.append(heading)
        all_time.append(time_)
        
        if plot_state:
            data_y = [all_speed, all_acc, all_st_angle]
            name = ['speed','acc','st_angle']
            for k in range(len(name)):
                plot(count+k, all_time[count], data_y[k][count], label_name=name[k])
                # plt.title(name[k])
            
            plt.show()
    
