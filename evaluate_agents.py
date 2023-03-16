import gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import core
import argparse
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from utils.mpi_tools import mpi_fork, mpi_sum, mpi_avg, num_procs, proc_id
import os
import random

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha=0.0003, state_dim=50, action_dim=5, fc1_dim=256, fc2_dim=256):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

def merge_eval(env_fn,
               render=False,
               Max_episode_len=1000,
               eval_episodes=100,
               logger:EpochLogger=None,
               print_freq=50,
               choose_agent="ppo_baseline",
               safe_protect=False,
               seed=0,
               data_file=None,
               exp_name=None):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # set env seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    merge_env = merge_env.unwrapped
    merge_env.action_space.seed(seed)
    # create merge env
    merge_env = env_fn()
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "screen_width": 2000,
        "screen_height": 600,
        "scaling": 10,
        "cooperative_prob": 0.8,
        "mpc_control": True
    })

    ##### choose an trained agent #######
    agent_path = "trained_models/" + data_file + '/' + choose_agent + ".pt"
    if choose_agent == 'Original_DuelingDQN_1':
        agent = DuelingDeepQNetwork()
        agent = torch.load(agent_path)
    else:
        agent = torch.load(agent_path)
    
    # save data to txt file
    density = exp_name.split(sep='_')[2]
    save_dir = os.path.join('eval_result', data_file, density)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_dir = os.path.join(save_dir, exp_name + '_' + choose_agent)
    head_list = ['Agent', 'Episode', 'Crashed', 'Success', 'Costs', 'Length\n']
    head_str = '\t'.join(head_list)
    with open(file_dir + '/process.txt', 'w+') as f:
        f.write(head_str)
    # Sync params across processes
    sync_params(agent)
    # logger.store(agent=choose_agent)

    # initialize counter
    episode_counter = 0
    crash_counter = 0
    success_counter = 0
    cost_counter = 0
    re_action = safe_protect

    # EVALUATE #
    while True:
        obs = merge_env.reset()
        ep_costs = 0
        ep_len = 0
        crash = 0
        reach = 0
        if episode_counter == int(eval_episodes / num_procs()):
            break
        for step in range(int(Max_episode_len / num_procs())):
            obs = obs[1:].flatten()
            if 'Original_DuelingDQN' in choose_agent :
                state = torch.tensor([obs], dtype=torch.float)
                _, A = agent.forward(state)
                a = torch.argmax(A).item()
            else:
                a, _, _, _ = agent.step(torch.from_numpy(obs).float())

            if re_action:
                a = merge_env.check_action(int(a))

            obs_, r, d, info = merge_env.step(int(a))

            cost = info['cost']
            ep_costs += cost
            ep_len += 1
            if render:
                merge_env.render()
            if info['crashed']:
                crash = 1
                crash_counter += 1
            elif info['success']:
                reach = 1
                if ep_costs >= 0.5:
                    cost_counter += 1
                else:
                    success_counter += 1
            obs = obs_
            if d:
                episode_counter += 1
                # print('current episode:',episode_counter, 
                #       'remain episodes:', int(eval_episodes / num_procs()) - episode_counter)
                head_list = [agent_name, str(episode_counter), str(crash), str(reach), str(ep_costs), str(ep_len)+'\n']
                head_str = '\t'.join(head_list)
                with open(file_dir + '/process.txt', 'a+') as f:
                    f.write(head_str)

                logger.store(AverageCost=ep_costs, AverageLen=ep_len)
                logger.log_tabular("Agent", choose_agent)
                logger.log_tabular("Episode", episode_counter)
                logger.log_tabular("Crashed", mpi_sum(crash))
                logger.log_tabular("Success", mpi_sum(reach))
                logger.log_tabular("Costs", mpi_avg(ep_costs))
                logger.log_tabular("Length", mpi_avg(ep_len))
                logger.dump_tabular()
                break
    
    print('CrashRatio', mpi_sum(crash_counter) / eval_episodes,
          'CrashCounter', mpi_sum(crash_counter),
          'SuccessRatio', mpi_sum(success_counter) / eval_episodes,
          'SuccessCounter', mpi_sum(success_counter),
          'HighCostsRatio', mpi_sum(cost_counter) / eval_episodes,
          'HighCostsCounter', mpi_sum(cost_counter))

    # logger.log_tabular("Agent", choose_agent)
    # logger.log_tabular("CrashRatio", mpi_sum(crash_counter) / eval_episodes)
    # logger.log_tabular("AverageCost", average_only=True)
    # logger.log_tabular("AverageLen", average_only=True)
    # logger.log_tabular("CrashCounter", mpi_sum(crash_counter))
    # logger.log_tabular("SuccessCounter", mpi_sum(success_counter))
    # logger.log_tabular("SuccessRatio", mpi_sum(success_counter) / eval_episodes)
    # logger.log_tabular("CostCounter", mpi_sum(cost_counter))
    # logger.log_tabular("CostRatio", mpi_sum(cost_counter) / eval_episodes)
    # logger.log_tabular("FailCounter", mpi_sum(cost_counter) + mpi_sum(crash_counter))
    # logger.dump_tabular()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="merge_eval_low_density-v0")
    parser.add_argument('--eval_episodes', type=int, default=400)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='original_in_low')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--safe_protect', type=bool, default=False)
    parser.add_argument('--print_freq', type=int, default=100)
    # parser.add_argument('--agents', nargs='+', default=['sac_baseline',
    #                                                     'Original_DuelingDQN_1',
    #                                                     'Original_SACD_original_1',
    #                                                     'sac_mpc',
    #                                                     'sac_mpc_nstep'])
    parser.add_argument('--agents', nargs='+', default=['sac_baseline',
                                                        # 'Original_SACD_original_100',
                                                        ])
    # parser.add_argument('--agents', nargs='+', default=['PPO_Baseline_PPO_5'])
    # parser.add_argument('--agents', nargs='+', default=['SAC_sac_mpc_nsteps_high_2',
    #                                                     'SAC_sac_mpc_nsteps_low_2',
    #                                                     'SAC_sac_baseline_high_1',
    #                                                     'SAC_sac_baseline_low_1',
    #                                                     ])
    
    parser.add_argument('--data_file', type=str, default='original')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    for i, agent_name in enumerate(args.agents):

        logger_kwargs = setup_logger_kwargs(args.exp_name + '_' + agent_name, 
                                        data_dir=os.path.join('eval_result',args.data_file, 'low'))

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)

        args.safe_protect = False
        print('agent_name:', agent_name)
        print('Safe protect', args.safe_protect)
        # if agent_name == 'sac_mpc' or agent_name == 'sac_mpc_nstep':
        if agent_name == 'SAC_sac_mpc_nsteps_high_2' or agent_name == 'SAC_sac_mpc_nsteps_low_2':
            args.safe_protect = True
            print('Safe protect', args.safe_protect)

        merge_eval(lambda: gym.make(args.env),
                   Max_episode_len=args.steps,
                   eval_episodes=args.eval_episodes,
                   print_freq=args.print_freq,
                   logger=logger,
                   choose_agent=agent_name,
                   safe_protect=args.safe_protect,
                   render=args.render,
                   seed=args.seed,
                   data_file=args.data_file,
                   exp_name=args.exp_name)
