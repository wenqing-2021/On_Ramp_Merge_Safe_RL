import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import highway_env
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
import wandb
import argparse
import time
import os
import core

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.batch_size = batch_size

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, ))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size

device = "cpu"

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

class DuelingDQN(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-4,
                 max_size=1000000, batch_size=256):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, isTrain=True):
        state = torch.tensor([observation], dtype=torch.float).to(device)
        _, A = self.q_eval.forward(state)
        action = torch.argmax(A).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, logger):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = torch.arange(self.batch_size, dtype=torch.long).to(device)
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(device)
        terminals_tensor = torch.tensor(terminals).to(device)

        with torch.no_grad():
            V_, A_ = self.q_target.forward(next_states_tensor)
            q_ = V_ + A_ - torch.mean(A_, dim=-1, keepdim=True)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * torch.max(q_, dim=-1)[0]
        V, A = self.q_eval.forward(states_tensor)
        q = (V + A - torch.mean(A, dim=-1, keepdim=True))[batch_idx, actions_tensor]

        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        mpi_avg_grads(self.q_eval)
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

        logger.store(Eval_Q=q.mean().item(), Target_Q=q_.mean().item(),
                     TD_error=loss.mean().item())

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model, save_dir + args.proj_name + '_' + args.run_name + save_name + str(ep) + ".pt")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def train(config, logger_kwargs=dict()):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed = config.seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(config.env)
    env = env.unwrapped
    obs_dim = (env.observation_space.shape[0] - 1) * env.observation_space.shape[1] # 50
    act_dim = env.action_space.n # 5

    env.action_space.seed(seed)

    steps = 0
    i = 1  # record episode num

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):

        agent = DuelingDQN(alpha=config.lr_q,
                           state_dim=obs_dim,
                           action_dim=act_dim,
                           fc1_dim=256,
                           fc2_dim=256,
                           max_size=config.buffer_size,
                           batch_size=config.batch_size)

        # Sync params across processes
        sync_params(agent)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [agent.q_eval])
        logger.log('\nNumber of parameters: \t v: %d\n' % var_counts)

        wandb.watch(agent, log="gradients", log_freq=10)

        # Set up model saving
        logger.setup_pytorch_saver(agent.q_eval)

        # env initial
        local_steps_per_epoch = config.steps_per_epoch // num_procs()
        state = env.reset()
        state = state[1:].flatten()
        episode_steps, rewards, costs, crash_counter, tra_counter = 0, 0, 0, 0, 0
        start_time = time.time()

        while steps < int(config.num_steps / num_procs()):
            action = agent.choose_action(state, isTrain=True)

            next_state, reward, done, info = env.step(int(action))
            cost = info['cost']
            if info['crashed']:
                crash_counter += 1
            next_state = next_state[1:].flatten()
            reward -= cost
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            rewards += reward
            costs += cost
            episode_steps += 1
            steps += 1

            if done or (episode_steps >= config.max_ep_len):
                i += 1
                tra_counter += 1
                logger.store(Episode=i, EpRet=rewards, EpCost=costs, EpLen=episode_steps)
                state = env.reset()
                state = state[1:].flatten()
                episode_steps, rewards, costs = 0, 0, 0

            if steps % config.update_freq == 0:
                for _ in range(config.update_freq):
                    agent.learn(logger)

            if steps % local_steps_per_epoch == 0:
                crash_ratio = mpi_sum(crash_counter) / mpi_sum(tra_counter)
                crash_counter, tra_counter = 0, 0

                wandb.log({"AverageRewards": logger.get_stats('EpRet')[0],
                           "AverageCost": logger.get_stats('EpCost')[0],
                           'EpLen': logger.get_stats('EpLen')[0],
                           "Crash_ratio": crash_ratio,
                           "Eval_Q": logger.get_stats('Eval_Q')[0],
                           "Target_Q": logger.get_stats('Target_Q')[0],
                           "TD_error": logger.get_stats('TD_error')[0],
                           "Steps": mpi_sum(steps)})

                logger.log_tabular('Epoch', steps // local_steps_per_epoch)
                logger.log_tabular('Crash_ratio', crash_ratio)
                logger.log_tabular('Eval_Q', average_only=True)
                logger.log_tabular('Target_Q', average_only=True)
                logger.log_tabular('TD_error', average_only=True)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpCost', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', mpi_sum(steps))
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()

            if steps % config.save_every == 0:
                logger.save_state({'env': env}, None)
                save(config, save_name="_", model=agent, wandb=wandb, ep=config.seed)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--proj_name", type=str, default="Baseline")
    parser.add_argument("--run_name", type=str, default="DuelingDQN", help="Run name, default: baseline")
    parser.add_argument("--env", type=str, default="merge_game_env-v0",
                        help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--buffer_size", type=int, default=100_000,
                        help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=10, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=20, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")

    parser.add_argument('--num_steps', type=int, default=5e5)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--lr_q', type=float, default=0.0001)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(data_dir=os.path.join('data',args.proj_name), 
                                        exp_name=args.run_name, 
                                        seed=args.seed)

    train(args, logger_kwargs=logger_kwargs)