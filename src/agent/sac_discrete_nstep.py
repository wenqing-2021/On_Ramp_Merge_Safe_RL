import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from core import SAC_Critic, SAC_Actor
import core
import copy
import random
import numpy as np
import wandb
import argparse
import time
import os
import gym
import highway_env
from collections import deque, namedtuple
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum


class sumtree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.tree_pointer = 0

    def add(self, priority):
        '''
        Add priority of new transition
        :param priority: TD-error + epsilon
        '''
        tree_idx = self.tree_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.tree_pointer += 1

        if self.tree_pointer >= self.capacity:
            self.tree_pointer = 0

    def update(self, tree_idx, priority):
        '''
        Update priority when TD-error has changed
        :param tree_idx:
        :param priority:
        '''
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        :param v: v is the chosen value
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx  # get the chosen priority index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root


class MultiStepBuff:

    def __init__(self, maxlen=3, gamma=0.99):
        self.maxlen = int(maxlen)
        self.gamma = gamma
        self.reset()

    def append(self, state, action, reward, cost):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.costs.append(cost)

    def get(self):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward, cost = self.n_step_return()
        return state, action, reward, cost

    def n_step_return(self):
        r = np.sum([r * (self.gamma ** i) for i, r in enumerate(self.rewards)])
        c = np.sum([c * (self.gamma ** i) for i, c in enumerate(self.costs)])
        self.rewards.popleft()
        self.costs.popleft()
        return r, c

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)
        self.costs = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, train_steps=4000, initial_beta=0.4, n_step=3,
                 gamma=0.99, use_per=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.size = buffer_size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "cost"])

        # binary tree for Prioritized Experience Replay
        self.use_per = use_per
        self.tree = sumtree(buffer_size)
        self.abs_error_upper = 1.
        self.beta = initial_beta
        self.alpha = 0.6
        self.epsilon = 1e3
        self.beta_diff = (1 - initial_beta) / train_steps

        self.n_step = n_step
        self.n_step_buff = MultiStepBuff(maxlen=n_step, gamma=gamma)

    def add(self, state, action, reward, next_state, done, cost):
        """Add a new experience to memory."""
        if self.n_step != 1:
            self.n_step_buff.append(state, action, reward, cost)
            if self.n_step_buff.is_full():
                state, action, reward, cost = self.n_step_buff.get()
                e = self.experience(state, action, reward, next_state, done, cost)
                self.memory.append(e)

            if done:
                while not self.n_step_buff.is_empty():
                    state, action, reward, cost = self.n_step_buff.get()
                    e = self.experience(state, action, reward, next_state, done, cost)
                    self.memory.append(e)
        else:
            e = self.experience(state, action, reward, next_state, done, cost)
            max_p = np.max(self.tree.tree[-self.tree.capacity:])  # the max priority for the new transition
            if max_p == 0:
                max_p = self.abs_error_upper
            if self.memory_counter < self.size:
                self.memory.append(e)
                self.tree.add(max_p)
            else:
                self.memory[self.memory_counter % self.size] = e
                self.tree.add(max_p)

        self.memory_counter += 1

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        tree_idx, ISweight = np.ones(self.batch_size), np.ones(self.batch_size)
        if self.use_per:
            experiences = deque(maxlen=self.batch_size)
            tree_idx, ISweight = np.zeros(self.batch_size), np.zeros(self.batch_size)
            priority_segment = self.tree.total_p / self.batch_size
            self.beta = np.min([1., self.beta + self.beta_diff])  # max = 1, beta increase after sample
            # find min priority
            if len(self.memory) == self.size:
                capacity_priority = self.tree.tree[-self.tree.capacity:]
            else:
                start = self.tree.capacity - 1
                capacity_priority = self.tree.tree[start:start + len(self.memory)]
            min_prob = np.min(capacity_priority) / self.tree.total_p
            for i in range(self.batch_size):
                a, b = priority_segment * i, priority_segment * (i + 1)
                v = random.uniform(a, b)
                idx, priority, data_idx = self.tree.get_leaf(v)
                prob = priority / self.tree.total_p
                ISweight[i] = np.power(prob / min_prob, -self.beta)
                tree_idx[i] = idx
                experiences.append(self.memory[data_idx])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        cost = torch.from_numpy(np.vstack([e.cost for e in experiences if e is not None]).astype(np.uint8)).float()

        ISweight = torch.as_tensor(ISweight.reshape(self.batch_size, 1), dtype=torch.float32)

        return (states, actions, rewards, next_states, dones, cost), tree_idx, ISweight

    def update_batch(self, tree_idx, abs_error):
        """update batch priority"""
        abs_error += self.epsilon
        clipped_errors = np.minimum(abs_error, self.abs_error_upper)
        priority = np.power(clipped_errors, self.alpha)  # convert TD-error to priority
        for ti, p in zip(tree_idx, priority):
            self.tree.update(int(ti), p)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SAC(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 config=None,
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        hidden_size = 256
        pi_learning_rate = 1e-4
        v_learning_rate = 1e-4
        penalty_init = 1.0
        penalty_lr_rate = 1e-4
        target_entropy_ratio = 0.98
        self.damping = 0

        self.max_ep_len = 1000
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.tau = 1e-2
        self.cost_limit = config.cost_limit
        self.penalty = torch.tensor(penalty_init, requires_grad=True).float()
        self.penalty_optimizer = optim.Adam(params=[self.penalty], lr=penalty_lr_rate)

        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)
        # self.target_entropy = \
        #     -np.log(1.0 / action_size) * target_entropy_ratio

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=pi_learning_rate)

        # Actor Network

        self.actor_local = SAC_Actor(state_size, action_size, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=pi_learning_rate)

        # Critic Network (w/ Target Network)

        self.critic1 = SAC_Critic(state_size, action_size, hidden_size)
        self.critic2 = SAC_Critic(state_size, action_size, hidden_size)
        self.cost = SAC_Critic(state_size, action_size, hidden_size)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = SAC_Critic(state_size, action_size, hidden_size)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = SAC_Critic(state_size, action_size, hidden_size)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.cost_target = SAC_Critic(state_size, action_size, hidden_size)
        self.cost_target.load_state_dict(self.cost.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=v_learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=v_learning_rate)
        self.cost_optimizer = optim.Adam(self.cost.parameters(), lr=v_learning_rate)

    def get_action(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float()

        with torch.no_grad():
            action, _, _, _ = self.actor_local.step(state)
        return action.numpy()

    def calc_policy_loss(self, states, ISweight, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        qc = self.cost(states)
        min_Q = torch.min(q1, q2)
        cost_constraint = self.cost_limit * (1 - self.gamma ** self.max_ep_len) / (1 - self.gamma) / self.max_ep_len
        damp = self.damping * (cost_constraint - (action_probs * self.cost(states)).sum(1).mean())
        soft_penalty = F.softplus(self.penalty)
        actor_loss = (action_probs * (alpha * log_pis - min_Q + (soft_penalty.item() - damp) * qc) * ISweight).sum(
            1).mean()
        log_action_pi = -torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, experiences, gamma, ISweight, logger=None):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, cost) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, cost = experiences

        # ---------------------------- loss actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, entropy = self.calc_policy_loss(states, ISweight, current_alpha)

        # Compute alpha loss
        alpha_loss = - (self.log_alpha * (self.target_entropy - entropy.detach()).detach() * ISweight).mean()

        # ---------------------------- loss critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (
                    torch.min(Q_target1_next, Q_target2_next) - self.alpha * log_pis)
            QC_target_next = action_probs * self.cost_target(next_states)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))
            QC_targets = cost + (gamma * (1 - dones) * QC_target_next.sum(dim=1).unsqueeze(-1))

            # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        qc = self.cost(states).gather(1, actions.long())

        critic1_loss = 0.5 * torch.mean((q1 - Q_targets).pow(2) * ISweight)
        critic2_loss = 0.5 * torch.mean((q2 - Q_targets).pow(2) * ISweight)
        cost_loss = 0.5 * torch.mean((qc - QC_targets).pow(2) * ISweight)

        # penalty loss
        # cost_states = action_probs * self.cost(states)
        cost_constraint = self.cost_limit * (1 - gamma ** self.max_ep_len) / (1 - gamma) / self.max_ep_len
        penalty_loss = - self.penalty * ((qc.detach() - cost_constraint) * ISweight).mean()
        # penalty_loss = - self.penalty * ((qc.detach() - self.cost_limit) * ISweight).mean()

        # ---------------------------- update ---------------------------- #
        # actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        mpi_avg_grads(self.actor_local)
        self.actor_optimizer.step()

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        mpi_avg_grads(self.critic1)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        mpi_avg_grads(self.critic2)
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # cost
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        mpi_avg_grads(self.cost)
        clip_grad_norm_(self.cost.parameters(), self.clip_grad_param)
        self.cost_optimizer.step()

        # alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        mpi_avg_grads(self.log_alpha)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # penalty
        self.penalty_optimizer.zero_grad()
        penalty_loss.backward()
        mpi_avg_grads(self.penalty)
        self.penalty_optimizer.step()
        logger.store(LossPi=actor_loss.item(), alpha_loss=alpha_loss.item(), critic1_loss=critic1_loss.item(),
                     critic2_loss=critic2_loss.item(), alpha=current_alpha.item(), cost_loss=cost_loss.item(),
                     penalty=self.penalty.item(), critic1=q1.mean().item(), critic2=q2.mean().item(),
                     cost=qc.mean().item(), entropy=entropy.mean().item())

        abs_error = abs((q1 + q2) / 2 - Q_targets).detach().numpy()  # batch abs error
        return abs_error

    # ----------------------- update target networks ----------------------- #
    def update_soft(self):
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        self.soft_update(self.cost, self.cost_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    state = state[1:].flatten()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(int(action))
        cost = info['cost']
        next_state = next_state[1:].flatten()
        dataset.add(state, action, reward, next_state, done, cost)
        state = next_state
        if done:
            state = env.reset()
            state = state[1:].flatten()


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

    env.action_space.seed(seed)

    steps = 0
    i = 1  # record episode num
    AverageRewards10 = deque(maxlen=10)
    AverageCost10 = deque(maxlen=10)

    re_action = config.safe_check

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):

        agent = SAC(state_size=(env.observation_space.shape[0] - 1) * env.observation_space.shape[1],
                    action_size=env.action_space.n, config=config)

        # Sync params across processes
        sync_params(agent)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [agent.actor_local, agent.critic1])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        wandb.watch(agent, log="gradients", log_freq=10)

        # Set up model saving
        logger.setup_pytorch_saver(agent.actor_local)

        buffer = ReplayBuffer(buffer_size=int(config.buffer_size / num_procs()),
                              batch_size=int(config.batch_size / num_procs()),
                              initial_beta=0.4,
                              train_steps=int(config.num_steps / num_procs() / config.update_freq),
                              n_step=config.n_step,
                              use_per=config.per)

        collect_random(env=env, dataset=buffer, num_samples=int(config.sample_steps / num_procs()))

        # env initial
        local_steps_per_epoch = config.steps_per_epoch // num_procs()
        state = env.reset()
        state = state[1:].flatten()
        episode_steps, rewards, costs, crash_counter, tra_counter = 0, 0, 0, 0, 0
        start_time = time.time()

        while steps < int(config.num_steps / num_procs()):
            action = agent.get_action(state)

            if re_action:
                action_origin = action
                # "simple" or "mpc"
                action = env.check_action(int(action), "simple")

            next_state, reward, done, info = env.step(int(action))
            cost = info['cost']
            if info['crashed']:
                crash_counter += 1
            next_state = next_state[1:].flatten()
            buffer.add(state, action, reward, next_state, done, cost)
            state = next_state
            rewards += reward
            costs += cost
            episode_steps += 1
            steps += 1

            if done or (episode_steps == config.max_ep_len):
                i += 1
                tra_counter += 1
                logger.store(Episode=i, EpRet=rewards, EpCost=costs, EpLen=episode_steps)
                AverageRewards10.append(rewards)  # compute average reward of recent 10 trajectories
                AverageCost10.append(costs)
                state = env.reset()
                state = state[1:].flatten()
                episode_steps, rewards, costs = 0, 0, 0

            if steps % config.update_freq == 0:
                for _ in range(config.update_freq):
                    experiences, tree_idx, ISweight = buffer.sample()
                    abs_error = agent.learn(
                        experiences, gamma=0.99, ISweight=ISweight, logger=logger)
                    if config.per:
                        buffer.update_batch(tree_idx, abs_error)

            if steps % config.update_target_freq == 0:
                agent.update_soft()

            if steps % local_steps_per_epoch == 0:
                crash_ratio = mpi_sum(crash_counter) / mpi_sum(tra_counter)
                crash_counter, tra_counter = 0, 0

                wandb.log({"AverageRewards": logger.get_stats('EpRet')[0],
                           "AverageCost": logger.get_stats('EpCost')[0],
                           'EpLen': logger.get_stats('EpLen')[0],
                           "Crash_ratio": crash_ratio,
                           "Critic1": logger.get_stats('critic1')[0],
                           "Critic2": logger.get_stats('critic2')[0],
                           "sac_cost": logger.get_stats('cost')[0],
                           "Entropy": logger.get_stats('entropy')[0],
                           "Policy Loss": logger.get_stats('LossPi')[0],
                           "Alpha Loss": logger.get_stats('alpha_loss')[0],
                           "Critic error 1": logger.get_stats('critic1_loss')[0],
                           "Critic error 2": logger.get_stats('critic2_loss')[0],
                           "Cost loss": logger.get_stats('cost_loss')[0],
                           "Alpha": logger.get_stats('alpha')[0],
                           "Penalty": logger.get_stats('penalty')[0],
                           "Steps": mpi_sum(steps),
                           "Buffer size": buffer.__len__()})

                logger.log_tabular('Epoch', steps // local_steps_per_epoch)
                logger.log_tabular('penalty', average_only=True)
                logger.log_tabular('Crash_ratio', crash_ratio)
                logger.log_tabular('alpha', average_only=True)
                logger.log_tabular('entropy', average_only=True)
                logger.log_tabular('critic1', average_only=True)
                logger.log_tabular('critic2', average_only=True)
                logger.log_tabular('cost', average_only=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('alpha_loss', average_only=True)
                logger.log_tabular('critic1_loss', average_only=True)
                logger.log_tabular('critic2_loss', average_only=True)
                logger.log_tabular('cost_loss', average_only=True)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpCost', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('Buffer_size', buffer.__len__())
                logger.log_tabular('TotalEnvInteracts', mpi_sum(steps))
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()

            if steps % config.save_every == 0:
                logger.save_state({'env': env}, None)
                save(config, save_name="_", model=agent.actor_local, wandb=wandb, ep=config.seed)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--proj_name", type=str, default="SAC_simple_check")
    parser.add_argument("--run_name", type=str, default="sac_simple_nstep_0.01", help="Run name, default: baseline")
    parser.add_argument("--env", type=str, default="merge_game_env-v0",
                        help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--buffer_size", type=int, default=1000_000,
                        help="Maximal training dataset size, default: 1000_000")
    parser.add_argument("--seed", type=int, default=2, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=20, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--safe_check", type=bool, default=True)
    parser.add_argument('--num_steps', type=int, default=5e5)
    parser.add_argument('--sample_steps', type=int, default=20000)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--update_target_freq', type=int, default=100)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--per', type=bool, default=False)
    parser.add_argument('--n_step', type=int, default=3, help='1 or 3')
    parser.add_argument('--cost_limit', type=float, default=0.01)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(data_dir=os.path.join('data',args.proj_name), 
                                        exp_name=args.run_name, 
                                        seed=args.seed)

    train(args, logger_kwargs=logger_kwargs)
