
import numpy as np
import torch
from torch.optim import Adam
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import highway_env
import gym
import time
import core
import wandb
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from torch.nn.functional import softplus

torch.autograd.set_detect_anomaly(True)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        obs_dim = (obs_dim[0] - 1) * obs_dim[1]
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.re_a_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size


    def store(self, obs, act, rew, crew, val, cval, logp, re_a):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.re_a_buf[self.ptr] = re_a

        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean)  # / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
                    adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf, re_a=self.re_a_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=1, cost_limit=25, re_action=False,
        config=None):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    env = env.unwrapped
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    damping = 0

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'], data['logp']
        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()

        # clip_cadv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * cadv
        # loss_cpi = (torch.min(ratio * cadv, clip_cadv)).mean()
        loss_cpi = ratio * cadv
        loss_cpi = loss_cpi.mean()

        p = softplus(penalty_param)
        penalty_item = p.item()

        damp = damping * (cost_limit - cur_cost)
        pi_objective = loss_rpi # - (penalty_item - damp) * loss_cpi
        # pi_objective = pi_objective / (1 + penalty_item - damp)
        loss_pi = -pi_objective

        cost_deviation = (cur_cost - cost_limit)

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        return ((ac.v(obs) - ret) ** 2).mean(), ((ac.vc(obs) - cret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_lr = 3e-4
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = torch.tensor(1.0, requires_grad=True).float()
    penalty = softplus(penalty_param)

    penalty_lr = 5e-2
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    vf_lr = 1e-3
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(re_action):
        cur_cost = logger.get_stats('EpCost')[0]
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)
        # print(penalty_param)
        loss_penalty = -penalty_param * cost_dev

        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()
        # print(penalty_param)

        # penalty = softplus(penalty_param)

        data['cur_penalty'] = penalty_param
        logger.store(Penalty=penalty_param.item())

        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item()

        # Train policy with multiple steps of gradient descent
        train_pi_iters = 80
        for i in range(train_pi_iters):
            if re_action:
                # pretrain pi
                obs, act, re_a = data['obs'], data['act'], data['re_a']
                _, logp = ac.pi(obs, act)
                _, re_logp = ac.pi(obs, re_a)
                pre_loss_pi = (0.5 * (logp - re_logp) ** 2).mean()
                pi_optimizer.zero_grad()
                pre_loss_pi.backward()
                mpi_avg_grads(ac.pi)
                pi_optimizer.step()

            pi_optimizer.zero_grad()
            loss_pi, _, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        train_v_iters = 80
        for i in range(train_v_iters):
            loss_v, loss_vc = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossVC=cv_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossVC=(loss_vc.item() - cv_l_old))

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

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_cret, ep_len, crash_counter, tra_counter = env.reset(), 0, 0, 0, 0, 0

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):
        wandb.watch(ac, log="gradients", log_freq=10)
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                o = o[1:].flatten()
                a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
                re_a = a

                if re_action:
                    re_a = env.check_action(int(a))
                    pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, info = env.step(int(a))
                c = info['cost']
                if info['crashed']:
                    crash_counter += 1

                r = r - c
                ep_ret += r
                ep_cret += c
                ep_len += 1

                # save and log
                buf.store(o, a, r, c, v, vc, logp, re_a)
                logger.store(VVals=v)
                logger.store(CVVals=vc)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        o = o[1:].flatten()
                        _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                        vc = 0
                    buf.finish_path(last_val=v, last_cval=vc)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cret)
                        tra_counter += 1
                    o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0

            # Perform PPO update!
            update(re_action)

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)
                save(config, save_name="_", model=ac, wandb=wandb, ep=config.seed)

            crash_ratio = mpi_sum(crash_counter) / mpi_sum(tra_counter)
            logger.store(Crash_counter=mpi_sum(crash_counter))
            crash_counter, tra_counter = 0, 0
            # Log info about epoch
            wandb.log({"AverageRewards": logger.get_stats('EpRet')[0],
                       "AverageCost": logger.get_stats('EpCost')[0],
                       'Penalty': logger.get_stats('Penalty')[0],
                       'EpLen': logger.get_stats('EpLen')[0],
                       "Steps": (epoch + 1) * steps_per_epoch,
                       "Policy Loss": logger.get_stats('LossPi')[0],
                       "Stop_iter": logger.get_stats('StopIter')[0],
                       "Crash_ratio": crash_ratio,
                       })

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Penalty')
            logger.log_tabular('Crash_counter')
            logger.log_tabular('Crash_ratio', crash_ratio)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossVC', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('DeltaLossVC', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='merge_game_env-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=1) # 1,2,3,4,5
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--max_ep_len', type=int, default=4000)
    parser.add_argument('--num_steps', type=int, default=5e5)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--cost_limit', type=float, default=0.01)
    parser.add_argument('--safe_check', type=bool, default=False)
    parser.add_argument('--run_name', type=str, default='PPO')
    parser.add_argument('--proj_name', type=str, default='Lagrangian')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.proj_name + '_' + args.run_name, args.seed)

    epochs = int(args.num_steps / args.steps_per_epoch)

    ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma, max_ep_len=args.max_ep_len,
        seed=args.seed, steps_per_epoch=args.steps_per_epoch, epochs=epochs, cost_limit=args.cost_limit,
        re_action=args.safe_check, logger_kwargs=logger_kwargs, config=args)
