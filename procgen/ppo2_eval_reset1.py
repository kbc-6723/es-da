import os
import time
import numpy as np
import os.path as osp

from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from procgen.policy_new import build_policy_reset
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
import tensorflow as tf

def constfn(val):
    def f(_):
        return val
    return f
    
DIR_NAME = 'test_log'

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, id = None,rep_count=1,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, filename=None,comm=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)
    
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    file_name = '%s/%s.txt'%(DIR_NAME, filename)
    f_io = open(file_name, 'a')
 
    policy = build_policy_reset(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from procgen.model_eval_reset import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)
                    
    agent = model.act_model
    should_eval = 1
    
    num_actions = env.action_space.n
  
    if load_path is not None:
        load_path = '/home/kbc/imp/procgen/checkpoints/' + load_path
        model.load(load_path)
    
    obs = env.reset()
    t_step = 0

    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return True

    state = agent.initial_state
    done = np.zeros(nenvs)
    
    
    while should_continue():
        actions, _ , _, _ = agent.new_step(obs, S=state, M=done)
        obs, rew, done, info = env.step(actions)    
            

        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1

                    if 'episode' in info[i]:
                        scores[i] += info[i].get('episode')['r']


        t_step += 1

        if done[0]:
            curr_rews[:] = 0

    result = 0

    if should_eval:
        mean_score = np.mean(scores) / rep_count
        max_idx = np.argmax(scores)

        result = mean_score
        
        f_io.write("{}\n".format(result))
        f_io.close()
        
    return result
    



