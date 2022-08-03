import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
import procgen.data_augs as rad
from baselines.common import explained_variance, set_global_seeds
from procgen.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from procgen.runner import Runner
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from procgen.model import UCBDrAC   
import torch
def constfn(val):
    def f(_):
        return val
    return f
DIR_NAME = 'train_log'

    
def learn(*, network, stu_network=None, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, run_id = None, vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,args, 
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, ucb_coef = 1.5, 
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
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

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    file_name_aug = '%s/%s.txt'%(DIR_NAME, args.run_id + "_aug")
    f_aug = open(file_name_aug, 'a') 
    '''
    file_name_gradient = '%s/%s.txt'%(DIR_NAME, run_id + "gradient")
    file_name_loss = '%s/%s.txt'%(DIR_NAME, run_id + "loss")
    file_name_netsize = '%s/%s.txt'%(DIR_NAME, run_id + "netsize")
    f_gradient = open(file_name_gradient, 'a')
    f_loss = open(file_name_loss, 'a')
    f_netsize = open(file_name_netsize, 'a')
    '''
   
    policy = build_policy(env, network, network, use_drac = True, **network_kwargs)
   
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from procgen.model import Model
        model_fn = Model
    
    model = UCBDrAC(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, args = args,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight, ucb_coef = ucb_coef)

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in variables:
            print(v.name)
    if load_path is not None:
        load_path = args.log_dir + 'checkpoints/' + load_path
        model.load(load_path)
    # Instantiate the runner object

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)
        
    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()
   
    # Start total timer
 
            
    tfirststart = time.perf_counter()
    
    checkpoint = list(range(0,args.timesteps + 1,1))
    check_index = 0
    saved_key_checkpoints = [False] * len(checkpoint)

    nupdates = total_timesteps//nbatch
    aug_list = [ 'rccj', None, 'crop']
    current_aug_id = 1
    current_aug = None

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
            
        
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')
        
        # Get minibatch
       
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        model.update_ucb_values(current_aug_id, returns.mean())
        current_aug_id, current_aug = model.select_ucb_aug()
        print(current_aug_id, current_aug)
        if args.save_obs:
            save_idx = np.random.choice(nbatch, 100, replace = False)
            if update == 1:
                save_obs = obs[save_idx]
            else:
                save_obs = np.concatenate((save_obs, obs[save_idx]), axis = 0)
     
        if current_aug is not None:       
            if current_aug == 'color_jitter':
                aug_func = rad.ColorJitterLayer(nbatch, p_rand = 1)
            elif current_aug == 'gray':
                aug_func = rad.Grayscale(nbatch, p_rand = 1)
            elif current_aug == 'flip':
                aug_func = rad.Rand_Flip(nbatch , p_rand = 1)    
            elif current_aug == 'rotate':
                aug_func = rad.Rand_Rotate(nbatch, p_rand = 1)    
            elif current_aug == 'crop':
                aug_func = rad.Crop(nbatch, p_rand = 1)
            elif current_aug == 'cutout':
                aug_func = rad.Cutout(nbatch, p_rand = 1)
            elif current_aug == 'cutout_color':
                aug_func = rad.Cutout_Color(nbatch, p_rand = 1)
            elif current_aug == 'random_conv':
                aug_func = rad.RandomConv(nbatch)
            elif current_aug == 'rccj':
                aug_func = rad.ColorJitterLayer(int(nbatch / 2), p_rand = 1)
                aug_func2 = rad.RandomConv(int(nbatch / 2))
            else:
                pass

            if current_aug == 'rccj':
                obs_buffer1 = np.array([obs[x :: 64] for x in range(32)])
                obs_buffer1 = obs_buffer1.reshape(-1,64,64,3)
                
                obs_buffer2 = np.array([obs[x :: 64] for x in range(32,64)])
                obs_buffer2 = obs_buffer2.reshape(-1,64,64,3)
                
                obs = np.concatenate((obs_buffer1, obs_buffer2), axis = 0 )

                aug_obs = aug_func.do_augmentation(obs[:int(nbatch/2)].copy())
                aug_obs = np.concatenate((aug_obs, aug_func2.do_augmentation(obs[int(nbatch/2):].copy())), axis = 0)

                #plt.imshow(obs[8345])
                #plt.show()
            else:
                aug_obs = aug_func.do_augmentation(obs.copy())
        else:
            aug_obs = obs
        
                
                # plt.imshow(aug_obs[8345])
                # plt.show()
        # Returns = R + yV(s')
        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = np.mean(advs, axis = 0)
        if eval_env is not None:
            eval_epinfos = eval_runner.eval() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        gradient = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    '''
                    if  update*nbatch >= (checkpoint[check_index] * 1e6) and logger.get_dir() and is_mpi_root and (not saved_key_checkpoints[check_index]):
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        gradient.append(model.grad(lrnow, cliprangenow, *slices))
                    '''                      
                    slices = (arr[mbinds] for arr in (obs, aug_obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                   
        else: # recurrent version
            print("Not implenmented yet")

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            mean_reward = safemean([epinfo['r'] for epinfo in epinfobuf])
            
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            
            logger.logkv('advantage', advs)
            logger.logkv("misc/explained_variance", float(explained_variance(values, returns)))
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)
            logger.logkv('eprewmean', mean_reward)
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('stu_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('stu_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            
                
            '''    
            if  update*nbatch >= (checkpoint[check_index] * 1e6) and (not saved_key_checkpoints[check_index]):
                f_gradient.write("{}\n".format(lossvals[-1]))
                f_gradient.flush()  
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    if lossname =='loss':
                        f_loss.write("{}\n".format(lossval))
                        f_loss.flush()
                    elif lossname == 'impala_network_size':
                        f_netsize.write("{}\n".format(lossval))
                        f_netsize.flush()    
                    logger.logkv('loss/' + lossname, lossval)
            else:
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv('loss/' + lossname, lossval)
            '''
            logger.dumpkvs()

        if  update*nbatch >= ((checkpoint[check_index]) * 1e6) and logger.get_dir() and is_mpi_root and (not saved_key_checkpoints[check_index]):
            saved_key_checkpoints[check_index] = True
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            ind = checkpoint[check_index]
            savepath = osp.join(checkdir, '%i'%checkpoint[check_index]) if run_id == None else osp.join(checkdir, run_id+'%i'%ind)
            print('Saving to', savepath)
            model.save(savepath)
            check_index =  check_index + 1
            f_aug.write('Total timesteps:'+ str(update*nbatch))
            f_aug.write('\n'+' rccj, None, crop') 
            f_aug.write('\n'+str([a  for a in model.num_action])+'\n' )
            f_aug.flush()
            if args.save_obs:
                obs_dir = osp.join(logger.get_dir(), 'saved_obs')
                os.makedirs(obs_dir, exist_ok=True)
                obs_path = osp.join(obs_dir, run_id)
                np.save( obs_path, save_obs)
    '''
    f_gradient.close()
    f_loss.close()
    f_netsize.close()
    '''
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
