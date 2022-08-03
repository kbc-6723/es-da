import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from procgen import ppo2
from procgen.nets import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from procgen.tf_util import initialize
from baselines import logger
from mpi4py import MPI
import argparse
import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
import procgen.data_augs as rad
from baselines.common import explained_variance, set_global_seeds
from procgen.policies import build_policy
from procgen.runner import Runner_distill, Runner
import matplotlib.pyplot as plt
import torch
DIR_NAME = 'train_log'
def constfn(val):
    def f(_):
        return val
    return f
from procgen.model import InDA  

def data_augmentation(obs, aug_func):
    obs = aug_func.do_augmentation(obs)
    return obs
    
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easybg', choices=["easy", "hard","easybg","easy-test", "easybg-test", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--rl_learning_rate', type=float, default=5e-4)
    parser.add_argument('--distill_learning_rate', type=float, default=1e-4)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--distill_epochs', type=int, default = 3)
    parser.add_argument('--ppo_epochs', type=int, default = 3)
    parser.add_argument('--gpu', type=str, default= '0')
    
    parser.add_argument('--DA_interval', type=int, default= 5)# interval time step of DA is nenvs*nsteps*DA_interval
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None) # save file name
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--log_dir', type=str, default= None)
    parser.add_argument('--begin_timestep', type=int, default=0) # beginning time of training when using load
    parser.add_argument('--timesteps', type=int, default=25) # total_timesteps
    parser.add_argument('--start_DA_timesteps', type=int, default=0) # start time of DA
    parser.add_argument('--terminal_DA_timesteps', type=int, default=26) # terminal time of DA
    parser.add_argument('--reinit', default = False , action = 'store_true')
    parser.add_argument('--buffer_size', type=int, default= 40960) # nenvs*nsteps*c
    parser.add_argument('--data_aug', type=str, default=None)
    parser.add_argument('--data_aug2', type=str, default=None)
    parser.add_argument('--save_obs', default = False , action = 'store_true')
    parser.add_argument('--use_pdtgrad', default = False , action = 'store_true')
    args = parser.parse_args()
    total_timesteps = (args.timesteps+1) * 1000000
    test_worker_interval = args.test_worker_interval
    DA_interval = args.DA_interval
    ent_coef = .01
    vf_coef=0.5
    max_grad_norm=0.5
    gamma = .999
    lam = .95
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    distill_lr = args.distill_learning_rate
    rl_lr = args.rl_learning_rate
    log_interval = args.log_interval
    nsteps = args.nsteps
    nminibatches = args.nminibatches
    buffer_size = args.buffer_size
    is_test_worker = False
    run_id = args.run_id
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)
    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir='/data/kbc/procgen/coinrun/', format_strs=format_strs)

    logger.info("creating environment")
    env = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    
    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(
        venv=env, filename=None, keep_buf=100,
    )
    env = VecNormalize(env, ob=False)
    
    
    
    eval_env = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    
    eval_env = VecExtractDictObs(eval_env, "rgb")

    eval_env = VecMonitor(
        venv=eval_env, filename=None, keep_buf=100,
    )
    eval_env = VecNormalize(eval_env, ob=False)
    
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()
    
    network = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    
    set_global_seeds(None)

    
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    
  
    
    file_name_loss = '%s/%s.txt'%(DIR_NAME, args.run_id + "loss")
    f_loss = open(file_name_loss, 'a')
   
    policy = build_policy(env, network)
    
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

  
    nbatch = nenvs * nsteps
        
    nbatch_train = nbatch // nminibatches 
    
    epinfobuf = deque(maxlen=100)
    eval_epinfobuf = deque(maxlen=100)
    
    # Start total timer
    if args.data_aug is not None:       
        
        if args.data_aug == 'color_jitter':
            aug_func = rad.ColorJitterLayer(nbatch, p_rand = 1)
        elif args.data_aug == 'gray':
            aug_func = rad.Grayscale(nbatch, p_rand = 1)
        elif args.data_aug == 'flip':
            aug_func = rad.Rand_Flip(nbatch , p_rand = 1)    
        elif args.data_aug == 'rotate':
            aug_func = rad.Rand_Rotate(nbatch, p_rand = 1)    
        elif args.data_aug == 'crop':
            aug_func = rad.Crop(nbatch, p_rand = 1)
        elif args.data_aug == 'cutout':
            aug_func = rad.Cutout(nbatch, p_rand = 1)
        elif args.data_aug == 'cutout_color':
            aug_func = rad.Cutout_Color(nbatch, p_rand = 1)
        elif args.data_aug == 'random_conv':
            aug_func = rad.RandomConv(nbatch)
        elif args.data_aug == 'rccj':
            aug_func1_1 = rad.ColorJitterLayer(int(nbatch / 2), p_rand = 1)
            aug_func1_2 = rad.RandomConv(int(nbatch / 2))
        elif args.data_aug == 'black':
            aug_func = rad.Black(nbatch, p_rand = 1)
        else:
            pass

    if args.data_aug2 is not None:       
        if args.data_aug2 == 'color_jitter':
            aug_func2 = rad.ColorJitterLayer(nbatch, p_rand = 1)
        elif args.data_aug2 == 'gray':
            aug_func2 = rad.Grayscale(nbatch, p_rand = 1)
        elif args.data_aug2 == 'flip':
            aug_func2 = rad.Rand_Flip(nbatch , p_rand = 1)    
        elif args.data_aug2 == 'rotate':
            aug_func2 = rad.Rand_Rotate(nbatch, p_rand = 1)    
        elif args.data_aug2 == 'crop':
            aug_func2 = rad.Crop(nbatch, p_rand = 1)
        elif args.data_aug2 == 'cutout':
            aug_func2 = rad.Cutout(nbatch, p_rand = 1)
        elif args.data_aug2 == 'cutout_color':
            aug_func2 = rad.Cutout_Color(nbatch, p_rand = 1)
        elif args.data_aug2 == 'random_conv':
            aug_func2 = rad.RandomConv(nbatch)
        elif args.data_aug2 == 'rccj':
            aug_func2_1 = rad.ColorJitterLayer(int(nbatch / 2), p_rand = 1)
            aug_func2_2 = rad.RandomConv(int(nbatch / 2))
        else:
            pass

    model = InDA(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, args = args,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)
    
    if args.res_id is not None:
        load_path = args.log_dir + 'checkpoints/' + args.res_id
        model.load(load_path)
        
    tfirststart = time.perf_counter()
    runner = Runner(env = env, model = model, nsteps = nsteps, gamma = gamma, lam = lam)
    eval_runner = Runner_distill(env = eval_env, model = model, nsteps = nsteps)
   
    checkpoint = list(range(args.begin_timestep,args.timesteps + 1,1))
    check_index = 0
    saved_key_checkpoints = [False] * len(checkpoint)
    nupdates = total_timesteps//nbatch   
    DA_time = False
    obs_buffer = None
    buffer_step = int(buffer_size / nbatch)

    for update in range(1, nupdates):
    
        assert nbatch % nminibatches == 0
            
        
        # Start timer
        tstart = time.perf_counter()
        cliprange = 0.2

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')
        
        # Get minibatch
       
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        if args.save_obs:
            save_idx = np.random.choice(nbatch, 100, replace = False)
            if update == 1:
                save_obs = obs[save_idx]
            else:
                save_obs = np.concatenate((save_obs, obs[save_idx]), axis = 0)
        
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
            for _ in range(args.ppo_epochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(rl_lr, cliprange, *slices))
        else: # recurrent version
            print("Not implenmented yet")
            
        if DA_time == False:
            if args.start_DA_timesteps == 0:
                DA_time = True
            else:
                if checkpoint[check_index] > args.start_DA_timesteps and update % args.DA_interval == args.DA_interval - buffer_step + 1:
                    DA_time = True
        else:
            if checkpoint[check_index] > args.terminal_DA_timesteps:
                DA_time = False
                 
        if DA_time:            
            if update % DA_interval == 0:
                if buffer_step == 1:
                    obs_buffer = obs
                else:
                    obs_buffer = np.concatenate((obs_buffer, obs),axis = 0)
            elif update % DA_interval == DA_interval - buffer_step + 1:
                obs_buffer = obs
                
            elif DA_interval - buffer_step + 1 < update % DA_interval and update % DA_interval < DA_interval:
                obs_buffer = np.concatenate((obs_buffer, obs),axis = 0)
               
                
        
            if update % DA_interval == 0 :
                if args.reinit:
                    initialize()
                if args.data_aug == 'rccj':
                    obs_buffer1 = np.array([obs_buffer[x :: 32] for x in range(16)])
                    obs_buffer1 = obs_buffer1.reshape(-1,64,64,3)
                    
                    obs_buffer2 = np.array([obs_buffer[x :: 32] for x in range(16,32)])
                    obs_buffer2 = obs_buffer2.reshape(-1,64,64,3)
                    
                    obs_buffer = np.concatenate((obs_buffer1, obs_buffer2), axis = 0 )
                for i in range(buffer_step * nminibatches):
                    if i == 0:
          
                        policy_buffer = model.pi(obs_buffer[nbatch_train*i:nbatch_train*(i+1)])
                        value_buffer = model.vf(obs_buffer[nbatch_train*i:nbatch_train*(i+1)])
                    else:                      
                        policy_buffer = np.concatenate((policy_buffer, model.pi(obs_buffer[nbatch_train*i:nbatch_train*(i+1)])), axis = 0)
                        value_buffer = np.concatenate((value_buffer, model.vf(obs_buffer[nbatch_train*i:nbatch_train*(i+1)])), axis = 0)    
                        
                if args.data_aug is not None:                  
                    if args.data_aug == 'rccj':
                        for i  in range(buffer_step * 2):
                            if i < 5:
                                obs_buffer = np.concatenate((obs_buffer, aug_func1_1.do_augmentation(obs_buffer[i*int(nbatch / 2):(i+1)*int(nbatch / 2)])),axis = 0)
                            else:
                                obs_buffer = np.concatenate((obs_buffer, aug_func1_2.do_augmentation(obs_buffer[i*int(nbatch / 2):(i+1)*int(nbatch / 2)])),axis = 0) 
                    else:
                        for i  in range(buffer_step):
                            obs_buffer = np.concatenate((obs_buffer, aug_func.do_augmentation(obs_buffer[i*nbatch:(i+1)*nbatch])),axis = 0)
                if args.data_aug2 is not None:
                    if args.data_aug2 == 'rccj':
                        for i  in range(buffer_step * 2):
                            if i < 5:
                                obs_buffer = np.concatenate((obs_buffer, aug_func2_1.do_augmentation(obs_buffer[i*int(nbatch / 2):(i+1)*int(nbatch / 2)])),axis = 0)
                            else:
                                obs_buffer = np.concatenate((obs_buffer, aug_func2_2.do_augmentation(obs_buffer[i*int(nbatch / 2):(i+1)*int(nbatch / 2)])),axis = 0) 
                    else:
                        for i  in range(buffer_step):
                            obs_buffer = np.concatenate((obs_buffer, aug_func2.do_augmentation(obs_buffer[i*nbatch:(i+1)*nbatch])),axis = 0)

                # Here what we're going to do is for each minibatch calculate the loss and append it.
                distill_mblossvals = []
                distill_batch = obs_buffer.shape[0]
              
                inds = np.arange(distill_batch)
              
                    
                for _ in range(args.distill_epochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, distill_batch , nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        if args.data_aug is None:
                            slices = (arr[mbinds] for arr in (obs_buffer,policy_buffer,value_buffer ))
                            distill_mblossvals.append(model.distill(distill_lr, *slices))
                        else:                    
                            slices = (arr[mbinds % buffer_size] for arr in (policy_buffer, value_buffer))
                            distill_mblossvals.append(model.distill(distill_lr, obs_buffer[mbinds], *slices))
        
                distill_lossvals = np.array(distill_mblossvals)
                distill_lossvals = distill_lossvals[-1]
                eval_epinfos = eval_runner.eval() #pylint: disable=E0632         
                eval_epinfobuf.extend(eval_epinfos)
            
            
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        
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
            if update % DA_interval == 0 and DA_time:
                logger.logkv('DA_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('DA_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
                for (lossval, lossname) in zip(distill_lossvals, model.distill_loss_names):
                    logger.logkv('loss/' + lossname, lossval)
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            
            logger.dumpkvs()

        if  update*nbatch >= ((checkpoint[check_index] - args.begin_timestep) * 1e6) and logger.get_dir() and is_mpi_root and (not saved_key_checkpoints[check_index]):
            saved_key_checkpoints[check_index] = True
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            ind = checkpoint[check_index]
            savepath = osp.join(checkdir, '%i'%checkpoint[check_index]) if run_id == None else osp.join(checkdir, run_id+'%i'%ind)
            print('Saving to', savepath)
            model.save(savepath)
            check_index =  check_index + 1

            if args.save_obs:
                obs_dir = osp.join(logger.get_dir(), 'saved_obs')
                os.makedirs(obs_dir, exist_ok=True)
                obs_path = osp.join(obs_dir, run_id)
                np.save( obs_path, save_obs)
                  
    return model
    
                         



if __name__ == '__main__':
    main()
