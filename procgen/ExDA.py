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
from procgen.runner import Runner_distill
import matplotlib.pyplot as plt
import torch
DIR_NAME = 'train_log'
def constfn(val):
    def f(_):
        return val
    return f
from procgen.model import ExDA   
    
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='climber')
    parser.add_argument('--distribution_mode', type=str, default='easybg', choices=["easy", "hard","easybg","easy-test", "easybg-test", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default= 64)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--nminibatches', type=int, default=4096)
    parser.add_argument('--buffer_size', type=int, default=491520)
    parser.add_argument('--aug_interval', type=int, default = 3)
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None) # save file name
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--res_id2', type=str, default=None) #load file name
    parser.add_argument('--log_dir', type=str, default='/home/kbc/pdad/')
    parser.add_argument('--epochs', type=int, default=10) # total_update
    parser.add_argument('--use_vdf', default = False , action = 'store_true') #value distance functinon
    parser.add_argument('--reinit', default = False , action = 'store_true')
    parser.add_argument('--saved_obs', default = False , action = 'store_true')
    parser.add_argument('--data_aug', type=str, default=None)
   
    
    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    lr_init = args.learning_rate
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
    logger.configure(dir=args.log_dir, format_strs=format_strs)

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
    
   
    policy = build_policy(env, network)
    
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space
    batch_aug = nenvs * nsteps 
    if args.saved_obs:
        obs_path = args.log_dir + 'saved_obs/' + args.res_id2 + '.npy'
        saved_obs = np.load(obs_path)
        num_saved_obs = saved_obs.shape[0]
        trash_obs = num_saved_obs % batch_aug
        saved_obs = saved_obs[0:-1 * trash_obs]
        num_saved_obs = saved_obs.shape[0]
        print(num_saved_obs)
        run_step = (args.buffer_size - num_saved_obs) // batch_aug
        total_run_step = args.buffer_size // batch_aug
    else:
        run_step = args.buffer_size  // batch_aug
        total_run_step = run_step

    nbatch = batch_aug * total_run_step
        
    nbatch_train = nbatch // nminibatches 
    
    epinfobuf = deque(maxlen=100)
    eval_epinfobuf = deque(maxlen=100)
    
    # Start total timer
    if args.data_aug is not None:       
        
        if args.data_aug == 'color_jitter':
            aug_func = rad.ColorJitterLayer(batch_aug, p_rand = 1)
        elif args.data_aug == 'gray':
            aug_func = rad.Grayscale(batch_aug, p_rand = 1)
        elif args.data_aug == 'flip':
            aug_func = rad.Rand_Flip(batch_aug , p_rand = 1)    
        elif args.data_aug == 'rotate':
            aug_func = rad.Rand_Rotate(batch_aug, p_rand = 1)    
        elif args.data_aug == 'crop':
            aug_func = rad.Crop(batch_aug, p_rand = 1)
        elif args.data_aug == 'cutout':
            aug_func = rad.Cutout(batch_aug, p_rand = 1)
        elif args.data_aug == 'cutout_color':
            aug_func = rad.Cutout_Color(batch_aug, p_rand = 1)
        elif args.data_aug == 'random_conv':
            aug_func = rad.RandomConv(batch_aug)
        elif args.data_aug == 'black':
            aug_func = rad.Black(batch_aug, p_rand = 1)
        else:
            pass
        nbatch = nbatch * 2
    
   
    model = ExDA(policy = policy, ob_space = ob_space,ac_space = ac_space, nbatch_act = nenvs, nbatch_train = nbatch_train, args = args, nsteps =  nsteps, mpi_rank_weight = mpi_rank_weight)
    if args.res_id is not None:
        load_path = args.log_dir + 'checkpoints/' + args.res_id
        model.load(load_path)
        
    tfirststart = time.perf_counter()
    runner = Runner_distill(env = env, model = model, nsteps = nsteps)
    eval_runner = Runner_distill(env = eval_env, model = model, nsteps = nsteps)
   
   
   
    for i in range(run_step):
        obs, epinfos = runner.run()
        if i == 0:
            if args.saved_obs:
                obs_buffer = np.concatenate((saved_obs,obs),axis = 0)
            else:
                obs_buffer = obs            
        else:
            obs_buffer = np.concatenate((obs_buffer,obs),axis = 0)
        epinfobuf.extend(epinfos)
        target_reward = safemean([epinfo['r'] for epinfo in epinfobuf])
        target_eplenmean  = safemean([epinfo['l'] for epinfo in epinfobuf])
    print("complete storing in buffer")     
    
    

    for i in range(nminibatches):
        if i == 0:
            policy_buffer = model.pi(obs_buffer[i*nbatch_train:(i+1)*nbatch_train])
        else:
            policy_buffer = np.concatenate((policy_buffer, model.pi(obs_buffer[i*nbatch_train:(i+1)*nbatch_train])),axis = 0)
            
    if args.reinit:
        model.initialize()
    if args.data_aug is not None:     
        if args.data_aug == 'color_jitter' or args.data_aug == 'gray':       
            for i  in range(total_run_step):
                if i == 0:
                    aug_obs = np.concatenate((obs_buffer, aug_func.do_augmentation(obs_buffer[i*batch_aug:(i+1)*batch_aug])),axis = 0)
                else:
                    aug_obs = np.concatenate((aug_obs, aug_func.do_augmentation(obs_buffer[i*batch_aug:(i+1)*batch_aug])),axis = 0)

    nupdates = int(args.epochs / args.aug_interval)
    print('nbatch')
    print(nbatch)
    print('obs_buffer')
    print(obs_buffer.shape[0])
    print('policy')
    print(policy_buffer.shape[0])
    current_epoch = 1


    for update in range(1, nupdates+1):
       
        frac = 1.0 - (update - 1.0) / nupdates
        lr = frac*lr_init
        assert nbatch % nminibatches == 0
        
        # Start timer
        tstart = time.perf_counter()
 
        # Calculate the learning rate
        
        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')
        
        # Get minibatch

            
        if args.data_aug is not None:    
            if args.data_aug != 'color_jitter' and args.data_aug != 'gray':           
                for i  in range(total_run_step):
                    if i == 0:
                        aug_obs = np.concatenate((obs_buffer, aug_func.do_augmentation(obs_buffer[i*batch_aug:(i+1)*batch_aug])),axis = 0)
                    else:
                        aug_obs = np.concatenate((aug_obs, aug_func.do_augmentation(obs_buffer[i*batch_aug:(i+1)*batch_aug])),axis = 0)
        print('aug_obs')
        print(aug_obs.shape[0])
        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        

        # Here what we're going to do is for each minibatch calculate the loss and append it.
         
        inds = np.arange(nbatch)
        for epoch in range(args.aug_interval):
            mblossvals = []
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            
            eval_epinfos = eval_runner.eval() #pylint: disable=E0632
            eval_epinfobuf.extend(eval_epinfos)
            
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                           
                if args.data_aug is None:
                    mblossvals.append(model.distill(lr, obs_buffer[mbinds], policy_buffer[mbinds]))
                else:                 
                    mblossvals.append(model.distill(lr, aug_obs[mbinds], policy_buffer[mbinds% buffer_size]))
                  
            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
    
            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
            
                logger.logkv('target_eprewmean', target_reward)
                logger.logkv('target_eplenmean', target_eplenmean)
                    
                logger.logkv('ExDA_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('ExDA_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
                logger.logkv("misc/nepochs", (update+epoch))
      
              
                for (lossval, lossname) in zip(lossvals, model.distill_loss_names):
                    logger.logkv('loss/' + lossname, lossval)
                logger.logkv('misc/time_elapsed', tnow - tfirststart)
                logger.dumpkvs()
    
            if  logger.get_dir() and is_mpi_root and update % args.save_interval == 0:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                
                savepath = osp.join(checkdir, '%i'%current_epoch) if run_id == None else osp.join(checkdir, run_id+'%i'%current_epoch)
                print('Saving to', savepath)
                model.save(savepath)  
                current_epoch += 1
    
   
    return model
    
                         



if __name__ == '__main__':
    main()
