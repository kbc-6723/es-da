import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from procgen import ppo2
from procgen.nets import build_impala_cnn, build_random_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common import set_global_seeds
from procgen import ProcgenEnv
from procgen.policies import build_policy
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
    VecVideoRecorder,
)
from gym3 import  VideoRecorderWrapper

from skimage import io
from matplotlib import pyplot as plt
import procgen.data_augs as rad
import scipy as sp
import numpy as np
from baselines import logger
from mpi4py import MPI
import argparse
import os
from procgen.tf_util import load_variables

    
DIR_NAME = 'test_log'
def normalize(x):
    x  = np.asarray(x)
    norm = (x - np.min(x)) / (np.max(x) - np.min(x)) 
    return norm

def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np
    x = []
    for ix, p_ in enumerate(p):
        q_ = q[ix]
        m = 1./2*(p_ + q_)
        x.append(sp.stats.entropy(p_,m, base=base,axis = -1)/2. +  sp.stats.entropy(q_, m, base=base, axis = -1)/2.)

    return x

def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "easybg","easybg_test","hardbg","hard","testbg", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--num_envs', type=int, default=50)
    parser.add_argument('--rep_count', type=int, default=1)
    parser.add_argument('--learning_rate', type=int, default=5e-4)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None) # save file name
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--log_dir', type=str, default='/home/kbc/pdad/')
    parser.add_argument('--use_randconv', default = False , action = 'store_true')
    parser.add_argument('--use_record', default = False , action = 'store_true')
    parser.add_argument('--use_drac', default = False , action = 'store_true')
    parser.add_argument('--use_rad', default = False , action = 'store_true')
    parser.add_argument('--jsd', default = False , action = 'store_true')
    parser.add_argument('--data_aug', type=str, default=None)
    
    
    args = parser.parse_args()
    rep_count = args.rep_count
    test_worker_interval = args.test_worker_interval  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=args.log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels,use_sequential_levels=False, start_level=args.start_level, distribution_mode=args.distribution_mode)
    
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv, ob=False)
    env = venv

    if args.use_record:
        if not os.path.exists('record'):
            os.makedirs('record')
        record_dir = 'record/' +  args.res_id
        env = VideoRecorderWrapper(
            env=env, directory=record_dir, ob_key=ob_key, info_key=info_key
        )
        #env = VecVideoRecorder(env, record_dir, record_video_trigger=lambda x: x % 1 == 0, video_length=500)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()
    if args.use_randconv:
        network = lambda x: build_random_impala_cnn(x, depths=[16,32,32], emb_size=256) 
    elif args.use_rad and args.data_aug == 'random_conv':
        network = lambda x: build_random_impala_cnn(x, depths=[16,32,32], emb_size=256)       
    else:  
        network = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)    
     
    logger.info("evaluation")
    
    set_global_seeds(None)
    
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    file_name = '%s/%s.txt'%(DIR_NAME, args.run_id)
    
    f_io = open(file_name, 'a')
    
    if args.jsd:
        f_jsd = open('%s/%s.txt'%(DIR_NAME, args.run_id+'_jsd'), 'w')
    nbatch_train = args.num_envs
    if args.data_aug is not None:
        if args.data_aug == 'color_jitter':
            aug_func = rad.ColorJitterLayer(nbatch_train, p_rand = 1)
        elif args.data_aug == 'gray':
            aug_func = rad.RandGray(nbatch_train, p_rand = 1)
        elif args.data_aug == 'flip':
            aug_func = rad.Rand_Flip(nbatch_train, p_rand = 1)    
        elif args.data_aug == 'rotate':
            aug_func = rad.Rand_Rotate(nbatch_train, p_rand = 1)    
        elif args.data_aug == 'crop':
            aug_func = rad.Crop(nbatch_train, p_rand = 1)
        elif args.data_aug == 'cutout':
            aug_func = rad.Cutout(nbatch_train, p_rand = 1)
        elif args.data_aug == 'cutout_color':
            aug_func = rad.Cutout_Color(nbatch_train, p_rand = 1)
        elif args.data_aug == 'random_conv':
            aug_func = rad.RandomConv(nbatch_train)
        elif args.data_aug == 'conv':
            aug_func = rad.Convfilter(nbatch_train)
        else:
            pass

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates act_model and train_model)
       
    policy = build_policy(env, network)
      
    
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        agent = policy(nenvs,1,sess)
            
    num_actions = env.action_space.n
  
    
    load_path = args.log_dir + 'checkpoints/' + args.res_id
    load_variables(load_path)
    
    
    obs = env.reset()
    t_step = 0     
        
    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        return np.sum(score_counts) < rep_count * nenvs


    state = agent.initial_state
    done = np.zeros(nenvs)
    
    jsd_list = []
    
    while should_continue():        
                    
        actions, _ , _ = agent.step_eval(obs, S=state, M=done) 
        if args.jsd:
            pi,_ = agent.get_softmax(obs)
            aug_obs = aug_func.do_augmentation(obs)
            aug_pi, _ = agent.get_softmax(aug_obs)
            print(pi)
            print(aug_pi)
            jsd_list.append(jsd(pi, aug_pi))
            print(jsd_list[-1])
            
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
    
    if args.jsd: 
        f_jsd.write(str(np.mean(np.array(jsd_list))))
    
    mean_score = np.mean(scores) / rep_count
    max_idx = np.argmax(scores)

    result = mean_score
    
    f_io.write("{}\n".format(result))    
    f_io.close()
    
    return result

if __name__ == '__main__':
    main()
