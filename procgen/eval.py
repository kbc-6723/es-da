import tensorflow as tf
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
)
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

def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "easybg","hardbg","hard","testbg", "exploration", "memory", "extreme"])
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
    parser.add_argument('--data_aug', type=str, default=None)
    
    
    args = parser.parse_args()
    rep_count = args.rep_count
    test_worker_interval = args.test_worker_interval
    use_gradcam = args.use_gradcam   
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
        record_dir = 'record/' +  args.run_id
        env = VecVideoRecorder(env, record_dir, record_video_trigger=lambda x: x % 1 == 0, video_length=500)

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
    
    
    while should_continue():        
                    
        actions, pi , v = agent.step_eval(obs, S=state, M=done) 
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

    
    mean_score = np.mean(scores) / rep_count
    max_idx = np.argmax(scores)

    result = mean_score
    
    f_io.write("{}\n".format(result))    
    f_io.close()
    
    return result

if __name__ == '__main__':
    main()
