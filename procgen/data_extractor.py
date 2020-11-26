import tensorflow as tf
import functools
import numpy as np
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen.policy_new import build_policy_reset
from procgen.tf_util import get_session, save_variables, load_variables
from procgen import ProcgenEnv
from procgen.policy_new import build_policy, build_policy_multiple_sample
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from sklearn.manifold import TSNE
from baselines.common import explained_variance, set_global_seeds
from baselines import logger
from mpi4py import MPI
import argparse
import os
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
LOG_DIR = '/home/kbc/imp/procgen_ppo/' #checkpoint location

def main():


    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "easy_plus", "hard_minus", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default= 1)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--gpu', type=str, default= '0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    log_comm = comm.Split(0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")
 
    venv = ProcgenEnv(num_envs=1, env_name=args.env_name, num_levels=1, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv,ob=False)

    
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    
    logger.info("training")
    
          
    #######################################################################################   
    set_global_seeds(None)
    
    
    
    # Get the nb of env
    nenvs = venv.num_envs

    
    load = functools.partial(load_variables, sess=sess)
    policy = build_policy(venv, conv_fn)
    
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        agent = policy(nenvs, 1, sess)
            
  
    obs = venv.reset()
    state = agent.initial_state
    done = np.zeros(nenvs)
    
    ppo_steps = range()
    
    load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try1_pre_epoch_3_level_50_step_300_' + '30'
    load(load_path)
    
    for _ in range(20):
        actions,_,_,_ = agent.step(obs, S=state, M=done)
        obs, rew, done, info = venv.step(actions)
    
    softout, out = agent.get_softmax(obs, S=state, M=done)
    data = out
    data_label = []

    for i in range(len(ppo_steps)) :
        load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try6_bigfish_pre_level_20_' + str(ppo_steps[i])
        load(load_path)
        softout, out = agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    data = data[1:]
    
    
    policy_multiple_sample = build_policy_multiple_sample(venv, conv_fn)
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        new_agent = policy_multiple_sample(nenvs, 1, sess)
    re_steps = [0,5,10,15,20,25,30,35,40,45,50,55, 60,65,70,75,80,85,90,95,100]
    
    for i in range(len(ppo_steps)) :
        load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try1_KL_shuffle_reinit_epoch_3_level_50_step_300_' + str(re_steps[i])
        load(load_path)
        softout, out = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    
    
  
    tsne = TSNE(perplexity = 6,  early_exaggeration = 15, random_state=0)
    digits_tsne = tsne.fit_transform(data)
    
    colors = 
    #colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
    #               '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
    
    # 시각화
    # 0부터  digits.data까지 정수
    # x, y 
    for i in range(len(data)): 
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(data_label[i]), 
                 color=colors[i], 
                 fontdict={'weight': 'bold', 'size':9}) 
                 
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()) # 최소, 최대
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()) # 최소, 최대
    plt.xlabel('t-SNE 0') # x축 이름
    plt.ylabel('t-SNE 1') # y축 이름
    plt.show() # 그래프 출력


    



        
if __name__ == '__main__':
    main()