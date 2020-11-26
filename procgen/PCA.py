import tensorflow as tf
import functools
import numpy as np
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen.policy_new import build_policy_reset
from procgen.tf_util import get_session, save_variables, load_variables, load_variables_a2c
from procgen import ProcgenEnv
from procgen.policy_new import build_policy, build_policy_multiple_sample, build_policy_reset
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from sklearn.decomposition import PCA

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
    
          
    ######################################################################################
    set_global_seeds(None)
    
    
    
    # Get the nb of env
    nenvs = venv.num_envs

    
    load = functools.partial(load_variables, sess=sess)
    load_a2c = functools.partial(load_variables_a2c, sess=sess)
    policy = build_policy(venv, conv_fn)
    
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        agent = policy(nenvs, 1, sess)
            
    
    obs = venv.reset()
    state = agent.initial_state
    done = np.zeros(nenvs)
    
    ppo_steps = range(0,21,1)
    
    load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try3_coinrun_ppo_level10_' + '10'
    load(load_path)
    
    
    for _ in range(20):
        actions,_,_,_ = agent.step(obs, S=state, M=done)
        obs, rew, done, info = venv.step(actions)
    
    softout, out = agent.get_softmax(obs, S=state, M=done)
    data = out
    data_label = []

    for i in range(len(ppo_steps)) :
        load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try3_coinrun_ppo_level10_' + str(ppo_steps[i])
        load(load_path)

      
        _ , out = agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    data = data[1:]
    
    
    
    
    new_policy = build_policy(venv, conv_fn)
    with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
        new_agent = new_policy(nenvs, 1, sess)
            
    
    a2c_steps = range(0,21,1)

    for i in range(len(a2c_steps)) :
        load_path = '/home/kbc/imp/procgen_ppo/checkpoints/try3_coinrun_a2c_level10_' + str(a2c_steps[i])
        load_a2c(load_path)
        _ , out = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)

    
    '''
    policy_re = build_policy_reset(venv, conv_fn)
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        new_agent = policy_re(nenvs, 1, sess)
    re1_steps = range(0,202,4)
    
    for i in range(len(re1_steps)) :
        load_path = '/home/postech/procgen/checkpoints/try1_new_KL_epoch_3_level_50_step_300_' + str(re1_steps[i])
        load(load_path)
        _ , out = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    re2_steps = range(0,192,4)    
    for i in range(len(re2_steps)) :
        load_path = '/home/postech/procgen/checkpoints/try1_new_L2_epoch_3_level_50_step_300_' + str(re2_steps[i])
        load(load_path)
        _ , out = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    
    
    
    policy_multiple_sample = build_policy_multiple_sample(venv, conv_fn)
    with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
        new_agent = policy_multiple_sample(nenvs, 1, sess)
    re_steps = range(0,52,5)
    for i in range(len(re_steps)) :
        load_path = '/home/postech/procgen/checkpoints/try1_KL_shuffle_reinit_epoch_3_level_50_step_300_' + str(re_steps[i])
        load(load_path)
        out, _ = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
        
    for i in range(len(re_steps)) :
        load_path = '/home/postech/procgen/checkpoints/try1_KL_shuffle_init_20_epoch_3_level_50_step_300_' + str(re_steps[i])
        load(load_path)
        out, _ = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
        
    for i in range(len(re_steps)) :
        load_path = '/home/postech/procgen/checkpoints/try1_KL_shuffle_init_60_epoch_3_level_50_step_300_' + str(re_steps[i])
        load(load_path)
        out, _ = new_agent.get_softmax(obs, S=state, M=done)
        data  = np.concatenate((data, out),axis = 0)
        data_label.append(i)
    
    print(data[75])
    print(data[86])
    print(data[97])
    print(data[108])
    '''
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    #tsne = TSNE(perplexity = 10,  early_exaggeration = 12, random_state=0)
    #digits_tsne = tsne.fit_transform(data)
    colors = ['#476A2A']* len(ppo_steps) + ['#BD3430'] * len(a2c_steps)
    #colors = ['#476A2A']* len(ppo_steps) + ['#7851B8'] * len(re_steps) + ['#BD3430'] * len(re_steps) + ['#4A2D4E'] * len(re_steps)
    #colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
    #               '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
    
    # 시각화
    # 0부터  digits.data까지 정수
    # x, y 
    for i in range(len(pca_data)): 
        plt.text(pca_data[i, 0], pca_data[i, 1], str(data_label[i]), 
                 color=colors[i], 
                 fontdict={'weight': 'bold', 'size':9}) 
    
    plt.xlim(pca_data[:, 0].min(), pca_data[:, 0].max()) # 최소, 최대
    plt.ylim(pca_data[:, 1].min(), pca_data[:, 1].max()) # 최소, 최대
    plt.xlabel('PCA 0') # x축 이름
    plt.ylabel('PCA 1') # y축 이름
    plt.show() # 그래프 출력


    



        
if __name__ == '__main__':
    main()