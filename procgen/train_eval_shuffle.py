import tensorflow as tf
import os
from procgen import ppo2_eval_shuffle
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse


LOG_DIR = '/home/kbc/imp/procgen_ppo/' #checkpoint location

def main():
    num_envs = 64
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    timesteps_per_proc = 30_020_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--run_id', type=str, default=None) #store file name
    parser.add_argument('--eval', type=int, default=None)  # train:0 test:1
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    if args.eval == 4:
        rep_count = 3
        num_eval = 500
        num_levels = 0
        distribution_mode = 'hard'
        start_level = 0
    elif args.eval == 1:
        rep_count = 3
        num_eval = 500
        num_levels = 0
        distribution_mode = 'easy'
        start_level = 0
    elif args.eval == 2:
        rep_count = 3
        num_eval = 500
        num_levels = 0
        distribution_mode = 'easy_plus'
        start_level = 0
    elif args.eval == 3:
        rep_count = 3
        num_eval = 500
        num_levels = 0
        distribution_mode = 'hard_minus'
        start_level = 0
    else:
        rep_count = 1
        num_levels = 20
        num_eval = 20
        start_level = 0
        distribution_mode = 'easy'
    comm = MPI.COMM_WORLD
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_eval, env_name=args.env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    
    
        
    logger.info("evaluating")
    ppo2_eval_shuffle.learn(
        filename = args.run_id,
        env=venv,
        rep_count=rep_count,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=1,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        eval = args.eval,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        clip_vf=use_vf_clipping,
        comm=comm,
        load_path=args.res_id ,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    

if __name__ == '__main__':
    main()
