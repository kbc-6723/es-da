import tensorflow as tf
from procgen import ppo2_image
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
import os
LOG_DIR = '/home/kbc/imp/procgen/' #checkpoint location

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
    timesteps_per_proc = 101_000_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard","easy_plus", "hard_minus", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=10)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None) # save file name
    parser.add_argument('--res_id', type=str, default=None) #load file name

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")
    
    venv1 = ProcgenEnv(num_envs=num_envs, env_name='bossfight', num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv1 = VecExtractDictObs(venv1, "rgb")

    venv1 = VecMonitor(
        venv=venv1, filename=None, keep_buf=100,
    )
    venv1 = VecNormalize(venv1, ob=False)
    
    venv2 = ProcgenEnv(num_envs=num_envs, env_name='caveflyer', num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv2 = VecExtractDictObs(venv2, "rgb")

    venv2 = VecMonitor(
        venv=venv2, filename=None, keep_buf=100,
    )
    venv2 = VecNormalize(venv2, ob=False)
    
    venv3 = ProcgenEnv(num_envs=num_envs, env_name='chaser', num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv3 = VecExtractDictObs(venv3, "rgb")

    venv3 = VecMonitor(
        venv=venv3, filename=None, keep_buf=100,
    )
    venv3 = VecNormalize(venv3, ob=False)
    
    
    
    
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2_image.learn(
        env1=venv1,
        env2=venv2,
        env3=venv3,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=1,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        id=args.run_id,
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
