import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from procgen import ppo2, ucb_drac
from procgen.nets import build_impala_cnn, build_random_impala_cnn
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



def main():
    ent_coef = .01
    gamma = .999
    lam = .95
    ppo_epochs = 3
    clip_range = .2
    
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easybg', choices=["easy", "hard","easybg","easy-test", "easybg-test", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--learning_rate', type=int, default=5e-4)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None) # save file name
    parser.add_argument('--res_id', type=str, default=None) #load file name
    parser.add_argument('--log_dir', type=str, default= None)
    parser.add_argument('--timesteps', type=int, default=25) # total_timesteps
    parser.add_argument('--ucb_coef', type=float, default=1.5)
    parser.add_argument('--use_rand_FM', default = False , action = 'store_true')
    parser.add_argument('--use_drac', default = False , action = 'store_true')
    parser.add_argument('--use_pcgrad', default = False , action = 'store_true')
    parser.add_argument('--use_pdtgrad', default = False , action = 'store_true')
    parser.add_argument('--use_rad', default = False , action = 'store_true')
    parser.add_argument('--save_obs', default = False , action = 'store_true')
    parser.add_argument('--ucb_drac', default = False , action = 'store_true')
    parser.add_argument('--only_distill', default = False , action = 'store_true')
    parser.add_argument('--data_aug', type=str, default=None)
    parser.add_argument('--data_aug2', type=str, default=None)
    args = parser.parse_args()
    timesteps_per_proc = (args.timesteps+1) * 1000000
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
    eval_venv = None
    
    
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.33
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()
    
    if args.use_rand_FM:
        conv_fn = lambda x: build_random_impala_cnn(x, depths=[16,32,32], emb_size=256)
    elif args.use_rad and args.data_aug == 'random_conv':
        conv_fn = lambda x: build_random_impala_cnn(x, depths=[16,32,32], emb_size=256)
    else:     
        conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256) 
    
    if args.use_rand_FM:
        stu_conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    else:
        stu_conv_fn = None
        
    logger.info("training")
    if args.ucb_drac:
        ucb_drac.learn(
            env=venv,
            eval_env = eval_venv,
            network=conv_fn,
            stu_network = stu_conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=1,
            nsteps=args.nsteps,
            nminibatches=args.nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            run_id=args.run_id,
            ucb_coef = args.ucb_coef,
            comm=comm,
            load_path=args.res_id ,
            lr=args.learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            args = args
        )
    else:
        ppo2.learn(
            env=venv,
            eval_env = eval_venv,
            network=conv_fn,
            stu_network = stu_conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=1,
            nsteps=args.nsteps,
            nminibatches=args.nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            run_id=args.run_id,
            comm=comm,
            load_path=args.res_id ,
            lr=args.learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            args = args
        )

if __name__ == '__main__':
    main()
