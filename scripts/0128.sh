#python -m procgen.ExDA --log_dir /data/kbc/procgen/heist/ --env_name heist --distribution_mode easybg --res_id try1_heist_easybg_ppo_20 --data_aug crop --run_id try1_heist_easybg_ExDA_crop_ --gpu 6
python -m procgen.ExDA --log_dir /data/kbc/procgen/coinrun/ --env_name coinrun --distribution_mode easybg --res_id try1_coinrun_easybg_200_ppo_20 --data_aug crop --run_id try1_coinrun_easybg_ExDA_crop_ --gpu 6
python -m procgen.ExDA --log_dir /data/kbc/procgen/coinrun/ --env_name chaser --distribution_mode easybg --res_id try1_chaser_easybg_200_ppo_20 --data_aug crop --run_id try1_chaser_easybg_ExDA_crop_ --gpu 6


