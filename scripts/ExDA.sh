#python -m procgen.ExDA_half --log_dir /data/kbc/procgen/coinrun/ --env_name fruitbot --distribution_mode easybg --res_id try1_fruitbot_easybg_200_ppo_20 --data_aug random_conv --data_aug2 color_jitter --run_id try1_fruitbot_easybg_ExDA_rccj_ --gpu 7
#python -m procgen.ExDA_half --log_dir /data/kbc/procgen/coinrun/ --env_name fruitbot --distribution_mode easybg --res_id try2_fruitbot_easybg_200_ppo_20 --data_aug random_conv --data_aug2 color_jitter --run_id try2_fruitbot_easybg_ExDA_rccj_ --gpu 7
#python -m procgen.ExDA_half --log_dir /data/kbc/procgen/coinrun/ --env_name fruitbot --distribution_mode easybg --res_id try3_fruitbot_easybg_200_ppo_20 --data_aug random_conv --data_aug2 color_jitter --run_id try3_fruitbot_easybg_ExDA_rccj_ --gpu 7



# python -m procgen.ExDA --log_dir /data/kbc/procgen/heist/ --env_name heist --distribution_mode easybg --res_id try1_heist_easybg_ppo_25 --data_aug black --run_id try1_heist_easybg_ExDA_black_ --gpu 7
# python -m procgen.ExDA --log_dir /data/kbc/procgen/heist/ --env_name heist --distribution_mode easybg --res_id try2_heist_easybg_ppo_25 --data_aug black --run_id try2_heist_easybg_ExDA_black_ --gpu 7
# python -m procgen.ExDA --log_dir /data/kbc/procgen/heist/ --env_name heist --distribution_mode easybg --res_id try3_heist_easybg_ppo_25 --data_aug black --run_id try3_heist_easybg_ExDA_black_ --gpu 7


python -m procgen.ExDA --log_dir /data/kbc/procgen/coinrun/ --env_name chaser --distribution_mode easybg --res_id try1_chaser_easybg_200_ppo_20 --data_aug black --run_id try1_chaser_easybg_ExDA_black_ --gpu 6
python -m procgen.ExDA --log_dir /data/kbc/procgen/coinrun/ --env_name chaser --distribution_mode easybg --res_id try2_chaser_easybg_200_ppo_20 --data_aug black --run_id try2_chaser_easybg_ExDA_black_ --gpu 6
python -m procgen.ExDA --log_dir /data/kbc/procgen/coinrun/ --env_name chaser --distribution_mode easybg --res_id try3_chaser_easybg_200_ppo_20 --data_aug black --run_id try3_chaser_easybg_ExDA_black_ --gpu 6


for temp_index in try1_ try2_ try3_
    do
        for agent_index in chaser_easybg_ExDA_black_
            do
                for file_index in 0 1 2 3 4 5 6 7 8 9 # 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 --gpu 6 #--use_record
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode easybg_test --res_id $realfilename --run_id $fn1 --gpu 6
                        python -m procgen.eval --env_name chaser --num_levels 0 --start_level 200 --log_dir /data/kbc/procgen/coinrun/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2 --gpu 6
                        
                    done
            done
    done
