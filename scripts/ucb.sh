python -m procgen.UCB_InDA --env_name chaser --buffer_size 40960 --pd_interval 5 --distribution_mode easybg --gpu 4 --run_id try1_chaser_easybg_ucb_inda_black_ --log_dir /data/kbc/procgen/coinrun 
python -m procgen.UCB_InDA --env_name chaser --buffer_size 40960 --pd_interval 5 --distribution_mode easybg --gpu 4 --run_id try2_chaser_easybg_ucb_inda_black_ --log_dir /data/kbc/procgen/coinrun 
python -m procgen.UCB_InDA --env_name chaser --buffer_size 40960 --pd_interval 5 --distribution_mode easybg --gpu 4 --run_id try3_chaser_easybg_ucb_inda_black_ --log_dir /data/kbc/procgen/coinrun 


for temp_index in try1_ try2_ try3_ #try2_
    do
        for agent_index in chaser_easybg_ucb_inda_black_ 
            do
                for file_index in 25
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 --gpu 4
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode easybg_test --res_id $realfilename --run_id $fn1 --gpu 4
                        python -m procgen.eval --env_name chaser --num_levels 0 --start_level 200 --log_dir /data/kbc/procgen/coinrun/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2 --gpu 4
                        
                    done
            done
    done




