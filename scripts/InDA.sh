python -m procgen.InDA --env_name chaser --distribution_mode easybg --data_aug black --timesteps 25 --run_id try1_chaser_easybg_InDA_black_ --num_levels 200 --log_dir /data/kbc/procgen/coinrun/ --gpu 2
python -m procgen.InDA --env_name chaser --distribution_mode easybg --data_aug black --timesteps 25 --run_id try2_chaser_easybg_InDA_black_ --num_levels 200 --log_dir /data/kbc/procgen/coinrun/ --gpu 2
python -m procgen.InDA --env_name chaser --distribution_mode easybg --data_aug black --timesteps 25 --run_id try3_chaser_easybg_InDA_black_ --num_levels 200 --log_dir /data/kbc/procgen/coinrun/ --gpu 2

for temp_index in try1_ try2_ try3_
    do
        for agent_index in chaser_easybg_InDA_black_ 
            do
                for file_index in 25 
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 --gpu 2
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/coinrun/ --num_envs 200 --rep_count 5 --distribution_mode hardbg --res_id $realfilename --run_id $fn1 --gpu 2
                        python -m procgen.eval --env_name chaser --num_levels 0 --start_level 200 --log_dir /data/kbc/procgen/coinrun/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2 --gpu 2
                        
                    done
            done
    done

