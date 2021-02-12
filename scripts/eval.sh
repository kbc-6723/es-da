for temp_index in try1_ try2_ try3_
    do
        for agent_index in climber_easybg_200_ppo_ 
            do
                for file_index in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name climber --num_levels 200 --start_level 0 --log_dir /home/kbc/pdad/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 
                        python -m procgen.eval --env_name climber --num_levels 200 --start_level 0 --log_dir /home/kbc/pdad/ --num_envs 200 --rep_count 5 --distribution_mode hardbg --res_id $realfilename --run_id $fn1 
                        python -m procgen.eval --env_name climber --num_levels 0 --start_level 200 --log_dir /home/kbc/pdad/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2
                        
                    done
            done
    done
