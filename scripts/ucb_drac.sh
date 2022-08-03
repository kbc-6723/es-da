python -m procgen.train --env_name chaser --ucb_drac --distribution_mode easybg --gpu 3 --run_id try1_chaser_easybg_ucb_drac_coef_2_ --log_dir /data/kbc/procgen/all/ --ucb_coef 2
python -m procgen.train --env_name chaser --ucb_drac --distribution_mode easybg --gpu 3 --run_id try2_chaser_easybg_ucb_drac_coef_2_ --log_dir /data/kbc/procgen/all/ --ucb_coef 2
python -m procgen.train --env_name chaser --ucb_drac --distribution_mode easybg --gpu 3 --run_id try3_chaser_easybg_ucb_drac_coef_2_ --log_dir /data/kbc/procgen/all/ --ucb_coef 2
python -m procgen.train --env_name chaser --ucb_drac --distribution_mode easybg --gpu 3 --run_id try4_chaser_easybg_ucb_drac_coef_2_ --log_dir /data/kbc/procgen/all/ --ucb_coef 2
python -m procgen.train --env_name chaser --ucb_drac --distribution_mode easybg --gpu 3 --run_id try5_chaser_easybg_ucb_drac_coef_2_ --log_dir /data/kbc/procgen/all/ --ucb_coef 2
#python -m procgen.train --env_name climber --distribution_mode easybg --gpu 3 --run_id try2_climber_easybg_ppo_ --log_dir /data/kbc/procgen/all/


for temp_index in try1_ try2_ try3_ try4_ try5_ #try2_
    do
        for agent_index in chaser_easybg_ucb_drac_coef_2_ 
            do
                for file_index in 25 
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/all/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 --gpu 3
                        python -m procgen.eval --env_name chaser --num_levels 200 --start_level 0 --log_dir /data/kbc/procgen/all/ --num_envs 200 --rep_count 5 --distribution_mode easybg_test --res_id $realfilename --run_id $fn1 --gpu 3
                        python -m procgen.eval --env_name chaser --num_levels 0 --start_level 200 --log_dir /data/kbc/procgen/all/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2 --gpu 3
                        
                    done
            done
    done




