python -m procgen.train --env_name chaser --use_drac --distribution_mode easybg --data_aug black --gpu 4 --run_id try1_chaser_easybg_drac_black_ --log_dir /data/kbc/procgen/coinrun/ --timesteps 25 
python -m procgen.train --env_name chaser --use_drac --distribution_mode easybg --data_aug black --gpu 4 --run_id try2_chaser_easybg_drac_black_ --log_dir /data/kbc/procgen/coinrun/ --timesteps 25 
python -m procgen.train --env_name chaser --use_drac --distribution_mode easybg --data_aug black --gpu 4 --run_id try3_chaser_easybg_drac_black_ --log_dir /data/kbc/procgen/coinrun/ --timesteps 25 
#python -m procgen.train --env_name maze --use_drac --only_distill --distribution_mode easybg --data_aug crop --gpu 4 --res_id try4_maze_easybg_ppo_20_drac_pdtgrad_crop_5_0 --run_id try4_maze_easybg_ppo_20_drac_only_ --log_dir /data/kbc/procgen/all/ --timesteps 5 
#python -m procgen.train --env_name maze --use_drac --only_distill --distribution_mode easybg --data_aug crop --gpu 4 --res_id try5_maze_easybg_ppo_20_drac_pdtgrad_crop_5_0 --run_id try5_maze_easybg_ppo_20_drac_only_ --log_dir /data/kbc/procgen/all/ --timesteps 5 

#python -m procgen.train --env_name climber --distribution_mode easybg --gpu 4 --run_id try2_climber_easybg_ppo_ --log_dir /data/kbc/procgen/all/


for temp_index in try1_ try2_ try3_ 
    do
        for agent_index in chaser_easybg_drac_black_ 
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




