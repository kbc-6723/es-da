python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug random_conv --timesteps 25 --run_id try1_fruitbot_easybg_InDA_random_conv_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Randconv
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug random_conv --timesteps 25 --run_id try2_fruitbot_easybg_InDA_random_conv_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Randconv
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug random_conv --timesteps 25 --run_id try3_fruitbot_easybg_InDA_random_conv_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Randconv
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug random_conv --timesteps 25 --run_id try4_fruitbot_easybg_InDA_random_conv_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Randconv
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug random_conv --timesteps 25 --run_id try5_fruitbot_easybg_InDA_random_conv_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Randconv

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try1_fruitbot_easybg_rad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # RAD Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try2_fruitbot_easybg_rad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # RAD Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try3_fruitbot_easybg_rad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # RAD Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try4_fruitbot_easybg_rad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # RAD Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try5_fruitbot_easybg_rad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # RAD Randconv

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try1_fruitbot_easybg_drac_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try2_fruitbot_easybg_drac_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try3_fruitbot_easybg_drac_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try4_fruitbot_easybg_drac_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try5_fruitbot_easybg_drac_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Randconv

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try1_fruitbot_easybg_drac_ptgrad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try2_fruitbot_easybg_drac_ptgrad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try3_fruitbot_easybg_drac_ptgrad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try4_fruitbot_easybg_drac_ptgrad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Randconv
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try5_fruitbot_easybg_drac_ptgrad_random_conv_ --data_aug random_conv --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Randconv

python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug color_jitter --timesteps 25 --run_id try1_fruitbot_easybg_InDA_color_jitter_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Colorjitter
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug color_jitter --timesteps 25 --run_id try2_fruitbot_easybg_InDA_color_jitter_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Colorjitter
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug color_jitter --timesteps 25 --run_id try3_fruitbot_easybg_InDA_color_jitter_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Colorjitter
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug color_jitter --timesteps 25 --run_id try4_fruitbot_easybg_InDA_color_jitter_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Colorjitter
python -m procgen.InDA --env_name fruitbot --distribution_mode easybg --data_aug color_jitter --timesteps 25 --run_id try5_fruitbot_easybg_InDA_color_jitter_ --num_levels 200 --log_dir /data/kbc/procgen/all/ --gpu 7  #InDA Colorjitter

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try1_fruitbot_easybg_rad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # RAD Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try2_fruitbot_easybg_rad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # RAD Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try3_fruitbot_easybg_rad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # RAD Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try4_fruitbot_easybg_rad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # RAD Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_rad --gpu 7 --run_id try5_fruitbot_easybg_rad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # RAD Colorjitter

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try1_fruitbot_easybg_drac_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try2_fruitbot_easybg_drac_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try3_fruitbot_easybg_drac_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try4_fruitbot_easybg_drac_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --gpu 7 --run_id try5_fruitbot_easybg_drac_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Colorjitter

python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try1_fruitbot_easybg_drac_ptgrad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try2_fruitbot_easybg_drac_ptgrad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try3_fruitbot_easybg_drac_ptgrad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try4_fruitbot_easybg_drac_ptgrad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Colorjitter
python -m procgen.train --env_name fruitbot --distribution_mode easybg --use_drac --use_pdtgrad --gpu 7 --run_id try5_fruitbot_easybg_drac_ptgrad_color_jitter_ --data_aug color_jitter --log_dir /data/kbc/procgen/all/ # DrAC Ptgrad Colorjitter

python -m procgen.train --env_name fruitbot --distribution_mode easybg --gpu 7 --run_id try1_fruitbot_easybg_ppo_ --log_dir /data/kbc/procgen/all/ # PPO
python -m procgen.train --env_name fruitbot --distribution_mode easybg --gpu 7 --run_id try2_fruitbot_easybg_ppo_ --log_dir /data/kbc/procgen/all/ # PPO
python -m procgen.train --env_name fruitbot --distribution_mode easybg --gpu 7 --run_id try3_fruitbot_easybg_ppo_ --log_dir /data/kbc/procgen/all/ # PPO
python -m procgen.train --env_name fruitbot --distribution_mode easybg --gpu 7 --run_id try4_fruitbot_easybg_ppo_ --log_dir /data/kbc/procgen/all/ # PPO
python -m procgen.train --env_name fruitbot --distribution_mode easybg --gpu 7 --run_id try5_fruitbot_easybg_ppo_ --log_dir /data/kbc/procgen/all/ # PPO


for temp_index in try1_ try2_ try3_ try4_ try5_
    do
        for agent_index in fruitbot_easybg_InDA_random_conv_ fruitbot_easybg_rad_random_conv_ fruitbot_easybg_drac_random_conv_ fruitbot_easybg_drac_ptgrad_random_conv_ fruitbot_easybg_InDA_color_jitter_ fruitbot_easybg_rad_color_jitter_ fruitbot_easybg_drac_color_jitter_ fruitbot_easybg_drac_ptgrad_color_jitter_ fruitbot_easybg_ppo_
            do
                for file_index in 25 
                    do
                        realfilename="$temp_index$agent_index$file_index"
                        fn0="$temp_index$agent_index$1_train"
                        fn1="$temp_index$agent_index$1_test_bg"
                        fn2="$temp_index$agent_index$1_test_lv" 
                        python -m procgen.eval --env_name fruitbot --num_levels 200 --gpu 7 --start_level 0 --log_dir /data/kbc/procgen/all/ --num_envs 200 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn0 
                        python -m procgen.eval --env_name fruitbot --num_levels 200 --gpu 7 --start_level 0 --log_dir /data/kbc/procgen/all/ --num_envs 200 --rep_count 5 --distribution_mode easybg_test --res_id $realfilename --run_id $fn1 
                        python -m procgen.eval --env_name fruitbot --num_levels 0 --gpu 7 --start_level 200 --log_dir /data/kbc/procgen/all/ --num_envs 500 --rep_count 5 --distribution_mode easybg --res_id $realfilename --run_id $fn2
                        
                    done
            done
    done
