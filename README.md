# Efficient Scheduling of Data Augmentation for Deep Reinforcment Learning (Neurips 2022)

We modified the Procgen (https://github.com/openai/procgen) to verify each generalization about change of backgrounds and levels.

Our data augmentation methods are from RAD(https://github.com/pokaxpoka/rad_procgen) and Auto-DrAC (https://github.com/rraileanu/auto-drac).

## Modified Procgen

Required Libraries

- tensorflow2
- https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a
- matplotlib
- pytorch
- kornia
- scipy
- scikit-image
- mpi4py
- pytest

## Download Modified Procgen from Source
```
git clone https://github.com/kbc-6723/Efficient-Scheduling-of-Data-Augmentation-for-Deep-Reinforcement-Learning.git Scheduling-DA
cd Scheduling-DA
conda env update --name da --file environment.yml
conda activate da
pip install -e .
# this should say "building procgen...done"
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"
# this should create a window where you can play the coinrun environment
python -m procgen.interactive
```
## InDA
python -m procgen.InDA --env_name climber --distribution_mode easybg --num_levels 200 --data_aug random_conv --run_id 'file_name' --log_dir 'your_path'
## ExDA
python -m procgen.ExDA --env_name climber --distribution_mode easybg --num_levels 200 --res_id 'pre-trained-model-path' --data_aug random_conv --run_id 'file_name' --log_dir 'your_path'

For tensorflow2, you should modify OpenAI baselines libararies using tensorflow.compat.v1. 
