# Time Matter in Using Data Augmentation in Vision-Based Reinforcement Learning

Modified Procgen


Supported platforms:

- Windows 10
- macOS 10.14 (Mojave), 10.15 (Catalina)
- Linux (manylinux2010)

Supported Pythons:

- 3.6 64-bit
- 3.7 64-bit
- 3.8 64-bit

Supported CPUs:

- Must have at least AVX

```
git clone git@github.com:openai/procgen.git
cd Time-DA
conda env update --name time-da --file environment.yml
conda activate time-da
pip install -e .
# this should say "building procgen...done"
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"
# this should create a window where you can play the coinrun environment
python -m procgen.interactive
```

The environment code is in C++ and is compiled into a shared library exposing the [`gym3.libenv`](https://github.com/openai/gym3/blob/master/gym3/libenv.h) C interface that is then loaded by python.  The C++ code uses [Qt](https://www.qt.io/) for drawing.


