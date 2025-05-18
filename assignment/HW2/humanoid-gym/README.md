# <a href="https://sites.google.com/view/humanoid-gym/">Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer</a>

Humanoid-Gym is an easy-to-use reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment. 

Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujoco that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies. In this assignment, we don't use the sim2sim part.

This codebase is verified by RobotEra's XBot-S (1.2-meter tall humanoid robot) and XBot-L (1.65-meter tall humanoid robot) in a real-world environment with zero-shot sim-to-real transfer.

## Installation
Isaac Gym only officially support Ubuntu. If you don't have access to a linux machine, we provide a WSL set-up procedure.

### Ubuntu Installation
1. Create Python env with 3.8. We recommend you to use conda with `conda create -n env_name python=3.8`
2. Install pytorch 1.13 with cuda-11.7
    - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
3. Install numpy-1.23 with `conda install numpy=1.23`
4. Install Isaac Gym
    - Download Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
    - Run an example with `cd examples && python 1080_balls_of_solitude.py`
    - Consult `isaacgym/docs/index.html` for troubleshooting
5. Install humanoid-gym
    - `cd humanoid-gym && pip install -e .`
    - Run an example `cd humanoid && python scripts/play_demo.py`.

### WSL Installation
If you do not have a Ubuntu system, you can install WSL2 (Windows Subsystems for Linux).
1. Install WSL
    - Follow the instructions on https://learn.microsoft.com/en-us/windows/wsl/install
2. Install CUDA Toolkit
    - You can refer to the link below which is for cuda 12.6.
    - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
    - Set path parameters (**check your cuda path**). You can write into `~/.bashrc` and `source ~/.bashrc`. Then run `nvcc -V` to check if your cuda is successfully installed.
```bash
export PATH=$PATH:/usr/local/cuda-12.6/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64
```
3. Install cudnn
    - You may need a NVIDIA account. Download Local Installer for Linux x86_64 (Tar) from https://developer.nvidia.com/rdp/cudnn-archive
    - Unzip the file, copy to cuda lib (**check your path**), and change permissions. Below is an example.
```bash
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
sudo cp -r lib/* /usr/local/cuda-12.6/lib64/
sudo cp -r include/* /usr/local/cuda-12.6/include/
# Permissions
sudo chmod a+r /usr/local/cuda-12.6/include/cudnn*
sudo chmod a+r /usr/local/cuda-12.6/lib64/libcudnn*
```
4. Install Vulkan
    - You can find instructions on https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html
    - On Ubuntu 24.04 (my PC), run the scripts below. Then run `vulkaninfo` to check your Vulkan is installed.
```bash
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt update
sudo apt install vulkan-sdk
```
5. Install Conda and follow the instructions in **Ubuntu Installation** to install the project. If you encounter GLFW error or segmentation fault when running examples in IsaacGym, it's normal. Try to install humanoid-gym and run `play_demo.py`, which should work on WSL.

## Usage Guide

#### Examples

```bash
# Launching PPO Policy Training for 'v1' Across 4096 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=humanoid_ppo --run_name v1

# Implementing Simulation-to-Simulation Model Transformation
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_1.pt

# Run our trained policy
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_example.pt
```

#### Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

#### Command-Line Arguments
For RL training, please refer to `humanoid/utils/helpers.py#L161`.
For the sim-to-sim process, please refer to `humanoid/scripts/sim2sim.py#L169`.

## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.


## Add a new environment 

The base environment `legged_robot` constructs a rough terrain locomotion task. The corresponding configuration does not specify a robot asset (URDF/ MJCF) and no reward scales.

1. If you need to add a new environment, create a new folder in the `envs/` directory with a configuration file named `<your_env>_config.py`. The new configuration should inherit from existing environment configurations.
2. If proposing a new robot:
    - Insert the corresponding assets in the `resources/` folder.
    - In the `cfg` file, set the path to the asset, define body names, default_joint_positions, and PD gains. Specify the desired `train_cfg` and the environment's name (python class).
    - In the `train_cfg`, set the `experiment_name` and `run_name`.
3. If needed, create your environment in `<your_env>.py`. Inherit from existing environments, override desired functions and/or add your reward functions.
4. Register your environment in `humanoid/envs/__init__.py`.
5. Modify or tune other parameters in your `cfg` or `cfg_train` as per requirements. To remove the reward, set its scale to zero. Avoid modifying the parameters of other environments!
6. If you want a new robot/environment to perform sim2sim, you may need to modify `humanoid/scripts/sim2sim.py`: 
    - Check the joint mapping of the robot between MJCF and URDF.
    - Change the initial joint position of the robot according to your trained policy.

## Setting for 4GB GPU memory
### Baseline
In `envs/custom/humanoid_config.py`
- XBotLCfg
    - `terrain.measure_heights=False`
    - `env.num_height_points=0`
    - `env.num_envs=800`
- XBotLCfgPPO
    - `policy.actor_hidden_dims=[256,256,128]`
    - `policy.critic_hidden_dims=[256,256,128]`

This setting will use about 3.5GB GPU memory. 

### RMA
Besides the changes above, in `envs/custom/humanoid_rma_config.py`,
- XBotLRMACfg
    - `terrain.mesh_type='plane'`
    - `env.num_height_points=0`
    - `commands.ranges` same with XBotLCfg
- XBotLRMACfgPPO
    - `policy.enc_hidden_dims=[256,256]`

This setting will use about 3.7GB GPU memory.

## Troubleshooting
Observe the following cases:
```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# solution
# set the correct path
export LD_LIBRARY_PATH="/${PATH_TO_CONDA_ENV}/lib:$LD_LIBRARY_PATH"
# example on my server
export LD_LIBRARY_PATH="/home/scm/miniconda3/env/hw2/lib:$LD_LIBRARY_PATH"

# OR
sudo apt install libpython3.8

# error
AttributeError: module 'distutils' has no attribute 'version'

# solution
# install pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# error, results from libstdc++ version distributed with conda differing from the one used on your system to build Isaac Gym
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# solution
mkdir ${PATH_TO_CONDA_ENV}/lib/_unused
mv ${PATH_TO_CONDA_ENV}/lib/libstdc++* ${PATH_TO_CONDA_ENV}/lib/_unused
```

Some problems on WSL
```bash
# error
*** Warning: failed to preload CUDA lib
*** Warning: failed to preload PhysX libs
...
/buildAgent/work/f3416cf82e3cf1ba/source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp (147) : internal error : libcuda.so!
...

# Find the path to libcuda.so and add the dir to LD_LIBRARY_PATH
which libcuda.so
/usr/lib/wsl/lib/libcuda.so
# Add via
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib/

# Vulkan issue
[Error] [carb.gym.plugin] Failed to create Nvf device in createNvfGraphics. Please make sure Vulkan is correctly installed.
# Run vulkaninfo and see if it outputs:
vulkaninfo
/.../miniconda3/envs/hgym/lib/libstdc++.so.6: version `GLIBCXX_3.4.32` not found (required by vulkaninfo)

# solution 
cd ${PATH_TO_CONDA_ENV}/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
# Reference: https://stackoverflow.com/a/73708979/1255535

# warning
/usr/lib/wsl/lib/libcuda.so.1 is not a symbolic link

# solution
cd /usr/lib/wsl/lib
sudo rm libcuda.so libcuda.so.1
sudo ln -s libcuda.so.1.1 libcuda.so.1
sudo ln -s libcuda.so.1.1 libcuda.so

# GLFW error
[Error] [carb.windowing-glfw.plugin] GLFW initialization failed.
[Error] [carb.windowing-glfw.plugin] GLFW window creation failed!
[Error] [carb.gym.plugin] Failed to create Window in CreateGymViewerInternal

# If you are running scripts in humanoid-gym, add --headless flag to avoid render.
# This error means your support for OpenGL is not installed. But even if you install it, you may encounter segmentation fault because IsaacGym render may not support WSL.
# You don't need to install OpenGL support for this homework. We use offscreen render which should support WSL.

# If you still want to use OpenGL, you can do the following
# On Windows side, install VcXsrv https://sourceforge.net/projects/vcxsrv/
# Choose multiple windows, display 0, start no client, disable native opengl (sic).
# On WSL side do following. Please note ubuntu-desktop is an integration of all apps so it's huge.
sudo apt install ubuntu-desktop mesa-utils
export DISPLAY=localhost:0
# Run an example
glxgears
# Reference: https://github.com/microsoft/WSL/issues/2855#issuecomment-358861903
```