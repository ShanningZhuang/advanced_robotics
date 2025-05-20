import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, task_registry, Logger, export_policy_as_jit
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.sim.max_gpu_contact_pairs = 2**10

    # terrain setting
    # env_cfg.terrain.mesh_type = 'trimesh' #'plane'
    env_cfg.terrain.terrain_proportions = [0]*7
    # plane; obstacles; uniform; slope_up; slope_down, stair_down, stair_up
    env_cfg.terrain.terrain_proportions[6] = 1 # 5: stair_down, 6: stair_up
    env_cfg.terrain.fix_level = 0.1 # step_height = fix_level * 10cm
    env_cfg.terrain.terrain_length = 10.
    env_cfg.terrain.terrain_width = 10.
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    task = args.task.split('_')[-1]
    train_cfg.seed = 123145
    if task == 'adaptation':
        train_cfg.runner.load_teacher = False
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # export policy as a jit module (used to run it from C++)
    if True:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + train_cfg.runner.experiment_name + '_' + args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 100.0, (1920, 1080))

    obs = env.get_observations()
    if task == 'ppo':
        rma_obs = None
    else:
        rma_obs = env.get_rma_observations()

    for i in tqdm(range(NUM_STEPS)):
        
        obs = obs.detach()
        if task == 'rma':
            actions = policy(obs, rma_obs.detach())
        elif task == 'adaptation':
            actions = policy(obs, obs)
        else:
            actions = policy(obs) # * 0.
        
        if FIX_COMMAND:
            env.commands[:, 0] = 1.0 if env_cfg.terrain.mesh_type=='trimesh' else 0.5
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        if task == 'ppo':
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        else:
            obs, critic_obs, rma_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
            env.gym.end_access_image_tensors(env.sim)
    
    if RENDER:
        video.release()
        print("Video saved to ", dir)

if __name__ == '__main__':
    RENDER = True
    FIX_COMMAND = True
    NUM_STEPS = 1000 # number of steps before plotting states
    args = get_args()
    play(args)
