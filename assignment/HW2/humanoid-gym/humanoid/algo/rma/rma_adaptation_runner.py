import time
import os
from collections import deque
import statistics
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb
from copy import deepcopy

from .rma_adaptation import RMAAdaptation
from .actor_critic_encoder import ActorCriticEncoder, ActorCriticAdaptation

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.utils import get_load_path
from humanoid.algo.vec_env import VecEnv
from humanoid.algo.ppo.on_policy_runner import OnPolicyRunner

class RMAAdaptationRunner(OnPolicyRunner):

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", use_wandb=False):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.use_wandb = use_wandb
        self.wandb_run_name = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        num_latents = train_cfg["rma"]["num_latents"]
        actor_critic_class = eval(self.cfg["policy_class_name"])
        print("Adaptation Module:")
        actor_critic: ActorCriticAdaptation = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_rma_obs, self.env.num_actions, num_latents, **self.policy_cfg
        ).to(self.device)
        print("RMA Teacher Module:")
        teacher_actor_critic = ActorCriticEncoder(
            self.env.num_obs, num_critic_obs, self.env.num_rma_obs, self.env.num_actions, num_latents, **self.policy_cfg
        ).to(self.device)

        if self.cfg['load_teacher']:
            teacher_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg["rma"]["teacher_experiment_name"])
            teacher_log_path = get_load_path(teacher_log_root, train_cfg["rma"]["teacher_load_run"], train_cfg["rma"]["teacher_checkpoint"], train_cfg['rma']['teacher_run_name'])
            loaded_dict = torch.load(teacher_log_path)
            teacher_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
            print("Load teacher model from ", teacher_log_path)
            actor_critic.actor.actor.load_state_dict(teacher_actor_critic.actor.actor.state_dict())

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: RMAAdaptation = alg_class(actor_critic, teacher_actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_rma_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            if self.use_wandb:
                wandb.init(
                    project="XBot",
                    sync_tensorboard=True,
                    name=self.wandb_run_name,
                    config=self.all_cfg,
                )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        rma_obs = self.env.get_rma_observations()
        obs, critic_obs, rma_obs = obs.to(self.device), critic_obs.to(self.device), rma_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, rma_obs)
                    obs, privileged_obs, rma_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rma_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rma_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_encoder_loss, mean_action_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # self.writer.add_scalar(
        #     "Loss/value_function", locs["mean_value_loss"], locs["it"]
        # )
        # self.writer.add_scalar(
        #     "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        # )
        self.writer.add_scalar(
            "Loss/encoder", locs["mean_encoder_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/action", locs["mean_action_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                # f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Encoder loss:':>{pad}} {locs['mean_encoder_loss']:.4f}\n"""
                f"""{'Action loss:':>{pad}} {locs['mean_action_loss']:.4f}\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                # f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Encoder loss:':>{pad}} {locs['mean_encoder_loss']:.4f}\n"""
                f"""{'Action loss:':>{pad}} {locs['mean_action_loss']:.4f}\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)
