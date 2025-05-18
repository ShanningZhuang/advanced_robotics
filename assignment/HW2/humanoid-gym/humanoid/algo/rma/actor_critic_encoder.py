import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid.algo.ppo.actor_critic import ActorCritic
from .actor_encoder import ActorEncoder

class ActorCriticEncoder(ActorCritic):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_rma_obs,
                        num_actions,
                        num_latents,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        enc_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        super(ActorCritic, self).__init__()

        self.create_actor(num_actor_obs, num_rma_obs, num_actions, num_latents, actor_hidden_dims, enc_hidden_dims, activation)
        
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.actor.encoder}")
        print(f"Actor MLP: {self.actor.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def create_actor(self, num_actor_obs, num_rma_obs, num_actions, num_latents, actor_hidden_dims, enc_hidden_dims, activation):
        self.actor = ActorEncoder(num_actor_obs, num_rma_obs, num_actions, num_latents, actor_hidden_dims, enc_hidden_dims, activation)

    def update_distribution(self, observations, rma_obs):
        mean = self.actor(observations, rma_obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, rma_obs, **kwargs):
        self.update_distribution(observations, rma_obs)
        return self.distribution.sample()

    def act_inference(self, observations, rma_obs):
        actions_mean = self.actor(observations, rma_obs)
        return actions_mean

    def encode(self, rma_obs):
        return self.actor.encode(rma_obs)


class ActorCriticAdaptation(ActorCriticEncoder):

    def create_actor(self, num_actor_obs, num_rma_obs, num_actions, num_latents, actor_hidden_dims, enc_hidden_dims, activation):
        self.actor = ActorEncoder(num_actor_obs, num_actor_obs, num_actions, num_latents, actor_hidden_dims, enc_hidden_dims, activation)
    
    def act(self, observations, rma_obs, **kwargs):
        mean = self.actor(observations, rma_obs)
        self.distribution = Normal(mean, mean*0. + self.std)
        return mean