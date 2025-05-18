import torch
import torch.nn as nn
import torch.optim as optim

class ActorEncoder(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_rma_obs,
                        num_actions,
                        num_latents,
                        actor_hidden_dims=[256, 256, 256],
                        enc_hidden_dims=[256, 256, 256],
                        activation=nn.ELU(),
                        **kwargs):
        super().__init__()

        # TODO ---------------------------------------------------
        # Please create actor and encoder
        # Both actor and encoder are MLPs with the given activation
        # Hidden dims of MLP layers are defined as actor_hidden_dims and enc_hidden_dims
        # You don't need to apply activation after the last layer.
        # Refer to algo/ppo/actor_critic.py for examples.
        # Take care of the input & output dims.
        # Encoder will get input of the rma obs (dim=num_rma_obs), and output a latent vector (dim=num_latents).
        # Actor will get input of actor obs (dim=num_actor_obs) as well as the latent vector, and output an action (dim=num_actions).
        
        self.actor = None
        self.encoder = None
        # --------------------------------------------------------

    def forward(self, observation, rma_obs):
        # TODO ---------------------------------------------------
        # Use your actor and encoder to get the action

        action_mean = None
        # --------------------------------------------------------
        return action_mean
    
    def encode(self, rma_obs):
        # TODO ---------------------------------------------------
        # Use your encoder to get the latent vector

        est_latent = None
        # --------------------------------------------------------
        return est_latent