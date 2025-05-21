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
        
        # Build encoder network
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_rma_obs, enc_hidden_dims[0]))
        encoder_layers.append(activation)
        for i in range(len(enc_hidden_dims)-1):
            encoder_layers.append(nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i+1]))
            encoder_layers.append(activation)
        encoder_layers.append(nn.Linear(enc_hidden_dims[-1], num_latents))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build actor network (takes in both observations and latents)
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs + num_latents, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for i in range(len(actor_hidden_dims)-1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i+1]))
            actor_layers.append(activation)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        # --------------------------------------------------------

    def forward(self, observation, rma_obs):
        # TODO ---------------------------------------------------
        # Use your actor and encoder to get the action

        latent = self.encode(rma_obs)
        action_mean = self.actor(torch.cat([observation, latent], dim=1))
        # --------------------------------------------------------
        return action_mean
    
    def encode(self, rma_obs):
        # TODO ---------------------------------------------------
        # Use your encoder to get the latent vector

        est_latent = self.encoder(rma_obs)
        # --------------------------------------------------------
        return est_latent