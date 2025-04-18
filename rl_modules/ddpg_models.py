# TODO: Check if this logic is correct against 
# https://github.com/TianhongDai/hindsight-experience-replay/blob/master/rl_modules/models.py

# also try dividng by max action to see if it makes a difference

import torch.nn as nn
import torch

# need to bound actor values iwth tanh




class DDPG_Actor(nn.Module):
    def __init__(self, env_params):
        super(DDPG_Actor, self).__init__()
        input_dim = env_params['obs'] + env_params['goal']
        hidden_dim = 256
        output_dim = env_params['action']    #here input dim should be: input space + action space
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return torch.tanh(self.network(x))
    
    
#returns expected reward given actor and critic
class DDPG_Critic(nn.Module):
    def __init__(self, env_params):
        super(DDPG_Critic, self).__init__()
        # Input: state (obs + goal) + action
        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        output_dim = 1  # Q-value is scalar
        hidden_dim = 256
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state, action):
        """
        Args:
            state: concatenated [observation, goal] tensor
            action: action tensor
        Returns:
            Q-value estimate (unbounded)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)