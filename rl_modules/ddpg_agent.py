import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl_modules.ddpg_models import DDPG_Actor, DDPG_Critic
from her_modules.replay_buffer import ReplayBuffer
from rl_modules.normalizer import Normalizer



#TODO: REMOVe all the unnecessary print statements 
#   figure out why reward isn't improving: mmore normalization?
class DDPG_HER_AGENT:
    def __init__(self, env_params, env, device, checkpoint_path=None, load=False):
        self.device = device
        self.env_params = env_params
        self.env = env
        
        self.actor = DDPG_Actor(env_params).to(device)
        self.critic = DDPG_Critic(env_params).to(device)
        self.actor_target = DDPG_Actor(env_params).to(device)
        self.critic_target = DDPG_Critic(env_params).to(device)
        
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = ReplayBuffer(env_params, buffer_size=int(1e6), device=device)
        
        self.batch_size = 256
        self.gamma = 0.98
        self.tau = 0.05
        self.checkpoint_path = checkpoint_path
        self.clip_range = 5
        
        self.o_norm = Normalizer(size=env_params['obs'], default_clip_range=self.clip_range)
        self.g_norm = Normalizer(size=env_params['goal'], default_clip_range=self.clip_range)
        
        if load:
            self.load_checkpoint()

    def select_action(self, state, goal, train_mode=True):
        with torch.set_grad_enabled(train_mode):
            input_tensor = torch.cat([state, goal], dim=-1)
            action = self.actor(input_tensor)
            
            random_actions = (2 * torch.rand(4, dtype=torch.float32) - 1).to(action.device)

            if train_mode:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
            #n -> num trials, p -> prob of 1, size -> num experiments
            #   returns number of times it was 1
            if np.random.binomial(1, 0.9, 1)[0] == 0:
                # print("ACTUALLY RANDOM")
                action += (random_actions - action)
            return action
    
    #shape of mb -> (batch size, time_steps, feature dim)
    def update_normalizer(self, mb):
        mb_obs, mb_ag, mb_g, mb_actions = mb
        mb_obs_next = mb_obs[:, 1:, :]  #future observation is just the observation at the next time step
        mb_ag_next = mb_ag[:, 1:, :]
        #TODO: FIGURE OUT WHY MB ACTIONS SHAPE IS (2, 1, 15, 4) instead of (2, 51, 15, 4)
        num_transitions = mb_actions.shape[1]
        print("NUM TRANSITIONS: " + str(num_transitions))
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        obs, g = self.buffer.sample_for_normalization(buffer_temp)
        #TODO Consider adding some preprocess for obs and g
        self.o_norm.update(obs)
        self.g_norm.update(g)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        
        

    def train(self):
        print("TRAINING")
        # TODO: CHECK THIS IS CORRECT< MAKE SURE IT DOES HER TRANSITIONS AND ALSO DO NORMALIZATRION
        
        # performs HER relabeling by relabeling goals with future achieved goals 
        # transitions include the newly calculated rewards
        transitions = self.buffer.sample(self.batch_size)
        
        obs = torch.FloatTensor(transitions['obs']).to(self.device)
        ag = torch.FloatTensor(transitions['ag']).to(self.device)
        g = torch.FloatTensor(transitions['g']).to(self.device)
        actions = torch.FloatTensor(transitions['actions']).to(self.device)
        obs_next = torch.FloatTensor(transitions['obs_next']).to(self.device)
        ag_next = torch.FloatTensor(transitions['ag_next']).to(self.device)
        rewards = torch.FloatTensor(transitions['r']).to(self.device)
        
        # Critic update
        with torch.no_grad():
            #target_Q
            next_input = torch.cat([obs_next, g], dim=-1)
            target_actions = self.actor_target(next_input) # -> returns actions
            target_Q = self.critic_target(next_input, target_actions) #-> expected reward
            #expected future reward 
            # target_Q = rewards + (1 - (rewards == 0).float()) * self.gamma * target_Q
            target_Q = rewards + self.gamma * target_Q      #current reward + expected future reward 
            target_Q = torch.clamp(target_Q, -1 / (1 - self.gamma), 0)

        current_input = torch.cat([obs, g], dim=-1)
        current_Q = self.critic(current_input, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        pred_actions = self.actor(current_input)
        actor_loss = -self.critic(current_input, pred_actions).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local, target):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, source, target):
        target.load_state_dict(source.state_dict())

    def save_checkpoint(self):
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_optim_state': self.actor_optim.state_dict(),
            'critic_optim_state': self.critic_optim.state_dict()
        }, self.checkpoint_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state'])
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)