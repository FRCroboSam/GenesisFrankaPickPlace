#TODO CHECK IMPORTS SEE IF THIS THING WORKS 
#TODO: Also try to use the mpi4py thing if this has problems
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rl_modules
from rl_modules.ddpg_models import DDPG_Actor, DDPG_Critic
from her_modules.replay_buffer import replay_buffer
class DDPG_HER_AGENT:
    def __init__(self, env_params, env, device, checkpoint_path=None, load=False):
        self.device = device
        self.env_params = env_params
        self.env = env
        
        # Initialize networks
        self.actor = DDPG_Actor(env_params).to(device)
        self.critic = DDPG_Critic(env_params).to(device)
        self.actor_target = DDPG_Actor(env_params).to(device)
        self.critic_target = DDPG_Critic(env_params).to(device)
        
        # Initialize targets
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Replay buffer with HER
        self.buffer = replay_buffer(
            env_params=env_params,
            buffer_size=int(1e6),
            sample_func=self._sample_her_transitions
        )
        
        # Training parameters
        self.batch_size = 256
        self.gamma = 0.98
        self.tau = 0.05
        self.checkpoint_path = checkpoint_path
        
        if load:
            self.load_checkpoint()

    def _sample_her_transitions(self, episode_batch, batch_size):
        """HER sampling with explicit key names"""
        T = self.env_params['max_timesteps']
        rollout_batch_size = episode_batch['state'].shape[0]
        
        # Select random episodes and timesteps
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        
        transitions = {
            'state': episode_batch['state'][episode_idxs, t_samples].copy(),
            'achieved_goal': episode_batch['achieved_goal'][episode_idxs, t_samples].copy(),
            'desired_goal': episode_batch['desired_goal'][episode_idxs, t_samples].copy(),
            'action': episode_batch['action'][episode_idxs, t_samples].copy(),
            'next_state': episode_batch['next_state'][episode_idxs, t_samples].copy(),
            'next_achieved_goal': episode_batch['next_achieved_goal'][episode_idxs, t_samples].copy(),
            'done': episode_batch['done'][episode_idxs, t_samples].copy()
        }
        
        # HER: Relabel goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < 0.8)[0]
        future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        future_ag = episode_batch['achieved_goal'][episode_idxs[her_indexes], future_t]
        transitions['desired_goal'][her_indexes] = future_ag
        
        # Compute rewards
        transitions['reward'] = np.expand_dims(
            
            self.env.compute_reward(transitions['next_achieved_goal'], transitions['desired_goal']), 
            axis=1
        )
        
        return transitions

    def store_transition(self, state, action, reward, next_state, desired_goal, done):
        """Store transition with all necessary components including reward"""
        episode_batch = {
            'state': np.array([state]),
            'action': np.array([action]),
            'reward': np.array([[reward]]),  # Now storing reward
            'next_state': np.array([next_state['observation']]),  # Added next_state
            'achieved_goal': np.array([next_state['achieved_goal']]),
            'next_achieved_goal': np.array([next_state['achieved_goal']]),  # For HER
            'desired_goal': np.array([desired_goal]),
            'done': np.array([[done]])
        }
        self.buffer.store_episode(episode_batch)

    def select_action(self, state, desired_goal, train_mode=True):
        """Select action with explicit inputs"""
        print("STATE SHAPE: " + str(state.shape))
        print(state)
        print("DEVICE IS: " + str(self.device))
        with torch.set_grad_enabled(train_mode):
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # goal_tensor = torch.FloatTensor(desired_goal).unsqueeze(0).to(self.device)
            input_tensor = torch.cat([state, desired_goal], dim=-1)
            actions = self.actor(input_tensor)#.cpu().data.numpy().flatten()
            
            if train_mode:
                # Generate noise directly on the same device as actions
                noise = torch.randn_like(actions) * 0.1
                actions = torch.clamp(actions + noise, -1, 1)
            
            # Return a tensor with original batch dimension
            return actions  # shape [10, 4]

    def train(self):
        """Train using explicit key names"""
        transitions = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(transitions['state']).to(self.device)
        action = torch.FloatTensor(transitions['action']).to(self.device)
        reward = torch.FloatTensor(transitions['reward']).to(self.device)
        next_state = torch.FloatTensor(transitions['next_state']).to(self.device)
        desired_goal = torch.FloatTensor(transitions['desired_goal']).to(self.device)
        done = torch.FloatTensor(transitions['done']).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_input = torch.cat([next_state, desired_goal], dim=-1)
            target_action = self.actor_target(next_input)
            target_Q = self.critic_target(next_input, target_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
        current_input = torch.cat([state, desired_goal], dim=-1)
        current_Q = self.critic(current_input, action)
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
        
        # Update targets
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
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
