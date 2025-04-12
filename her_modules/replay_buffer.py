import threading
import numpy as np
import torch
from copy import deepcopy as dc

class ReplayBuffer:
    def __init__(self, env_params, buffer_size, device='cpu'):
        self.clip_value = 200
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        print("SIZE IS: " + str(self.size))
        self.device = device
        replay_k = 4
        self.future_p = 1 - (1. / (1 + replay_k))
        
        self.buffers = {
            'obs': torch.empty(
                [self.size, self.T + 1, self.env_params['num_envs'], env_params['obs']],
                dtype=torch.float32,
                device=device
            ),
            'ag': torch.empty(
                [self.size, self.T + 1, self.env_params['num_envs'], env_params['goal']],
                dtype=torch.float32,
                device=device
            ),
            'g': torch.empty(
                [self.size, self.T, self.env_params['num_envs'], env_params['goal']],
                dtype=torch.float32,
                device=device
            ),
            'actions': torch.empty(
                [self.size, self.T, self.env_params['num_envs'], env_params['action']],
                dtype=torch.float32,
                device=device
            ),
        }
        print("BUFFER OBS SHAPE")
        print(self.buffers['obs'].shape)
        
        self.lock = threading.Lock()
        self.current_size = 0
        self.n_transitions_stored = 0

    def store_episode(self, episode_batch):
        print("EPISODE BATCH")
        print(len(episode_batch))
        
        #shape of each is [2,51, 15 or 3 or 4]
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        print("JUST OBS SHAPE: " + str(mb_obs.shape))
        print(self.buffers['obs'].shape)
        print("OBS SHAPE: " + str(mb_obs.shape))
        print("AG SHAPE: " + str(mb_ag.shape))

        batch_size = mb_obs.shape[0]
        
        with self.lock:
            # should be like [0 1] or something
            idxs = self._get_storage_idx(batch_size)
            if isinstance(mb_obs, np.ndarray):
                mb_obs = torch.FloatTensor(mb_obs).to(self.device)
                mb_ag = torch.FloatTensor(mb_ag).to(self.device)
                mb_g = torch.FloatTensor(mb_g).to(self.device)
                mb_actions = torch.FloatTensor(mb_actions).to(self.device)
            
            print("OBS IDX SHAPE: " + str(self.buffers['obs'][idxs].shape))
            print("MB OBS SHAPE: " + str(mb_obs.shape))
            #idx shape is: [2, 51, 15] or something since
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += batch_size * self.T

    def clip_obs(self, obs):
        return np.clip(obs, -self.clip_value, self.clip_value)

 

    #TODO INTEGRATE THIS METHOD INTO THE CODE, ie. need to make the code implement batching
    #ALSO IMPLEMENT THE STATE AND GOAL NORMALIZERS 
    def sample_for_normalization(self, batch):
        # batch is now a dictionary with keys like 'obs', 'ag', 'g', etc.
        # Each value is a numpy array of shape (num_episodes, num_timesteps, feature_dim)
        
        num_episodes = batch['obs'].shape[0]
        max_timesteps = batch['g'].shape[1]
        print("MAX TIMESTEPS: " + str(max_timesteps))
        size = max_timesteps  # or whatever sample size you want
        
        # Randomly select episodes and timesteps
        ep_indices = np.random.randint(0, num_episodes, size)
        time_indices = np.random.randint(0, max_timesteps, size)
        
        # Get current states and desired goals
        states = np.array([batch['obs'][ep, t] for ep, t in zip(ep_indices, time_indices)])
        desired_goals = np.array([batch['g'][ep, t] for ep, t in zip(ep_indices, time_indices)])
        
        # Apply HER (Hindsight Experience Replay)
        her_indices = np.where(np.random.uniform(size=size) < self.future_p)[0]
        future_offset = np.random.uniform(size=size) * (max_timesteps - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset)[her_indices]  # removed the +1 since mb_obs_next is already offset
        
        # Get future achieved goals for HER
        future_ag = np.array([batch['ag_next'][ep, t] for ep, t in zip(ep_indices[her_indices], future_t)])
        desired_goals[her_indices] = future_ag
        
        return self.clip_obs(states), self.clip_obs(desired_goals)

    
    #sasmple a batch for use for trainig
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        
        episode_idxs = np.random.randint(0, self.current_size, batch_size)
        t_samples = np.random.randint(0, self.T, batch_size)
        
        transitions = {
            'obs': temp_buffers['obs'][episode_idxs, t_samples].cpu().numpy(),
            'ag': temp_buffers['ag'][episode_idxs, t_samples].cpu().numpy(),
            'g': temp_buffers['g'][episode_idxs, t_samples].cpu().numpy(),
            'actions': temp_buffers['actions'][episode_idxs, t_samples].cpu().numpy(),
            'obs_next': temp_buffers['obs_next'][episode_idxs, t_samples].cpu().numpy(),
            'ag_next': temp_buffers['ag_next'][episode_idxs, t_samples].cpu().numpy(),
        }
        
        her_indexes = np.where(np.random.uniform(size=batch_size) < 0.8)[0]
        future_offset = np.random.uniform(size=batch_size) * (self.T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = temp_buffers['ag'][episode_idxs[her_indexes], future_t].cpu().numpy()
        transitions['g'][her_indexes] = future_ag
        
        transitions['r'] = np.expand_dims(
            self.env_params['reward_func'](transitions['ag_next'], transitions['g'], None), 
            axis=1
        )
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        
        self.current_size = min(self.size, self.current_size + inc)
        print("STORAGE IDX IS: " + str(idx))
        return idx

    def __len__(self):
        return self.n_transitions_stored