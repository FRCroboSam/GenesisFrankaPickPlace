import threading
import torch
import numpy as np

class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, device='cpu'):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.sample_func = sample_func
        self.device = device
        
        # Initialize buffers with explicit key names using PyTorch tensors
        self.buffers = {
            'state': torch.empty([self.size, self.T + 1, self.env_params['obs']], dtype=torch.float32, device=self.device),
            'achieved_goal': torch.empty([self.size, self.T + 1, self.env_params['goal']], dtype=torch.float32, device=self.device),
            'desired_goal': torch.empty([self.size, self.T, self.env_params['goal']], dtype=torch.float32, device=self.device),
            'action': torch.empty([self.size, self.T, self.env_params['action']], dtype=torch.float32, device=self.device),
            'reward': torch.empty([self.size, self.T, 1], dtype=torch.float32, device=self.device),
            'done': torch.empty([self.size, self.T, 1], dtype=torch.bool, device=self.device)
        }
        
        self.lock = threading.Lock()
        self.current_size = 0
        self.n_transitions_stored = 0
        
        
    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def store_single_transition(self, transition):
        """Store a batch of transitions (compatible with your training code)"""
        # Convert numpy arrays to tensors if needed
        if isinstance(transition_batch['state'], np.ndarray):
            transition_batch = {k: torch.from_numpy(v).to(self.device) 
                              for k, v in transition_batch.items()}
        
        batch_size = transition_batch['state'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            for key in self.buffers.keys():
                if key in transition_batch:
                    # Handle single timestep storage (expand dims if needed)
                    if transition_batch[key].ndim == len(self.buffers[key].shape) - 1:
                        transition_batch[key] = transition_batch[key].unsqueeze(1)
                    self.buffers[key][idxs] = transition_batch[key]
            
            self.n_transitions_stored += batch_size

    
    
    #sample a batch of temp buffers
    def sample_batch(self, batch_size):
        """Sample with explicit key names, returning PyTorch tensors"""
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        
        # Prepare next states
        temp_buffers['next_state'] = temp_buffers['state'][:, 1:, :]
        temp_buffers['next_achieved_goal'] = temp_buffers['achieved_goal'][:, 1:, :]
        
        # Sample transitions using HER
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = torch.arange(self.current_size, self.current_size + inc, device=self.device)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = torch.arange(self.current_size, self.size, device=self.device)
            idx_b = torch.randint(0, self.current_size, (overflow,), device=self.device)
            idx = torch.cat([idx_a, idx_b])
        else:
            idx = torch.randint(0, self.size, (inc,), device=self.device)
        
        self.current_size = min(self.size, self.current_size + inc)
        return idx[0] if inc == 1 else idx

    def __len__(self):
        return self.n_transitions_stored