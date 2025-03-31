# for the stuff thats common between eval and train ddpg 


import argparse
import numpy as np
import torch
import genesis as gs
from fetch_pick_place_env import FrankaPickPlaceDDPG_Env

class BaseDDPG:
    def __init__(self, args):
        self.args = args
        self.task_to_class = {
            'FrankaPickPlaceDDPGHer': FrankaPickPlaceDDPG_Env,
            # Add other tasks...
        }
        
    def _create_env(self, task_name, vis=False, device='cuda', num_envs=1):
        env_class = self.task_to_class.get(task_name)
        if not env_class:
            raise ValueError(f"Task '{task_name}' not recognized")
        return env_class(vis=vis, device=device, num_envs=num_envs)
    
    def process_inputs(self, o, g, o_mean, o_std, g_mean, g_std, clip_obs=5, clip_range=5):
        o_clip = np.clip(o, -clip_obs, clip_obs)
        g_clip = np.clip(g, -clip_obs, clip_obs)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
        return np.concatenate([o_norm, g_norm])
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Common arguments
        parser.add_argument("-v", "--vis", action="store_true", help="Enable visualization")
        parser.add_argument("-d", "--device", default="cuda", help="Device: cpu, cuda, or mps")
        parser.add_argument("-t", "--task", default="FrankaPickPlaceDDPGHer", help="Task name")
        parser.add_argument("--clip_obs", type=float, default=5, help="Observation clipping")
        parser.add_argument("--clip_range", type=float, default=5, help="Normalization range")
        
        # Training-specific args would be added in TrainDDPG
        # Evaluation-specific args would be added in EvalDDPG
        
        return parser.parse_args()