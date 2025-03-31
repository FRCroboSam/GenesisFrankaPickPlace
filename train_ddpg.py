import time
import psutil
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from rl_modules.ddpg_agent import DDPG_HER_AGENT
from fetch_pick_place_env import FrankaPickPlaceDDPG_Env
import genesis as gs
from copy import deepcopy as dc


gs.init(backend=gs.gpu, precision="32")

class TrainDDPG:
    def __init__(self, args):
        self.args = args
        self.env = FrankaPickPlaceDDPG_Env(
            vis=args.vis, 
            device=args.device, 
            num_envs=args.num_envs
        )
        
        env_params = {
            'obs': self.env.state_dim,
            'goal': self.env.goal_dim,
            'action': self.env.action_dim,
            'action_max': 1.0,
            'max_timesteps': 50  # Added max timesteps for buffer
        }
        
        self.agent = DDPG_HER_AGENT(
            env_params=env_params,
            env=self.env,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            load=args.load
        )
        
        # Training parameters
        self.MAX_EPOCHS = 50
        self.MAX_CYCLES = 50
        self.MAX_EPISODES = 2
        self.num_updates = 40
        
        # Tracking metrics
        self.t_success_rate = []
        self.total_ac_loss = []
        self.total_cr_loss = []
        
        if args.device == "mps":
            print("Running with mps")
            gs.tools.run_in_another_thread(
                fn=lambda: self.train(), 
                args=()
            )            
            self.env.scene.viewer.start()
            
    # corresponds to the first two forloops
    #   stores transitions,  from just running the policy 50 timesteps
    #   samples additional goals (achieved ones, makes them for the Future time
    #   timesteps using the HER algorithm)
    #   next achieved goal represents goal achieved at next time step (s_t+1)
    #   here st represents -> observation and achieved goal at s_t
    
    #TODO: CHange this to use the minibatch model of adding it and then calling update normalizer
    def run_episode(self, mb):
        """Exactly matches HER pseudocode with proper tensor handling"""
        # Initialize - sample goal g and initial state s0
        obs_dict = self.env.reset()
        state = obs_dict['observation']
        original_goal = obs_dict['desired_goal']
        achieved_goal = obs_dict['achieved_goal']
        
        # Ensure valid initial state
        while torch.norm(achieved_goal - original_goal) <= 0.05:
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            original_goal = obs_dict['desired_goal']
            achieved_goal = obs_dict['achieved_goal']
        
        # Phase 1: Collect trajectory (first loop in pseudocode)
        # Initialize episode dictionary with empty lists
        episode_dict = {
            'state': [],
            'action': [],
            'achieved_goal': [],
            'desired_goal': [],
            'next_state': [],
            'next_achieved_goal': [],
            'done': []
        }

        # Run episode for up to 50 timesteps
        # equivalent of the sample action, execute action, observe next state
        for t in range(50):
            with torch.no_grad():  # Disable gradients for action selection
                # Sample action using behavior policy (returns tensor)
                action = self.agent.select_action(state, original_goal, train_mode=True)
                
                # Execute action and observe next state
                next_obs_dict, _, done, _ = self.env.step(action)
                
                # Store transition components (detach tensors to prevent gradient accumulation)
                episode_dict['state'].append(state.detach().cpu() if torch.is_tensor(state) else state)
                episode_dict['action'].append(action.detach().cpu() if torch.is_tensor(action) else action)
                episode_dict['achieved_goal'].append(achieved_goal.detach().cpu() if torch.is_tensor(achieved_goal) else achieved_goal)
                episode_dict['desired_goal'].append(original_goal.detach().cpu() if torch.is_tensor(original_goal) else original_goal)
                episode_dict['next_state'].append(next_obs_dict['observation'].detach().cpu() if torch.is_tensor(next_obs_dict['observation']) else next_obs_dict['observation'])
                episode_dict['next_achieved_goal'].append(next_obs_dict['achieved_goal'].detach().cpu() if torch.is_tensor(next_obs_dict['achieved_goal']) else next_obs_dict['achieved_goal'])
                episode_dict['done'].append(done)
                
                # Break if episode is done
                if done.all():
                    break
                    
                # Update current state and achieved goal
                state = next_obs_dict['observation']
                achieved_goal = next_obs_dict['achieved_goal']

        # Convert lists to stacked tensors/numpy arrays
        for key in episode_dict:
            if episode_dict[key] and torch.is_tensor(episode_dict[key][0]):
                episode_dict[key] = torch.stack(episode_dict[key])
            else:
                episode_dict[key] = np.array(episode_dict[key])
                
        #store the entire episode in the minibatch
        mb.append(dc(episode_dict))
                    
                    
                    
                    
        
    def store_transition_and_her_transition(self, mb):  
        for batch in mb:
            self.agent.buffer.
            
        
        
              
        #TODO: Tmrw figure out mini batch logic, check if this is correct
        for t, transition in enumerate(transitions):
            # 1. Original transition (sₜ||g, aₜ, rₜ, sₜ₊₁||g)
            self._store_single_transition(
                transition=transition,
                achieved_goal=transition['next_achieved_goal'],  # For reward calculation
                desired_goal=original_goal
            )
            
            # 2. HER: Sample 4 future goals (G = S(current episode))
            future_indices = np.random.randint(t, len(transitions), size=4)
            for idx in future_indices:
                g_prime = transitions[idx]['achieved_goal']  # g' = s_{future}
                
                self._store_single_transition(
                    transition=transition,
                    achieved_goal=transition['next_achieved_goal'],  # Still use next_achieved_goal from current transition
                    desired_goal=g_prime  # New desired goal
                )
        
        return len(transitions)

    def _store_single_transition(self, transition, achieved_goal, desired_goal):
        """Helper to store a single transition with proper reward calculation"""
        # Compute reward using next_achieved_goal vs desired_goal
        reward = self.env.compute_reward(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )
        
        # Concatenate state and goal (sₜ||g)
        
        
        # STORe single transition: 
        self.agent.buffer.store_single_transition({
            'state': transition['state'],
            'action': transition['action'],
            'reward': torch.tensor([reward], 
                                dtype=torch.float32, 
                                device=self.args.device),
            'next_state': transition['next_state'],
            'achieved_goal': transition['achieved_goal'],
            'next_achieved_goal': transition['next_achieved_goal'],
            'desired_goal': desired_goal,
            'done': torch.tensor([transition['done']], 
                                dtype=torch.bool, 
                                device=self.args.device)
        })

    def train(self):
        for epoch in range(self.MAX_EPOCHS):
            start_time = time.time()
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            
            for cycle in range(self.MAX_CYCLES):
                mb = []   #collect the mini batch for each sample
                # Collect episodes (innermost loop)

                # Training updates
                cycle_actor_loss = 0
                cycle_critic_loss = 0
                
                for _ in range(self.MAX_EPISODES):
                    self.run_episode(mb)  # Collect and store transitions
                    
                for n_update in range(self.num_updates):
                    actor_loss, critic_loss = self.agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss
                
                self.agent.update_networks()
                
                # Accumulate losses for epoch stats
                epoch_actor_loss += cycle_actor_loss / self.num_updates
                epoch_critic_loss += cycle_critic_loss / self.num_updates
            
            # Epoch evaluation and logging
            success_rate, avg_reward = self.evaluate()
            self.t_success_rate.append(success_rate)
            self.total_ac_loss.append(epoch_actor_loss)
            self.total_cr_loss.append(epoch_critic_loss)
            
            print(f"Epoch:{epoch}| "
                f"Success:{success_rate:.3f}| "
                f"Average REward:{avg_reward:.3f}| "
                f"Actor_Loss:{epoch_actor_loss:.3f}| "
                f"Critic_Loss:{epoch_critic_loss:.3f}| "
                f"Time:{time.time()-start_time:.1f}s")
            
            if epoch % 10 == 0:
                self.agent.save_checkpoint()
                self._log_metrics(epoch)

    def evaluate(self, num_episodes=10):
        total_success = 0
        total_reward = 0
        
        for _ in range(num_episodes):
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            desired_goal = obs_dict['desired_goal']
            episode_success = 0
            
            for t in range(50):
                with torch.no_grad():
                    action = self.agent.select_action(state, desired_goal, train_mode=False)
                next_obs_dict, reward, done, info = self.env.step(action)
                
                total_reward += reward.mean().item()
                episode_success += info.get('is_success', 0)
                
                if done.all():
                    break
                state = next_obs_dict['observation']
            
            total_success += 1 if episode_success > 0 else 0
        
        success_rate = total_success / num_episodes
        avg_reward = total_reward / num_episodes
        return success_rate, avg_reward

    def _log_metrics(self, epoch):
        """Log metrics to TensorBoard and save plots"""
        with SummaryWriter("logs") as writer:
            writer.add_scalar("Success_rate", self.t_success_rate[-1], epoch)
            writer.add_scalar("Actor_loss", self.total_ac_loss[-1], epoch)
            writer.add_scalar("Critic_loss", self.total_cr_loss[-1], epoch)
        
        plt.style.use('ggplot')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(np.arange(0, epoch+1), self.t_success_rate)
        ax1.set_title("Success Rate")
        
        ax2.plot(np.arange(0, epoch+1), self.total_ac_loss)
        ax2.set_title("Actor Loss")
        
        ax3.plot(np.arange(0, epoch+1), self.total_cr_loss)
        ax3.set_title("Critic Loss")
        
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.close()

    @staticmethod
    def _to_gb(in_bytes):
        return in_bytes / 1024 / 1024 / 1024

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", help="Enable visualization")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--checkpoint_path", default="checkpoint.pth", help="Path to save/load model")
    parser.add_argument("--load", action="store_true", help="Load checkpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = TrainDDPG(args)
    # trainer.train() 