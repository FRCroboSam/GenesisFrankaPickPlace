import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from rl_modules.ddpg_agent import DDPG_HER_AGENT, DDPG_Actor, DDPG_Critic
from fetch_pick_place_env import FrankaPickPlaceDDPG_Env
import genesis as gs
from copy import deepcopy as dc
import argparse

gs.init(backend=gs.gpu, precision="32")


#TODOS:
#   debug why this doesnt work in the single environment case -> spend like 30 min
#   make a better policy -> try training for a lot longer, see if it improves at all by running eval
class TrainDDPG:
    def __init__(self, args):
        self.train_metrics = {
            'epochs': [],
            'success_rates': [],
            'actor_losses': [],
            'critic_losses': [],
            'avg_rewards': []
        }
        self.args = args
        self.env = FrankaPickPlaceDDPG_Env(vis=args.vis, device=args.device, num_envs=args.num_envs)
        
        self.env_params = {
            'obs': self.env.state_dim,
            'goal': self.env.goal_dim,
            'action': self.env.action_dim,
            'action_max': 1.0,
            'max_timesteps': 50,
            'num_envs': args.num_envs,
            'reward_func': self.env.compute_reward
        }
        
        self.agent = DDPG_HER_AGENT(
            env_params=self.env_params,
            env=self.env,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            load=args.load
        )
        
        self.MAX_EPOCHS = 50 #TODO: Should be 50
        self.MAX_CYCLES = 1 #TODO: SHould be 50
        self.MAX_EPISODES = 1 #50
        self.num_updates = 40 #TODO: should be 40
        
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
    def run_episode(self):
        #TODO: check if this is the right loop order
        """Exactly matches HER pseudocode with proper tensor handling"""
        # Initialize - sample goal g and initial state s0
        print("RUNNING EPISODE")
        
        
        obs_dict = self.env.reset()
        state = obs_dict['observation']
        goal = obs_dict['desired_goal']
        achieved_goal = obs_dict['achieved_goal']
        previous_done = None
        #TODO: CHANGE THESE TO BE EPISODES
        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
        
        # Ensure valid initial state -> reset if the achieved goal is already desired goal
        while torch.norm(achieved_goal - goal) <= 0.05:
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            goal = obs_dict['desired_goal']
            achieved_goal = obs_dict['achieved_goal']
        print("STARTING GOAL IS: " + str(goal))
        
        for t in range(self.env_params['max_timesteps']):
            with torch.no_grad():
                    action = self.agent.select_action(
                    state=state, goal=goal, done_mask=previous_done
                    ).cpu().numpy()
            
            ep_obs.append(state.clone().cpu())
            ep_ag.append(achieved_goal.clone().cpu())
            ep_g.append(goal.clone().cpu()) 
            ep_actions.append(action.copy())
            
            next_obs_dict, _, done, _ = self.env.step(action)
            state = next_obs_dict['observation']
            achieved_goal = next_obs_dict['achieved_goal']
            
            
            #TODO: make a policy smart so that when it achieves a goal in an environment
                #it knows to not do anything -> look at pseudocode
            done[0] = True
            previous_done = done
            if done.any():
                print("ONE OF THEM IS DONE")
            if done.all():
                print("ALL OF THEM ARE DONE")
                # break
            
        
        ep_obs.append(state.clone().cpu())
        ep_ag.append(achieved_goal.clone().cpu())
        # ep_g.append(goal.clone().cpu()) #keep this if u want shape 51
        # ep_actions.append(action.copy())
        # print("PRINTING EP OBS")
        # print(str(ep_obs))
        #TODO: FIGURE out how to 
        ep_obs = np.array(ep_obs)
        ep_ag = np.array(ep_ag)
        ep_g = np.array(ep_g)
        ep_actions = np.array(ep_actions)
        
        return [ep_obs, ep_ag, ep_g, ep_actions]
        

    def train(self):
        for epoch in range(self.MAX_EPOCHS):
            start_time = time.time()
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            
            for cycle in range(self.MAX_CYCLES):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []

                for _ in range(self.MAX_EPISODES):
                    #TODO :FIGURE OUT IF THIS IS THE CORRECT PLACE TO STORE MB 
                    ep_batch = self.run_episode()
                    mb_obs.append(ep_batch[0])
                    mb_ag.append(ep_batch[1])
                    mb_g.append(ep_batch[2])
                    mb_actions.append(ep_batch[3])
                print("LEN MB_OBS: " + str(len(mb_obs)))
                print(mb_obs)
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb = [mb_obs, mb_ag, mb_g, mb_actions]
                
                #TODO: Fix shape bug tmrw
                self.agent.buffer.store_episode(mb)
                #END OF STAGE 1: storing transition for Standard experience replay

                #TODO: ADD STUFF TO MINIBATCH
                self.agent.update_normalizer(mb)
                    #END OF STAGE 1
                cycle_actor_loss = 0
                cycle_critic_loss = 0
                for _ in range(self.num_updates):
                    #TODO: FIX BUGS
                    actor_loss, critic_loss = self.agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss
                
                epoch_actor_loss += cycle_actor_loss / self.num_updates
                epoch_critic_loss += cycle_critic_loss / self.num_updates
            
            success_rate, avg_reward = self.evaluate()
            self.t_success_rate.append(success_rate)
            self.total_ac_loss.append(epoch_actor_loss)
            self.total_cr_loss.append(epoch_critic_loss)
            
            print(f"Epoch:{epoch}| Success:{success_rate:.3f}| Avg Reward:{avg_reward:.3f}| "
                  f"Actor Loss:{epoch_actor_loss:.3f}| Critic Loss:{epoch_critic_loss:.3f}| "
                  f"Time:{time.time()-start_time:.1f}s")
            
            # if epoch % 10 == 0:
            self.agent.save_checkpoint()
            self._log_metrics(epoch, success_rate, epoch_actor_loss, epoch_critic_loss, avg_reward)
        self.plot_metrics()

    def evaluate(self, num_episodes=10):
        total_success = 0
        total_reward = 0
        previous_done = None

        for _ in range(num_episodes):
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            goal = obs_dict['desired_goal']
            
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    
                    action = self.agent.select_action(
                    state=state, goal=goal, done_mask=previous_done
                    ).cpu().numpy()
                
                next_obs_dict, reward, done, info = self.env.step(action)
                previous_done = done
                total_reward += reward.mean().item()
                state = next_obs_dict['observation']
                done[0] = True
                if done.all():
                    total_success += info.get('is_success', 0)
                    break
        
        return total_success / num_episodes, total_reward / num_episodes

    def _log_metrics(self, epoch, success_rate, actor_loss, critic_loss, avg_reward):
        """Store metrics without plotting"""
        self.train_metrics['epochs'].append(epoch)
        self.train_metrics['success_rates'].append(success_rate)
        self.train_metrics['actor_losses'].append(actor_loss)
        self.train_metrics['critic_losses'].append(critic_loss)
        self.train_metrics['avg_rewards'].append(avg_reward)
        
        # Still write to tensorboard if needed
        with SummaryWriter("logs") as writer:
            writer.add_scalar("Success_rate", success_rate, epoch)
            writer.add_scalar("Actor_loss", actor_loss, epoch)
            writer.add_scalar("Critic_loss", critic_loss, epoch)
    @staticmethod
    def _to_gb(in_bytes):
        return in_bytes / 1024 / 1024 / 1024
    def plot_metrics(self):
        print("Plotting metrics")
        """Plot metrics after training is complete"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 12))
        
        # Success Rate
        plt.subplot(3, 1, 1)
        plt.plot(self.train_metrics['epochs'], self.train_metrics['success_rates'])
        plt.title("Success Rate")
        plt.xlabel("Epoch")
        
        # Actor Loss
        plt.subplot(3, 1, 2)
        plt.plot(self.train_metrics['epochs'], self.train_metrics['actor_losses'])
        plt.title("Actor Loss")
        plt.xlabel("Epoch")
        
        # Critic Loss
        plt.subplot(3, 1, 3)
        plt.plot(self.train_metrics['epochs'], self.train_metrics['critic_losses'])
        plt.title("Critic Loss")
        plt.xlabel("Epoch")
        
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.close()
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
    trainer.train() 