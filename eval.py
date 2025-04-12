from fetch_pick_place_env import FrankaPickPlaceDDPG_Env
# from main import BaseDDPG
import torch
import genesis as gs


#TODO: MAKE TJHIS WORK
from rl_modules.ddpg_agent import DDPG_HER_AGENT, DDPG_Actor, DDPG_Critic
# something like: https://github.com/alirezakazemipour/DDPG-HER/blob/master/play.py
import argparse

gs.init(backend=gs.gpu, precision="32")


class EvalDDPG:
    def __init__(self, args):
        # super().__init__(args)
        self.env = FrankaPickPlaceDDPG_Env(vis=args.vis, device=args.device, num_envs=args.num_envs)
        print("LOADING FROM PATH: " + str(args.load_path))
        self.env_params = {
            'obs': self.env.state_dim,
            'goal': self.env.goal_dim,
            'action': self.env.action_dim,
            'action_max': 1.0,
            'max_timesteps': 50,
            'num_envs': args.num_envs,
            'reward_func': self.env.compute_reward
        }
        args.load = True
        self.agent = DDPG_HER_AGENT(
            env_params=self.env_params,
            env=self.env,
            device=args.device,
            checkpoint_path=args.load_path,
            load=args.load
        )
        
        if args.device == "mps":
            print("Running with mps")
            gs.tools.run_in_another_thread(
                fn=lambda: self.run_evaluation(), 
                args=()
            )            
            self.env.scene.viewer.start()
        
    # def _load_model(self, model_path):
    #     o_mean, o_std, g_mean, g_std, model = torch.load(model_path)
    #     actor_network = actor(env_params)  # Your actor network
    #     actor_network.load_state_dict(model)
    #     actor_network.eval()
    #     return {
    #         'network': actor_network,
    #         'normalization': (o_mean, o_std, g_mean, g_std)
    #     }
    
    def run_evaluation(self):
        i = 0
        while True:
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            goal = obs_dict['desired_goal']
        
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    action = self.agent.select_action(
                    state=state, goal=goal
                    ).cpu().numpy()
                
                next_obs_dict, _, done, _ = self.env.step(action)
                state = next_obs_dict['observation']
                
                if done.all():
                    break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, 
                       help="Enable visualization") 
    parser.add_argument("-l", "--load_path", type=str, nargs='?', default=None,
                       help="Path for loading model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1,
                       help="Number of environments to create") 
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock",
                       help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                       help="device: cpu or cuda:x or mps for macos")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = EvalDDPG(args)
    evaluator.run_evaluation()