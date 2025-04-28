#TODO: fix this

import os
import subprocess

# Directory containing all checkpoint files
models_dir = "models/fullrun"

# List all .pth files
checkpoints = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

for ckpt in checkpoints:
    ckpt_path = os.path.join(models_dir, ckpt)
    
    print(f"Running evaluation for checkpoint: {ckpt_path}")
    
    # Run eval.py with appropriate arguments
    subprocess.run([
        "python", "eval.py",
        "--vis",
        "--load_path", ckpt_path,
        "--task", "FrankaPickPlace",
        "--device", "mps"
    ])



class EvalDDPG:
    def __init__(self, args):
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

        # Track success
        self.total_cycles = 20
        self.success_count = 0
        
        if args.device == "mps":
            print("Running with mps")
            gs.tools.run_in_another_thread(
                fn=lambda: self.run_evaluation(), 
                args=()
            )            
            self.env.scene.viewer.start()
    
    def run_evaluation(self):
        prev_done = None
        for cycle in range(self.total_cycles):
            obs_dict = self.env.reset()
            state = obs_dict['observation']
            goal = obs_dict['desired_goal']
            success = False
        
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    action = self.agent.select_action(
                        state=state, goal=goal, done_mask=prev_done, train_mode=False
                    ).cpu().numpy()
                
                next_obs_dict, _, done, _ = self.env.step(action)
                prev_done = done
                state = next_obs_dict['observation']
                
                if done.all():  # All environments report done
                    success = True
                    break

            if success:
                self.success_count += 1

        print(f"Evaluation complete. Successes: {self.success_count}/{self.total_cycles}")
