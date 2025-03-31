from main import BaseDDPG
import torch

class EvalDDPG(BaseDDPG):
    def __init__(self, args):
        super().__init__(args)
        self.env = self._create_env(args.task, args.vis, args.device)
        self.model = self._load_model(args.model_path)
        
    def _load_model(self, model_path):
        o_mean, o_std, g_mean, g_std, model = torch.load(model_path)
        actor_network = actor(env_params)  # Your actor network
        actor_network.load_state_dict(model)
        actor_network.eval()
        return {
            'network': actor_network,
            'normalization': (o_mean, o_std, g_mean, g_std)
        }
    
    def run_evaluation(self):
        for i in range(self.args.demo_length):
            obs, goal = self.env.reset()
            for t in range(self.env._max_episode_steps):
                inputs = self.process_inputs(
                    obs, goal, *self.model['normalization'],
                    self.args.clip_obs, self.args.clip_range
                )
                with torch.no_grad():
                    action = self.model['network'](inputs).numpy().squeeze()
                obs_new, _, _, info = self.env.step(action)
                obs = obs_new
                if info['is_success']:
                    break
            print(f'Episode {i}, Success: {info["is_success"]}')

def parse_eval_args():
    parser = BaseDDPG().parse_args()
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--demo_length", type=int, default=10, help="Number of demo episodes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = EvalDDPG(args)
    evaluator.run_evaluation()