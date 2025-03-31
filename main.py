# main.py
import argparse
from train_ddpg import TrainDDPG  # Assuming the class is in train_ddpg.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, 
                       help="Enable visualization") 
    parser.add_argument("-l", "--load_path", type=str, nargs='?', default=None,
                       help="Path for loading model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1,
                       help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("-r", "--replay_size", type=int, default=None,
                       help="Size of replay buffer")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64,
                       help="Hidden dimension for the network")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock",
                       help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                       help="device: cpu or cuda:x or mps for macos")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = TrainDDPG(args)
    trainer.start_training()