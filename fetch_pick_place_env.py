import numpy as np
import genesis as gs
import torch
from numpy import random 

#TODO TMRW 
#1. rewrite this stuff to actually be based on the environment
# https://robotics.farama.org/envs/fetch/pick_and_place/
# once ur done, go through the entire training loop of the DDPG her repo and 
# fix stuff accordingly. 
class FrankaPickPlaceDDPG_Env:
    def __init__(self, vis=False, device='mps', num_envs=1, place_only=False):
        self.place_only = place_only
        print("PLACE ONLY: " + str(place_only))
        self.goal_index = 0
        self.device = device
        self.action_space = 4  # end effector x, y, z, finger disp.
        # gripper pos (3) + block pos (3) + block to gripper (3) + finger displacement (2)
        self.state_dim = 11   # doesn't include goal/action dims
        self.goal_dim = 3      # goal position
        self.num_envs = num_envs
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=120,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,
        )
        
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.65, 0.0, 0.02),
            )
        )
        
        self.goal_target = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                visualization=True,
                collision=False,
                requires_jac_and_IK=False,
                fixed=True,
                radius=0.04
            )
        )
        
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.build_env()
        
        # For DDPG compatibility
        self.observation_space = (self.state_dim,)
        self.action_dim = self.action_space
        self.action_space = (self.action_space,)
        
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        
        # Initial franka position -> change 1.3662 to 1.35 to be lower
        # [0] Shoulder yaw, [1] Shoulder pitch, [2] Elbow pitch, [3] Forearm yaw, [4] Wrist pitch, [5] Wrist yaw, [6] Wrist pitch (near gripper)
        franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.0, 0.0]).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        
        # Initial end effector target original 0.135
        pos = torch.tensor([1.65, -1.2, 0.135], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        
        
        
        # if self.place_only:
        #     pos = torch.tensor([0.65, 0.0, 0.02], dtype=torch.float32, device=self.device)
        #     self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        #     quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        #     self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
            
        
        # Pre-generate goal positions
        self.target_poses = []
        default_pos = np.array([0.7, 0.0, 0])
        
        
        #random version
        for _ in range(12):
            # default range
            offset = np.array([random.rand() * 0.2, random.rand() * 0.6 - 0.3, 0.35 * random.rand() + 0.1])
            #less picky range
            # offset = np.array([random.rand() * 0.1, random.rand() * 0.4 - 0.2, 0.2 * random.rand() + 0.1])

            target_pos = default_pos + offset
            target_pos = np.repeat(target_pos[np.newaxis], self.num_envs, axis=0)
            self.target_poses.append(target_pos)
        
        
        # offsets = [
        #     np.array([0.10, -0.20, 0.30]),  # max in all directions
        #     np.array([0.00,  0.20, 0.28]),  # min x, max y
        #     np.array([0.09, -0.18, 0.12]),  # high x, low y, low z
        #     np.array([0.01,  0.15, 0.25]),  # low x, high y
        # ]

        # # Cycle through these offsets for each iteration
        # for i in range(2000):
        #     offset = offsets[i % 4]  # Cycle through the 4 offsets
        #     target_pos = default_pos + offset
        #     target_pos = np.repeat(target_pos[np.newaxis], self.num_envs, axis=0)
        #     self.target_poses.append(target_pos)
        # self.goal_index = 0
        
    def reset(self):
        
        # Reset cube position -> forward, side to side, vertical
        if not self.place_only: 
            cube_pos = np.array([0.65, 0.0, 0.02])
        else:
            cube_pos = np.array([0.65, 0.0, 0.06])

        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        
        self.build_env()

        # Set new goal position
        print("GOAL INDEX: " + str(self.goal_index))
        print("INDEX: " + str(self.goal_index % len(self.target_poses)))
        goal_pos = self.target_poses[self.goal_index % len(self.target_poses)]
        self.goal_target.set_pos(goal_pos, envs_idx=self.envs_idx)  #we already did the repeat earlier
        self.goal_index += 1
        
        # Reset end effector position
        
        # self.pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
        # self.pos = self.pos.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Get initsial observation
        obs = self._get_obs()
        
        # Return dictionary format for HER
        return {
            'observation': obs['observation'],
            'achieved_goal': obs['achieved_goal'],
            'desired_goal': obs['desired_goal']
        }
    
    
    #TODO: BEFORE TESTING< TEST IF THESE VALUES ARE ACCURATE
    def _get_obs(self):
        """Returns observation in dictionary format for HER"""
        left_pos = self.franka.get_link("left_finger").get_pos()
        right_pos = self.franka.get_link("right_finger").get_pos()
        
        gripper_pos = (left_pos + right_pos) / 2
        cube_pos = (self.cube.get_pos())
        goal_pos = self.goal_target.get_pos()
        
        # print("gripper_pos shape:", gripper_pos.shape)       # Expected shape: (3,) or (batch_size, 3)
        # print("cube_pos shape:", cube_pos.shape)             # Expected shape: (3,) or (batch_size, 3)
        # print("AFTER")
        
        # Convert to numpy arrays  -> this removes [1,3] -> [3]
        # gripper_pos = gripper_pos.squeeze(0)
        # cube_pos = cube_pos.squeeze(0)
        # goal_pos = goal_pos.squeeze(0)
        
        
        
        cube_distance = cube_pos - gripper_pos
        
        grip_midpoint = gripper_pos[:, 1]  # Multi-env: slice
        lfinger_disp = (left_pos[:, 1] - grip_midpoint).unsqueeze(-1)  # Will be negative
        rfinger_disp = (right_pos[:, 1] - grip_midpoint).unsqueeze(-1)  # Will be positive

        observation = torch.cat([
            gripper_pos,      # Gripper position (3)
            cube_pos,         # Cube position (3)
            cube_distance,    # Distance cube to grippers
            rfinger_disp,
            lfinger_disp
        ], dim=-1)
        # print("gripper_pos shape:", gripper_pos.shape)       # Expected shape: (3,) or (batch_size, 3)
        # print("cube_pos shape:", cube_pos.shape)             # Expected shape: (3,) or (batch_size, 3)
        # print("cube_distance shape:", cube_distance.shape)   # Expected shape: (1,) or (batch_size, 1)
        # print("rfinger_disp shape:", rfinger_disp.shape)     # Expected shape: (1,) or (batch_size, 1)
        # print("lfinger_disp shape:", lfinger_disp.shape)     # Expected shape: (1,) or (batch_size, 1))



        
        # The achieved goal is the cube position
        achieved_goal = cube_pos.clone()
        
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': goal_pos
        }
        
        
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute reward for HER, handling both NumPy arrays and PyTorch tensors"""
        # print("DESIRED GOAL: " + str(desired_goal))

        # Convert NumPy arrays to PyTorch tensors
        if isinstance(achieved_goal, np.ndarray):
            achieved_goal = torch.tensor(achieved_goal, dtype=torch.float32)
        if isinstance(desired_goal, np.ndarray):
            desired_goal = torch.tensor(desired_goal, dtype=torch.float32)

        # Ensure both tensors are float and on the same device
        achieved_goal = achieved_goal.to(torch.float32)
        desired_goal = desired_goal.to(torch.float32)

        if achieved_goal.device != desired_goal.device:
            desired_goal = desired_goal.to(achieved_goal.device)

        # Compute distance
        distances = torch.norm(achieved_goal - desired_goal, dim=-1)
        # print("DISTANCE:", distances)
        
        # Compute reward
        reward = -(distances > 0.08).float()
        any_zero = (distances.eq(0)).any()
        print("DISTANCES: " + str(distances) + " REWARD IS :", str(reward))

        if any_zero:
            print("FOUND ZERO: IS:", reward)
            print(distances)
        return reward
    

    def _check_done(self):
        """Returns done flags for all environments as a tensor [num_envs, 1]"""
        goal_pos = self.goal_target.get_pos()  # shape [num_envs, 3]
        cube_pos = self.cube.get_pos()  # shape [num_envs, 3]
        # print("GOAL POS IS: " + str(goal_pos))

        # Compute distances for all environments
        distances = torch.norm(cube_pos - goal_pos, dim=1, keepdim=True)  # shape [num_envs, 1]
        
        # Return boolean tensor indicating which envs are done
        return distances < 0.08  # shape [num_envs, 1]
    
    #TODO: Check if the action space is accurates -> compare with their code and try to see if 
    #   you are able to run their demo 
    def step(self, action):
        # print("ACTION: " + str(action))

        # Convert action to tensor if it's numpy
        if isinstance(action, np.ndarray):
            # print("ACTION: " + str(action))
            # print("ACTION SHAPE: " + str(action.shape))
            action = torch.from_numpy(action).to(self.device)
        
        # Reshape action if needed
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        # Scale actions to real-world units
        delta_pos = action[:, :3] * 0.05  #should be 5cm max movement
        # delta_pos[:, 0] = 0   #TODO: debug arm moving right 
        # delta_pos[:, 1] = 0.5    # - moves left, + moves right
        # delta_pos[:, 2] = 0
        
        action[:, 3] = -1  #force close, TODO: UNcomment this
        gripper_cmd = action[:, 3]

        # print(gripper_cmd)
        
        # Update position
        self.pos += delta_pos
        # print(self.pos)
        
        # Continuous gripper control (0=closed, 0.04=open)
        finger_width = (1 + gripper_cmd) * 0.02  # Map [-1,1]→[0,0.04]
        finger_pos = torch.stack([finger_width, finger_width], dim=1)  # Both fingers
        
        # Inverse kinematics
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        
        # Execute movements
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        # if not self.place_only:
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        done = (reward == 0)

        # print("REWARD VS DONE")
        # print(reward)
        # print(done)

        if (done == 1).any():
            print("IS DONE")


        info = {
            'is_success': done,
            'achieved_goal': obs['achieved_goal'],
            'desired_goal': obs['desired_goal']
        }
        
        return obs, reward, done, info

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]