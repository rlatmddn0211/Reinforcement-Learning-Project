import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco

class T1ShootingStage2Env(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200, 
    }

    def __init__(self, xml_file='t1.xml', render_mode=None):
        utils.EzPickle.__init__(self, xml_file, render_mode)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(xml_file):
            xml_path = xml_file
        else:
            xml_path = os.path.join(current_dir, xml_file)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float64)
        
        MujocoEnv.__init__(self, model_path=xml_path, frame_skip=5, observation_space=observation_space, default_camera_config=None, render_mode=render_mode)

        self.left_foot_id = self.data.body("left_foot_link").id
        self.right_foot_id = self.data.body("right_foot_link").id
        
        ball_joint = self.model.joint("ball_free_joint")
        self.ball_qpos_adr = int(ball_joint.qposadr)
        self.ball_qvel_adr = int(ball_joint.dofadr)
        
        self.goal_pos = np.array([11.0, 0.0, 0.0])
        
        self.last_action = np.zeros(23)
        self.step_counter = 0

    def step(self, action):
        self.step_counter += 1

        action[:11] = 0.0  

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        trunk_pos = self.data.body("Trunk").xpos
        
        # 공 정보
        ball_pos = self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        
        reward = 0.0
        
        is_healthy = trunk_pos[2] > 0.53
        
        if is_healthy:
            # 1. 생존 보상
            reward += 1.0 
            
            # 2. 자세 보상 (완화됨)
            trunk_rot = self.data.body("Trunk").xmat.reshape(3, 3)
            trunk_z_axis = trunk_rot[:, 2]
            orientation_reward = 1.0 * np.exp(-5.0 * (1.0 - trunk_z_axis[2])**2)
            reward += orientation_reward
            
            # (A) 공 속도 보상 (유지)
            ball_velocity_x = ball_vel[0]
            
            # (A) 슈팅 보상 
            if ball_velocity_x > 0.1: 
                reward += 5.0 * ball_velocity_x 
            
            # (B) 접근 보상 
            
            if ball_velocity_x <= 0.1:  # 공이 멈춰있거나 느릴 때만 유효
                dist_to_ball = np.linalg.norm(trunk_pos[:2] - ball_pos[:2])
                approach_reward = 2.0 * np.exp(-5.0 * dist_to_ball)
                reward += approach_reward

            # (C) 킥 존 보너스 (유지)
            l_foot_pos = self.data.xpos[self.left_foot_id]
            r_foot_pos = self.data.xpos[self.right_foot_id]
            
            dist_l = np.linalg.norm(l_foot_pos - ball_pos)
            dist_r = np.linalg.norm(r_foot_pos - ball_pos)
            
            if dist_l < 0.12 or dist_r < 0.12: # 12cm 이내 (거의 접촉)
                reward += 5.0  
            elif dist_l < 0.2 or dist_r < 0.2: # 20cm 이내 (접근 중)
                reward += 1.0


        reward -= 0.05 * np.square(action).sum()

        terminated = not is_healthy
        if terminated:
            reward -= 100.0 

        self.last_action = action.copy()

        return observation, reward, terminated, False, {}

    def _get_obs(self):
        ball_pos = self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        n_joints = 23
        qpos_robot = self.data.qpos.flat[7 : 7+n_joints] 
        qvel_robot = self.data.qvel.flat[6 : 6+n_joints]
        observation_noise = np.random.normal(0, 0.005, size=55)
        vec_to_goal = self.goal_pos - ball_pos
        current_obs = np.concatenate([ball_pos, ball_vel, qpos_robot, qvel_robot, vec_to_goal])
        current_obs += observation_noise 
        target_dim = 68
        if len(current_obs) < target_dim:
            current_obs = np.concatenate([current_obs, np.zeros(target_dim - len(current_obs))])
        elif len(current_obs) > target_dim:
            current_obs = current_obs[:target_dim]
        return current_obs

    def reset_model(self):
        self.last_action = np.zeros(23) 
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[7:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq-7)
        
        ball_x = 0.13 + self.np_random.uniform(0.0, 0.05) 
        
        ball_y = self.np_random.uniform(-0.05, 0.05)
        
        qpos[self.ball_qpos_adr] = ball_x
        qpos[self.ball_qpos_adr+1] = ball_y
        qpos[self.ball_qpos_adr+2] = 0.11
        qvel[self.ball_qvel_adr:self.ball_qvel_adr+6] = 0.0
        
        self.set_state(qpos, qvel)
        self.step_counter = 0
        return self._get_obs()