import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco


class T1ShootingEnv(MujocoEnv, utils.EzPickle):
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

        # 68차원 관측 공간
        observation_space = Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float64)
        
        MujocoEnv.__init__(self, model_path=xml_path, frame_skip=5, observation_space=observation_space, default_camera_config=None, render_mode=render_mode)

        self.left_foot_id = self.data.body("left_foot_link").id
        self.right_foot_id = self.data.body("right_foot_link").id
        self.ball_body_id = self.data.body("ball").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        
        ball_joint = self.model.joint("ball_free_joint")
        self.ball_qpos_adr = int(ball_joint.qposadr)
        self.ball_qvel_adr = int(ball_joint.dofadr)
        
        self.goal_pos = np.array([11.0, 0.0, 0.0])
        self.goal_width = 5.0
        
        # 변수 초기화
        self.last_action = np.zeros(23)
        self.has_touched_ball = False
        self.last_action_touched = False
        self.step_counter = 0

    def check_contact(self, body1_name, body2_name):

        body1_id = self.data.body(body1_name).id
        body2_id = self.data.body(body2_name).id
        
        for contact in self.data.contact:
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            if (geom1_body == body1_id and geom2_body == body2_id) or \
               (geom1_body == body2_id and geom2_body == body1_id):
                return True
        return False
    
    def _quat_to_yaw(self, quat):

        w, x, y, z = quat
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def step(self, action):
        self.step_counter += 1

        terminated = False
        
        # 2~11 ( 팔만 고정 )
        action[2:11] = 0.0 

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        

        trunk_pos = self.data.body("Trunk").xpos
        trunk_quat = self.data.body("Trunk").xquat 
        l_foot_pos = self.data.xpos[self.left_foot_id]
        r_foot_pos = self.data.xpos[self.right_foot_id]
        ball_pos = self.data.body("ball").xpos
        
        ball_vel_vec = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        ball_speed = np.linalg.norm(ball_vel_vec)
        
        # 2. Active Sensing: Gaze Alignment (Numpy) (논문 발췌)
        trunk_yaw = self._quat_to_yaw(trunk_quat)
        head_yaw_angle = self.data.qpos[7] 
        global_head_yaw = trunk_yaw + head_yaw_angle
        
        vec_to_ball = ball_pos - trunk_pos
        target_yaw = np.arctan2(vec_to_ball[1], vec_to_ball[0])
        
        yaw_error = target_yaw - global_head_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi 

        # 볼 터치 유무 판단
        is_contact = self.check_contact("left_foot_link", "ball")
        if is_contact and not self.has_touched_ball:
            self.has_touched_ball = True

        reward = 0.0
        is_healthy = trunk_pos[2] > 0.55
        
        l_foot_grounded = l_foot_pos[2] < 0.05
        r_foot_grounded = r_foot_pos[2] < 0.05
        
        if not l_foot_grounded and not r_foot_grounded:
            reward -= 5.0 

        # 넘어짐 판단 (논문 발췌)
        if is_healthy: 
            reward += 1.0 # Alive Reward
            
            # [Gaze Reward]
            reward += 1.0 * np.exp(-5.0 * yaw_error**2)

            # [Feet Distance]
            foot_dist = np.linalg.norm(l_foot_pos - r_foot_pos)
            reward += 1.0 * np.exp(-20.0 * (foot_dist - 0.25)**2)

            # [Action Smoothness]
            smoothness_penalty = np.sum(np.square(action - self.last_action))
            reward -= 0.1 * smoothness_penalty 

            # [Height Maintenance]
            height_err = abs(trunk_pos[2] - 0.8)
            reward += 0.5 * np.exp(-10.0 * height_err)
            
        if not self.has_touched_ball:
            dist_lfoot_ball = np.linalg.norm(l_foot_pos - ball_pos)
            reward += 2.0 * np.exp(-10.0 * dist_lfoot_ball)
            
            if self.step_counter > 1000: 
                reward -= 0.1 

        if is_contact and self.step_counter > 1: 
            reward += 1.0
            if self.has_touched_ball and (self.last_action_touched == False): 
                reward += 50.0

        self.last_action_touched = self.has_touched_ball

        # 민혁승우태중 이거 잘 확인해서 고쳐야함 셋 다 다름
        if self.has_touched_ball:
            
            if ball_speed > 0.1:
                reward += 3.0 * (np.exp(ball_speed) - 1.0)
            
            vec_to_goal = self.goal_pos - ball_pos
            vec_to_goal_norm = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-8)
            ball_dir_norm = ball_vel_vec / (ball_speed + 1e-8)
            direction_cosine = np.dot(ball_dir_norm, vec_to_goal_norm)
            
            if ball_speed > 0.5:
                reward += 2.0 * direction_cosine
                
               
                if direction_cosine < 0.2: 
                    reward -= 50.0
                    terminated = True
                    return observation, reward, terminated, False, {}

            # 공을 찬 후에는 움직이지 말고 버텨라는 조건
            joint_vel_sum = np.square(self.data.qvel[6:]).sum()
            if joint_vel_sum < 2.0:
                reward += 2.0

                if l_foot_grounded and r_foot_grounded:
                    reward += 2.0
            

            if ball_pos[0] > 10.5 and abs(ball_pos[1]) < 2.5:
                reward += 2000.0
                terminated = True
                return observation, reward, terminated, False, {}


        # 나는 공이 골대 라인 기준 좌우 3.0m를 벗어나면 즉시 종료하도록 함
        if abs(ball_pos[1]) > 3.0:
            reward -= 1000.0
            terminated = True
            return observation, reward, terminated, False, {}

        reward -= 0.05 * np.square(action).sum()

        if not is_healthy:
            if not self.has_touched_ball:
                reward -= 1500.0
                terminated = True
            else:
                reward -= 100.0 
                terminated = False 

        if self.step_counter >= 5000:
            if not self.has_touched_ball:
                reward -= 2000.0 # 공도 못 차고 시간만 끔
            else:
                reward -= 1000.0 # 찼는데 골 못 넣음
            terminated = True

        self.last_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, {}
    
    def _get_obs(self):
        ball_pos = self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        n_joints = 23
        qpos_robot = self.data.qpos.flat[7 : 7+n_joints] 
        qvel_robot = self.data.qvel.flat[6 : 6+n_joints]
        vec_to_goal = self.goal_pos - ball_pos
        current_obs = np.concatenate([ball_pos, ball_vel, qpos_robot, qvel_robot, vec_to_goal])
        
        target_dim = 68
        if len(current_obs) < target_dim:
            current_obs = np.concatenate([current_obs, np.zeros(target_dim - len(current_obs))])
        elif len(current_obs) > target_dim:
            current_obs = current_obs[:target_dim]
        return current_obs

    def reset_model(self):
        self.last_action = np.zeros(23) 
        self.has_touched_ball = False
        self.last_action_touched = False
        self.step_counter = 0
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[7:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq-7)
        
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data) 
        
        l_foot_pos = self.data.xpos[self.left_foot_id]
        
        # 볼 위치 노이즈 추가
        ball_x = l_foot_pos[0] + 0.45 + self.np_random.uniform(-0.05, 0.05)
        ball_y = l_foot_pos[1] + self.np_random.uniform(-0.05, 0.05)
        
        qpos[self.ball_qpos_adr] = ball_x
        qpos[self.ball_qpos_adr+1] = ball_y
        qpos[self.ball_qpos_adr+2] = 0.11
        qvel[self.ball_qvel_adr:self.ball_qvel_adr+6] = 0.0
        
        self.set_state(qpos, qvel)
        return self._get_obs()