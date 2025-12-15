import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco
from mujoco import MjModel, MjData

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'booster_t1')
MODEL_XML_PATH = os.path.join(ASSET_DIR, 't1.xml')
class T1KickEnv3(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.model = MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = MjData(self.model)

        # 초기화 및 ID 찾기
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.target_height = self.data.qpos[2]

        try:
            self.left_foot_id = self.model.body("left_foot_link").id
            self.right_foot_id = self.model.body("right_foot_link").id
            ball_joint = self.model.joint("ball_free_joint")
            self.ball_qpos_adr = int(ball_joint.qposadr)
            self.ball_qvel_adr = int(ball_joint.dofadr)
        except KeyError as e:
            print(f"ID 찾기 실패: {e}")
            raise e

        # Action Mapping
        self.ctrl_qpos_indices = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0] 
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.ctrl_qpos_indices.append(qpos_adr)
        self.ctrl_qpos_indices = np.array(self.ctrl_qpos_indices, dtype=np.int32)

        # Spaces
        n_actuators = self.model.nu 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32)
        
        # Obs: Robot(46) + Ball(6) + Goal(3) = 55
        obs_dim = (self.model.nq - 7) + (self.model.nv - 6) + 4 + 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.mujoco_renderer = None
        self.prev_action = np.zeros(self.model.nu)
        self.step_counter = 0
        self.max_steps = 7500 # 500Hz * 15s
        self.goal_pos = np.array([11.0, 0.0, 0.0])
        self.goal_width = 5.0
        self.goal_height = 2.0
        self.has_touched_ball = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qvel[:] = 0.0
        
        # 로봇 노이즈
        self.data.qpos[7:] += np.random.uniform(-0.01, 0.01, size=self.model.nq-7)

        # 공 위치 (발 앞 0.45m)
        ball_adr = self.ball_qpos_adr
        self.data.qpos[ball_adr] = 0.45 + np.random.uniform(-0.05, 0.05)
        self.data.qpos[ball_adr+1] = np.random.uniform(-0.05, 0.05)
        self.data.qpos[ball_adr+2] = 0.1
        self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.model.nu)
        self.has_touched_ball = False
        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1

        alpha = 0.8
        filtered_action = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = filtered_action

        default_ctrl = self.initial_qpos[self.ctrl_qpos_indices].copy()
        target_ctrl = default_ctrl + (filtered_action * 1.2)
        target_ctrl = np.clip(target_ctrl, -5.0, 5.0)
        self.data.ctrl[:] = target_ctrl
        
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_done()
        

        truncated = False
        if self.step_counter >= self.max_steps:
            truncated = True

        if self.render_mode == "human": self._render_frame()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos_robot = self.data.qpos.flatten()[7:]
        qvel_robot = self.data.qvel.flatten()[6:]
        trunk_quat = self.data.qpos[3:7]
        trunk_vel = self.data.qvel[:3]
        trunk_ang_vel = self.data.qvel[3:6]
        ball_pos = self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        vec_to_goal = self.goal_pos - ball_pos
        return np.concatenate([qpos_robot, qvel_robot, trunk_quat, trunk_vel, trunk_ang_vel, ball_pos, ball_vel, vec_to_goal])

    def _compute_reward_and_done(self):
        reward = 0.0
        terminated = False
        
        ball_pos = self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel = self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        trunk_pos = self.data.body("Trunk").xpos
        trunk_upright = self.data.body("Trunk").xmat.reshape(3, 3)[2, 2]

        # 1. 종료 조건
        #if trunk_pos[2] < 0.4 or trunk_upright < 0.5:
        #   reward = -500.0
        #    terminated = True
        #   return reward, terminated
        if ball_vel[0] == 0.0:
            if trunk_pos[2] < 0.4 or trunk_upright < 0.5:
                reward = -500.0
                terminated = True
                return reward, terminated

        # 2. 서기 보상 (Alive + Stability)
        reward += 0.1
        reward += 0.1 * trunk_upright # (기존 0.2 -> 0.1로 줄여서 움직임 유도)

        # 3. 접근 보상
        l_foot_pos = self.data.xpos[self.left_foot_id]
        dist = np.linalg.norm(l_foot_pos - ball_pos)
        reward += 0.05 * (1.0 - np.tanh(2.0 * dist))

        # 4. 킥 속도 보상
        if self.has_touched_ball and trunk_upright > 0.8:
        # 공의 전진 속도(x)에 대해 큰 가중치를 줍니다.
        # 공이 5m/s로 날아가면 -> +10점 (매 스텝!)
            if ball_vel[0] > 0.1:
                reward += 2.0 * ball_vel[0]

        # 공이 앞으로(x축) 많이 나갈수록 추가 점수!!!!
        # 11m까지 가는 동기를 부여
        # (단, 공이 뒤로 가면 감점)
        if ball_pos[0] > 0.5: # 초기 위치(0.45)보다 나갔을 때
            reward += 1.0 * ball_pos[0] # 5m 가면 +5점, 10m 가면 +10점

        # 5. 터치
        if not self.has_touched_ball and dist < 0.15:
            reward += 100.0
            self.has_touched_ball = True

        # 6. 골인
        if ball_pos[0] > 11.0:
            is_goal = abs(ball_pos[1]) < (self.goal_width / 2.0) and ball_pos[2] < self.goal_height
            if is_goal:
                reward += 1000.0
                if trunk_upright > 0.6:
                    reward += 500.0
                    terminated = True
            else:
                reward -= 10.0
                terminated = True

        # 7. 페널티
        reward -= 0.001 * np.square(self.data.ctrl).sum()
        reward -= 0.001 * np.square(self.data.qvel).sum()
        reward -= 0.05 * np.sum(np.square(self.data.body("Trunk").cvel[3:]))

        l_foot_y = self.data.xpos[self.left_foot_id][1]
        r_foot_y = self.data.xpos[self.right_foot_id][1]
        if r_foot_y > l_foot_y:
            reward -= 1.0

        return reward, terminated

    def render(self):
        if self.render_mode == "rgb_array": return self._render_rgb()
        elif self.render_mode == "human": self._render_frame()
    def _render_rgb(self):
        if self.mujoco_renderer is None: self.mujoco_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.mujoco_renderer.update_scene(self.data)
        return self.mujoco_renderer.render()
    def _render_frame(self):
        if self.viewer is None and self.render_mode == "human":
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer: self.viewer.sync()
    def close(self):
        if self.viewer is not None: self.viewer.close()
        if self.mujoco_renderer is not None: self.mujoco_renderer.close()

class T1StandPretrainEnv(T1KickEnv3):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.max_steps = 2500 # 5초

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        # [핵심] 공을 5m 밖으로 치워버림
        ball_adr = self.ball_qpos_adr
        self.data.qpos[ball_adr] = 5.0
        self.data.qpos[ball_adr+1] = 5.0
        self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def _compute_reward_and_done(self):
        """서기 전용 보상 함수 (이미지 기반)"""

        l_foot_pos = self.data.xpos[self.left_foot_id]
        r_foot_pos = self.data.xpos[self.right_foot_id]
        reward = 0.0
        terminated = False
        trunk_z = self.data.qpos[2]

        reward += 5.0 # Alive
        
        foot_dist = np.linalg.norm(l_foot_pos - r_foot_pos)
        target_foot_dist = 0.25 
        reward += 1.0 * np.exp(-10.0 * (foot_dist - target_foot_dist)**2)
        
        trunk_vel = self.data.qvel[0:3]
        reward += 3.0 * np.exp(-5.0 * np.linalg.norm(trunk_vel)) # Statuesque

        trunk_quat = self.data.qpos[3:7]
        reward += 1.0 * (trunk_quat[0] ** 2) # Upright

        height_err = abs(trunk_z - self.target_height)
        reward += 1.0 * np.exp(-20.0 * height_err) # Height

        pos_err = np.linalg.norm(self.data.qpos[:2])
        reward += 0.2 * np.exp(-5.0 * pos_err) # Position

        reward -= 0.08 * np.square(self.data.ctrl).sum() # Effort
        
        joint_angles = self.data.qpos[7:]
        default_angles = self.initial_qpos[7:]
        reward -= 0.02 * np.square(joint_angles - default_angles).sum() # Posture

        qvel = self.data.qvel[6:]
        reward -= 0.001 * np.square(qvel).sum() # Damping


        if trunk_z < 0.35: 
            reward = -500.0
            terminated = True
        
        return reward, terminated