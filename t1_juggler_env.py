import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco
from mujoco import MjModel, MjData, mj_step

# MuJoCo Assets 경로 설정
ASSET_DIR = os.path.join(os.path.dirname(__file__), 'booster_t1')
MODEL_XML_PATH = os.path.join(ASSET_DIR, 't1.xml')

class T1JugglerEnv(gym.Env):
    """
    Booster T1 로봇을 위한 축구공 리프팅 Gymnasium 환경
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # 1. MuJoCo 모델 로드
        self.model = MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = MjData(self.model)
        ball_joint_name = "ball_free_joint"
        
        try:
            # joint(name) 메서드를 사용하여 관절 객체를 가져옵니다.
            ball_joint_obj = self.model.joint(ball_joint_name)
        except IndexError:
            raise ValueError(f"XML 파일에서 조인트 '{ball_joint_name}'를 찾을 수 없습니다. 이름 확인 필요.")
            
        # 관절 객체에서 qpos(위치)와 qvel(속도) 데이터 주소를 추출합니다.
        self.ball_qpos_adr = int(ball_joint_obj.qposadr)
        self.ball_qvel_adr = int(ball_joint_obj.dofadr)
        # 2. 로봇의 관절 수 (액추에이터 수)
        # Actuator 정의가 position control이므로, action은 목표 위치입니다.
        n_actuators = self.model.nu 
        
        # 3. 행동 공간 (Action Space) 정의: 목표 관절 위치
        # 각 액추에이터는 [-1, 1] 사이의 값을 받아 목표 관절 위치로 변환됩니다.
        # (이 값은 Stable Baselines3에서 사용되는 일반적인 스케일입니다.)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32)
        
        # 4. 관측 공간 (Observation Space) 정의: 로봇 상태 + 공 상태
        # (로봇 관절 각도, 관절 속도, 공 위치, 공 속도 등을 포함)
        # 임시로 큰 값으로 설정하고, 실제 훈련 시 디버깅을 통해 크기를 확정합니다.
        obs_dim = 6+(self.model.nq-7)+(self.model.nv-6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 5. 렌더링 설정
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.viewer = None
        
        # 초기 위치 저장을 위해 keyframe을 활용할 수 있습니다.
        self.initial_qpos = self.data.qpos.copy()


    def _get_obs(self):
        """환경의 현재 상태를 관측 벡터로 추출"""
        ball_pos=self.data.qpos[self.ball_qpos_adr: self.ball_qpos_adr+3]
        ball_vel=self.data.qvel[self.ball_qvel_adr: self.ball_qvel_adr+3]
        # 로봇 관절 각도 및 속도
        qpos_robot_joints = self.data.qpos.flatten()[7:]
        qvel_robot_joints = self.data.qvel.flatten()[6:]
        
        # 축구공의 위치 (ball body의 qpos)와 속도 (qvel)
        # qpos에서 'ball_free_joint'의 7개 값(위치3, 회전4)을 찾습니다.
        # qpos_ball, qvel_ball을 찾아 여기에 포함시켜야 합니다.
        
        # *******************************************************************
        # 임시 관측 벡터 (나중에 실제 데이터로 대체해야 함)
        # *******************************************************************
        
        return np.concatenate([
            ball_pos,
            ball_vel,
            qpos_robot_joints,
            qvel_robot_joints,
        ])

    
    def step(self, action):
        """에이전트의 행동을 실행하고 다음 상태, 보상, 종료 여부를 반환"""
        
        # 1. Action 스케일링 및 적용
        
        # 액추에이터는 'position' 제어를 사용하므로,
        # action은 목표 관절 위치를 결정합니다.
        
        # RL 에이전트의 action은 보통 [-1, 1] 범위입니다. 
        # 이를 각 관절의 허용 범위(range)에 맞춰 스케일링하거나,
        # 현재 목표 위치에 작은 델타(변화량)로 적용하는 방법이 있습니다.
        
        # 여기서는 가장 간단하게, action을 이전 제어값에 더해 목표 위치를 업데이트합니다.
        # action에 스케일링 계수 0.1을 곱하여 목표 위치를 미세 조정합니다.
        
        current_ctrl = self.data.ctrl.copy()
        
        # Trunk free joint를 제외한 나머지 관절의 ctrl 값만 업데이트합니다.
        # Trunk free joint는 제어할 수 없습니다. (actuator가 정의되어 있지 않음)
        self.data.ctrl[:] = current_ctrl + action * 0.1
        
        # 2. MuJoCo 시뮬레이션 한 단계 진행
        mujoco.mj_step(self.model, self.data)

        # 3. 관측 (Observation)
        observation = self._get_obs()

        # 4. 보상 (Reward) 계산
        reward = self._compute_reward()

        # 5. 종료 조건 (Done / Truncated)
        terminated = self._is_terminated() 
        truncated = False 

        if self.render_mode == "human":
            self._render_frame()

        # MuJoCo의 경우, mj_forward를 수동으로 호출할 필요는 없습니다.
        # mj_step이 자동으로 다음 상태로 갱신해 줍니다.
        
        return observation, reward, terminated, truncated, {}


    def reset(self, seed=None, options=None):
        """환경을 초기 상태로 재설정"""
        super().reset(seed=seed)
        
        # MuJoCo 데이터 초기화 (qpos를 keyframe home으로 설정)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.data.qacc[:] = np.zeros(self.model.nv)
        
        mujoco.mj_forward(self.model, self.data) # 순방향 물리 계산

        observation = self._get_obs()
        info = {}
        return observation, info


    def _compute_reward(self):
        """축구공 리프팅 목표에 대한 보상 함수"""
        
        # 1. 공 상태 추출 (관측 벡터와 동일)
        ball_joint_name = "ball_free_joint"
        ball_qpos_adr = self.ball_qpos_adr
        ball_qvel_adr = self.ball_qvel_adr
        ball_z_vel = self.data.qvel[ball_qvel_adr + 2] # vz는 3번째 값 (index 2)
        
        # 공의 위치 (x, y, z)
        ball_pos = self.data.qpos[ball_qpos_adr: ball_qpos_adr + 3]
        ball_x, ball_y, ball_z = ball_pos
        
        # 2. 보상 항목 계산
        
        # A. 공 높이 보상 (Lift Reward): 공이 높을수록 긍정 보상
        # 공의 Z 높이에 비례하며, 땅 (z=0)에 가까울수록 보상이 작아지게 합니다.
        lift_reward = 0.1 * max(0, ball_z - 0.1) # 0.1m 이하에서는 보상X (땅 근처)
        lift_momentum_reward = 0.2*ball_z_vel
        if ball_z_vel > 0.1:
            lift_momentum_reward = 20.0 * ball_z_vel # 튕겨 올리는 속도에 비례
        # B. 수평 위치 페널티 (Centering Penalty): 공이 로봇의 수평 중심에서 멀어질수록 페널티
        # 로봇은 (0, 0, Z) 근처에 서 있으므로, x^2 + y^2 에 비례하여 페널티를 줍니다.
        centering_penalty = -0.005 * (ball_x**2 + ball_y**2)

        # C. 로봇 안정성 보상 (Stability Reward): 로봇이 균형을 유지할 경우 보상
        # Trunk의 Z 높이가 안정적일 때 긍정 보상 (넘어지면 Trunk Z가 급격히 작아짐)
        trunk_z_position = self.data.qpos[2] 
        stability_reward = 0.05 * trunk_z_position 
        
        # D. 행동 평활화 페널티 (Control Smoothness Penalty)
        # 에이전트가 너무 격렬하게 행동하지 않도록 작은 페널티를 줍니다.
        action_penalty = -1e-8 * np.sum(self.data.ctrl**2)
        
        # 3. 최종 보상 합산
        reward = lift_reward + centering_penalty + stability_reward + action_penalty + lift_momentum_reward
        
        return reward


    def _is_terminated(self):
        """종료 조건 검사 (로봇 넘어짐 또는 공 분실)"""
        
        # 1. 로봇 넘어짐 종료 조건
        # Trunk body의 Z 위치가 너무 낮으면 종료 (0.3m)
        trunk_z_position = self.data.qpos[2] 
        if trunk_z_position < 0.35: # 로봇의 초기 높이(0.618)의 절반 이하로 내려가면
            return True
            
        # 2. 공 분실 종료 조건
        # 공이 로봇의 수평 반경 1m 밖으로 나가면 종료
        ball_joint_name = "ball_free_joint"
        ball_qpos_adr = self.ball_qpos_adr
        ball_pos = self.data.qpos[ball_qpos_adr: ball_qpos_adr + 3]
        ball_x, ball_y = ball_pos[0], ball_pos[1]
        
        horizontal_distance = np.sqrt(ball_x**2 + ball_y**2)
        if horizontal_distance > 1.0:
            return True
            
        return False
        
        
    def _render_frame(self):
        """MuJoCo Viewer로 시각화"""
        if self.viewer is None and self.render_mode == "human":
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer:
            self.viewer.sync()


    def close(self):
        """환경 종료 및 Viewer 해제"""
        if self.viewer is not None:
            self.viewer.close()