import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from t1_juggler_env import T1JugglerEnv  # 우리가 만든 환경 클래스 임포트

# 1. 환경 등록 및 생성
# Gym에 환경을 등록하지 않고 클래스를 직접 전달하여 사용합니다.
# make_vec_env는 여러 환경(여기서는 4개)을 병렬로 실행하여 학습 속도를 높입니다.
vec_env = make_vec_env(T1JugglerEnv, n_envs=1, env_kwargs=dict(render_mode='human'))

# 2. 체크포인트 콜백 설정
# 100,000 스텝마다 모델을 저장하여 학습 중간의 결과를 보존합니다.
checkpoint_callback = CheckpointCallback(
  save_freq=100000, 
  save_path="./models/ppo_t1_juggler/",
  name_prefix="t1_juggler_model"
)

# 3. PPO 알고리즘 설정
# MlpPolicy: 다층 퍼셉트론 (표준 신경망) 정책
# policy_kwargs: 신경망의 구조를 정의합니다. (히든 레이어 64x64)
# verbose=1: 학습 진행 상황을 출력합니다.
model = PPO(
    "MlpPolicy", 
    vec_env, 
    learning_rate=3e-4,          # 학습률 (Learning Rate)
    n_steps=2048,                # 한 번에 수집할 경험 데이터 수
    batch_size=64,               # 배치 사이즈
    gamma=0.99,                  # 할인율 (Discount Factor)
    gae_lambda=0.95,             # GAE (Generalized Advantage Estimation) 파라미터
    clip_range=0.2,              # PPO 클리핑 범위
    policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]), # 정책 네트워크 구조
    verbose=1
)

# 4. 학습 시작
# 총 500만 스텝 동안 학습을 진행합니다. (필요에 따라 조절)
TIMESTEPS = 5_000_000
print(f"Starting training for {TIMESTEPS} timesteps...")

model.learn(
    total_timesteps=TIMESTEPS, 
    callback=checkpoint_callback
)

# 5. 최종 모델 저장
model.save("final_t1_juggler_model.zip")
print("Training finished and final model saved!")