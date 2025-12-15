import gymnasium as gym
from stable_baselines3 import SAC
from t1_env_2 import T1ShootingEnv
import os

def main():
    model_path = "models/ppo_swing_T2_0/swing_model_T2_0_5000000_steps.zip" 

    if not os.path.exists(model_path):
        print(f"오류{model_path}")
        return


    # 1. 환경 생성 (Human 모드 = 화면에 보여줌)
    env = T1ShootingEnv(xml_file='booster_t1/t1.xml', render_mode="human")

    # 2. 모델 로드
    model = SAC.load(model_path)

    obs, _ = env.reset()

    print(" 로봇 행동 확인")

    # 3. 시뮬레이션 반복
    for i in range(5000): 
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            print(f"{i}프레임: 넘어짐 (Reset)")
            obs, _ = env.reset()

    env.close()

if __name__ == '__main__':
    main()