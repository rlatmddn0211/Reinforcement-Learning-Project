import gymnasium as gym
from stable_baselines3 import SAC
import os
import pickle
from t1_env_2 import T1ShootingStage2Env  

def main():
    experiment_name = "t1_shooting_ball_contact_phase2"
    log_dir = f"./logs/{experiment_name}"
    models_dir = f"./models/{experiment_name}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Stage 2 환경 생성 
    # (상체는 여전히 고정되지만, 다리는 자유롭고 공 속도 보상이 추가됨)
    env = T1ShootingStage2Env(xml_file='booster_t1/t1.xml', render_mode=None) 
    load_path = "./models/t1_shooting_ball_contact/t1_shoot_1000000.zip" 
    
    if not os.path.exists(load_path):
        print(f"오류:{load_path}")
        return

    
    model = SAC.load(load_path, env=env, device="cpu", tensorboard_log=log_dir)

    # 4. 학습 루프 (슈팅 학습)
    TIMESTEPS = 50000 
    TOTAL_LOOPS = 20  
    
    for i in range(1, TOTAL_LOOPS + 1): 
        model.learn(
            total_timesteps=TIMESTEPS, 
            log_interval=10, 
            reset_num_timesteps=False, 
            tb_log_name="SAC_Shooting_PaperBased"
        )
        
        current_steps = TIMESTEPS * i
        base_name = f"{models_dir}/t1_shoot_{current_steps}"
        
        # zip 저장
        model.save(base_name + ".zip")
        
        # pkl 저장
        try:
            with open(base_name + ".pkl", "wb") as f:
                pickle.dump(model.policy, f)
        except Exception as e:
            pass
        
        print(f"💾 [{i}/{TOTAL_LOOPS}] 슈팅 모델 저장 완료: {base_name}.zip")

    env.close()
    print("학습 종료")

if __name__ == '__main__':
    main()