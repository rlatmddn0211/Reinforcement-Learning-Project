import gymnasium as gym
from stable_baselines3 import SAC
import os
import pickle 
from t1_env import T1ShootingEnv  

def main():
    experiment_name = "t1_standing_last"
    log_dir = f"./logs/{experiment_name}"
    models_dir = f"./models/{experiment_name}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = T1ShootingEnv(xml_file='booster_t1/t1.xml', render_mode=None) 


    print(f"✨ 실험 시작: {experiment_name}")
    print("🧠 초기화된 새로운 모델을 생성합니다...")


    model = SAC(
        "MlpPolicy",         
        env, 
        verbose=1,           
        device="cpu",        
        tensorboard_log=log_dir,
        learning_rate=3e-4,   
        batch_size=256        
    )
    TIMESTEPS = 50000  
    TOTAL_LOOPS = 20   
    
    for i in range(1, TOTAL_LOOPS + 1): 

        model.learn(
            total_timesteps=TIMESTEPS, 
            log_interval=10, 
            reset_num_timesteps=False, 
            tb_log_name="SAC_Stage1_Standing"
        )
        
        # 파일명 생성
        current_steps = TIMESTEPS * i
        base_name = f"{models_dir}/t1_stage1_{current_steps}"
        
        model.save(base_name + ".zip")
        
        try:
            with open(base_name + ".pkl", "wb") as f:
                pickle.dump(model.policy, f)
        except Exception as e:
            print(f"저장 과정 중 오류 : {e}")
        
        print(f"[{i}/{TOTAL_LOOPS}] 저장 완료: {base_name}.zip")

    env.close()
    print("학습 종료")

if __name__ == '__main__':
    main()