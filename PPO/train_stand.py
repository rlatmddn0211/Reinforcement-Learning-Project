import gymnasium as gym
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from t1_kick_env import T1StandPretrainEnv # 서기 전용 환경

def main():
    save_dir = "models/phase2_stand"
    log_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("🚀 [Phase 1] 서기(Standing) 학습을 준비합니다...")

    env = make_vec_env(
        T1StandPretrainEnv, 
        n_envs=8, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(render_mode=None)
    )
    
    # 정규화 적용
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        tensorboard_log=log_dir,
        device="cuda"
    )
    
    # 50만 번마다 자동 저장
    callback = CheckpointCallback(save_freq=500000, save_path=save_dir, name_prefix="stand")
    print("서기 학습 시작")

    try:
        model.learn(
            total_timesteps=10000000, 
            callback=callback, 
            tb_log_name="phase1_stand"
        )
        print("학습 완료")

    except KeyboardInterrupt:
        print("현재 상태 저장")
        
        model.save(f"{save_dir}/stand_model_interrupted")
        env.save(f"{save_dir}/vec_normalize.pkl") # 통계도 저장 (중요)
        
        print("   -> 비상 저장 완료.")

    except Exception as e:
        print(f"\n[오류 발생] {e}")
        try:
            model.save(f"{save_dir}/stand_model_error")
            env.save(f"{save_dir}/vec_normalize_error.pkl")
        except:
            pass

    finally:
        
        # 정상적으로 끝났든, 도중에 껐든 'final' 이름으로 저장해둠
        model.save(f"{save_dir}/stand_expert_final")
        env.save(f"{save_dir}/vec_normalize.pkl")
        
        print(f"최종 모델 저장 완료: {save_dir}/stand_expert_final.zip")
        
        try:
            env.close()
            print("환경 종료 완료.")
        except:
            pass

if __name__ == "__main__":
    main()