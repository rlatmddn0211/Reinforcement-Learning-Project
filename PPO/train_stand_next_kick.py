import gymnasium as gym
import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback 
from t1_kick_env import T1KickEnv  # 킥 전용 환경

class SaveModelAndVecNormalize(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "kick", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.training_env is not None:
                self.training_env.save(f"{path}.pkl")
            if self.verbose > 1:
                print(f"Saved model and stats to {path}")
        return True

# 전역 변수 (비상 저장용)
model = None
vec_env = None 

# 저장 경로 설정
models_dir = "models/phase2_kick8"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def signal_handler(sig, frame):
    print("\n\n!!! 강제 종료 신호 감지 !!!")
    if model is not None:
        save_path = os.path.join(models_dir, "kick_model_emergency.zip")
        stats_path = os.path.join(models_dir, "vec_normalize_emergency.pkl")
        
        print(f"비상 저장 중... -> {save_path}")
        model.save(save_path)
        if vec_env is not None:
            vec_env.save(stats_path)
        print("저장 완료!")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    pretrained_model_path = "models/phase1_stand/stand_expert_final.zip"
    pretrained_stats_path = "models/phase1_stand/vec_normalize.pkl"

    if not os.path.exists(pretrained_model_path) or not os.path.exists(pretrained_stats_path):

        print(f"   확인: {pretrained_model_path}")
        sys.exit(1)
    num_cpu = 8  
    total_timesteps = 10_000_000
    
    print(f"🚀 1단계 모델을 불러와서 킥(Kick) 학습을 시작합니다...")

    vec_env = make_vec_env(
        T1KickEnv, 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=dict(render_mode=None)
    )
    
    # [★ 핵심 수정 1] 1단계 통계 불러오기 (안경 착용)
    print(f"📥 통계 로드 중: {pretrained_stats_path}")
    vec_env = VecNormalize.load(pretrained_stats_path, vec_env)
    vec_env.training = True  # 계속 학습해야 하므로 True
    vec_env.norm_reward = True

    # -----------------------------------------------------------
    # 3. 모델 로드
    # -----------------------------------------------------------
    # [★ 핵심 수정 2] PPO.load로 모델 불러오기
    model = PPO.load(
        pretrained_model_path, 
        env=vec_env, 
        device="cuda", 
        tensorboard_log=log_dir,
        # (선택) 킥은 섬세해야 하므로 학습률을 살짝 낮춰서 시작 (2e-4)
        learning_rate=2e-4 
    )
    
    # 콜백 설정 (50만 번마다 저장)
    save_freq = 500000 // num_cpu
    callback = SaveModelAndVecNormalize(
        save_freq=save_freq, 
        save_path=models_dir, 
        name_prefix="phase2_kick7"
    )

    print(f"킥 학습 시작 (Total: {total_timesteps})")


    # 4. 학습 루프
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback, 
            tb_log_name="kick_phase2_kick6"
        )
        print("학습 완료.")

    except KeyboardInterrupt:
        pass 
        
    except Exception as e:
        print(f"\n [오류 발생] 에러: {e}")
        try:
            model.save(os.path.join(models_dir, "kick_model_error.zip"))
            vec_env.save(os.path.join(models_dir, "vec_normalize_error.pkl"))
        except:
            pass

    finally:
        
        
        final_model_path = os.path.join(models_dir, "kick_model_final.zip")
        final_stats_path = os.path.join(models_dir, "vec_normalize.pkl")
        
        if model is not None:
            model.save(final_model_path)
            if vec_env is not None:
                vec_env.save(final_stats_path)
            print(f"최종 저장 완료: {final_model_path}")
        
        try:
            vec_env.close()
            print("환경 종료 완료.")
        except:
            pass

if __name__ == '__main__':
    main()