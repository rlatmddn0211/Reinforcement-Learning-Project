import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import pickle
import numpy as np
from t1_juggler_env import T1ShootingEnv

# =================================================================
# 설정 파일 경로, !! 민혁승우태중 이건 다르게 해야됨
# ========================================================
PROJECT_DIR = r"C:\Reinforce\RL_Project\booster_t1"
XML_FILE = "t1.xml"
XML_PATH = os.path.join(PROJECT_DIR, XML_FILE)

#시작지점 수정 전이학습
LOAD_STEPS = 71000000 
#몇번 더 학습할건지
ADDITIONAL_STEPS = 30000000
TIMESTEPS_PER_SAVE = 500000 

MODELS_DIR_PATH = r"C:\Reinforce\RL_Project\models\shooting_stage1_standing"
LOAD_PATH = os.path.join(MODELS_DIR_PATH, f"t1_stage1_{LOAD_STEPS}.zip")
TARGET_TOTAL_STEPS = LOAD_STEPS + ADDITIONAL_STEPS

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:

        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for idx, done in enumerate(dones):
            if done:
                self.episode_count += 1 # 에피소드 카운트 증가
                
                info = infos[idx]
                if 'episode' in info:
                    ep_rew = info['episode']['r']
                    ep_len = info['episode']['l']
                    self.episode_rewards.append(ep_rew)
                    self.episode_lengths.append(ep_len)
                    
                    if len(self.episode_rewards) > 1000:
                        self.episode_rewards = self.episode_rewards[-1000:]
                        self.episode_lengths = self.episode_lengths[-1000:]
                    
                    if self.episode_count % 1000 == 0:
                        mean_rew = np.mean(self.episode_rewards)
                        mean_len = np.mean(self.episode_lengths)
                        
                        print(f"[{self.num_timesteps} steps] Episode {self.episode_count} | Mean Rew (1000): {mean_rew:.2f} | Last Ep Rew: {ep_rew:.2f}")
                    
        return True

def make_env():
    # 렌더모드 human으로 하면 학습과정 보임, 근데 느려짐
    return T1ShootingEnv(xml_file=XML_PATH, render_mode=None)

def main():
    #cpu 코어 병렬학습 처리함. 민혁승우태중 다 다르게 설정해야햄
    num_cpu = 12  
    
    log_base_dir = r"C:\Reinforce\RL_Project" 
    experiment_name = "shooting_stage1_standing"  
    log_dir = os.path.join(log_base_dir, "logs", experiment_name)
    models_dir = os.path.join(log_base_dir, "models", experiment_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = make_vec_env(
        make_env, 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv
    )

    model = SAC.load(LOAD_PATH, env=env, device="cuda", tensorboard_log=log_dir)
    print("로드 성공")

    #속도 최적화
    model.train_freq = TrainFreq(frequency=64, unit=TrainFrequencyUnit.STEP)
    model.gradient_steps = 64
    
    start_cycle = int(LOAD_STEPS / TIMESTEPS_PER_SAVE)
    end_cycle = int(TARGET_TOTAL_STEPS / TIMESTEPS_PER_SAVE)
    
    callback = TensorboardCallback()


    for i in range(start_cycle + 1, end_cycle + 1): 
        current_target_step = TIMESTEPS_PER_SAVE * i

        
        model.learn(
            total_timesteps=TIMESTEPS_PER_SAVE, 
            callback=callback,
            reset_num_timesteps=False, 
            tb_log_name="Stage1_Standing"
        )
        
        save_path = os.path.join(models_dir, f"t1_stage1_{current_target_step}")
        
        model.save(save_path + ".zip")
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump(model.policy, f)
        
        print(f"완료")

    env.close()
    print("\n 종료")

if __name__ == '__main__':
    main()