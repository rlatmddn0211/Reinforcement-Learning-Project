# Humanoid Soccer Kicker: RL with Curriculum Learning

### 휴머노이드 로봇(Booster T1)이 스스로 균형을 잡고 강력한 슛을 날리도록 학습시킨 강화학습 프로젝트

## 🎯 프로젝트 목표 (Project Goal)

본 프로젝트의 목표는 불안정한 이족보행 로봇이 넘어져 있는 상태(Zero-base)에서 시작하여, 스스로 균형을 잡고 공을 인식해 강력한 슈팅을 날리는 것입니다.

이를 위해 2단계 커리큘럼 학습(Two-Stage Curriculum Learning) 전략을 사용하여, 복잡한 제어 문제를 '보행 안정화'와 '슈팅'이라는 두 가지 하위 문제로 분할해 해결했습니다.

### 🎞️ 학습 과정 및 변화 (Evolution of Robot)

저희 로봇은 수백만 번의 시행착오 끝에 다음과 같이 성장했습니다.

####  초기 상태 (The "Drunk" Robot)
https://github.com/user-attachments/assets/64798587-3ce5-4aad-9bde-eba55fe7b4e7
####  최종 완성: 임팩트 & 슈팅 (Impact & Shooting)
https://github.com/user-attachments/assets/b5cc8e7c-ab8f-4248-88c8-5801147fee29
## 🧠 알고리즘 및 방법론 (Methodology)

### 1. 사용 알고리즘 (Algorithm)

#### SAC (Soft Actor-Critic): 연속적인 행동 공간(Continuous Action Space)을 가진 로봇 제어에 적합하며, 높은 샘플 효율성과 탐색(Exploration) 능력을 가짐.

Framework: Gymnasium, Stable-Baselines3, MuJoCo Physics Engine.

2. 커리큘럼 학습 전략 (Two-Stage Curriculum)

논문 *"Shooting Master"*의 핵심 아이디어를 차용하여 학습 단계를 분리했습니다.

Stage 1

기초 체력 (Standing)

• Frozen Upper Body: 상체 관절(11개)을 고정하여 하체 제어에 집중.



• Strict Orientation Reward: 상체가 수직($Z$-axis)이 아니면 보상 0점 처리.



• Stand Still: 불필요한 떨림 방지.

Stage 2

슈팅 (Shooting)

• Unconstrained Legs: 다리의 가동 범위를 확보하기 위해 자세 강제 보상 제거.



• Magnet Reward: 공과 로봇의 거리가 20cm 이내일 때만 높은 보상 부여.



• Velocity Reward: 공이 앞으로 나가는 속도에 비례해 폭발적인 보상 부여 ($R \propto v_{ball}$).



## 📚 References

Paper: 
•Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018.
•Zhang, X. et al. (2025). Whole-Body Model-Predictive Control of Legged Robots with MuJoCo. arXiv preprint.
•Wang, Z., Zhou, J., & Wu, Q. (2025). Dribble Master: Learning Agile Humanoid Dribbling Through Legged Locomotion.
•Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint.

Physics Engine: MuJoCo

RL Library: Stable-Baselines3
