# Humanoid Soccer Kicker: RL with Curriculum Learning

### 휴머노이드 로봇(Booster T1)이 스스로 균형을 잡고 강력한 슛을 날리도록 학습시킨 강화학습 프로젝트입니다.

## 🎯 프로젝트 목표 (Project Goal)

본 프로젝트의 목표는 불안정한 이족보행 로봇이 넘어져 있는 상태(Zero-base)에서 시작하여, 스스로 균형을 잡고 공을 인식해 강력한 슈팅을 날리는 것입니다.

이를 위해 2단계 커리큘럼 학습(Two-Stage Curriculum Learning) 전략을 사용하여, 복잡한 제어 문제를 '보행 안정화'와 '슈팅'이라는 두 가지 하위 문제로 분할해 해결했습니다.

### 🎞️ 학습 과정 및 변화 (Evolution of Robot)

저희 로봇은 수백만 번의 시행착오 끝에 다음과 같이 성장했습니다.

#### Phase 1. 초기 상태 (The "Drunk" Robot)

초기에는 균형을 잡지 못하고 흐느적거리거나 뒤로 눕는(Reward Hacking) 현상이 발생했습니다.

<!-- 여기에 처음에 비틀거리는 gif나 이미지를 넣으세요. 예:  -->

문제점: 생존 보상에만 의존하여 편하게 누워서 점수를 얻으려는 경향 발생.

#### Phase 2. Stage 1 성공: "호랑이 교관" 훈련 (Stable Standing)

엄격한 자세 보상(Strict Orientation Reward)과 상체 고정(Frozen Upper Body) 전략을 도입하여 완벽한 차렷 자세를 학습했습니다.

<!-- 여기에 꼿꼿하게 서 있는 gif를 넣으세요. 예:  -->

해결책: 자세가 1도라도 흐트러지면 가차 없이 감점하는 Reward Shaping 적용.

#### Phase 3. Stage 2 시도: 헛발질 단계 (Air Kick)

다리의 자유를 주었으나, 공과의 거리 조절에 실패하여 허공에 발을 휘두르는 단계입니다.

<!-- 여기에 공 근처에서 헛발질하는 gif를 넣으세요. 예:  -->

현상: 킥을 하려는 의도는 보이나, 임팩트 정확도가 떨어짐.

#### Phase 4. 최종 완성: 임팩트 & 슈팅 (Impact & Shooting)

"자석 보상(Magnet Reward)"과 "임팩트 보너스"를 통해 공에 바짝 붙어 강력하게 차는 동작을 완성했습니다.

<!-- 여기에 시원하게 슛을 날리는 gif를 넣으세요. 예:  -->

성과: 공의 속도($v_{ball}$)에 비례한 보상을 통해 강력한 킥 동작 유도 성공.

## 🧠 알고리즘 및 방법론 (Methodology)

### 1. 사용 알고리즘 (Algorithm)

#### SAC (Soft Actor-Critic): 연속적인 행동 공간(Continuous Action Space)을 가진 로봇 제어에 적합하며, 높은 샘플 효율성과 탐색(Exploration) 능력을 가짐.

Framework: Gymnasium, Stable-Baselines3, MuJoCo Physics Engine.

2. 커리큘럼 학습 전략 (Two-Stage Curriculum)

논문 *"Dribble Master"*의 핵심 아이디어를 차용하여 학습 단계를 분리했습니다.

단계

목표

핵심 전략 (Key Strategy)

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
