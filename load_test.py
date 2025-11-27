import mujoco
import mujoco.viewer
import os
import time

# 1. MJCF 파일 경로 설정
# 현재 실행 폴더(RL_Project) 안에 booster_t1/t1.xml 파일이 있습니다.
model_dir = 'booster_t1'
model_file = 't1.xml'
model_path = os.path.join(model_dir, model_file)

# 파일이 존재하는지 확인하는 안전 장치
if not os.path.exists(model_path):
    print(f"오류: 모델 파일 경로를 찾을 수 없습니다. 경로 확인: {model_path}")
    exit()

try:
    # 2. MuJoCo 모델 로드
    # MjModel.from_xml_path()를 사용하여 XML 파일을 MuJoCo 모델로 변환합니다.
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"모델 '{model_file}' 로드 성공!")
    
    # 3. Viewer 실행 및 시뮬레이션
    # launch_passive()를 사용하여 MuJoCo Viewer 창을 띄웁니다.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 시뮬레이션 시간을 10초로 설정합니다.
        duration = 10 
        start_time = time.time()

        print("MuJoCo Viewer가 실행되었습니다. 10초 동안 시뮬레이션을 진행합니다.")
        
        while viewer.is_running() and time.time() - start_time < duration:
            # MuJoCo 물리 엔진의 한 스텝을 계산합니다.
            mujoco.mj_step(model, data)
            
            # 뷰어 창을 업데이트합니다.
            viewer.sync()
            
            # 시뮬레이션 속도를 실제 시간과 비슷하게 맞추기 위해 약간의 딜레이를 줍니다.
            time.sleep(model.opt.timestep * 0.5)

        print("시뮬레이션 종료.")

except Exception as e:
    print(f"시뮬레이션 로드 중 오류 발생: {e}")