# 🤖 RobotPal Jupyter

**RobotPal Jupyter**는 Jetson Nano 기반의 AI 로봇(JetTank 등)을 제어하고 다양한 인공지능 비전 프로젝트를 실습할 수 있는 라이브러리 및 예제 코드 저장소입니다. Jupyter Notebook 환경에서 로봇의 모터 제어, 카메라 영상 스트리밍, 딥러닝 모델 학습 및 자율 주행 실습을 단계별로 진행할 수 있습니다.

---

## 📂 폴더 구조

이 저장소는 다음과 같은 주요 디렉토리로 구성되어 있습니다.

- **`robotpal/`**: 로봇 제어를 위한 핵심 Python 라이브러리입니다.
  - 모터 및 서보 제어 (`motor.py`, `SCSCtrl/`)
  - 카메라 영상 스트리밍 및 서버 통신 (`camera/`, `_core/server.py`)
  - 로봇 기본 동작 정의 (`robot.py`)
- **`vision/`**: 다양한 AI 비전 인식 엔진을 통합한 패키지입니다.
  - 객체 감지 및 문자 인식 (Detector, OCR)
  - 지원 엔진: Clova Vision API, PaddleOCR 등 (`engines/`)
- **`exmape_code/`**: 단계별 실습을 위한 Jupyter Notebook 예제 코드들이 포함되어 있습니다. (PJT07 ~ PJT12)
- **`data/`**: 실습에 사용되는 데이터셋, 학습된 모델 파일, 데모 이미지 등이 저장됩니다.

---

## 🚀 주요 기능

1.  **하드웨어 제어**
    - DC 모터 및 서보 모터의 정밀한 제어
    - 게임패드(조이스틱)를 이용한 수동 조작 지원
2.  **AI 비전 및 영상 처리**
    - 실시간 카메라 영상 스트리밍 (Websocket 기반)
    - 색상 인식 및 추적 (Color Tracking)
    - 도로 주행(Road Following)을 위한 데이터 수집 및 자율 주행 모델 학습
    - 작업 영역 검사 및 객체 인식 (Inspection & Object Detection)
3.  **확장 가능한 아키텍처**
    - 다양한 비전 엔진(Clova, Paddle 등)을 플러그인 형태로 사용 가능
    - 시뮬레이터 및 실제 로봇 환경 지원

---

## 📚 학습 커리큘럼 (Example Code)

`exmape_code` 폴더 내의 노트북 파일을 통해 다음 내용을 순서대로 학습할 수 있습니다.

### 1. 기본 동작 및 제어 (PJT07)
- **[5-1] basicmotion**: 로봇의 전후좌우 기본 이동 제어
- **[5-2] JETANK_1_servos**: 카메라 짐벌 등 서보 모터 제어
- **[5-4] motionDetect**: 카메라를 이용한 움직임 감지
- **[5-5] ~ [5-6]**: 색상 인식 및 특정 색상 추적하기
- **[5-7] gamepadCtrl**: 게임패드를 이용한 로봇 원격 조종

### 2. 자율 주행 실습 (PJT09)
- **DataCollection**: 도로 주행 데이터 수집
- **Train Model**: 수집된 데이터를 바탕으로 자율 주행 모델 학습
- **RoadFollowingFeedback**: 학습된 모델을 적용하여 라인 트레이싱(Road Following) 구현

### 3. 비전 검사 및 응용 (PJT11 ~ PJT12)
- **Camera 세부 동작**: 카메라 설정 및 이미지 캡처 테스트
- **Working Area Inspection**: 특정 영역의 상태를 검사하고 판단하는 로직 구현
- **RoadFollowing + WorkingArea**: 자율 주행과 작업 영역 인식을 결합한 종합 프로젝트

---

## 🛠 설치 및 시작하기

1.  이 저장소를 Jetson Nano 또는 실습 환경에 클론합니다.
    ```bash
    git clone [https://github.com/fastturtle7892/robotpal-jupyter.git](https://github.com/fastturtle7892/robotpal-jupyter.git)
    cd robotpal-jupyter
    ```

2.  Jupyter Notebook을 실행하여 `exmape_code` 폴더의 노트북을 엽니다.
    ```bash
    jupyter notebook
    ```

3.  필요한 Python 라이브러리가 설치되어 있는지 확인하세요. (주요 의존성: `opencv-python`, `numpy`, `websockets`, `torch` 또는 `tensorflow` 등 AI 프레임워크)

---

## 📝 라이선스

이 프로젝트는 오픈 소스 라이선스를 따릅니다. 자세한 내용은 라이선스 파일을 참고하세요.
