# 🤖 RobotPal

![RobotPal Demo](./.github/assets/demo.webp)

[![Emscripten Build](https://github.com/fastturtle7892/robotpal/actions/workflows/emscripten-build.yml/badge.svg)](https://github.com/fastturtle7892/robotpal/actions/workflows/emscripten-build.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RobotPal**은 웹(WebAssembly)과 데스크톱 환경 모두에서 동작하는 고성능 로봇 시뮬레이션 및 제어 프레임워크입니다. C++로 작성된 코어 엔진을 기반으로 하며, Python 바인딩을 통해 로봇 제어 알고리즘과 인공지능(AI) 모델을 손쉽게 테스트할 수 있습니다.

🔗 **[Live Web Demo 보기](https://fastturtle7892.github.io/robotpal/)**
> *웹 브라우저에서 별도의 설치 없이 시뮬레이션을 바로 실행해보세요!*

---

## ✨ 주요 기능 (Key Features)

* **🌐 크로스 플랫폼 지원 (Cross-Platform)**
    * **WebAssembly (WASM)**: Emscripten을 통해 웹 브라우저에서 네이티브에 준하는 성능으로 실행됩니다.
    * **Desktop (Native)**: Windows, Linux, macOS 환경에서 고성능 시뮬레이션이 가능합니다.
* **🎨 사실적인 렌더링 (Realistic Rendering)**
    * OpenGL/WebGL 기반의 PBR (Physically Based Rendering) 적용
    * HDR Skybox 및 IBL (Image Based Lighting) 지원
    * 실시간 그림자 및 조명 효과
* **🐍 Python API 및 Jupyter 연동**
    * `RobotPal-python` 모듈을 통해 파이썬으로 로봇을 제어할 수 있습니다.
    * Jupyter Notebook 환경에서 카메라 피드를 받아와 YOLO, 라인 트레이싱 등 컴퓨터 비전(CV) 알고리즘을 실시간으로 실습할 수 있습니다.
* **🏎️ 시뮬레이션 및 실물 제어**
    * 가상 환경(SimController)과 실제 로봇(RealController)을 동일한 API로 제어하는 하이브리드 아키텍처를 지원합니다.

## 📂 프로젝트 구조 (Project Structure)

* `RobotPal/`: C++ 코어 엔진 소스 코드 (렌더링, 시스템, 로직)
* `RobotPal-python/`: 로봇 제어를 위한 Python 바인딩 및 라이브러리
* `RobotPal-web/`: 웹 빌드를 위한 리소스 및 템플릿 (Service Worker 포함)
* `robotpal-jupyter/`: AI 모델 학습 및 제어 예제 노트북 (YOLO, Line Following 등)
* `.github/workflows/`: GitHub Actions 빌드 및 배포 자동화 스크립트

## 🚀 시작하기 (Getting Started)

### 웹 빌드 (Web Build)
이 프로젝트는 **Emscripten**을 사용하여 웹용으로 빌드됩니다.

```bash
# 1. 빌드 디렉토리 생성
mkdir build-web && cd build-web

# 2. Emscripten CMake 설정
emcmake cmake .. -G Ninja

# 3. 빌드 실행
ninja

# 4. 로컬 테스트 (보안 헤더 설정 필요)
# coi-serviceworker가 적용된 index.html을 실행해야 합니다.
python run_server.py
