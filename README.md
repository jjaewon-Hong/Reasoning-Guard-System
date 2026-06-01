# Reasoning Guard 
딥러닝 기반 미확인 공중 표적 정밀 판별 시스템


![Reasoning Guard Logo](./docs/Logo.png)


* **Author:** 22212026 홍재원
* **Course:** OSS설계 최종

## 🛡️ Project Overview
**Reasoning-Guard-System**은 `Reasoning-Guard` 기반으로 프론트엔드 UI와 백엔드(C++ 전술 엔진 및 AI 모델)가 완전히 결합된 **웹 기반 통합 시뮬레이션 환경**과, 실제 방공 하드웨어 탑재 및 검증을 목적으로 하는 **C++ 네이티브 코어 환경** 두 가지 형태로 구축되었습니다.

<br>

## 🌐 Web Visualization (Frontend / Backend Integrated Demo)
웹 브라우저 상에서 실제 방공 시스템의 동작을 시각적으로 확인할 수 있는 메인 시뮬레이션 환경입니다. UI/UX를 담당하는 프론트엔드와 표적을 판별하는 백엔드(ONNX AI 모델 및 C++ 전술 엔진)가 WebAssembly(WASM)를 통해 매끄럽게 상호작용합니다. 실제 방공 장비의 카메라나 레이더에 새로운 표적이 포착되는 상황을 `+1 Hour (시뮬레이션 진행)` 버튼으로 구현하였으며, 버튼 클릭 시 백엔드 모델이 즉각적으로 표적을 분석하고 그 결과를 프론트엔드로 반환하여 화면에 렌더링합니다.
* **`docs/index.html`** : 사용자의 조작(`+1 Hour` 버튼)을 입력받아 백엔드로 전달하고, 분석된 전술 결과를 화면에 띄워주는 프론트엔드 메인 인터페이스
* **`docs/wasm/tactical_wasm.cpp`** : C++로 작성된 백엔드 전술 코어를 웹 환경과 연결해주는 핵심 브릿지 모듈
* **`reasoning_guard_engine.onnx`** : 전달받은 표적의 종류와 위협 확률을 정확히 판별해 내는 백엔드의 핵심 두뇌(AI 모델)

<br>

## ⚙️ Tactical Core Engine (Embedded / Hardware Target)
실제 임베디드 장비나 하드웨어 기반의 방공 시스템에 직접 적용 및 이식하기 쉬운 형태로 구현된 C++ 코어 모듈입니다. 현재는 로컬 환경에서의 테스트용으로 구성되어 있습니다.
* **`main.cpp`** : C++ 전술 엔진 및 모델 추론 독립 테스트를 위한 엔트리 포인트
* **`preprocess.h`** : 레이더 등에서 수집된 센서 데이터를 AI 모델의 입력 텐서 형태로 변환하는 전처리 모듈
* **`tactical_engine.cpp`** : 표적의 위협 수준을 판별하고 최적의 교전 시나리오를 결정하는 핵심 전술 로직
