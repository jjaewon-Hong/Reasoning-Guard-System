# Reasoning Guard 
딥러닝 기반 미확인 공중 표적 정밀 판별 시스템


![Reasoning Guard Logo](./docs/Logo.png)


* **Author:** 22212026 홍재원
* **Course:** OSS설계 

## 🛡️ Project Overview
Reasoning Guard는 PyTorch 기반의 고도화된 딥러닝 학습 파이프라인과 C++ 기반의 고속 전술 인터페이스를 결합한 지능형 방공 시스템으로, 모델 가중치를 ONNX 포맷으로 변환하여 전술적 실시간성과 시스템 이식성을 극대화한 것이 핵심입니다. 브라우저 단에서 직접 추론과 시각적 HUD를 구현하는 독립 실행형 웹 시뮬레이션 환경을 제공하며, 이를 통해 미식별 객체의 정밀 식별부터 3단계 전술 대응(SAFE, CAUTION, DANGER)에 이르는 의사결정 프로세스를 고신뢰성 엔진으로 자동화합니다.

## 🛠️ Technology Stack
* **AI & Inference:** PyTorch, ONNX, ONNX Runtime (C++ / Web)
* **Backend:** C++ (Tactical Engine), Python (Model Export & Test API)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Browser Inference)
* **Build & Deploy:** CMake, GitHub Pages
