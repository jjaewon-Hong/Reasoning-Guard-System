# Reasoning Guard 
딥러닝 기반 미확인 공중 표적 정밀 판별 시스템


![Reasoning Guard Logo](./docs/Logo.png)


* **Author:** 22212026 홍재원
* **Course:** OSS설계 최종

## 🛡️ Project Overview
**Reasoning-Guard-System**은 `Reasoning-Guard`에서 진행된 PyTorch 기반의 모델 학습과 체계적인 시스템 설계(개념화·분석·디자인)를 바탕으로, 실제 전술 환경에서의 운용까지 C++로 구현해 낸 최종 통합 시스템입니다. 학습 완료된 모델을 ONNX로 추출 및 최적화하여 C++ 전술 엔진에 이식한 고신뢰성 지능형 방공 솔루션으로서, 3단계 전술 로직으로 의사결정을 자동화합니다. 또한 연구용 Public Dataset을 활용하여 보안 무결성을 견지함과 동시에, 전술 데이터의 집약적 관리와 확장성을 극대화한 실전형 사격 통제 아키텍처를 완성했습니다.

## 🛠️ Technology Stack
* **AI & Inference:** PyTorch, ONNX, ONNX Runtime (C++ / Web)
* **Backend:** C++ (Tactical Engine), Python (Model Export & Test API)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Browser Inference)
* **Build & Deploy:** CMake, GitHub Pages
