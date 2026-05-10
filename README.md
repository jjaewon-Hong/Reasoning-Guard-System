# Reasoning Guard 
딥러닝 기반 미확인 공중 표적 정밀 판별 시스템


![Reasoning Guard Logo](./docs/Logo.png)


* **Author:** 22212026 홍재원
* **Course:** OSS설계 

## 🛡️ Project Overview
현대 전장의 하이브리드 공중 위협에 대응하여, PyTorch 기반의 딥러닝 모델을 활용해 공중 객체(미사일, 전투기, 드론)를 정밀 탐지하고 요격 의사결정(Shoot/Hold)을 지원하는 고신뢰성 소프트웨어 엔진입니다.

## 📂 Project Documents
본 프로젝트는 시스템 생명 주기에 따른 4단계 산출물을 기반으로 관리됩니다.

1. **[개념화 (Conceptualization) 문서]** - 프로젝트 기획 배경, 핵심 가치 및 목표 정의
2. **[분석 (Analysis) 문서]** - 요구사항 도출, 위협 모델링 및 세부 기능 명세
3. **[설계 (Design) 문서]** - 시스템 아키텍처, C++/웹 인터페이스 설계 및 데이터 파이프라인
4. **[구현 (Implementation) 문서]** - 알고리즘 구현, 모델 ONNX 최적화 및 최종 엔진 배포

---

## ⚙️ Core Architecture
Reasoning Guard System은 PyTorch 기반으로 학습된 AI 모델을 최적화하여 고속 추론이 가능하도록 설계되었습니다.

* **AI Model Pipeline:** PyTorch에서 학습된 ResNet 기반 표적 분류 모델을 **ONNX 포맷**으로 변환하여 이식성과 처리 속도를 극대화했습니다.
* **Tactical Engine (C++):** ONNX Runtime 기반 고성능 C++ 백엔드로 구현되었으며, 실시간 표적 식별 및 교전 판단(Shoot/Hold) 전술 로직을 자동화합니다.
* **Tactical Web Dashboard:** 정적 웹 환경(HTML/JS)에서 **ONNX Runtime Web**을 이용해 서버 없이 브라우저 상에서 추론과 시각적 HUD 인터페이스를 직접 렌더링합니다.

## 🚀 Key Features
* **고정밀 공중 표적 식별:** 전투기, 드론, 미사일 등 다양한 공중 객체를 신속하게 판별.
* **크로스 플랫폼 고속 추론:** 서버(C++)와 클라이언트(Web) 모두에서 ONNX를 통한 최적화된 하드웨어 가속 추론 지원.
* **자율 교전 의사결정 (Rule Engine):** 식별된 타겟의 위협도를 분석하여 실시간으로 요격 여부(Shoot/Hold)를 도출하는 룰 기반 시스템.
* **Zero-Server 웹 시뮬레이션:** 별도의 백엔드 연산 서버 없이 정적 웹 호스팅(GitHub Pages 등)만으로 구동되는 브라우저 독립 실행 환경 제공.

## 🛠️ Technology Stack
* **AI & Inference:** PyTorch, ONNX, ONNX Runtime (C++ / Web)
* **Backend:** C++ (Tactical Engine), Python (Model Export & Test API)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Browser Inference)
* **Build & Deploy:** CMake, GitHub Pages
