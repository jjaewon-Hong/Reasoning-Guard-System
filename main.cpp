/*
=============================================================================
  [Standalone C++ Engine Validation & Testbed]
  
  본 파일(main.cpp)은 웹(WASM) 환경과 독립적으로, 순수 C++ 환경에서 
  AI 추론(ONNX Runtime) 및 전술 판정 로직(Tactical Engine)의 
  정확성, 성능, 메모리 안정성을 사전 검증하기 위한 '콘솔형 디버깅/테스트베드'입니다.

  - 개발 목적: 핵심 로직(preprocess, tactical_engine)을 WebAssembly로 
    컴파일하여 HTML에 올리기 전, 네이티브(Native) 환경에서 알고리즘의 
    무결성을 빠르게 테스트하기 위해 필수적인 파일입니다.
  - 브라우저 오버헤드 없이 순수 알고리즘의 동작 속도 및 결과를 측정하는 용도입니다.
  - 실제 웹 서비스 배포(HTML 렌더링)에는 직접 관여하지 않는 개발 검증용 코드입니다.
=============================================================================
*/
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// 프로젝트 헤더
#include "preprocess.h"
#include "tactical_engine.h"

// std 네임스페이스 전역 사용
using namespace std;
namespace fs = filesystem; 

//  [이미지 파일 목록 수집]
vector<string> collect_images(const string& dir) {
    vector<string> images;

    // directory_iterator 사용해 dir 경로 내부 모든 파일/폴더를 하나씩 순회함
    for (const auto& entry : fs::directory_iterator(dir)) {
        string ext = entry.path().extension().string();

        // 대소문자 구분 없이 매칭하기 위해 확장자 모두 소문자로 변환
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // 조건에 만족하는 이미지 파일인것을 확인하면 전체경로를 문자열로 바꿔줌
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            images.push_back(entry.path().string());
    }
    return images;
}

//  [판정 상태별 ANSI 색상]
string colorize(const string& status, const string& msg) {
    if (status == "DANGER")  return "\033[91m" + msg + "\033[0m";
    // Danger -> 텍스트를 붉은 빨간색으로 감싸서 반환

    if (status == "CAUTION") return "\033[93m" + msg + "\033[0m";
    // CAUTION -> 텍스트를 밝은 노란색으로 감싸서 반환

    return "\033[92m" + msg + "\033[0m"; 
    // 위 조건에 해당되지 않는 경우(SAFE) -> 텍스트를 밝은 초록색으로 감싸서 반환
}

//  << 메인 진입점 >>
int main() {
    // Windows 콘솔 UTF-8 설정
    system("chcp 65001 > nul");

    const string MODEL_PATH  = "reasoning_guard_engine.onnx";
    const string IMG_DIR     = "./test_pool"; // 테스트용 이미지 12개 

    // 1. ONNX Runtime 세션 초기화 
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "reasoning_guard");
    // 학습 도메인(ONNX) 문제인지 운용 도메인(C++) 문제인지 식별하기 위해 추가 

    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    // 실시간 추론 속도 극대화 => 스레드간 문맥교환 오버헤드를 줄이기 위해 스레드 1개 사용

    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 연산 노드 통합 방식을 통해 C++ 추론 속도 극대화 (GraphOptimizationLevel::ORT_ENABLE_ALL)

    if (!fs::exists(MODEL_PATH)) {
        cerr << "[ERROR] 모델 파일을 찾을 수 없습니다: " << MODEL_PATH << "\n";
        return 1;
    }

    // [Windows: wstring 경로 사용]
    wstring wmodel_path(MODEL_PATH.begin(), MODEL_PATH.end());
    // 일반 문자열 -> 윈도우 전용 와이드 문자열로 변환
    Ort::Session session(env, wmodel_path.c_str(), session_opts);
    // AI 구동환경 + 모델 경로 + 최적화 옵션 => 실제 추론 세션

    cout << "[OK] ONNX 모델 로드 완료: " << MODEL_PATH << "\n";

    // 2. 이미지 파일 목록 수집 
    auto images = collect_images(IMG_DIR);
    if (images.empty()) {
        cerr << "[ERROR] test_pool 폴더에 이미지가 없습니다.\n";
        return 1;
    }
    cout << "[OK] 표적 이미지 " << images.size() << "개 확보\n";

    // 3. 랜덤 엔진
    random_device rd; // HW 기준 고유한 시드값 생성
    mt19937 rng(rd()); // '메르센 트위스터 알고리즘' 기반 난수 생성 엔진 초기화
    // *Mersenne Prime : 거대한 메르센 소수 주기 바탕으로 뛰어난 무작위성 보장 

    uniform_int_distribution<size_t> dist(0, images.size() - 1); 
    // 난수범위 지정 : test_pool 폴더 내부의 12장의 이미지 => 0~11

    // 4. ONNX 입출력 메타데이터 
    const char* input_names[]  = {"x"}; // 입력 노드 이름
    const char* output_names[] = {"sigmoid"}; // 출력 노드 이름
    array<int64_t, 4> input_shape{1, 3, 224, 224}; 
    // Tensor : 배치사이즈, 채널수, 이미지 가로/세로 크기

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // CPU 전용 고속 메모리 할당기(Arena) 설정을 생성하여 텐서 메모리 관리 효율화

    // 5. 메인 루프 
    cout << "\n" << string(55, '=') << "\n";
    cout << " Reasoning Guard: Tactical Operational System [C++]\n";
    cout << " [v3] ONNX Runtime C++ API — DANGER 발사 조건 적용\n";
    cout << string(55, '=') << "\n";

    while (true) {
        cout << "\n[Enter]: +1 HOUR 시뮬레이션 진행 | [q]: 종료 -> ";
        string cmd;
        getline(cin, cmd);
        if (cmd == "q" || cmd == "Q") break;

        //  5-1. 랜덤 표적 선택 
        const string& target_path = images[dist(rng)];
        string target_name = fs::path(target_path).filename().string();

        //  5-2. 이미지 전처리 (Zero-Padding → 224×224 → Normalize)
        vector<float> input_data;
        try {
            input_data = preprocess(target_path);
        } catch (const exception& e) {
            cerr << "[ERROR] 전처리 실패: " << e.what() << "\n";
            continue;
        }

        // 5-3. ONNX 추론 
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,  &input_tensor, 1,
            output_names, 1
        ); 

        const float* probs = output_tensors[0].GetTensorData<float>();
        // probs[0]=FIGHTER, probs[1]=DRONE, probs[2]=MISSILE, probs[3]=ETC

        // 5-4. 전술 판정 
        TacticalResult result = TacticalEngine::evaluate_threat(probs);

        // 5-5. 결과 출력 
        cout << "\n[TIME +1H] 감지된 표적: " << target_name << "\n";
        cout << "AI 분류 결과: " << result.predicted << "\n";
        cout << fixed << setprecision(2);
        cout << "  - FIGHTER:  " << setw(6) << probs[0] * 100.0f << "%\n";
        cout << "  - DRONE:    " << setw(6) << probs[1] * 100.0f << "%\n";
        cout << "  - MISSILE:  " << setw(6) << probs[2] * 100.0f << "%\n";
        cout << "  - ETC:      " << setw(6) << probs[3] * 100.0f << "%\n";
        cout << "최종 전술 판정: "
             << colorize(result.status, result.message) << "\n";
        cout << string(55, '-') << "\n";
    }

    cout << "\n[시스템 종료] Reasoning Guard 오프라인\n";
    return 0;
}