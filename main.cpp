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

namespace fs = std::filesystem;

// ============================================================
//  이미지 파일 목록 수집
// ============================================================
std::vector<std::string> collect_images(const std::string& dir) {
    std::vector<std::string> images;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        // 소문자 변환
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            images.push_back(entry.path().string());
    }
    return images;
}

// ============================================================
//  판정 상태별 ANSI 색상
// ============================================================
std::string colorize(const std::string& status, const std::string& msg) {
    if (status == "DANGER")  return "\033[91m" + msg + "\033[0m";
    if (status == "CAUTION") return "\033[93m" + msg + "\033[0m";
    return "\033[92m" + msg + "\033[0m"; // SAFE
}

// ============================================================
//  메인 진입점
// ============================================================
int main() {
    // Windows 콘솔 UTF-8 설정
    system("chcp 65001 > nul");

    const std::string MODEL_PATH  = "reasoning_guard_engine.onnx";
    const std::string IMG_DIR     = "./test_pool";

    // ── 1. ONNX Runtime 세션 초기화 ─────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "reasoning_guard");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (!fs::exists(MODEL_PATH)) {
        std::cerr << "[ERROR] 모델 파일을 찾을 수 없습니다: " << MODEL_PATH << "\n";
        return 1;
    }

    // Windows: wstring 경로 사용
    std::wstring wmodel_path(MODEL_PATH.begin(), MODEL_PATH.end());
    Ort::Session session(env, wmodel_path.c_str(), session_opts);

    std::cout << "[OK] ONNX 모델 로드 완료: " << MODEL_PATH << "\n";

    // ── 2. 이미지 파일 목록 수집 ────────────────────────────
    auto images = collect_images(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[ERROR] test_pool 폴더에 이미지가 없습니다.\n";
        return 1;
    }
    std::cout << "[OK] 표적 이미지 " << images.size() << "개 확보\n";

    // ── 3. 랜덤 엔진 ─────────────────────────────────────────
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, images.size() - 1);

    // ── 4. ONNX 입출력 메타데이터 ───────────────────────────
    const char* input_names[]  = {"x"};
    const char* output_names[] = {"sigmoid"};
    std::array<int64_t, 4> input_shape{1, 3, 224, 224};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── 5. 메인 루프 ─────────────────────────────────────────
    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << " Reasoning Guard: Tactical Operational System [C++]\n";
    std::cout << " [v3] ONNX Runtime C++ API — DANGER 발사 조건 적용\n";
    std::cout << std::string(55, '=') << "\n";

    while (true) {
        std::cout << "\n[Enter]: +1 HOUR 시뮬레이션 진행 | [q]: 종료 -> ";
        std::string cmd;
        std::getline(std::cin, cmd);
        if (cmd == "q" || cmd == "Q") break;

        // ── 5-1. 랜덤 표적 선택 ────────────────────────────
        const std::string& target_path = images[dist(rng)];
        std::string target_name = fs::path(target_path).filename().string();

        // ── 5-2. 이미지 전처리 (Zero-Padding → 224×224 → Normalize)
        std::vector<float> input_data;
        try {
            input_data = preprocess(target_path);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] 전처리 실패: " << e.what() << "\n";
            continue;
        }

        // ── 5-3. ONNX 추론 ─────────────────────────────────
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

        // ── 5-4. 전술 판정 ─────────────────────────────────
        TacticalResult result = TacticalEngine::evaluate_threat(probs);

        // ── 5-5. 결과 출력 ─────────────────────────────────
        std::cout << "\n[TIME +1H] 감지된 표적: " << target_name << "\n";
        std::cout << "AI 분류 결과: " << result.predicted << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  - FIGHTER:  " << std::setw(6) << probs[0] * 100.0f << "%\n";
        std::cout << "  - DRONE:    " << std::setw(6) << probs[1] * 100.0f << "%\n";
        std::cout << "  - MISSILE:  " << std::setw(6) << probs[2] * 100.0f << "%\n";
        std::cout << "  - ETC:      " << std::setw(6) << probs[3] * 100.0f << "%\n";
        std::cout << "최종 전술 판정: "
                  << colorize(result.status, result.message) << "\n";
        std::cout << std::string(55, '-') << "\n";
    }

    std::cout << "\n[시스템 종료] Reasoning Guard 오프라인\n";
    return 0;
}
