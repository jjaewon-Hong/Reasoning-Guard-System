// ============================================================
//  tactical_wasm.cpp — Reasoning Guard WebAssembly Module
//  C++ → WASM via Emscripten
//
//  기능:
//   1) preprocess()  : RGBA 픽셀 → CHW [1,3,224,224] float 텐서
//                      (Zero-Padding + Bilinear Resize + ImageNet Normalize)
//   2) evaluate()    : sigmoid 확률값 → 전술 판정 (DANGER/CAUTION/SAFE)
//
//  이 모듈은 브라우저에서 onnxruntime-web과 함께 동작하여
//  GitHub Pages 정적 호스팅 환경에서도 실제 ONNX 추론을 수행한다.
// ============================================================

#include <emscripten/emscripten.h>
#include <cmath>
#include <cstring>
#include <algorithm>

// ── 상수 정의 ─────────────────────────────────────────────────
static constexpr int TARGET_H = 224;
static constexpr int TARGET_W = 224;
static constexpr int CHANNELS = 3;
static constexpr int TENSOR_SIZE = CHANNELS * TARGET_H * TARGET_W; // 150528

// ImageNet 정규화 파라미터
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

// 전술 판정용 클래스명
static const char* CLASS_NAMES[4] = {"FIGHTER", "DRONE", "MISSILE", "ETC"};

// ── 내부 버퍼 ─────────────────────────────────────────────────
static float g_tensor[TENSOR_SIZE];  // 전처리된 CHW 텐서

// 전술 판정 결과 버퍼
struct TacticalResultWasm {
    int   status;        // 0=SAFE, 1=CAUTION, 2=DANGER
    int   predicted_idx; // 0=FIGHTER, 1=DRONE, 2=MISSILE, 3=ETC
    float fighter;
    float drone;
    float missile;
    float etc;
    float tactical_max;
};
static TacticalResultWasm g_result;

// ============================================================
//  Bilinear Interpolation (C++ 구현)
//  OpenCV 없이 순수 C++로 이미지 리사이즈 수행
// ============================================================
static void bilinear_resize(
    const unsigned char* src, int srcW, int srcH,
    unsigned char* dst, int dstW, int dstH, int channels)
{
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float srcX = (x + 0.5f) * srcW / dstW - 0.5f;
            float srcY = (y + 0.5f) * srcH / dstH - 0.5f;

            srcX = std::max(0.0f, std::min(srcX, (float)(srcW - 1)));
            srcY = std::max(0.0f, std::min(srcY, (float)(srcH - 1)));

            int x0 = (int)srcX;
            int y0 = (int)srcY;
            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);

            float fx = srcX - x0;
            float fy = srcY - y0;

            for (int c = 0; c < channels; ++c) {
                float v00 = src[(y0 * srcW + x0) * channels + c];
                float v10 = src[(y0 * srcW + x1) * channels + c];
                float v01 = src[(y1 * srcW + x0) * channels + c];
                float v11 = src[(y1 * srcW + x1) * channels + c];

                float val = v00 * (1 - fx) * (1 - fy)
                          + v10 * fx * (1 - fy)
                          + v01 * (1 - fx) * fy
                          + v11 * fx * fy;

                dst[(y * dstW + x) * channels + c] = (unsigned char)(val + 0.5f);
            }
        }
    }
}

// ============================================================
//  preprocess() — 브라우저 Canvas RGBA → ONNX 입력 텐서
//
//  Python/C++ preprocess()와 동일한 로직:
//   1) RGBA → RGB 추출
//   2) 긴 변 기준 Zero-Padding (정사각형)
//   3) 224×224 Bilinear Resize
//   4) [0,255] → [0,1] float 변환
//   5) ImageNet Normalize: (pixel - mean) / std
//   6) HWC → CHW 텐서 변환
//
//  @param rgba_data  Canvas.getImageData()의 RGBA 픽셀 배열
//  @param width      원본 이미지 너비
//  @param height     원본 이미지 높이
//  @return           CHW float 텐서 포인터 (크기: 150528 floats)
// ============================================================
extern "C" {

EMSCRIPTEN_KEEPALIVE
float* preprocess(const unsigned char* rgba_data, int width, int height) {
    // 1) RGBA → RGB 추출
    int pixel_count = width * height;
    unsigned char* rgb = new unsigned char[pixel_count * 3];
    for (int i = 0; i < pixel_count; ++i) {
        rgb[i * 3 + 0] = rgba_data[i * 4 + 0]; // R
        rgb[i * 3 + 1] = rgba_data[i * 4 + 1]; // G
        rgb[i * 3 + 2] = rgba_data[i * 4 + 2]; // B
        // A 채널은 무시
    }

    // 2) Zero-Padding (긴 변 기준 정사각형)
    int max_dim = std::max(width, height);
    unsigned char* padded = new unsigned char[max_dim * max_dim * 3];
    std::memset(padded, 0, max_dim * max_dim * 3); // 검은색 패딩

    int x_offset = (max_dim - width) / 2;
    int y_offset = (max_dim - height) / 2;

    for (int y = 0; y < height; ++y) {
        std::memcpy(
            padded + ((y + y_offset) * max_dim + x_offset) * 3,
            rgb + (y * width) * 3,
            width * 3
        );
    }
    delete[] rgb;

    // 3) Bilinear Resize → 224×224
    unsigned char* resized = new unsigned char[TARGET_H * TARGET_W * 3];
    bilinear_resize(padded, max_dim, max_dim, resized, TARGET_W, TARGET_H, 3);
    delete[] padded;

    // 4-6) float 변환 + ImageNet 정규화 + HWC→CHW 변환
    for (int c = 0; c < CHANNELS; ++c) {
        for (int h = 0; h < TARGET_H; ++h) {
            for (int w = 0; w < TARGET_W; ++w) {
                float pixel = resized[(h * TARGET_W + w) * 3 + c] / 255.0f;
                g_tensor[c * TARGET_H * TARGET_W + h * TARGET_W + w] =
                    (pixel - MEAN[c]) / STD[c];
            }
        }
    }
    delete[] resized;

    return g_tensor;
}

// ============================================================
//  get_tensor_size() — 텐서 크기 반환 (JS에서 Float32Array 생성용)
// ============================================================
EMSCRIPTEN_KEEPALIVE
int get_tensor_size() {
    return TENSOR_SIZE;
}

// ============================================================
//  evaluate_threat() — 전술 판정 엔진 (C++ 구현)
//
//  tactical_engine.h와 동일한 판정 로직:
//   - DANGER  : 전투기/드론/미사일 최대값 >= 90% AND ETC <= 50%
//   - CAUTION : 전투기/드론/미사일 최대값 >= 40%
//   - SAFE    : 그 외
//
//  @param probs  sigmoid 출력 [FIGHTER, DRONE, MISSILE, ETC]
//  @return       TacticalResultWasm 포인터
// ============================================================
EMSCRIPTEN_KEEPALIVE
TacticalResultWasm* evaluate_threat(const float* probs) {
    float fighter = probs[0];
    float drone   = probs[1];
    float missile = probs[2];
    float etc     = probs[3];

    // 최고 확률 클래스 인덱스
    int pred_idx = 0;
    float max_val = fighter;
    if (drone   > max_val) { max_val = drone;   pred_idx = 1; }
    if (missile > max_val) { max_val = missile;  pred_idx = 2; }
    if (etc     > max_val) { max_val = etc;      pred_idx = 3; }

    // 전술 타격 대상 최대 확률
    float tactical_max = std::max({fighter, drone, missile});

    g_result.fighter      = fighter;
    g_result.drone        = drone;
    g_result.missile      = missile;
    g_result.etc          = etc;
    g_result.predicted_idx = pred_idx;
    g_result.tactical_max = tactical_max;

    // 전술 판정
    if (tactical_max >= 0.90f && etc <= 0.50f) {
        g_result.status = 2; // DANGER
    } else if (tactical_max >= 0.40f) {
        g_result.status = 1; // CAUTION
    } else {
        g_result.status = 0; // SAFE
    }

    return &g_result;
}

// ============================================================
//  get_class_name() — 클래스 인덱스 → 문자열 반환
// ============================================================
EMSCRIPTEN_KEEPALIVE
const char* get_class_name(int idx) {
    if (idx >= 0 && idx < 4) return CLASS_NAMES[idx];
    return "UNKNOWN";
}

} // extern "C"
