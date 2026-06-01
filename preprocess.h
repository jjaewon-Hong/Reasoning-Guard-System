/*
=============================================================================
  [Native C++ Image Preprocessing Module]
  
  본 파일(preprocess.h)은 웹(WASM) 환경과 독립적으로, 데스크톱 및 서버 등 
  순수 네이티브 C++ 환경(main.cpp)에서 AI 추론을 수행하기 전
  이미지 입력 데이터를 규격에 맞게 변환(Preprocessing)하는 모듈입니다.

  - 개발 목적: OpenCV 라이브러리를 활용하여, Python 기반 딥러닝 모델의 
    학습 시 사용된 전처리 로직(정사각형 Zero-Padding, 224x224 리사이즈, 
    ImageNet 정규화 등)을 C++에서 정확하게 재현하기 위함입니다.
  - 참고 사항: WebAssembly(웹) 빌드 시에는 무거운 OpenCV 종속성을 피하기 위해 
    자체 구현된 가벼운 전처리 로직(tactical_wasm.cpp)을 대체 사용하므로, 
    본 파일은 웹 서비스 배포(HTML)에는 직접 관여하지 않는 네이티브 전용 코드입니다.
=============================================================================
*/

#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>

static std::vector<float> preprocess(const std::string& img_path) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty())
        throw std::runtime_error("이미지를 불러올 수 없습니다: " + img_path);

    // 1) BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 2) Zero-Padding (정사각형)
    int max_dim = std::max(img.rows, img.cols);
    cv::Mat square = cv::Mat::zeros(max_dim, max_dim, CV_8UC3);
    int x_off = (max_dim - img.cols) / 2;
    int y_off = (max_dim - img.rows) / 2;
    img.copyTo(square(cv::Rect(x_off, y_off, img.cols, img.rows)));

    // 3) Resize → 224x224
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(224, 224));

    // 4) float 변환 [0, 1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // 5) ImageNet 정규화 (mean/std)
    const float mean[3]   = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    // 6) CHW 텐서로 변환 (ONNX Runtime 입력 포맷)
    constexpr int H = 224, W = 224, C = 3;
    std::vector<float> tensor(C * H * W);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                tensor[c * H * W + h * W + w] =
                    (float_img.at<cv::Vec3f>(h, w)[c] - mean[c]) / std_dev[c];

    return tensor;
}
