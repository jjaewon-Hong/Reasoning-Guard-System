#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>

// ============================================================
//  preprocess() — Python 코드와 동일한 Zero-Padding 전처리
//  1) BGR -> RGB 변환
//  2) 긴 변 기준 정사각형 Zero-Padding
//  3) 224x224 리사이즈
//  4) ImageNet 정규화 (mean/std)
//  반환: CHW float 텐서 [1, 3, 224, 224] (size = 150528)
// ============================================================

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
