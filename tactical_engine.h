/*
=============================================================================
  [Native C++ Tactical Logic Engine]
  
  본 파일(tactical_engine.h)은 웹(WASM) 환경과 독립적으로, 데스크톱 및 서버 등 
  순수 네이티브 C++ 환경(main.cpp)에서 AI 모델의 예측 결과(확률값)를 바탕으로 
  최종 전술 판정(DANGER / CAUTION / SAFE)을 내리는 모듈입니다.

  - 개발 목적: AI 추론 결과값(0.0~1.0)을 기반으로 어떤 상황에서 발포(FIRE)를 
    승인할지, 혹은 대기(HOLD)할지를 결정하는 핵심 알고리즘을 C++ 콘솔에서 
    검증하고 테스트하기 위함입니다.
  - 참고 사항: WebAssembly(웹) 빌드 시에는 자바스크립트와의 데이터 연동(C++ 문자열 처리 등) 
    오버헤드를 줄이기 위해, 숫자(int, float) 기반으로 경량화된 자체 로직(tactical_wasm.cpp 내 
    evaluate_threat 함수)을 대체 사용합니다. 따라서 본 파일은 웹 배포(HTML)에는 
    직접 관여하지 않는 네이티브 전용 코드입니다.
=============================================================================
*/

#pragma once
#pragma execution_character_set("utf-8")
#include <string>
#include <algorithm>

// ============================================================
//  TacticalEngine — 실전 전술 판정 엔진
//  probs[4]: [FIGHTER, DRONE, MISSILE, ETC] sigmoid 확률값
//  판정 규칙:
//    DANGER  : FIGHTER/DRONE/MISSILE 최대값 >= 90% AND ETC <= 50%
//    CAUTION : FIGHTER/DRONE/MISSILE 최대값 >= 40%
//    SAFE    : 그 외
// ============================================================

static const char* CLASS_NAMES[4] = {"FIGHTER", "DRONE", "MISSILE", "ETC"};

struct TacticalResult {
    std::string status;       // "DANGER" | "CAUTION" | "SAFE"
    std::string message;      // 출력 메시지
    std::string predicted;    // 최고 확률 클래스명
    float       tactical_max; // FIGHTER/DRONE/MISSILE 중 최대 확률
    float       etc_conf;     // ETC 확률
};

class TacticalEngine {
public:
    static TacticalResult evaluate_threat(const float probs[4]) {
        float fighter = probs[0];
        float drone   = probs[1];
        float missile = probs[2];
        float etc     = probs[3];

        // 최고 확률 클래스 선택
        int pred_idx = 0;
        float max_val = fighter;
        if (drone   > max_val) { max_val = drone;   pred_idx = 1; }
        if (missile > max_val) { max_val = missile;  pred_idx = 2; }
        if (etc     > max_val) { max_val = etc;      pred_idx = 3; }

        float tactical_max = std::max({fighter, drone, missile});

        TacticalResult r;
        r.predicted    = CLASS_NAMES[pred_idx];
        r.tactical_max = tactical_max;
        r.etc_conf     = etc;

        // [DANGER] 타격 대상 식별 90% 이상 AND 노이즈(ETC) 50% 이하
        if (tactical_max >= 0.90f && etc <= 0.50f) {
            r.status  = "DANGER";
            r.message = "[DANGER  (FIRE)    ] " + r.predicted
                        + " : Threat identified - Interceptor assigned";
        }
        else if (tactical_max >= 0.40f) {
            r.status  = "CAUTION";
            r.message = "[CAUTION (OVERRIDE)] " + r.predicted
                        + " : Suspected target - Awaiting fire approval";
        }
        else {
            r.status  = "SAFE";
            r.message = "[SAFE    (HOLD)    ] Background/noise - Tracking mode";
        }

        return r;
    }
};
