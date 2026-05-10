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
