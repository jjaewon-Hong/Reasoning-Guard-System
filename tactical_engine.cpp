extern "C" {
    // main_conf: THREAT/FRIENDLY/UNKNOWN 중 최대값, etc_conf: ETC 확률
    __declspec(dllexport) const char* evaluate_threat(float main_conf, float etc_conf) {
        if (main_conf >= 0.90f && etc_conf <= 0.50f) return "DANGER";  // 즉각 요격
        if (main_conf >= 0.40f) return "CAUTION";                      // 수동 승인
        return "SAFE";                                                  // 대기
    }
}
