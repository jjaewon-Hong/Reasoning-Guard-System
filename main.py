import os, random
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# 클래스 인덱스 매핑 (실전 전술 타격 대상 기준)
CLASS_NAMES = ["FIGHTER", "DRONE", "MISSILE", "ETC"]

# [Tactical Logic] 설계서의 OperationalResponse를 파이썬으로 구현
class TacticalEngine:
    @staticmethod
    def evaluate_threat(probs):
        """
        4-class sigmoid 출력 기반 실전 전술 판정.
        probs: [FIGHTER, DRONE, MISSILE, ETC] 각각의 sigmoid 확률값

        판정 규칙:
          1) 전투기/드론/미사일 중 하나가 90% 이상 AND ETC가 50% 이하 → DANGER (즉각 요격)
          2) 위 3개 객체 중 최대값이 40% 이상 → CAUTION (수동 교전 승인 대기)
          3) 그 외 → SAFE (추적 대기)
        """
        fighter_conf = float(probs[0])
        drone_conf   = float(probs[1])
        missile_conf = float(probs[2])
        etc_conf     = float(probs[3])

        predicted_idx = int(np.argmax(probs))
        predicted_cls = CLASS_NAMES[predicted_idx]

        # 실전 타격 대상(FIGHTER, DRONE, MISSILE) 중 최대 신뢰도
        tactical_target_max = max(fighter_conf, drone_conf, missile_conf)

        # [DANGER 발사 조건] 타격 대상 식별 90% 이상 AND 노이즈(ETC) 50% 이하
        if tactical_target_max >= 0.90 and etc_conf <= 0.50:
            return ("DANGER",
                    f"\033[91mDANGER (FIRE)\033[0m - {predicted_cls} 위협 식별, 요격 미사일 할당 완료",
                    predicted_cls, tactical_target_max, etc_conf)
        elif tactical_target_max >= 0.40:
            return ("CAUTION",
                    f"\033[93mCAUTION (OVERRIDE)\033[0m - {predicted_cls} 의심 표적, 사격 승인 대기",
                    predicted_cls, tactical_target_max, etc_conf)
        else:
            return ("SAFE",
                    "\033[92mSAFE (HOLD)\033[0m - 배경/노이즈 또는 식별 불가, 추적 모드 유지",
                    predicted_cls, tactical_target_max, etc_conf)

# [Pre-processing] 형상 보존형 Zero-Padding 로직 (노트북 이식)
def preprocess(path):
    img = Image.open(path).convert('RGB')
    max_dim = max(img.size)
    new_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(new_img).unsqueeze(0).numpy()

def main():
    # 1. 모델 로드 (노트북에서 도출한 ONNX 엔진)
    model_path = "reasoning_guard_engine.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path}를 찾을 수 없습니다.")
        return

    session = ort.InferenceSession(model_path)
    img_dir = './test_pool'
    
    # 2. 이미지 파일 리스트 확보
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("Error: test_pool 폴더에 이미지가 없습니다.")
        return

    print("\n" + "="*55)
    print(" Reasoning Guard: Tactical Operational System (Integrated)")
    print(" [v2] ETC 데이터셋 학습 반영 - DANGER 자동 발사 조건 적용")
    print("="*55)

    while True:
        cmd = input("\n[Enter]: +1 HOUR 시뮬레이션 진행 | [q]: 종료 -> ")
        if cmd.lower() == 'q': break
        
        # 랜덤 샘플링 (+1 HOUR 시나리오)
        target = random.choice(images)
        input_data = preprocess(os.path.join(img_dir, target))
        
        # 3. AI 추론 (변경된 ONNX IO: input="x", output="sigmoid")
        outputs = session.run(["sigmoid"], {"x": input_data})
        probs = outputs[0][0]  # [FIGHTER, DRONE, MISSILE, ETC]
        
        # 4. 전술 판정 호출 (DANGER 발사 조건 적용)
        status, message, pred_cls, tactical_max, etc_conf = TacticalEngine.evaluate_threat(probs)
        
        print(f"\n[TIME +1H] 감지된 표적: {target}")
        print(f"AI 분류 결과: {pred_cls}")
        print(f"  - FIGHTER:  {float(probs[0])*100:6.2f}%")
        print(f"  - DRONE:    {float(probs[1])*100:6.2f}%")
        print(f"  - MISSILE:  {float(probs[2])*100:6.2f}%")
        print(f"  - ETC:      {float(probs[3])*100:6.2f}%")
        print(f"최종 전술 판정: {message}")
        print("-" * 55)

if __name__ == "__main__":
    main()
