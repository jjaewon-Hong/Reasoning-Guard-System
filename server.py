import os, random
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from flask import Flask, jsonify, send_from_directory

# ── 설정 ────────────────────────────────────────────────────
MODEL_PATH = "reasoning_guard_engine.onnx"
IMG_DIR    = "./test_pool"
CLASS_NAMES = ["FIGHTER", "DRONE", "MISSILE", "ETC"]

app = Flask(__name__, static_folder="docs", static_url_path="")

# ── 모델 로드 ────────────────────────────────────────────────
print("[RG] ONNX 모델 로드 중...")
session = ort.InferenceSession(MODEL_PATH)
print("[RG] 모델 로드 완료")

# ── 전처리 ──────────────────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert("RGB")
    max_dim = max(img.size)
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(new_img).unsqueeze(0).numpy()

# ── 전술 판정 ────────────────────────────────────────────────
def evaluate(probs):
    f, d, m, e = float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])
    tactical_max = max(f, d, m)
    pred_idx  = int(np.argmax(probs))
    predicted = CLASS_NAMES[pred_idx]
    if tactical_max >= 0.90 and e <= 0.50:
        status = "DANGER"
    elif tactical_max >= 0.40:
        status = "CAUTION"
    else:
        status = "SAFE"
    return status, predicted, f, d, m, e

# ── 라우트 ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("docs", "index.html")

@app.route("/api/simulate")
def simulate():
    images = [f for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        return jsonify({"error": "No images"}), 500

    target = random.choice(images)
    input_data = preprocess(os.path.join(IMG_DIR, target))
    outputs = session.run(["sigmoid"], {"x": input_data})
    probs   = outputs[0][0]

    status, predicted, f, d, m, e = evaluate(probs)

    return jsonify({
        "target":    target,
        "predicted": predicted,
        "status":    status,
        "probs": {
            "fighter": round(f * 100, 1),
            "drone":   round(d * 100, 1),
            "missile": round(m * 100, 1),
            "etc":     round(e * 100, 1)
        }
    })

@app.route("/api/image/<path:filename>")
def get_image(filename):
    return send_from_directory(IMG_DIR, filename)

# ── 진입점 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("[RG] 서버 시작: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
