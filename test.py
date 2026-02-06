import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

import torch
import torch.nn as nn

# --- CONFIG ---
DETECTOR_WEIGHTS = "BallModel.pt"
BOUNCE_WEIGHTS   = "TouchModel.pt"   # your trained BounceNet checkpoint

VIDEO_PATH  = "Video.mp4"            # or 0 for webcam
OUTPUT_PATH = "Video_annotated.mp4"  # exported video path

CONF = 0.1
IMGSZ = 640

# BounceNet trigger tuning
BOUNCE_TH = 0.60
COOLDOWN_FRAMES = 7
# --------------

class BounceNet(nn.Module):
    # Must match the architecture used during training
    def __init__(self, in_feats=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_feats, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2)  # -> [B, F, T]
        return self.net(x).squeeze(1)  # [B]

def put_top_right_text(frame, text, y=35, scale=0.9, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = frame.shape[1] - tw - 15
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    # --- Load YOLO detector ---
    detector = YOLO(DETECTOR_WEIGHTS, task="detect")

    # --- Load BounceNet ---
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BOUNCE_WEIGHTS, map_location=device)

    T = int(ckpt.get("T", 21))  # window length
    feat_names = ckpt.get("feat_names", ["y_f", "vy", "ay", "conf_f"])
    in_feats = len(feat_names)

    bounce_model = BounceNet(in_feats=in_feats).to(device)
    bounce_model.load_state_dict(ckpt["state_dict"])
    bounce_model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH if isinstance(VIDEO_PATH, str) else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_PATH}")

    # --- Setup video writer (export annotated video) ---
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or np.isnan(fps):
        fps = 30.0  # fallback for some webcams/files

    # MP4 writer (try avc1 first; fallback to mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Could not open VideoWriter. Try OUTPUT_PATH ending with .avi and fourcc 'XVID'.")

    # --- Trajectory buffers for features ---
    feat_buf = deque(maxlen=T)
    prev_y = None
    prev_vy = 0.0

    touch_count = 0
    cooldown = 0
    last_prob = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- Detect ball ---
        results = detector.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

        ball_center = None
        det_conf = 0.0

        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            b = results.boxes[best_idx]

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            det_conf = float(b.conf[0].cpu().numpy())

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            rpx = int(min((x2 - x1), (y2 - y1)) / 2)

            ball_center = (cx, cy)

            # draw ball
            cv2.circle(frame, ball_center, max(rpx, 2), (255, 0, 0), 2)
            cv2.circle(frame, ball_center, 2, (0, 0, 255), -1)

        # --- Update feature buffer ---
        touch_happened = False

        if ball_center is not None:
            y = float(ball_center[1])

            if prev_y is None:
                vy = 0.0
                ay = 0.0
            else:
                vy = y - prev_y
                ay = vy - prev_vy

            prev_y = y
            prev_vy = vy

            feat = []
            for n in feat_names:
                if n in ("y", "y_f"):             feat.append(y)
                elif n == "vy":                    feat.append(vy)
                elif n == "ay":                    feat.append(ay)
                elif n in ("conf", "conf_f"):      feat.append(det_conf)
                else:                               feat.append(0.0)

            feat_buf.append(np.array(feat, dtype=np.float32))

        else:
            # push "missing" feature vector to keep timing consistent
            y = prev_y if prev_y is not None else 0.0
            vy = 0.0
            ay = 0.0
            feat = []
            for n in feat_names:
                if n in ("y", "y_f"):             feat.append(float(y) if y is not None else 0.0)
                elif n == "vy":                    feat.append(float(vy))
                elif n == "ay":                    feat.append(float(ay))
                elif n in ("conf", "conf_f"):      feat.append(0.0)
                else:                               feat.append(0.0)

            feat_buf.append(np.array(feat, dtype=np.float32))

        # --- Predict touch prob ---
        if len(feat_buf) == T:
            X = np.stack(list(feat_buf), axis=0)  # [T, F]
            X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,F]

            with torch.no_grad():
                logit = bounce_model(X_t)[0]
                prob = float(torch.sigmoid(logit).cpu().numpy())
                last_prob = prob

            if cooldown > 0:
                cooldown -= 1
            else:
                if prob >= BOUNCE_TH:
                    touch_count += 1
                    cooldown = COOLDOWN_FRAMES
                    touch_happened = True

        # --- Overlay UI ---
        put_top_right_text(frame, f"Touches: {touch_count}", y=35)
        put_top_right_text(frame, f"p(touch): {last_prob:.2f}", y=70, scale=0.75, thickness=2)

        if touch_happened:
            cv2.putText(frame, "Touch!", (15, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

        if ball_center is not None:
            cv2.putText(frame, f"center: {ball_center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Write annotated frame to output video ---
        out.write(frame)

        # --- Live preview ---
        cv2.imshow("Ball Detection + TouchNet (exporting)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
