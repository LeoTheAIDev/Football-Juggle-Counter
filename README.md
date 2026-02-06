# Tennis Ball Detection + Touch Counter (with Video Export)

This project detects a tennis ball in a video using a YOLO model and predicts **touch/bounce events** (shown as “Touch!”) using a small PyTorch temporal model (“BounceNet”). It exports an **annotated video** showing the ball marker, probability, and a running touch count.

---

## What it does

For each frame:

1. Detects the ball with YOLO (`BallModel.pt`)
2. Builds a rolling feature window (`T` frames) from the ball trajectory (e.g., `y`, `vy`, `ay`, `conf`)
3. Uses **BounceNet** (`TouchModel.pt`) to predict `p(touch)`
4. Triggers a **Touch!** event when probability crosses a threshold (with cooldown to avoid duplicates)
5. Draws overlays and writes an exported annotated video

---

## Folder structure

Example:

.
├── main.py
├── BallModel.pt
├── TouchModel.pt
├── Video.mp4
└── Video_annotated.mp4   # generated


---

## Requirements

- Python 3.9+ recommended
- Works on macOS / Windows / Linux
- GPU optional:
  - Apple Silicon: uses **MPS** if available
  - NVIDIA: uses **CUDA** if available
  - otherwise runs on CPU

Install dependencies:

python3 -m pip install --upgrade pip
python3 -m pip install numpy opencv-python ultralytics torch torchvision torchaudio

Optional (if you need extra OpenCV modules):

python3 -m pip install opencv-contrib-python

---

## Quick start

1) Put these files in the project folder:

- BallModel.pt
- TouchModel.pt
- Video.mp4

2) Run:

python3 main.py

3) Output:

- A live preview window (press **q** to quit)
- An exported annotated video saved as `Video_annotated.mp4` (default)

---

## Configuration

Edit these in `main.py` if wanted:

DETECTOR_WEIGHTS = "BallModel.pt"
BOUNCE_WEIGHTS   = "TouchModel.pt"
VIDEO_PATH       = "Video.mp4"
OUTPUT_PATH      = "Video_annotated.mp4"

Key tuning parameters:

CONF = 0.1          # YOLO confidence threshold
IMGSZ = 640         # YOLO inference image size

BOUNCE_TH = 0.60    # probability threshold to trigger "Touch!"
COOLDOWN_FRAMES = 7 # ignore triggers for N frames after a touch

---

## About the Touch/Bounce model checkpoint (TouchModel.pt)

The script expects the checkpoint to be a dict with:

- state_dict: model weights

Optional keys supported:

- T: window length (default 21)
- feat_names: list of feature names (default ["y_f","vy","ay","conf_f"])

Supported feature names:

- y or y_f
- vy
- ay
- conf or conf_f

If the checkpoint contains unknown feature names, the script fills them with 0.0.

---

## Troubleshooting

### Exported MP4 won’t play / is black

Some OpenCV builds can’t write MP4/H.264 reliably.

Fix: export AVI + XVID.

1) Change:
OUTPUT_PATH = "Video_annotated.avi"

2) And set the writer codec to:
fourcc = cv2.VideoWriter_fourcc(*"XVID")

### Too many false “Touch!” events
- Increase BOUNCE_TH (try 0.70–0.85)
- Increase COOLDOWN_FRAMES (try 10–15)
- Ensure ball detection is stable (false detections cause false touches)

### Missing touches
- Lower BOUNCE_TH
- Reduce COOLDOWN_FRAMES
- Check if the ball detector loses the ball during key frames

### Ball detection is unstable
- Increase CONF slightly (e.g. 0.2)
- Use a higher quality video (less blur)
- Retrain/finetune BallModel.pt for your court/lighting

---

## Ideas for improvements
- Add a tracker (Kalman/SORT/ByteTrack) instead of selecting the highest-confidence box each frame
- Normalize vy/ay by FPS to make predictions stable across different frame rates
- Export detected touch events to CSV (frame index + timestamp)

---

## License
Add a LICENSE file (MIT is a common choice).
