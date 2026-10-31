# Hand Gesture Recognition for Device Control 🖐️🖱️

High-performance, low-latency hand gesture recognition and touchless device control system for macOS (optimized for Apple Silicon / M-series chips).

Built with **MediaPipe Tasks API**, **Native macOS Quartz Events (PyObjC)**, **One-Euro Filtering**, and an **Automated Multi-State FSM**.

---

## 🚀 Key Features

- **Sub-10ms Pipeline Latency:** Ultra-fast detection pipeline yielding p50 ~ 5ms, p95 < 8ms on Apple Silicon CPU.
- **Micro-Jitter Free One-Euro Filter:** Uses actual $\Delta t$ and velocity-adaptive cutoff frequency to deliver butter-smooth cursor motion without latency lag.
- **Trackpad-Style Relative Mapping + Clutch:**
  - Dynamic acceleration curve based on hand velocity.
  - **Clutch Mode (Fist / Pause):** Freeze cursor position to comfortably reposition your hand in the air without moving the on-screen mouse.
- **Pinch-Drift Prevention:**
  - Cursor is anchored to the Index Finger MCP joint (`INDEX_FINGER_MCP`), not the fingertip, preventing movement while closing fingers.
  - Micro-freeze window upon pinch contact prevents click-sliding.
- **Guaranteed `mouseUp` Safety:** Unconditional hardware release when hand is lost (>200ms timeout), on application exit, or upon camera stutter.
- **Multi-Mode Control & Touchless Switching:**
  - **Touchless Mode Switch:** Hold a **3-Finger W-Pose** (Index, Middle, Ring extended) for **0.5s** to cycle modes (`MOUSE` $\to$ `PRESENTATION` $\to$ `MEDIA`) with an animated HUD progress ring and single-fire latching.
  - **Mouse Mode:** Cursor navigation, Left Click, Drag & Drop, Right Click (Middle-Pinch), Two-Finger Pixel Scroll, Fist Clutch.
  - **Presentation Mode:** Keynote / PowerPoint slide navigation via horizontal directional swipes (Next / Previous slide).
  - **4-Way Media Mode:** Native macOS `NSSystemDefined` events:
    - 🖐️➡️ **Swipe Right:** Next Track (`NEXT_TRACK = 17`)
    - 🖐️⬅️ **Swipe Left:** Previous Track (`PREV_TRACK = 18`)
    - 🖐️⬆️ **Swipe Up:** Volume Up (`VOLUME_UP = 0`)
    - 🖐️⬇️ **Swipe Down:** Volume Down (`VOLUME_DOWN = 1`)
    - 👍 **Thumbs-Up:** Play / Pause (`PLAY_PAUSE = 16`)
- **Headless Record & Replay:** Every session can be recorded to JSONL and replayed deterministically without a webcam for automated CI/CD testing.
- **Cyberpunk HUD Visualizer:** 21-landmark hand skeleton, active anchor tracking, reactive feedback rings, real-time FPS, and latency telemetry.

---

## 🛠️ Project Structure

```text
hand_gesture_control/
├── pyproject.toml              # Build & test configurations
├── requirements.txt            # Pinned dependency manifest
├── models/
│   └── hand_landmarker.task    # MediaPipe Tasks API model bundle
├── src/
│   ├── camera/
│   │   └── stream.py           # Threaded single-element atomic buffer capture
│   ├── controller/
│   │   ├── base.py             # Base & Mock mouse controllers
│   │   ├── quartz_mouse.py     # Native macOS Quartz mouse driver
│   │   ├── scroll.py           # Deadzone-stabilized pixel scroll driver
│   │   ├── media.py            # macOS NSSystemDefined media controller
│   │   └── presentation.py     # Slide arrow key presentation driver
│   ├── tracking/
│   │   ├── one_euro.py         # Variable dt 1€ filter
│   │   ├── cursor_controller.py# Relative acceleration, clutch & pinch freeze
│   │   └── state_machine.py    # 7-State FSM with hysteresis & safety timeout
│   ├── vision/
│   │   ├── detector.py         # MediaPipe Tasks HandLandmarker wrapper
│   │   ├── geometry.py         # Palm-normalized metrics & joint angles
│   │   ├── classifier.py       # Static pose classifier with temporal voting
│   │   ├── dynamic_gestures.py # Dominant direction swipe detector
│   │   ├── recorder.py         # JSONL landmark recorder
│   │   └── replayer.py         # Deterministic JSONL mock stream player
│   ├── ui/
│   │   └── hud.py              # OpenCV real-time telemetry HUD overlay
│   └── main.py                 # Integrated application loop & CLI
└── tests/                      # 529 automated tests (100% passing)
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation
Ensure you are on macOS with Python 3.11+.

```bash
# Clone the repository
cd hand_gesture_control

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Granting macOS Permissions
Because this tool drives native mouse and keyboard events on macOS, your terminal host (Terminal, iTerm2, or PyCharm) requires Accessibility permissions:
1. Open **System Settings > Privacy & Security > Accessibility**.
2. Enable your terminal or IDE application.
3. Open **Privacy & Security > Camera** and allow camera access.

### 3. Running the Application

#### Standard Mouse Mode (Webcam + HUD):
```bash
.venv/bin/python -m src.main --mode mouse
```

#### Presentation Mode (Keynote / Slides):
```bash
.venv/bin/python -m src.main --mode presentation
```

#### Media Mode (Volume & Play/Pause):
```bash
.venv/bin/python -m src.main --mode media
```

#### Recording a Session:
```bash
.venv/bin/python -m src.main --record session.jsonl
```

#### Replaying a Session (No Camera Needed):
```bash
.venv/bin/python -m src.main --replay session.jsonl
```

#### Running Headless (Background Daemon):
```bash
.venv/bin/python -m src.main --mode mouse --headless
```

---

## ⌨️ Interactive Controls & Gestures

| Gesture / Key | Action | Function |
|---|---|---|
| 🖖 **3-Finger (W-Pose)** | Hold 0.5s | Cycle mode (`MOUSE` $\to$ `PRESENTATION` $\to$ `MEDIA`) with HUD progress fill |
| ✊ **Fist** | Hold | Clutch mode (freeze cursor tracking to reposition hand) |
| `m` | Key press | Backup keyboard shortcut to switch operational mode |
| `q` or `ESC` | Key press | Safe exit and clean release of all mouse buttons / keys |

---

## 🧪 Running Automated Tests

Run the full test suite of **529 automated unit and integration tests**:

```bash
.venv/bin/pytest -v
```

---

## 📜 Authorship
- **Author:** Berkay Yağız Eren (<berkaybbgn@gmail.com>)
- **License:** MIT
