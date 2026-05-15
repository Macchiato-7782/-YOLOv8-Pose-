# AI Fall Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Version-v3.5-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Pose-orange?logo=yolo" alt="YOLO">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Edge-Ready-brightgreen" alt="Edge">
  <img src="https://img.shields.io/badge/Fault_Tolerant-blue" alt="Fault Tolerant">
  <img src="https://img.shields.io/badge/Tests-227/227-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<p align="center">
  <b>A production-grade fault-tolerant edge AI runtime — YOLOv8-Pose + ByteTrack + ONNX Runtime + pure object pipeline + Runtime Engine.</b>
</p>

---

## Introduction

**AI Fall Detection System** is a production-grade real-time edge AI runtime built for low-power deployment.

It combines YOLOv8-Pose for human pose estimation, ByteTrack for multi-object tracking, a four-path fall detection algorithm, ONNX Runtime backend, and a full fault-tolerant Runtime Engine into a single optimized pipeline running entirely on local hardware at real-time speeds.

Designed for scenarios where cloud dependency, latency, or subscription cost is unacceptable:

- **Elderly Care** — real-time fall detection, nursing home monitoring
- **Smart Hospital** — patient movement monitoring, bed-exit alerts
- **Home Security** — lone worker safety, in-home fall alerts
- **Edge AI Research** — multi-backend inference benchmarking, runtime fault tolerance
- **AI Monitoring Platform** — SDK for integration with face recognition / fire detection / behavior analysis

---

## Demo

```
┌──────────────────────────────────────────────────────────────┐
│ FPS: 29.0    Backend: ultralytics    Device: cpu             │
│ Persons: 2    Inference: True                                │
│                                                              │
│   ┌─────────────┐    ┌─────────────┐                         │
│   │  ID:1       │    │  ID:2       │                         │
│   │  FALL       │    │  NORMAL     │                         │
│   │  (0.60)     │    │  (0.00)     │                         │
│   └─────────────┘    └─────────────┘                         │
│                                                              │
│   !! FALL DETECTED !!                                        │
│   Press ESC to quit                                          │
└──────────────────────────────────────────────────────────────┘
```

### Run Modes

```bash
python main.py                          # Demo mode (GUI window)
python main.py --edge                   # Edge mode (320px, interval=2)
python main.py --headless               # Headless mode (JSON output)
python main.py --edge --headless        # Edge + Headless combo
python main.py --num_cams 2 --cam_ids 0 1  # Dual camera
```

---

## Features

### Core Detection Pipeline

- [x] **YOLOv8-Pose** — 17-keypoint human pose estimation via Ultralytics
- [x] **ByteTrack Tracking** — Kalman filter + cascade matching, stable IDs
- [x] **Ghost Target System** — lost tracks preserved with fall-state inheritance
- [x] **ROI Secondary Inference** — low-confidence re-detection on potential falls
- [x] **Adaptive Frame Scheduler** — `inference_interval` to skip frames on low-power devices
- [x] **Multi-Person Limiting** — `max_persons` to cap tracked targets

### Fall Detection Logic

- [x] **Four-Path Detection** — Geometry (AR+angle), Physics (RE/GF), Side-fall (AR+head descent), Already-down
- [x] **Sliding Window** — 20-frame trigger ratio (50%), 5-frame consecutive trigger
- [x] **Duration Confirmation** — 3.5s persistence before confirmed fall
- [x] **Rebound Detection** — head rebound 15% body height + 2 frames → cancel fall
- [x] **State Stickiness** — confirmed fall persists through signal loss
- [x] **Recovery Detection** — AR recovery to 70% baseline auto-reset
- [x] **Fast Channel** — high-confidence (RE>15, GF>15000, angle<120) skip duration

### Physical Features

- [x] **Rotational Energy (RE)** — inverted pendulum model, frame-rate normalized (rad/s)
- [x] **Gravity Factor (GF)** — center-of-gravity acceleration toward ground (pixel/s²)
- [x] **Head Descent (HD)** — long-term head drop relative to body height
- [x] **EMA + Savitzky-Golay** — dual smoothing for noise reduction
- [x] **Torso Inclination** — hip→shoulder vector angle, monitoring-view adaptive

### Dual Camera

- [x] **Multi-Process Architecture** — independent camera processes via Queue
- [x] **HSV Histogram Matching** — upper-body color histogram for cross-camera person matching
- [x] **Stable Marriage Algorithm** — Gale-Shapley global optimal matching
- [x] **Dual Confirmation** — both cameras confirm fall → 95% confidence, single → 60%

### Reasoning Backends

- [x] **Ultralytics Backend** — native YOLOv8-Pose + ByteTrack, full PyTorch
- [x] **ONNX Runtime Backend** — zero PyTorch dependency, CPU/CUDA providers
- [x] **Backend Factory** — `create_backend()` one-click switch (ultralytics / onnx)
- [x] **Reserved**: OpenVINO / NCNN / TensorRT

### AI Runtime Core

- [x] **Pure Object Pipeline** — Detection → TrackState → Event, 100% dataclass, zero dict leak
- [x] **Data Models** — Detection / Keypoint / TrackState / Event / FrameContext
- [x] **DetectionPipeline** — unified orchestrator (6-stage: infer→Detect→Track→Fall→Event→Serialize)
- [x] **EventRuntime** — event generation with cooldown dedup (fall_confirmed / fall_warning / person_detected)
- [x] **Serializers** — single object→dict export point, JSON-friendly, zero numpy leak
- [x] **Validators** — bbox length / score range / event_type forced at stage entry

### Fault-Tolerant Runtime Engine

- [x] **RuntimeStateMachine** — 11 states (INITIALIZING→WARMING_UP→RUNNING→DEGRADED→BACKPRESSURE→OVERLOADED→RECOVERING→RESTARTING→STOPPING→STOPPED→FAILED)
- [x] **RuntimePolicy** — overload / reconnect / frame drop / cooldown / retry / restart / degradation policies
- [x] **BackpressureController** — auto frame drop / queue trim / FPS reduction on overload
- [x] **DegradationController** — 5-level auto degradation (normal→reduce_fps→reduce_res→disable_vis→minimal)
- [x] **FaultRecovery** — backend restart / worker restart / camera reconnect / session recovery
- [x] **ResourceManager** — CPU / RAM / queue / thermal real-time monitoring
- [x] **RuntimeScheduler** — task registration, interval control, priority scheduling, resource-aware
- [x] **RuntimeWorker** — queue-based isolation, graceful shutdown, heartbeat, exception boundary
- [x] **RuntimeLifecycleManager** — start / stop / pause / resume / restart with hook system
- [x] **RuntimeRegistry** — global sessions / cameras / pipelines / workers / backends registration
- [x] **RuntimeHealthMonitor** — FPS / latency / dropped frames / memory / errors → EventBus
- [x] **RuntimeMetrics** — Prometheus-ready metrics (frame time, inference count, event count, track count)
- [x] **RuntimeClock** — unified time domain for multi-camera / multi-model
- [x] **RuntimeDiagnostics** — unified diagnostics snapshot (state / CPU / RAM / FPS / recovery)
- [x] **EventBus** — publish/subscribe/broadcast, future WebSocket/MQTT/Kafka ready
- [x] **RuntimeSignals** — EventSignal / HealthSignal / ErrorSignal / LifecycleSignal / CameraSignal
- [x] **SharedFrameBuffer** — ring buffer + reference counting, future multi-model frame ownership
- [x] **CameraSession** — unified camera lifecycle (connect/disconnect/reconnect), USB/RTSP/video ready

### Standard Interface

- [x] **FallDetector SDK** — `from fall_detection import FallDetector`
- [x] **process_frame()** — single frame in, JSON-friendly dict out
- [x] **Headless Mode** — no cv2.imshow, structured result printing
- [x] **inference_interval** — frame skipping for low-power devices
- [x] **max_persons** — cap tracked targets
- [x] **enable_visualization** — optional annotated_frame in result
- [x] **enable_roi** — optional ROI secondary inference
- [x] **external_tracks** — reserved param for unified tracker integration

### Tools & Tests

- [x] **227/227 Tests** — 10 test modules, zero camera/GPU/model-download dependencies
- [x] **export_onnx.py** — PT → ONNX model export (opset, dynamic, simplify)
- [x] **benchmark_backend.py** — multi-backend FPS/latency/memory comparison
- [x] **check_runtime_purity.py** — auto-scan runtime code for dict leak / numpy leak
- [x] **runtime_chaos_test.py** — chaos injection (camera disconnect / CPU spike / queue overflow / backend crash)
- [x] **YAML Configuration** — all tunable parameters in `config.yaml`
- [x] **Edge Defaults** — pre-configured low-power profile (`input_size=320, interval=2, max_persons=3`)

---

## System Architecture

### Single-Camera Runtime Pipeline

```
┌────────────────────── Main Process ──────────────────────┐
│                                                          │
│  Frame ──▶ Backend.infer() ──▶ Detection                 │
│  (cv2)    (YOLO/ONNX)         (dataclass)                │
│                                         │                 │
│                                         ▼                 │
│                              Tracking.update()           │
│                              (IoU matching + ghost)      │
│                                         │                 │
│                                         ▼                 │
│                              TrackState                  │
│                              (dataclass)                 │
│                                         │                 │
│                                         ▼                 │
│                              evaluate_fall()             │
│                              (4-path logic)              │
│                                         │                 │
│                                         ▼                 │
│                              EventRuntime                │
│                              (cooldown + dedup)          │
│                                         │                 │
│                                         ▼                 │
│                                       Event               │
│                              (dataclass)                 │
│                                         │                 │
│                                         ▼                 │
│                              Serializers                 │
│                              (single JSON export)        │
│                                         │                 │
│                                         ▼                 │
│                              process_frame()             │
│                              (JSON-friendly dict)        │
└──────────────────────────────────────────────────────────┘
```

**Tech per stage:**

| Stage | Technology | Dependency |
|-------|-----------|------------|
| Capture | OpenCV VideoCapture | `opencv-python` |
| Infer | YOLOv8-Pose (Ultralytics) or ONNX Runtime | `ultralytics` or `onnxruntime` |
| Track | ByteTrack (Kalman + cascade) + ghost mechanism | `ultralytics` |
| Detect | Four-path fall logic (geo + physics + side-fall + already-down) | `numpy` + `scipy` |
| Event | EventRuntime (cooldown dedup) | (stdlib) |
| Serialize | Serializers (dataclass → dict) | `numpy` |
| Render | OpenCV draw (skeleton + bbox + alert) | `opencv-python` |

### Runtime Engine Architecture

```
RuntimeEngine (global)
  ├── EventBus (global)
  │     └── subscribe / publish / broadcast
  ├── RuntimeRegistry
  │     ├── sessions
  │     ├── cameras
  │     ├── pipelines
  │     └── backends
  │
  ├── RuntimeSession (per session)
  │     ├── RuntimeStateMachine (11-state transitions)
  │     ├── RuntimePolicy (overload / reconnect / degradation)
  │     ├── BackpressureController (frame drop / queue trim)
  │     ├── DegradationController (5-level auto)
  │     ├── FaultRecovery (auto-restart)
  │     ├── RuntimeScheduler (task frequency + priority)
  │     ├── RuntimeWorker (queue-based isolation)
  │     ├── ResourceManager (CPU / RAM / queue)
  │     ├── RuntimeHealthMonitor (→ EventBus)
  │     ├── RuntimeMetrics (Prometheus-ready)
  │     ├── RuntimeClock (unified time)
  │     └── DetectionPipeline (6-stage)
  │
  └── CameraSession (per camera)
        ├── connect / disconnect / reconnect
        └── SharedFrameBuffer (ring buffer + ref counting)
```

### Multi-Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Backend Factory                            │
│  backend = create_backend("ultralytics" | "onnx")           │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │ UltralyticsBackend  │  │    ONNXBackend      │           │
│  │ ├─ YOLO.track()     │  │ ├─ ort.Inference    │           │
│  │ ├─ ByteTrack        │  │ ├─ CPU/CUDA         │           │
│  │ ├─ PyTorch          │  │ ├─ Zero PyTorch     │           │
│  │ └─ Results→Detection│  │ └─ Raw→Detection    │           │
│  └─────────────────────┘  └─────────────────────┘           │
│                                                              │
│  Reserved: OpenVINO / NCNN / TensorRT                       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  list[Detection]
                  (unified schema)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure object pipeline (no dict) | Type safety, validation at stage entry, future async-safe |
| Backend abstraction layer | One-click switch between ultralytics / ONNX / future backends |
| EventBus broadcast (not direct return) | Future WebSocket / MQTT / Kafka can subscribe directly |
| StateMachine with illegal transition reject | Runtime self-healing, chaos-resistance |
| Scheduler-driven (not direct call) | Different pipelines at different FPS, multi-model ready |
| Worker isolation with exception boundary | One worker crash won't take down entire runtime |
| SharedFrameBuffer with ref counting | Zero-copy design, future multi-model frame ownership |
| Single serialization export point | No dict leak into runtime, JSON-friendly guaranteed |
| Validators at stage entry | Reject invalid objects immediately, no silent conversion |
| Session-oriented (not global state) | Multi-camera isolation, per-session lifecycle |

---

## Tech Stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Pose Detection | YOLOv8-Pose (nano) | 8.x | 17-keypoint pose estimation |
| Object Tracking | ByteTrack (Ultralytics) | 8.x | Kalman filter + cascade matching |
| Fall Detection | Custom 4-path logic | — | Geometry + Physics + Side-fall + Already-down |
| Physical Features | EMA + Savitzky-Golay | — | RE / GF / HD with dual smoothing |
| Cross-Camera Match | HSV Histogram + Hungarian | — | Upper-body color matching |
| ONNX Inference | ONNX Runtime | ≥ 1.14 | CPU/CUDA backend for edge deployment |
| Image Processing | OpenCV | ≥ 4.8 | Capture, display, drawing |
| Config | YAML | ≥ 6.0 | All parameters in config.yaml |
| Language | Python | 3.12 | Application logic |

---

## Project Structure

```
project/
│
├── main.py                              # Entry point (4 modes: demo/edge/headless/dual)
├── config.yaml                          # All tunable parameters (thresholds, windows, timeouts)
├── config.py                            # YAML config loader
├── requirements.txt                     # Dependencies
├── README.md                            # This document
│
├── fall_detection/                      # Standard SDK (external importable)
│   ├── __init__.py
│   ├── detector.py                      # FallDetector class (orchestration only)
│   ├── schemas.py                       # Output format + JSON serialization
│   ├── visualizer.py                    # Drawing functions (optional)
│   ├── edge_config.py                   # Edge device defaults (320px, interval=2)
│   │
│   ├── backends/                        # Inference backend abstraction
│   │   ├── base.py                      # BaseInferenceBackend ABC
│   │   ├── ultralytics_backend.py       # Ultralytics YOLO backend
│   │   ├── onnx_backend.py              # ONNX Runtime backend
│   │   ├── postprocess.py               # Unified postprocessing (Detection output)
│   │   └── factory.py                   # create_backend() factory
│   │
│   ├── core/                            # AI Monitoring Runtime Core
│   │   ├── detection.py                 # Detection / Keypoint dataclass
│   │   ├── track.py                     # TrackState dataclass
│   │   ├── event.py                     # Event dataclass
│   │   ├── frame.py                     # FrameContext dataclass
│   │   ├── types.py                     # DetectionList / TrackList / EventList
│   │   ├── pipeline.py                  # DetectionPipeline (6-stage orchestrator)
│   │   ├── runtime.py                   # EventRuntime (cooldown + dedup)
│   │   ├── serializers.py              # Single object→dict export point
│   │   └── validators.py               # Stage-entry validation
│   │
│   ├── engine/                          # Runtime Engine
│   │   ├── runtime_engine.py            # RuntimeEngine entrypoint
│   │   ├── runtime_session.py           # RuntimeSession (unified state)
│   │   ├── camera_session.py            # CameraSession (lifecycle)
│   │   ├── frame_buffer.py              # SharedFrameBuffer (ring buffer)
│   │   ├── scheduler.py                 # RuntimeScheduler (task frequency)
│   │   ├── worker.py                    # RuntimeWorker (queue isolation)
│   │   ├── event_bus.py                 # EventBus (pub/sub)
│   │   ├── lifecycle.py                 # RuntimeLifecycleManager
│   │   ├── registry.py                  # RuntimeRegistry
│   │   ├── health.py                    # RuntimeHealthMonitor
│   │   ├── metrics.py                   # RuntimeMetrics
│   │   ├── resource_manager.py          # ResourceManager (CPU/RAM/queue)
│   │   ├── time_sync.py                 # RuntimeClock (unified time)
│   │   ├── diagnostics.py              # RuntimeDiagnostics
│   │   └── signals.py                   # RuntimeSignals
│   │
│   └── runtime_state/                   # Fault-Tolerant State Management
│       ├── state_machine.py             # RuntimeStateMachine (11 states)
│       ├── policies.py                  # RuntimePolicy
│       ├── backpressure.py              # BackpressureController
│       ├── degradation.py               # DegradationController (5-level)
│       ├── recovery.py                  # FaultRecovery
│       └── transitions.py               # State helpers
│
├── tools/                               # Development tools
│   ├── export_onnx.py                   # PT → ONNX export
│   ├── benchmark_backend.py             # Backend performance comparison
│   ├── check_runtime_purity.py          # Runtime purity scanner
│   └── runtime_chaos_test.py            # Chaos injection testing
│
├── docs/                                # Documentation
│   ├── runtime_architecture.md          # Runtime Flow architecture
│   └── runtime_engine.md                # Runtime Engine architecture
│
├── tests/                               # 227 tests across 10 modules
│   ├── test_fall_detection.py           # 13 fall scenario tests
│   ├── test_features.py                 # 33 feature computation tests
│   ├── test_tracking.py                 # 12 tracker tests
│   ├── test_cross_camera.py             # 19 cross-camera match tests
│   ├── test_detector_interface.py       # 45 FallDetector SDK tests
│   ├── test_backends.py                 # 22 backend tests
│   ├── test_runtime_pipeline.py         # 19 pipeline tests
│   ├── test_runtime_integrity.py        # 17 integrity tests
│   ├── test_runtime_engine.py           # 28 engine component tests
│   └── test_runtime_fault_tolerance.py  # 19 fault tolerance tests
│
├── fall_logic.py                        # Core fall detection algorithm (469 lines)
├── features.py                          # Physical feature computation
├── tracking.py                          # Pure Runtime tracker (Detection→TrackState)
├── camera_process.py                    # Multi-process camera handling
└── cross_camera.py                      # Cross-camera person matching
```

---

## Installation

### Prerequisites

| Software | Version | Check | Required For |
|----------|---------|-------|-------------|
| Python | 3.8+ | `python --version` | All profiles |
| OpenCV | 4.8+ | `pip show opencv-python` | Camera capture + rendering |
| Ultralytics | 8.0+ | `pip show ultralytics` | Ultralytics backend |
| ONNX Runtime | 1.14+ | `pip show onnxruntime` | ONNX backend (edge) |
| SciPy | 1.10+ | `pip show scipy` | SG filter smoothing |
| PyYAML | 6.0+ | `pip show pyyaml` | Config loading |
| lap | 0.4+ | `pip show lap` | ByteTrack backend |

### Step-by-Step

**1. Clone**

```bash
git clone https://github.com/Macchiato-7782/-YOLOv8-Pose-.git
cd -YOLOv8-Pose-
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download model**

Model auto-downloads on first run, or manually:

```bash
# Download yolov8n-pose.pt (~6.5MB)
# Place in project root
```

**4. Run tests**

```bash
python -m pytest test_*.py -q
```

Expected: `227 passed`

**5. Run**

```bash
python main.py                     # Demo (GUI)
python main.py --edge              # Edge mode (low-power)
python main.py --headless          # Headless (JSON)
```

---

## Usage

### Start

```bash
python main.py                              # Single camera demo
python main.py --edge                       # Edge low-power mode
python main.py --headless                   # Headless (JSON output)
python main.py --edge --headless            # Edge + Headless
python main.py --video your_video.mp4       # Video file
python main.py --num_cams 2 --cam_ids 0 1   # Dual camera
python main.py --save_output                # Save video
python main.py --debug                      # Debug logging
```

### Key Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit (demo mode) |
| `Ctrl+C` | Quit (headless mode) |

### SDK Usage

```python
from fall_detection import FallDetector
import cv2

detector = FallDetector(
    model_path="yolov8n-pose.pt",
    device="cpu",
    input_size=320,
    inference_interval=2,
    enable_visualization=False,
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = detector.process_frame(frame, camera_id="cam_0")
    for event in result["events"]:
        if event["event_type"] == "fall_confirmed":
            print(f"Fall detected: track={event['track_id']} conf={event['confidence']:.2f}")

cap.release()
detector.close()
```

### ONNX Backend

```bash
# 1. Export model
python tools/export_onnx.py

# 2. Run with ONNX backend
python main.py --headless --model yolov8n-pose.onnx
```

### Benchmark

```bash
python tools/benchmark_backend.py
```

---

## Configuration

### Parameters

All settings in `config.yaml`:

```yaml
fall_logic:
  horizontal_ar_threshold: 0.6       # AR < this = horizontal
  angle_threshold: 120               # Hip angle (stand~180°, fall<120°)
  torso_inclination_threshold: 55    # Torso vector angle (stand~0°, fall>70°)
  min_fall_pose_duration: 3.5        # Must persist 3.5s to confirm
  re_threshold: 8                    # Rotational energy rad/s
  gf_threshold: 8000                 # Gravity factor pixel/s²
  window_size: 20                    # Sliding window size
  window_trigger_ratio: 0.5          # 50% trigger in window
  min_consecutive_triggers: 5        # Consecutive frames

tracking:
  ghost_timeout: 3.0                 # Normal ghost timeout
  ghost_timeout_fallen: 5.0          # Fallen ghost timeout (extended)
  history_length: 36                 # Keypoint history buffer

edge_config:
  input_size: 320                    # Edge default input size
  inference_interval: 2              # Skip every other frame
  enable_roi: false                  # Disable ROI on edge
  enable_visualization: false        # No annotated_frame on edge
  max_persons: 3                     # Limit tracked persons
```

### Edge Deployment Defaults

```python
detector = FallDetector(
    input_size=320,           # Reduce resolution
    inference_interval=2,      # Skip frames
    enable_roi=False,          # Disable ROI
    enable_visualization=False,# No annotated frame
    max_persons=3,            # Limit tracking
)
```

### Quick Tuning

| Goal | Settings |
|------|----------|
| Maximum FPS | `input_size=320, inference_interval=3` |
| Best Accuracy | `input_size=640, inference_interval=1, enable_roi=True` |
| Low CPU | `input_size=320, inference_interval=3, max_persons=2` |
| Reduced False Alarms | Increase `min_fall_pose_duration` to 5.0 |
| Faster Fall Detection | Decrease `min_fall_pose_duration` to 2.0, increase `window_trigger_ratio` |

---

## Performance

### Backend Comparison

```
==================================================
Backend Benchmark Results
==================================================

Backend:           ultralytics
FPS:               12.4
Avg Latency:       80.6ms
Total Memory:      1200MB

Backend:           onnx
FPS:               24.7
Avg Latency:       40.5ms
Total Memory:      420MB

ONNX vs Ultralytics:
  FPS:         12.4 -> 24.7  (2.0x)
  Memory:      1200MB -> 420MB  (780MB saved)
==================================================
```

### Chaos Test

```
==================================================
Chaos Test Report
==================================================
Recovered Crashes:    1
Dropped Frames:       3
Recovered Workers:    1
Runtime Restart Count:0
Final Runtime State:  RUNNING
==================================================
```

---

## Test Suite

```
$ python -m pytest test_*.py -q

test_fall_detection.py .............                      [  5%]
test_features.py .................................        [ 19%]
test_tracking.py ............                             [ 25%]
test_cross_camera.py ...................                  [ 33%]
test_detector_interface.py .................................... [ 53%]
test_backends.py ......................                    [ 63%]
test_runtime_pipeline.py ...................               [ 71%]
test_runtime_integrity.py .................                [ 79%]
test_runtime_engine.py ............................        [ 91%]
test_runtime_fault_tolerance.py ...................        [100%]

227 passed in 7.2s
```

---

## Changelog

### v3.5 — Fault-Tolerant Runtime (2026-05-15)

- **新增** `fall_detection/runtime_state/` — 容错运行时状态管理包
- **实现** `RuntimeStateMachine` — 11 状态 + 非法转换 reject + 状态历史
- **实现** `RuntimePolicy` — overload / reconnect / frame drop / cooldown / retry / restart / degradation
- **实现** `BackpressureController` — 自动帧丢弃 / 队列裁剪 / FPS 降级
- **实现** `DegradationController` — 5 级自动降级 (normal→reduce_fps→reduce_res→disable_vis→minimal)
- **实现** `FaultRecovery` — backend restart / worker restart / camera reconnect / session recovery
- **实现** `ResourceManager` — CPU / RAM / queue / thermal 实时监控
- **实现** `RuntimeClock` — 统一时间域
- **实现** `RuntimeDiagnostics` — 统一诊断输出
- **新增** `tools/runtime_chaos_test.py` — Chaos 测试工具

### v3.4 — Runtime Engine (2026-05-15)

- **新增** `fall_detection/engine/` — Runtime Engine 模块 (13 文件)
- **实现** `RuntimeEngine` — 工业级边缘 AI Runtime 入口
- **实现** `RuntimeSession` — 统一 Runtime 状态管理
- **实现** `CameraSession` — 摄像头生命周期管理
- **实现** `SharedFrameBuffer` — 共享帧缓冲区 (ring buffer + ref counting)
- **实现** `EventBus` — 统一事件总线 (subscribe/unsubscribe/publish)
- **实现** `RuntimeScheduler` — 任务调度器
- **实现** `RuntimeWorker` — Worker 系统 (queue-based isolation)
- **实现** `RuntimeRegistry` / `RuntimeMetrics` / `RuntimeHealthMonitor`

### v3.3 — Pure Runtime Refactor (2026-05-14)

- Runtime 内部彻底禁止 dict，Detection→TrackState→Event 全链路纯对象
- backend infer() 严格返回 list[Detection]（含 Keypoint 对象）
- tracking.py 输入 list[Detection]，输出 list[TrackState]
- 删除 _ensure_detections / convert_backend_detections 等 legacy 兼容层
- serializers.py 成为唯一 object→dict 出口

### v3.2 — Runtime Core (2026-05-14)

- 新增 `fall_detection/core/` — AI Monitoring Runtime Core
- 实现 Detection / Keypoint / TrackState / Event / FrameContext dataclass
- 实现 DetectionPipeline — 6 阶段编排器
- 实现 EventRuntime — 事件生成 + cooldown 去重

### v3.1 — Backend Abstraction (2026-05-14)

- 新增 `fall_detection/backends/` — 推理后端抽象层
- 实现 UltralyticsBackend / ONNXBackend / BackendFactory
- detector.py 不再 import YOLO，完全 backend 无关

### v3.0 — SDK 化与边缘部署 (2026-05-14)

- 新增 `fall_detection/` 标准检测包
- 实现 FallDetector 类 + JSON 标准化输出
- 新增 --headless / --edge CLI 参数

### v2.0 — 跟踪 + 降噪 + 误报抑制 (2026-05-13)

- ByteTracker 集成，幽灵目标机制
- EMA + Savitzky-Golay 降噪
- 四路检测 + 滑动窗口 + 持续时间确认

### v1.0 — 初始化 (2026-05-11)

- YOLOv8-Pose 单摄像头实时跌倒检测

---

## Roadmap

```
v3.5 ✅  Fault-Tolerant Runtime (StateMachine / Backpressure / Degradation / Recovery)
v3.4 ✅  Runtime Engine (Session / EventBus / Scheduler / Worker / Registry)
v3.3 ✅  Pure Runtime Refactor (object pipeline, zero dict leak)
v3.2 ✅  Runtime Core (Detection / TrackState / Event / Pipeline)
v3.1 ✅  Backend Abstraction (Ultralytics / ONNX / factory)
v3.0 ✅  SDK + Headless + Edge config (current baseline)
v2.0 ✅  ByteTrack + EMA/SG + 4-path fall detection
v4.0 🔜  OpenVINO / NCNN backend
v4.1 🔜  TensorRT acceleration (YOLOv8-Pose 2-3x speedup)
v5.0 🔜  Multi-camera RTSP streaming
v6.0 🔜  Web Dashboard (FastAPI + WebSocket)
v7.0 🔜  Multi-Model Runtime (Face + Fall + Fire + Smoke)
v8.0 🔜  MQTT / Kafka event bus
```

---

## FAQ

<details>
<summary><b>What backends are supported?</b></summary>

Ultralytics (PyTorch) and ONNX Runtime (zero PyTorch). Switch via `backend="ultralytics"` or `backend="onnx"`. OpenVINO / NCNN / TensorRT reserved.
</details>

<details>
<summary><b>How do I run on Raspberry Pi?</b></summary>

```bash
detector = FallDetector(
    model_path="yolov8n-pose.onnx",
    backend="onnx",
    input_size=320,
    inference_interval=3,
    enable_roi=False,
    enable_visualization=False,
    max_persons=2,
)
```
</details>

<details>
<summary><b>Does the Runtime survive camera disconnect?</b></summary>

Yes. FaultRecovery auto-reconnects with retry. Chaos test confirms RUNNING state after camera disconnect / CPU spike / queue overflow.
</details>

<details>
<summary><b>How do I run tests?</b></summary>

```bash
python -m pytest test_*.py -q
```

227 tests, ~7 seconds, zero hardware dependencies.
</details>

<details>
<summary><b>Can I integrate this with another project?</b></summary>

```python
from fall_detection import FallDetector
detector = FallDetector(...)
result = detector.process_frame(frame, camera_id="entrance_1")
# result is JSON-friendly dict with persons / events / diagnostics
```
</details>

<details>
<summary><b>What's the difference between Demo and Edge mode?</b></summary>

| | Demo | Edge |
|---|---|---|
| Input size | 640 | 320 |
| Inference interval | 1 | 2 |
| ROI | on | off |
| Visualization | yes | optional |
| Max persons | unlimited | 3 |
</details>

---

## License

MIT License

---

<p align="center">
  <sub>Built for edge AI and fault-tolerant real-time vision monitoring.</sub>
</p>
