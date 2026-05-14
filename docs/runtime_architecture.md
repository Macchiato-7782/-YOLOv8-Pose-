# AI Monitoring Runtime Architecture

## Runtime Flow

```
Frame (OpenCV BGR)
  │
  ▼
Backend.infer(frame)
  │  ┌─ UltralyticsBackend
  │  └─ ONNXBackend (后续: OpenVINO, NCNN)
  ▼
list[Detection]
  │  ├─ bbox: [x1,y1,x2,y2]
  │  ├─ score: float
  │  ├─ keypoints: list[Keypoint]
  │  └─ track_id: int | None
  ▼
Tracking.convert_backend_detections()
  │  IoU matching + ghost management
  ▼
list[TrackState]
  │  ├─ track_id, bbox, center
  │  ├─ state: normal | potential_fall | fall
  │  ├─ confidence: float
  │  └─ is_ghost, age, lost_frames
  ▼
Fall Logic (evaluate_fall)
  │  四路检测: 几何 + 物理 + 侧倒 + 地面
  │  滑动窗口 + 持续时间确认
  ▼
updated list[TrackState]
  │
  ▼
EventRuntime.process_tracks()
  │  防重复 + cooldown + event generation
  ▼
list[Event]
  │  ├─ fall_confirmed
  │  ├─ fall_warning
  │  ├─ person_detected
  │  └─ track_lost
  ▼
Serializers
  │  dataclass → JSON-friendly dict
  ▼
process_frame output dict
  {
    "module": "fall_detection",
    "camera_id": "...",
    "timestamp": ...,
    "frame_id": ...,
    "persons": [...],
    "events": [...],
    "diagnostics": {...}
  }
```

## Design Principles

### 1. Backend Independence
- Backend 只负责 `infer(frame) → list[Detection]`
- detector.py / tracking.py / pipeline.py 完全不 import ultralytics
- 切换后端只需改 `backend` 参数

### 2. Dataclass Pipeline
- 内部统一使用 dataclass (Detection, TrackState, Event, FrameContext)
- 不允许 dict 在模块间传递
- 只有最终输出时 serialize

### 3. Pipeline Orchestration
- DetectionPipeline 编排整个流程
- detector.py 只做 orchestration，不写业务逻辑
- 每一层职责清晰、可替换

### 4. Event Runtime
- 独立的事件生成和管理
- 支持 cooldown（防止重复事件）
- 为未来 websocket / mqtt 等事件总线做准备

## Future Extensions

### Multi-Model Runtime
```
Frame → [
    FallDetectionPipeline  → events
    FaceDetectionPipeline  → events
    FireDetectionPipeline  → events
] → EventBus → External
```

只需:
1. 实现新的 DetectionPipeline 子类
2. 添加 MultiModelRuntime 调度
3. 事件总线分发

### External Communication
```
EventRuntime
  ├─ WebSocket → 前端实时推送
  ├─ MQTT      → IoT 设备
  ├─ Kafka     → 大数据平台
  └─ HTTP      → REST API
```

只需:
1. 实现 EventBus 接口
2. 订阅 EventRuntime 事件

### New Detection Tasks
```
BaseDetectionPipeline
  ├─ FallDetectionPipeline
  ├─ FaceDetectionPipeline
  ├─ SmokeDetectionPipeline
  ├─ FireDetectionPipeline
  └─ BehaviorDetectionPipeline
```

只需:
1. 实现新的 Backend (或复用)
2. 实现新的 Logic 模块
3. 继承 DetectionPipeline
