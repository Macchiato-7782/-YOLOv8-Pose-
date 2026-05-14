# Runtime Engine Architecture

## Engine Architecture

```
RuntimeEngine
  ├── RuntimeRegistry (全局注册中心)
  │     ├── sessions
  │     ├── cameras
  │     ├── pipelines
  │     ├── workers
  │     └── backends
  │
  ├── EventBus (全局事件总线)
  │     ├── subscribe / unsubscribe / publish
  │     └── future: websocket / mqtt / kafka
  │
  ├── RuntimeSession (每会话)
  │     ├── LifecycleManager (start/stop/pause/resume/restart)
  │     ├── Scheduler (task frequency control)
  │     ├── Metrics (FPS/latency/events/tracks)
  │     ├── HealthMonitor (health signals → EventBus)
  │     ├── FrameBuffer (shared frame ownership)
  │     └── Pipeline (DetectionPipeline)
  │
  ├── CameraSession (每摄像头)
  │     ├── connect / disconnect / reconnect
  │     ├── read_frame → FrameBuffer
  │     └── USB / RTSP / video / future WebRTC
  │
  └── RuntimeWorker (每 worker)
        ├── submit / process / drain
        └── inference / tracking / event
```

## Flow

```
CameraSession.read_frame()
  → FrameBuffer.push(frame, ctx)
  → Scheduler.tick()
    → fall_detection task (every 1/N frames)
      → Pipeline.run(frame, ctx)
        → backend.infer() → Detection
        → tracker.update() → TrackState
        → evaluate_fall → updated TrackState
        → EventRuntime → Event
        → serialize → dict
    → health_check task (every 30 frames)
      → HealthMonitor.check()
        → HealthSignal → EventBus
  → EventBus.publish(EventSignal)
  → Metrics.record_frame/record_inference/record_event
```

## Design Principles

### 1. Session-Oriented
- RuntimeSession 是唯一 Runtime 状态持有者
- 不允许 global state, scattered state
- 多 session 同时运行（multi-camera by design）

### 2. Scheduler-Driven
- 所有计算由 scheduler tick 驱动
- 不同 pipeline 不同频率（face=10fps, fall=1fps, health=1/30fps）
- 优先级调度

### 3. Event-Driven
- EventBus 作为唯一事件出口
- 不再直接 return events list
- Future websocket/mqtt/kafka 可直接订阅

### 4. Shared Frame Ownership
- SharedFrameBuffer 统一管理帧生命周期
- Ring buffer + reference counting
- Future multi-modal runtime 共享同一帧

### 5. Worker Isolation
- RuntimeWorker 隔离异常
- Queue-based 异步安全
- Graceful shutdown + drain

### 6. Async-Ready Design
- Queue-based architecture
- Event-driven communication
- Worker separation
- Non-blocking by design

## Future Extension

### Multi-Model Runtime
```python
engine = RuntimeEngine()

session = engine.create_session("main", "cam_0", pipeline, backend)
session.scheduler.register("face_detection", face_task, interval_frames=3, priority=5)
session.scheduler.register("smoke_detection", smoke_task, interval_frames=10, priority=2)
```

### Multi-Camera Runtime
```python
engine = RuntimeEngine()
engine.create_session("cam_0", "cam_0", pipeline_a, backend_a)
engine.create_session("cam_1", "cam_1", pipeline_b, backend_b)
```

### External Event Bus
```python
from fall_detection.engine import EventBus, EventSignal

bus = EventBus()
bus.subscribe("fall_confirmed", lambda s: websocket.send(s.to_dict()))
bus.subscribe_all(lambda s: kafka.produce(s.to_dict()))
```
