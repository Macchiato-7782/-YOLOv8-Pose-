"""
RuntimeSession -- 统一运行时状态管理
管理 tracking state / frame counter / event cooldown / backend lifecycle / scheduler / metrics / camera
"""

import time
import logging
from typing import Optional
from enum import Enum

from fall_detection.core.frame import FrameContext
from fall_detection.core.serializers import serialize_result
from fall_detection.engine.event_bus import EventBus
from fall_detection.engine.lifecycle import RuntimeLifecycleManager, RuntimeStatus
from fall_detection.engine.scheduler import RuntimeScheduler
from fall_detection.engine.metrics import RuntimeMetrics
from fall_detection.engine.health import RuntimeHealthMonitor
from fall_detection.engine.signals import EventSignal, LifecycleSignal
from fall_detection.core.event import Event

logger = logging.getLogger(__name__)


class RuntimeSession:
    """统一运行时会话 -- 唯一 Runtime 状态持有者"""

    def __init__(
        self,
        session_id: str,
        camera_id: str,
        pipeline,
        backend,
        config: dict = None,
    ):
        self.session_id = session_id
        self.camera_id = camera_id
        self._pipeline = pipeline
        self._backend = backend
        self._config = config or {}

        # 事件总线
        self.event_bus = EventBus()

        # 生命周期
        self.lifecycle = RuntimeLifecycleManager(event_bus=self.event_bus)

        # 调度器
        self.scheduler = RuntimeScheduler()

        # 指标
        self.metrics = RuntimeMetrics()

        # 健康
        self.health = RuntimeHealthMonitor(event_bus=self.event_bus)

        # 运行时状态（唯一持有者）
        self._frame_id = 0
        self._fps = 0.0
        self._fps_t0 = time.time()
        self._inference_interval = self._config.get("inference_interval", 1)
        self._last_result: Optional[dict] = None

        # 注册 fall detection task
        self.scheduler.register(
            name="fall_detection",
            callback=lambda: self._run_pipeline_step(),
            interval_frames=1,
            priority=10,
        )

        # 注册 health check task
        self.scheduler.register(
            name="health_check",
            callback=lambda: self._run_health_check(),
            interval_frames=30,
            priority=1,
        )

    def start(self):
        self.lifecycle.start()
        self.event_bus.publish(LifecycleSignal(
            signal_type="lifecycle",
            lifecycle_event="session_started",
            session_id=self.session_id,
        ))

    def stop(self):
        self.lifecycle.stop()
        self.scheduler.clear()
        self.event_bus.publish(LifecycleSignal(
            signal_type="lifecycle",
            lifecycle_event="session_stopped",
            session_id=self.session_id,
        ))

    def pause(self):
        self.lifecycle.pause()

    def resume(self):
        self.lifecycle.resume()

    def reset(self):
        self._frame_id = 0
        self._fps_t0 = time.time()
        self.scheduler.reset()
        self.metrics = RuntimeMetrics()
        self._last_result = None
        if hasattr(self._pipeline, 'reset'):
            self._pipeline.reset()

    def restart(self):
        self.stop()
        self.reset()
        self.start()

    def process_frame(self, frame, camera_id: str = None, timestamp: float = None) -> dict:
        """处理单帧 -- 兼容原有 process_frame API"""
        if not self.lifecycle.is_running:
            return self._empty_result(camera_id or self.camera_id, timestamp or time.time())

        self._frame_id += 1
        now = timestamp or time.time()

        # FPS
        elapsed = time.time() - self._fps_t0
        self._fps = self._frame_id / (elapsed + 1e-8)

        # Scheduler tick
        t_start = time.time()
        task_results = self.scheduler.tick()
        t_ms = (time.time() - t_start) * 1000

        # Metrics
        self.metrics.record_frame(t_ms)

        # Health
        self.health.record_fps(self._fps)
        self.health.record_latency(t_ms)

        result = self._last_result
        if result is None:
            result = self._empty_result(camera_id or self.camera_id, now)
        return result

    def _run_pipeline_step(self):
        """Pipeline 执行（由 scheduler 调用）"""
        frame, ctx = self._get_current_frame()
        if frame is None or ctx is None:
            return

        t0 = time.time()
        result = self._pipeline.run(frame, ctx)
        t_ms = (time.time() - t0) * 1000

        # Record
        self.metrics.record_inference(t_ms)
        for evt in result.get("events", []):
            self.metrics.record_event(evt.get("event_type", "unknown"))
            self.event_bus.publish(EventSignal(
                signal_type=evt.get("event_type", "event"),
                track_id=evt.get("track_id"),
                camera_id=evt.get("camera_id", self.camera_id),
                confidence=evt.get("confidence", 0),
                bbox=evt.get("bbox", []),
                session_id=self.session_id,
            ))

        n_persons = len(result.get("persons", []))
        self.metrics.record_tracks(n_persons)
        self.health.check(dropped_frames=0, inference_latency_ms=t_ms,
                          queue_backlog=0)

        self._last_result = result

    def _run_health_check(self):
        self.health.check()

    def _get_current_frame(self):
        # 子类可覆盖以接入 CameraSession 的 frame_buffer
        return None, None

    def _empty_result(self, camera_id: str, timestamp: float) -> dict:
        return serialize_result(
            tracked=[], events=[], camera_id=camera_id,
            timestamp=timestamp, frame_id=self._frame_id,
            diagnostics={"fps": self._fps, "backend": "runtime", "device": "cpu", "inference_ran": False},
        )
