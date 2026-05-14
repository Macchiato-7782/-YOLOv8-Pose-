"""
Runtime Health Monitor
监控 FPS / latency / dropped frames / queue backlog / memory / errors
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

from fall_detection.engine.signals import HealthSignal

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    fps: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0
    memory_mb: float = 0.0
    queue_backlog: int = 0
    inference_latency_ms: float = 0.0
    backend_errors: int = 0
    timestamp: float = field(default_factory=time.time)


class RuntimeHealthMonitor:
    """运行时健康监控"""

    def __init__(self, event_bus=None, interval_seconds: float = 5.0):
        self._event_bus = event_bus
        self._interval = interval_seconds
        self._last_check = time.time()
        self._fps_samples: list = []
        self._latency_samples: list = []
        self._error_count: int = 0
        self._snapshot = HealthSnapshot()

    def record_fps(self, fps: float):
        self._fps_samples.append(fps)
        if len(self._fps_samples) > 100:
            self._fps_samples = self._fps_samples[-100:]

    def record_latency(self, ms: float):
        self._latency_samples.append(ms)
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]

    def record_error(self):
        self._error_count += 1

    def check(self, dropped_frames: int = 0, inference_latency_ms: float = 0.0,
              queue_backlog: int = 0, memory_mb: float = 0.0):
        """执行健康检查"""
        now = time.time()
        if now - self._last_check < self._interval:
            return
        self._last_check = now

        avg_fps = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0
        avg_latency = sum(self._latency_samples) / len(self._latency_samples) if self._latency_samples else 0

        self._snapshot = HealthSnapshot(
            fps=avg_fps,
            latency_ms=avg_latency,
            dropped_frames=dropped_frames,
            memory_mb=memory_mb,
            queue_backlog=queue_backlog,
            inference_latency_ms=inference_latency_ms,
            backend_errors=self._error_count,
        )

        if self._event_bus:
            self._event_bus.publish(HealthSignal(
                signal_type="health_check",
                fps=avg_fps,
                latency_ms=avg_latency,
                dropped_frames=dropped_frames,
                memory_mb=memory_mb,
                queue_backlog=queue_backlog,
                inference_latency_ms=inference_latency_ms,
                backend_errors=self._error_count,
            ))

    @property
    def snapshot(self) -> HealthSnapshot:
        return self._snapshot

    @property
    def avg_fps(self) -> float:
        return sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0
