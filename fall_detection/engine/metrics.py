"""
Runtime Metrics -- Prometheus-ready 指标收集
记录 avg FPS / latency / frame time / event count / inference count / track count
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RuntimeMetricsSnapshot:
    avg_fps: float = 0.0
    avg_latency_ms: float = 0.0
    total_frames: int = 0
    total_inferences: int = 0
    total_events: int = 0
    total_tracks: int = 0
    event_counts: dict = field(default_factory=dict)
    tracked_persons_avg: float = 0.0
    uptime_seconds: float = 0.0


class RuntimeMetrics:
    """Runtime 指标收集"""

    def __init__(self):
        self._start_time = time.time()
        self._frame_times: list = []
        self._inference_times: list = []
        self._total_frames = 0
        self._total_inferences = 0
        self._total_events = 0
        self._total_tracks = 0
        self._event_counts: dict = defaultdict(int)
        self._track_counts: list = []

    def record_frame(self, processing_time_ms: float):
        self._total_frames += 1
        self._frame_times.append(processing_time_ms)
        if len(self._frame_times) > 600:
            self._frame_times = self._frame_times[-600:]

    def record_inference(self, latency_ms: float):
        self._total_inferences += 1
        self._inference_times.append(latency_ms)
        if len(self._inference_times) > 600:
            self._inference_times = self._inference_times[-600:]

    def record_event(self, event_type: str):
        self._total_events += 1
        self._event_counts[event_type] += 1

    def record_tracks(self, count: int):
        self._total_tracks += count
        self._track_counts.append(count)
        if len(self._track_counts) > 600:
            self._track_counts = self._track_counts[-600:]

    def snapshot(self) -> RuntimeMetricsSnapshot:
        uptime = time.time() - self._start_time
        avg_ft = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0
        avg_it = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0
        avg_tracks = sum(self._track_counts) / len(self._track_counts) if self._track_counts else 0

        return RuntimeMetricsSnapshot(
            avg_fps=1000.0 / avg_ft if avg_ft > 0 else 0,
            avg_latency_ms=avg_it,
            total_frames=self._total_frames,
            total_inferences=self._total_inferences,
            total_events=self._total_events,
            total_tracks=self._total_tracks,
            event_counts=dict(self._event_counts),
            tracked_persons_avg=avg_tracks,
            uptime_seconds=uptime,
        )

    def to_dict(self) -> dict:
        s = self.snapshot()
        return {
            "avg_fps": s.avg_fps,
            "avg_latency_ms": s.avg_latency_ms,
            "total_frames": s.total_frames,
            "total_inferences": s.total_inferences,
            "total_events": s.total_events,
            "total_tracks": s.total_tracks,
            "event_counts": s.event_counts,
            "tracked_persons_avg": s.tracked_persons_avg,
            "uptime_seconds": s.uptime_seconds,
        }
