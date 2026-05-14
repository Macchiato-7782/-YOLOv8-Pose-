"""
Runtime Policies
统一 Runtime 行为策略：overload / reconnect / frame drop / cooldown / retry / restart / degradation
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RuntimePolicy:
    """统一 Runtime 行为策略"""

    def __init__(self, config: dict = None):
        c = config or {}
        self.overload_cpu_threshold: float = c.get("overload_cpu_threshold", 0.90)
        self.overload_queue_threshold: int = c.get("overload_queue_threshold", 200)
        self.overload_latency_threshold_ms: float = c.get("overload_latency_threshold_ms", 500)

        self.degradation_cpu: float = c.get("degradation_cpu", 0.80)
        self.degradation_ram_mb: float = c.get("degradation_ram_mb", 3500)

        self.backpressure_queue: int = c.get("backpressure_queue", 100)
        self.backpressure_latency_ms: float = c.get("backpressure_latency_ms", 300)

        self.reconnect_max_retries: int = c.get("reconnect_max_retries", 5)
        self.reconnect_interval_seconds: float = c.get("reconnect_interval_seconds", 2.0)

        self.frame_drop_mode: str = c.get("frame_drop_mode", "oldest")
        self.max_frame_age_seconds: float = c.get("max_frame_age_seconds", 3.0)

        self.worker_restart_max: int = c.get("worker_restart_max", 3)
        self.worker_heartbeat_seconds: float = c.get("worker_heartbeat_seconds", 5.0)

        self.cooldown_seconds: float = c.get("cooldown_seconds", 5.0)

        self.degradation_reduce_fps_to: float = c.get("degradation_reduce_fps_to", 0.5)
        self.degradation_reduce_input_size_to: int = c.get("degradation_reduce_input_size_to", 320)

    def should_overload(self, cpu: float, queue_backlog: int, latency_ms: float) -> bool:
        return (cpu > self.overload_cpu_threshold or
                queue_backlog > self.overload_queue_threshold or
                latency_ms > self.overload_latency_threshold_ms)

    def should_degrade(self, cpu: float, ram_mb: float) -> bool:
        return cpu > self.degradation_cpu or ram_mb > self.degradation_ram_mb

    def should_backpressure(self, queue_backlog: int, latency_ms: float) -> bool:
        return queue_backlog > self.backpressure_queue or latency_ms > self.backpressure_latency_ms

    def to_dict(self) -> dict:
        return {
            "overload_cpu_threshold": self.overload_cpu_threshold,
            "overload_queue_threshold": self.overload_queue_threshold,
            "reconnect_max_retries": self.reconnect_max_retries,
            "worker_restart_max": self.worker_restart_max,
            "degradation_reduce_fps_to": self.degradation_reduce_fps_to,
        }
