"""
Resource Manager
实时监控 CPU / RAM / queue backlog / thermal / worker load / inference latency
"""

import os
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResourceManager:
    """实时资源监控，scheduler 的决策输入"""

    def __init__(self, interval_seconds: float = 2.0):
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._cpu_usage: float = 0.0
        self._ram_usage_mb: float = 0.0
        self._queue_backlog: int = 0
        self._inference_latency_ms: float = 0.0
        self._worker_loads: dict = {}
        self._frame_delay_ms: float = 0.0
        self._last_check: float = 0
        self._sample_count: int = 0

    def update(self, cpu: float = None, ram_mb: float = None,
               queue_backlog: int = None, inference_latency_ms: float = None,
               frame_delay_ms: float = None):
        with self._lock:
            if cpu is not None:
                self._cpu_usage = cpu
            if ram_mb is not None:
                self._ram_usage_mb = ram_mb
            if queue_backlog is not None:
                self._queue_backlog = queue_backlog
            if inference_latency_ms is not None:
                self._inference_latency_ms = inference_latency_ms
            if frame_delay_ms is not None:
                self._frame_delay_ms = frame_delay_ms
            self._last_check = time.time()
            self._sample_count += 1

    def update_cpu(self):
        """自动检测 CPU 使用率"""
        try:
            import psutil
            self.update(cpu=psutil.cpu_percent(interval=0.1) / 100.0)
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info().rss / (1024 * 1024)
            self.update(ram_mb=mem)
        except ImportError:
            pass

    def set_worker_load(self, worker_name: str, load: float):
        with self._lock:
            self._worker_loads[worker_name] = load

    @property
    def cpu_usage(self) -> float:
        return self._cpu_usage

    @property
    def ram_usage_mb(self) -> float:
        return self._ram_usage_mb

    @property
    def queue_backlog(self) -> int:
        return self._queue_backlog

    @property
    def inference_latency_ms(self) -> float:
        return self._inference_latency_ms

    @property
    def is_overloaded(self) -> bool:
        return self._cpu_usage > 0.90 or self._queue_backlog > 200

    def snapshot(self) -> dict:
        return {
            "cpu_usage": self._cpu_usage,
            "ram_usage_mb": self._ram_usage_mb,
            "queue_backlog": self._queue_backlog,
            "inference_latency_ms": self._inference_latency_ms,
            "frame_delay_ms": self._frame_delay_ms,
            "worker_loads": dict(self._worker_loads),
            "sample_count": self._sample_count,
        }
