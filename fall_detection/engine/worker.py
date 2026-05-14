"""
Runtime Worker System
Inference / Tracking / Event workers with lifecycle management
"""

import logging
from typing import Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class RuntimeWorker:
    """统一 Worker，支持 graceful shutdown / exception isolation"""

    def __init__(self, name: str, handler: Callable, queue_size: int = 10):
        self.name = name
        self._handler = handler
        self._status = WorkerStatus.IDLE
        self._queue: list = []
        self._queue_size = queue_size
        self._error_count = 0
        self._processed_count = 0

    def start(self):
        self._status = WorkerStatus.RUNNING

    def stop(self):
        self._status = WorkerStatus.STOPPED
        self._queue.clear()

    def pause(self):
        self._status = WorkerStatus.PAUSED

    def resume(self):
        if self._status == WorkerStatus.PAUSED:
            self._status = WorkerStatus.RUNNING

    def submit(self, item) -> bool:
        """提交任务到 worker"""
        if self._status != WorkerStatus.RUNNING:
            return False
        if len(self._queue) >= self._queue_size:
            self._queue.pop(0)
        self._queue.append(item)
        return True

    def process(self) -> Optional[list]:
        """处理队列中的任务"""
        if self._status != WorkerStatus.RUNNING or not self._queue:
            return None

        results = []
        while self._queue:
            item = self._queue.pop(0)
            try:
                result = self._handler(item)
                results.append(result)
                self._processed_count += 1
            except Exception as e:
                self._error_count += 1
                self._status = WorkerStatus.ERROR
                logger.error(f"Worker '{self.name}' error: {e}")
        return results if results else None

    def drain(self):
        """清空队列"""
        self._queue.clear()

    def reset(self):
        self._queue.clear()
        self._error_count = 0
        self._processed_count = 0
        self._status = WorkerStatus.IDLE

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def queue_backlog(self) -> int:
        return len(self._queue)

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def processed_count(self) -> int:
        return self._processed_count
