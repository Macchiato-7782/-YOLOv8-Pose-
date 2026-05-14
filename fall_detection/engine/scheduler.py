"""
Runtime Scheduler -- Task scheduling with frequency control
支持不同 pipeline 不同频率，future multimodal runtime
"""

import time
import logging
from typing import Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    name: str
    callback: Callable
    interval_frames: int = 1
    priority: int = 0
    frame_count: int = 0
    last_run: float = field(default_factory=time.time)
    enabled: bool = True

    def should_run(self) -> bool:
        if not self.enabled:
            return False
        self.frame_count += 1
        return self.frame_count % self.interval_frames == 0

    def mark_run(self):
        self.last_run = time.time()


class RuntimeScheduler:
    """任务调度器，管理不同 pipeline 频率"""

    def __init__(self):
        self._tasks: dict = {}
        self._frame_counter: int = 0
        self._running = False

    def register(self, name: str, callback: Callable, interval_frames: int = 1,
                 priority: int = 0):
        """注册调度任务"""
        self._tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            interval_frames=interval_frames,
            priority=priority,
        )
        logger.debug(f"Scheduler: registered task '{name}' interval={interval_frames}")

    def unregister(self, name: str):
        self._tasks.pop(name, None)

    def enable(self, name: str):
        if name in self._tasks:
            self._tasks[name].enabled = True

    def disable(self, name: str):
        if name in self._tasks:
            self._tasks[name].enabled = False

    def tick(self) -> list:
        """每帧调用，返回应该执行的任务列表"""
        self._frame_counter += 1
        results = []

        # 按优先级排序
        sorted_tasks = sorted(self._tasks.values(), key=lambda t: -t.priority)
        for task in sorted_tasks:
            if task.should_run():
                try:
                    result = task.callback()
                    results.append((task.name, result))
                    task.mark_run()
                except Exception as e:
                    logger.error(f"Scheduler task '{task.name}' error: {e}")

        return results

    def reset(self):
        self._frame_counter = 0
        for task in self._tasks.values():
            task.frame_count = 0

    def clear(self):
        self._tasks.clear()
        self._frame_counter = 0

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def task_count(self) -> int:
        return len(self._tasks)
