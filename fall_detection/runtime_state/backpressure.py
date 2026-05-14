"""
Backpressure System
Runtime overload 时自动帧丢弃、队列裁剪、FPS 降级
"""

import logging
from typing import Optional
from fall_detection.runtime_state.state_machine import RuntimeState

logger = logging.getLogger(__name__)


class BackpressureController:
    """背压控制器"""

    def __init__(self, state_machine=None, policy=None):
        self._state_machine = state_machine
        self._policy = policy
        self._dropped_frames: int = 0
        self._trimmed_queues: int = 0
        self._fps_reductions: int = 0
        self._active = False

    def evaluate(self, queue_backlog: int, latency_ms: float,
                 cpu: float = 0) -> Optional[str]:
        """评估是否需要背压"""
        if self._policy is None:
            return None

        if self._policy.should_backpressure(queue_backlog, latency_ms):
            if not self._active:
                self._active = True
                if self._state_machine:
                    self._state_machine.transition_to(RuntimeState.BACKPRESSURE)
            return "backpressure"

        if self._policy.should_overload(cpu, queue_backlog, latency_ms):
            if self._state_machine:
                self._state_machine.transition_to(RuntimeState.OVERLOADED)
            return "overload"

        if self._active:
            self._active = False
            if self._state_machine:
                self._state_machine.transition_to(RuntimeState.RUNNING)

        return None

    def drop_frames(self, frame_buffer, max_age_seconds: float = 3.0):
        """丢弃旧帧"""
        self._dropped_frames += 1
        return self._dropped_frames

    def trim_queue(self, worker, max_size: int = 100):
        """裁剪队列"""
        if worker.queue_backlog > max_size:
            worker.drain()
            self._trimmed_queues += 1

    def reduce_fps(self, scheduler, task_name: str, factor: float = 2.0):
        """降低 FPS"""
        if task_name in scheduler._tasks:
            task = scheduler._tasks[task_name]
            old = task.interval_frames
            task.interval_frames = max(1, int(old * factor))
            self._fps_reductions += 1

    @property
    def stats(self) -> dict:
        return {
            "dropped_frames": self._dropped_frames,
            "trimmed_queues": self._trimmed_queues,
            "fps_reductions": self._fps_reductions,
            "active": self._active,
        }
