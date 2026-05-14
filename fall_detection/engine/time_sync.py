"""
Runtime Clock -- 统一时间域
frame / inference / event / multi-model / multi-camera timing
"""

import time


class RuntimeClock:
    """统一 Runtime 时钟"""

    def __init__(self):
        self._start_time: float = time.time()
        self._frame_times: dict = {}  # camera_id → timestamp
        self._inference_times: dict = {}
        self._last_frame_time: dict = {}

    def tick(self, camera_id: str) -> float:
        """记录帧时间"""
        now = time.time()
        self._frame_times[camera_id] = now
        self._last_frame_time[camera_id] = now
        return now

    def mark_inference(self, name: str) -> float:
        now = time.time()
        self._inference_times[name] = now
        return now

    def frame_age(self, camera_id: str) -> float:
        """帧年龄（秒）"""
        return time.time() - self._last_frame_time.get(camera_id, time.time())

    def elapsed(self) -> float:
        return time.time() - self._start_time

    def uptime_seconds(self) -> float:
        return self.elapsed()

    def reset(self):
        self._start_time = time.time()
        self._frame_times.clear()
        self._inference_times.clear()
        self._last_frame_time.clear()
