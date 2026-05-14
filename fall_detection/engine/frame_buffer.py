"""
SharedFrameBuffer -- 统一帧缓冲区
支持 latest frame / ring buffer / snapshot / reference counting
为 future 多模型融合做 frame ownership 统一
"""

import threading
from typing import Optional
import numpy as np

from fall_detection.core.frame import FrameContext


class SharedFrameBuffer:
    """统一帧缓冲区，保证 future 多 pipeline 共享同一帧"""

    def __init__(self, ring_size: int = 8):
        self._ring_size = ring_size
        self._ring: list = []
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_context: Optional[FrameContext] = None
        self._lock = threading.Lock()
        self._ref_count: dict = {}  # id(frame) → count
        self._dropped_count: int = 0

    def push(self, frame, context: FrameContext):
        """推入一帧"""
        with self._lock:
            # Ring buffer
            self._ring.append((frame, context))
            if len(self._ring) > self._ring_size:
                old_frame, _ = self._ring.pop(0)
                self._release(old_frame)

            # Latest
            old = self._latest_frame
            self._latest_frame = frame
            self._latest_context = context
            self._add_ref(frame)
            if old is not None:
                self._release(old)

    def latest(self) -> Optional[tuple]:
        """获取最新帧"""
        with self._lock:
            if self._latest_frame is not None:
                self._add_ref(self._latest_frame)
            return (self._latest_frame, self._latest_context) if self._latest_frame is not None else None

    def snapshot(self, offset: int = 0) -> Optional[tuple]:
        """从 ring buffer 获取历史帧"""
        with self._lock:
            idx = len(self._ring) - 1 - offset
            if 0 <= idx < len(self._ring):
                frame, ctx = self._ring[idx]
                self._add_ref(frame)
                return (frame, ctx)
            return None

    def release(self, frame):
        """释放帧引用"""
        self._release(frame)

    def _add_ref(self, frame):
        fid = id(frame)
        self._ref_count[fid] = self._ref_count.get(fid, 0) + 1

    def _release(self, frame):
        fid = id(frame)
        if fid in self._ref_count:
            self._ref_count[fid] -= 1
            if self._ref_count[fid] <= 0:
                del self._ref_count[fid]

    @property
    def dropped_frames(self) -> int:
        return self._dropped_count

    @property
    def ref_count(self) -> int:
        return sum(1 for v in self._ref_count.values() if v > 0)

    def clear(self):
        with self._lock:
            self._ring.clear()
            self._latest_frame = None
            self._latest_context = None
            self._ref_count.clear()
