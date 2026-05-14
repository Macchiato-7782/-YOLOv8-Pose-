"""
帧上下文数据结构
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameContext:
    """统一的帧上下文"""
    frame_id: int
    timestamp: float
    camera_id: str
    width: int
    height: int
    fps: Optional[float] = None

    def __post_init__(self):
        self.frame_id = int(self.frame_id)
        self.timestamp = float(self.timestamp)
        self.width = int(self.width)
        self.height = int(self.height)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    @classmethod
    def from_frame(cls, frame, frame_id: int, camera_id: str,
                   timestamp: float = None, fps: float = None) -> "FrameContext":
        import time
        h, w = frame.shape[:2]
        return cls(
            frame_id=frame_id,
            timestamp=timestamp or time.time(),
            camera_id=camera_id,
            width=w,
            height=h,
            fps=fps,
        )
