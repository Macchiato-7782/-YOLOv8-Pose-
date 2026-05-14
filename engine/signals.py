"""
Runtime Signals
统一运行时信号系统，所有 runtime 内部消息通过此系统传递
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class RuntimeSignal:
    """运行时信号基类"""
    signal_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class EventSignal(RuntimeSignal):
    """事件信号（fall_confirmed, fall_warning, person_detected, track_lost）"""
    track_id: Optional[int] = None
    camera_id: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
        })
        return d


@dataclass
class HealthSignal(RuntimeSignal):
    """健康监控信号"""
    fps: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0
    memory_mb: float = 0.0
    queue_backlog: int = 0
    inference_latency_ms: float = 0.0
    backend_errors: int = 0


@dataclass
class ErrorSignal(RuntimeSignal):
    """错误信号"""
    error_type: str = ""
    error_message: str = ""
    component: str = ""
    recoverable: bool = True


@dataclass
class LifecycleSignal(RuntimeSignal):
    """生命周期信号"""
    lifecycle_event: str = ""  # started, stopped, paused, resumed, restarted
    reason: str = ""


@dataclass
class CameraSignal(RuntimeSignal):
    """摄像头信号"""
    camera_id: str = ""
    camera_event: str = ""  # connected, disconnected, reconnected, error
    source: str = ""
