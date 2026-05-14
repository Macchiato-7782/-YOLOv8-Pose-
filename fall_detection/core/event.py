"""
事件数据结构
所有运行时事件统一使用此格式
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Event:
    """统一运行时事件"""
    event_type: str           # fall_confirmed | fall_warning | person_detected | track_lost
    timestamp: float
    track_id: Optional[int]
    camera_id: str
    confidence: float
    bbox: list                # [x1, y1, x2, y2]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = float(self.confidence)
        self.timestamp = float(self.timestamp)
        self.bbox = [float(v) for v in self.bbox]

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "state": "fall" if "fall" in self.event_type else "normal",
            "metadata": self.metadata,
        }

    @classmethod
    def from_track(cls, event_type: str, track, camera_id: str, timestamp: float) -> "Event":
        from fall_detection.core.track import TrackState
        return cls(
            event_type=event_type,
            track_id=track.track_id,
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=track.confidence,
            bbox=track.bbox,
        )
