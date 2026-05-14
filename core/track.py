"""
跟踪状态数据结构
Backend 无关，只接收 Detection，输出 TrackState
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    """统一的跟踪目标状态"""
    track_id: int
    bbox: list          # [x1, y1, x2, y2]
    center: list        # [cx, cy]
    confidence: float
    state: str = "normal"          # normal | potential_fall | fall
    keypoints: list = field(default_factory=list)  # list[Keypoint] or raw list
    is_ghost: bool = False
    age: int = 0
    lost_frames: int = 0
    aspect_ratio: float = 0.0
    fall_detected: bool = False
    history: list = field(default_factory=list)
    fall_state: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bbox = [float(v) for v in self.bbox]
        self.center = [float(v) for v in self.center]
        self.confidence = float(self.confidence)
        self.track_id = int(self.track_id)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center": self.center,
            "state": self.state,
            "fall_detected": self.fall_detected,
            "confidence": self.confidence,
            "is_ghost": self.is_ghost,
            "age": self.age,
            "lost_frames": self.lost_frames,
            "keypoints": [
                kp.to_list() if hasattr(kp, 'to_list') else kp
                for kp in self.keypoints
            ],
        }

    @classmethod
    def from_detection(cls, detection, track_id: int) -> "TrackState":
        from fall_detection.core.detection import Detection
        bbox = detection.bbox
        center = [
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        ]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ar = h / w if w > 0 else 0.0
        return cls(
            track_id=track_id,
            bbox=bbox,
            center=center,
            confidence=detection.score,
            keypoints=detection.keypoints,
            aspect_ratio=ar,
        )
