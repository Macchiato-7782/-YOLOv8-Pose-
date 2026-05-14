"""
AI Monitoring Runtime Core
统一 AI 检测流水线的核心模块
"""

from fall_detection.core.detection import Detection, Keypoint
from fall_detection.core.track import TrackState
from fall_detection.core.event import Event
from fall_detection.core.frame import FrameContext
from fall_detection.core.pipeline import DetectionPipeline
from fall_detection.core.runtime import EventRuntime
from fall_detection.core.serializers import (
    serialize_detection, serialize_track, serialize_event,
    serialize_frame_context, serialize_result,
)
from fall_detection.core.validators import (
    validate_detection, validate_track, validate_event, validate_state,
)

__all__ = [
    "Detection", "Keypoint",
    "TrackState",
    "Event",
    "FrameContext",
    "DetectionPipeline",
    "EventRuntime",
    "serialize_detection", "serialize_track", "serialize_event",
    "serialize_frame_context", "serialize_result",
    "validate_detection", "validate_track", "validate_event", "validate_state",
]
