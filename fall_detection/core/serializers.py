"""
序列化器
将所有 dataclass 对象转换为 JSON-friendly dict
不允许 numpy / tensor / dataclass 泄露到 API
"""

from typing import Optional
import numpy as np


def serialize_keypoint(kp) -> dict:
    return {"x": float(kp.x), "y": float(kp.y), "confidence": float(kp.confidence)}


def serialize_detection(det) -> dict:
    return {
        "bbox": [float(v) for v in det.bbox],
        "score": float(det.score),
        "class_id": int(det.class_id),
        "keypoints": [serialize_keypoint(k) for k in det.keypoints],
        "track_id": det.track_id,
        "label": det.label,
    }


def serialize_track(track) -> dict:
    kps = []
    for kp in track.keypoints:
        if hasattr(kp, 'to_list'):
            kps.append(kp.to_list())
        elif hasattr(kp, 'x'):
            kps.append([kp.x, kp.y, kp.confidence])
        else:
            kps.append(list(kp) if isinstance(kp, (list, tuple)) else kp)
    return {
        "track_id": int(track.track_id),
        "bbox": [float(v) for v in track.bbox],
        "center": [float(v) for v in track.center],
        "state": str(track.state),
        "fall_detected": bool(track.fall_detected),
        "confidence": float(track.confidence),
        "is_ghost": bool(track.is_ghost),
        "keypoints": kps,
    }


def serialize_event(event) -> dict:
    return {
        "event_type": str(event.event_type),
        "track_id": event.track_id,
        "camera_id": str(event.camera_id),
        "timestamp": float(event.timestamp),
        "confidence": float(event.confidence),
        "bbox": [float(v) for v in event.bbox],
        "state": "fall" if "fall" in event.event_type else "normal",
    }


def serialize_frame_context(ctx) -> dict:
    return ctx.to_dict()


def serialize_result(
    tracked: list,
    events: list,
    camera_id: str,
    timestamp: float,
    frame_id: int,
    diagnostics: dict,
    module_name: str = "fall_detection",
) -> dict:
    """构建标准化 process_frame 返回值，兼容原有 JSON 格式"""
    from fall_detection.core.serializers import serialize_track, serialize_event

    persons = [serialize_track(t) for t in tracked]
    serialized_events = [serialize_event(e) for e in events]

    return {
        "module": module_name,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "frame_id": frame_id,
        "persons": persons,
        "events": serialized_events,
        "diagnostics": diagnostics,
    }
