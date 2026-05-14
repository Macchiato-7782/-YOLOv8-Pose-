"""
校验器
确保所有 dataclass 对象结构合法
"""

from typing import Optional


def validate_detection(det) -> tuple:
    """验证 Detection 结构，返回 (valid, error_msg)"""
    if not hasattr(det, 'bbox') or det.bbox is None:
        return False, "missing bbox"
    if len(det.bbox) != 4:
        return False, f"bbox length must be 4, got {len(det.bbox)}"
    if not hasattr(det, 'score'):
        return False, "missing score"
    if det.score < 0 or det.score > 1:
        return False, f"score out of range: {det.score}"
    return True, ""


def validate_track(track) -> tuple:
    """验证 TrackState 结构"""
    if not hasattr(track, 'track_id'):
        return False, "missing track_id"
    if not hasattr(track, 'bbox'):
        return False, "missing bbox"
    if len(track.bbox) != 4:
        return False, f"bbox length must be 4, got {len(track.bbox)}"
    if not hasattr(track, 'state'):
        return False, "missing state"
    valid_states = {"normal", "potential_fall", "fall"}
    if track.state not in valid_states:
        return False, f"invalid state '{track.state}', expected one of {valid_states}"
    return True, ""


def validate_event(event) -> tuple:
    """验证 Event 结构"""
    if not hasattr(event, 'event_type'):
        return False, "missing event_type"
    valid_types = {"fall_confirmed", "fall_warning", "person_detected", "track_lost"}
    if event.event_type not in valid_types:
        return False, f"invalid event_type '{event.event_type}'"
    if not hasattr(event, 'timestamp'):
        return False, "missing timestamp"
    if not hasattr(event, 'bbox'):
        return False, "missing bbox"
    if len(event.bbox) != 4:
        return False, f"bbox length must be 4"
    return True, ""


def validate_state(state: str) -> bool:
    """验证状态字符串"""
    return state in ("normal", "potential_fall", "fall")
