"""
fall_detection 包
提供标准化的摔倒检测模块，可被外部项目直接 import 调用
"""

from fall_detection.detector import FallDetector
from fall_detection.schemas import format_result, format_person, normalize_state, to_json_friendly
from fall_detection.edge_config import EDGE_DEFAULTS

__all__ = [
    "FallDetector",
    "format_result",
    "format_person",
    "normalize_state",
    "to_json_friendly",
    "EDGE_DEFAULTS",
]
