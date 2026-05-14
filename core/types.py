"""
Runtime Type System
强类型别名和 Protocol 定义
"""

from typing import List, Protocol, TypeAlias
from dataclasses import dataclass

DetectionList: TypeAlias = List['Detection']
TrackList: TypeAlias = List['TrackState']
EventList: TypeAlias = List['Event']


class InferenceBackendProtocol(Protocol):
    """推理后端协议"""
    def infer(self, frame) -> List['Detection']:
        ...
    def warmup(self) -> None:
        ...
    def close(self) -> None:
        ...


class TrackerProtocol(Protocol):
    """跟踪器协议"""
    def update(self, detections: List['Detection'], current_time: float) -> List['TrackState']:
        ...
    def cleanup(self, current_time: float) -> None:
        ...
