"""
推理后端子包
提供多种推理后端实现和工厂函数
"""

from fall_detection.backends.base import BaseInferenceBackend
from fall_detection.backends.factory import create_backend
from fall_detection.backends.postprocess import (
    ultralytics_results_to_detections,
    onnx_outputs_to_detections,
)

__all__ = [
    "BaseInferenceBackend",
    "create_backend",
    "ultralytics_results_to_detections",
    "onnx_outputs_to_detections",
]
