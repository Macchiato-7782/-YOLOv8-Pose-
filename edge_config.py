"""
边缘设备默认配置
提供低功耗边缘设备的推荐参数，后续可扩展 ONNX / OpenVINO / NCNN 后端
"""

EDGE_DEFAULTS = {
    "device": "cpu",
    "backend": "ultralytics",
    "input_size": 320,
    "inference_interval": 2,
    "enable_roi": False,
    "enable_visualization": False,
    "max_persons": 3,
    "conf": 0.35,
    "tracker": "bytetrack.yaml",
}
