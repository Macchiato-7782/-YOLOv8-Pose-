"""
FallDetector 标准接口
AI Monitoring Runtime Core -- pipeline orchestration only
"""

import os
import time
import logging

from tracking import SingleCameraTracker
from config import load_config

from fall_detection.core.pipeline import DetectionPipeline
from fall_detection.core.frame import FrameContext
from fall_detection.edge_config import EDGE_DEFAULTS
from fall_detection.backends.factory import create_backend

logger = logging.getLogger(__name__)


class FallDetector:
    """
    摔倒检测器标准接口 -- Runtime Orchestration Only

    内部使用 DetectionPipeline 编排:
    infer → Detection → tracking → TrackState → fall logic → Event → serialize
    """

    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        device: str = "cpu",
        backend: str = "ultralytics",
        enable_tracking: bool = True,
        enable_visualization: bool = False,
        enable_roi: bool = False,
        inference_interval: int = 1,
        input_size: int = 640,
        max_persons: int = None,
        conf: float = None,
        tracker: str = None,
    ):
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._root_dir = os.path.dirname(self._script_dir)

        if model_path is None:
            ext = ".onnx" if backend == "onnx" else ".pt"
            model_path = os.path.join(self._root_dir, f"yolov8n-pose{ext}")
        self.model_path = model_path

        if config_path is None:
            config_path = os.path.join(self._root_dir, "config.yaml")
        self._cfg = load_config(config_path) if os.path.exists(config_path) else {}

        self.device = device
        self.backend = backend
        self.enable_tracking = enable_tracking
        self.enable_visualization = enable_visualization
        self.enable_roi = enable_roi
        self.inference_interval = max(1, int(inference_interval))
        self.input_size = int(input_size)
        self.max_persons = int(max_persons) if max_persons is not None else None

        if conf is None:
            conf = self._cfg.get("camera_process", {}).get("roi_conf", 0.35)
        self.conf = float(conf)
        self._tracker_config = tracker or "bytetrack.yaml"

        # 创建推理后端
        self._backend = create_backend(
            backend=self.backend,
            model_path=self.model_path,
            device=self.device,
            conf=self.conf,
            tracker=self._tracker_config,
            input_size=self.input_size,
            enable_tracking=enable_tracking,
        )

        # 创建跟踪器
        self._tracker = SingleCameraTracker() if enable_tracking else None

        # 创建 Pipeline
        self._pipeline = DetectionPipeline(
            backend=self._backend,
            tracker=self._tracker,
            enable_roi=enable_roi,
            max_persons=self.max_persons,
            enable_visualization=enable_visualization,
        )

        self._frame_id = 0
        self._fps = 0.0
        self._fps_t0 = time.time()

    def process_frame(
        self,
        frame,
        camera_id: str = "cam_0",
        timestamp: float = None,
        external_tracks: list = None,
    ) -> dict:
        """处理单帧图像，返回标准化检测结果"""
        if timestamp is None:
            timestamp = time.time()

        self._frame_id += 1

        # 推理间隔
        do_inference = (self._frame_id - 1) % self.inference_interval == 0
        if external_tracks is not None:
            do_inference = False

        if not do_inference:
            if self._tracker is not None:
                self._tracker.cleanup(timestamp)
                for data in self._tracker.person_history.values():
                    data["last_seen"] = timestamp
            cached = self._cached_result
            if cached is not None:
                # 深拷贝避免修改原缓存
                result = dict(cached)
                result["diagnostics"] = dict(cached["diagnostics"])
                result["diagnostics"]["inference_ran"] = False
                result["diagnostics"]["fps"] = round(self._fps, 1)
                return result
            return {
                "module": "fall_detection",
                "camera_id": camera_id,
                "timestamp": timestamp,
                "frame_id": self._frame_id,
                "persons": [],
                "events": [],
                "diagnostics": {"fps": 0, "backend": self.backend, "device": self.device, "inference_ran": False},
            }

        # 更新 FPS
        elapsed = time.time() - self._fps_t0
        self._fps = self._frame_id / (elapsed + 1e-8)

        # 帧上下文
        ctx = FrameContext.from_frame(
            frame, self._frame_id, camera_id, timestamp, self._fps,
        )

        # Pipeline 运行
        result = self._pipeline.run(frame, ctx)
        self._cached_result = result
        return result

    def reset(self):
        self._tracker = SingleCameraTracker() if self.enable_tracking else None
        self._pipeline = DetectionPipeline(
            backend=self._backend,
            tracker=self._tracker,
            enable_roi=self.enable_roi,
            max_persons=self.max_persons,
            enable_visualization=self.enable_visualization,
        )
        self._frame_id = 0
        self._fps_t0 = time.time()
        self._cached_result = None

    def close(self):
        self._backend.close()
        self._tracker = None
