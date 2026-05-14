"""
FallDetector 标准接口
可被外部项目直接 import 调用的唯一主入口
Backend 无关：通过工厂函数创建推理后端
"""

import os
import time
import logging
import traceback

import cv2
import numpy as np

from tracking import SingleCameraTracker
from fall_logic import evaluate_fall
from config import load_config

from fall_detection.schemas import format_result
from fall_detection.visualizer import draw_person_info, draw_fall_alert, draw_skeleton
from fall_detection.edge_config import EDGE_DEFAULTS
from fall_detection.backends.factory import create_backend

logger = logging.getLogger(__name__)


class FallDetector:
    """
    摔倒检测器标准接口

    示例用法:
        detector = FallDetector(model_path="yolov8n-pose.pt")
        result = detector.process_frame(frame, camera_id="cam_0")
        for event in result["events"]:
            print("Fall detected:", event)
        detector.close()

        # ONNX backend
        detector = FallDetector(backend="onnx", model_path="yolov8n-pose.onnx")
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
        if os.path.exists(config_path):
            self._cfg = load_config(config_path)
        else:
            self._cfg = {}

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

        # 通过工厂函数创建推理后端
        self._backend = create_backend(
            backend=self.backend,
            model_path=self.model_path,
            device=self.device,
            conf=self.conf,
            tracker=self._tracker_config,
            input_size=self.input_size,
            enable_tracking=enable_tracking,
        )

        # 跟踪器和内部状态
        self._tracker = SingleCameraTracker() if enable_tracking else None
        self._frame_id = 0
        self._fps = 0.0
        self._fps_t0 = time.time()
        self._low_conf_rois = []

        self._cached_result = None

    def process_frame(
        self,
        frame,
        camera_id: str = "cam_0",
        timestamp: float = None,
        external_tracks: list = None,
    ) -> dict:
        """
        处理单帧图像，返回标准化检测结果
        """
        if timestamp is None:
            timestamp = time.time()

        from camera_process import extract_angle_keypoints, run_roi_inference  # noqa: E402

        self._frame_id += 1
        current_time = timestamp

        do_inference = (self._frame_id - 1) % self.inference_interval == 0
        if external_tracks is not None:
            do_inference = False

        diagnostics = {
            "fps": 0.0,
            "backend": self.backend,
            "device": self.device,
            "inference_ran": do_inference,
        }

        if do_inference:
            try:
                # 后端推理 → 统一 detection schema
                detections = self._backend.infer(frame)

                # 转换为跟踪器内部格式
                internal_dets = self._tracker.convert_backend_detections(detections)
                tracked = self._tracker.update(internal_dets, current_time)

                # 限制人数
                if self.max_persons is not None and len(tracked) > self.max_persons:
                    tracked.sort(
                        key=lambda p: (
                            (p["bbox"][2] - p["bbox"][0])
                            * (p["bbox"][3] - p["bbox"][1])
                        ),
                        reverse=True,
                    )
                    keep_pids = {p["pid"] for p in tracked[:self.max_persons]}
                    tracked = [p for p in tracked if p["pid"] in keep_pids]

                # ROI 二次推理
                if self.enable_roi and self._low_conf_rois:
                    roi_detections = run_roi_inference(
                        self._backend, frame, self._low_conf_rois, self._tracker
                    )
                    self._tracker.update_roi(roi_detections, current_time)

                self._tracker.cleanup(current_time)
            except Exception:
                logger.warning(f"推理异常: {traceback.format_exc()}")
                tracked = self._get_cached_tracked()

            self._cached_tracked = tracked
        else:
            if self._tracker is not None:
                self._tracker.cleanup(current_time)
                for tid, data in self._tracker.person_history.items():
                    data["last_seen"] = current_time
            tracked = self._get_cached_tracked()

        # 跌倒判断
        for person in tracked:
            try:
                if person.get("is_ghost"):
                    if "fall_result" not in person:
                        person["fall_result"] = {
                            "fall_detected": False,
                            "confidence": 0,
                            "state": "Normal",
                        }
                    if do_inference and person.get("fall_result", {}).get("fall_detected"):
                        person["fall_result"]["state"] = "FALL"
                    continue

                kpts = person.get("keypoints")
                confs = person.get("confs")
                if kpts is not None and confs is not None:
                    person["angle_keypoints"] = extract_angle_keypoints(kpts, confs)
                else:
                    person["angle_keypoints"] = None

                if do_inference:
                    fall_result = evaluate_fall(person, current_time)
                    person["fall_result"] = fall_result
                else:
                    if "fall_result" not in person:
                        person["fall_result"] = {
                            "fall_detected": False,
                            "confidence": 0,
                            "state": "Normal",
                        }
            except Exception:
                logger.debug(f"跌倒判断异常: {traceback.format_exc()}")
                if "fall_result" not in person:
                    person["fall_result"] = {
                        "fall_detected": False,
                        "confidence": 0,
                        "state": "Normal",
                    }

        elapsed = time.time() - self._fps_t0
        self._fps = self._frame_id / (elapsed + 1e-8)
        diagnostics["fps"] = round(self._fps, 1)

        if do_inference and self.enable_roi:
            self._low_conf_rois = []
            for person in tracked:
                if person.get("fall_result", {}).get("state") in (
                    "Potential Fall",
                    "FALL",
                ):
                    predicted = self._tracker._predict_position(
                        self._tracker.person_history.get(person["pid"], {})
                    )
                    self._low_conf_rois.append({
                        "bbox": person["bbox"],
                        "pid": person["pid"],
                        "predicted_center": predicted,
                    })

        result = format_result(
            tracked=tracked,
            camera_id=camera_id,
            timestamp=timestamp,
            frame_id=self._frame_id,
            diagnostics=diagnostics,
        )

        if self.enable_visualization:
            annotated = frame.copy()
            for person in tracked:
                fr = person.get(
                    "fall_result",
                    {"fall_detected": False, "confidence": 0, "state": "Normal"},
                )
                annotated = draw_person_info(annotated, person, fr)

            fall_results = []
            for p in tracked:
                fr = p.get(
                    "fall_result",
                    {"fall_detected": False, "confidence": 0, "state": "Normal"},
                )
                fr_copy = dict(fr)
                fr_copy["is_ghost"] = p.get("is_ghost", False)
                fall_results.append(fr_copy)
            annotated = draw_fall_alert(annotated, fall_results)

            cv2.putText(
                annotated, f"FPS: {self._fps:.0f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            cv2.putText(
                annotated, timestamp_str, (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )
            result["annotated_frame"] = annotated

        self._cached_result = result
        return result

    def _get_cached_tracked(self):
        if not hasattr(self, "_cached_tracked") or self._cached_tracked is None:
            return []
        return self._cached_tracked

    def reset(self):
        self._tracker = SingleCameraTracker() if self.enable_tracking else None
        self._frame_id = 0
        self._fps_t0 = time.time()
        self._low_conf_rois = []
        self._cached_result = None
        if hasattr(self, "_cached_tracked"):
            del self._cached_tracked

    def close(self):
        self._backend.close()
        self._tracker = None
