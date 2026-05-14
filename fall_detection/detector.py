"""
FallDetector 标准接口
可被外部项目直接 import 调用的唯一主入口
"""

import os
import time
import logging
import traceback

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from tracking import SingleCameraTracker
from fall_logic import evaluate_fall
from camera_process import extract_angle_keypoints, run_roi_inference
from config import CAM_PROC, load_config

from fall_detection.schemas import format_result, to_json_friendly
from fall_detection.visualizer import draw_person_info, draw_fall_alert, draw_skeleton
from fall_detection.edge_config import EDGE_DEFAULTS

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
        """
        Args:
            model_path: YOLOv8 姿态估计模型路径，默认使用内置 yolov8n-pose.pt
            config_path: YAML 配置文件路径，默认使用 config.yaml
            device: 推理设备 ("cpu", "cuda", "cuda:0", "mps")
            backend: 推理后端 ("ultralytics", 预留 "onnx", "openvino", "ncnn")
            enable_tracking: 是否启用 ByteTracker 多目标跟踪
            enable_visualization: 是否在返回结果中包含 annotated_frame
            enable_roi: 是否启用 ROI 二次推理（边缘设备建议关闭）
            inference_interval: 每 N 帧推理一次，非推理帧返回缓存状态
            input_size: 模型输入分辨率（默认 640，边缘设备建议 320）
            max_persons: 最多跟踪人数（按 bbox 面积排序取前 N）
            conf: YOLO 检测置信度阈值
            tracker: 跟踪器配置文件（bytetrack.yaml）
        """
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._root_dir = os.path.dirname(self._script_dir)

        # 模型路径
        if model_path is None:
            model_path = os.path.join(self._root_dir, "yolov8n-pose.pt")
        self.model_path = model_path
        self._model_check()

        # 配置
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

        # 初始化模型
        self._model = None
        self._init_model()

        # 跟踪器和内部状态
        self._tracker = SingleCameraTracker() if enable_tracking else None
        self._frame_id = 0
        self._fps = 0.0
        self._fps_t0 = time.time()
        self._low_conf_rois = []

        # 缓存：非推理帧复用上一次结果
        self._cached_result = None

    def _model_check(self):
        """检查模型文件是否存在，不存在则自动下载"""
        if os.path.exists(self.model_path):
            return
        logger.info(f"模型文件不存在，正在下载到 {self.model_path} ...")
        YOLO("yolov8n-pose.pt")
        # ultralytics 会自动下载到当前目录，如果路径不同则移动
        downloaded = os.path.join(os.getcwd(), "yolov8n-pose.pt")
        if os.path.exists(downloaded) and downloaded != self.model_path:
            try:
                os.rename(downloaded, self.model_path)
            except OSError:
                pass

    def _init_model(self):
        """初始化 YOLO 模型"""
        try:
            self._model = YOLO(self.model_path)
        except Exception:
            logger.warning(f"无法加载模型 {self.model_path}，尝试自动下载")
            YOLO("yolov8n-pose.pt")
            downloaded = os.path.join(os.getcwd(), "yolov8n-pose.pt")
            if os.path.exists(downloaded) and downloaded != self.model_path:
                try:
                    os.rename(downloaded, self.model_path)
                except OSError:
                    self.model_path = downloaded
            self._model = YOLO(self.model_path)

    def process_frame(
        self,
        frame,
        camera_id: str = "cam_0",
        timestamp: float = None,
        external_tracks: list = None,
    ) -> dict:
        """
        处理单帧图像，返回标准化检测结果

        Args:
            frame: OpenCV BGR 图像 (numpy array)
            camera_id: 摄像头标识符，用于多摄像头场景
            timestamp: Unix 时间戳，默认使用 time.time()
            external_tracks: 外部跟踪结果（预留，当前未使用）

        Returns:
            dict: 标准化检测结果，包含 persons / events / diagnostics
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_id += 1
        current_time = timestamp

        # 是否本帧执行推理（帧 1, 1+N, 1+2N, ...）
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
                results = self._model.track(
                    frame,
                    conf=self.conf,
                    persist=True,
                    tracker=self._tracker_config,
                    verbose=False,
                )[0]

                main_detections = self._tracker.extract_detections(results)
                tracked = self._tracker.update(main_detections, current_time)

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
                        self._model, frame, self._low_conf_rois, self._tracker
                    )
                    self._tracker.update_roi(roi_detections, current_time)

                self._tracker.cleanup(current_time)
            except Exception:
                logger.warning(f"推理异常: {traceback.format_exc()}")
                tracked = self._get_cached_tracked()

            self._cached_tracked = tracked
        else:
            # 非推理帧：仅做 tracker 清理，不更新检测
            if self._tracker is not None:
                self._tracker.cleanup(current_time)
                # 更新幽灵目标时间
                for tid, data in self._tracker.person_history.items():
                    data["last_seen"] = current_time
            tracked = self._get_cached_tracked()

        # 为每个目标计算跌倒状态
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

        # 更新 FPS
        elapsed = time.time() - self._fps_t0
        self._fps = self._frame_id / (elapsed + 1e-8)
        diagnostics["fps"] = round(self._fps, 1)

        # 更新 ROI 低置信度区域
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
                    self._low_conf_rois.append(
                        {
                            "bbox": person["bbox"],
                            "pid": person["pid"],
                            "predicted_center": predicted,
                        }
                    )

        # 构建标准化输出
        result = format_result(
            tracked=tracked,
            camera_id=camera_id,
            timestamp=timestamp,
            frame_id=self._frame_id,
            diagnostics=diagnostics,
        )

        # 可选：可视化
        if self.enable_visualization:
            annotated = frame.copy()
            if do_inference:
                try:
                    annotated = draw_skeleton(annotated, results)
                except Exception:
                    pass
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
                annotated,
                f"FPS: {self._fps:.0f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            cv2.putText(
                annotated,
                timestamp_str,
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            result["annotated_frame"] = annotated

        self._cached_result = result
        return result

    def _get_cached_tracked(self):
        """从 tracker 的 person_history 构建缓存跟踪列表"""
        if not hasattr(self, "_cached_tracked") or self._cached_tracked is None:
            if self._tracker is not None:
                return []
            return []
        return self._cached_tracked

    def reset(self):
        """重置内部状态（切换视频源时调用）"""
        self._tracker = SingleCameraTracker() if self.enable_tracking else None
        self._frame_id = 0
        self._fps_t0 = time.time()
        self._low_conf_rois = []
        self._cached_result = None
        if hasattr(self, "_cached_tracked"):
            del self._cached_tracked

    def close(self):
        """释放资源"""
        self._model = None
        self._tracker = None
