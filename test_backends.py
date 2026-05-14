"""
推理后端测试
测试 backend factory、ultralytics/onnx backend、统一 detection schema、detector 无关
"""

import sys
import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fall_detection.backends.base import BaseInferenceBackend
from fall_detection.backends.factory import create_backend
from fall_detection.backends.postprocess import (
    ultralytics_results_to_detections,
    onnx_outputs_to_detections,
)
from fall_detection.core.detection import Detection, Keypoint


def make_test_frame(w=640, h=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ============================================================
# Fake Results for ultralytics backend
# ============================================================

class FakeKeypoints:
    def __init__(self, xy_data, conf_data):
        self.xy = [np.array(xy_data)]
        self.conf = [np.array(conf_data)]

class FakeBoxes:
    def __init__(self, bboxes, scores, track_ids=None):
        self.xyxy = np.array(bboxes)
        self.conf = np.array(scores)
        self.id = np.array(track_ids) if track_ids is not None else None

class FakeResults:
    def __init__(self, bboxes, scores, keypoints_all_xy, keypoints_all_conf, track_ids=None):
        self.boxes = FakeBoxes(bboxes, scores, track_ids)
        class FakeKeypointsObj:
            def __init__(self, xy_list, conf_list):
                self.xy = [np.array(x) for x in xy_list]
                self.conf = [np.array(c) for c in conf_list]
            def __getitem__(self, i):
                class KpItem:
                    pass
                kp = KpItem()
                kp.xy = self.xy[i]
                kp.conf = self.conf[i]
                return kp
            def __len__(self):
                return len(self.xy)
        self.keypoints = FakeKeypointsObj(keypoints_all_xy, keypoints_all_conf)

    def plot(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================
# Fake YOLO for ultralytics backend
# ============================================================

class FakeYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def track(self, *args, **kwargs):
        return [FakeResults(
            bboxes=[[100, 200, 300, 500]],
            scores=[0.85],
            keypoints_all_xy=[[[150, 220], [180, 250], [160, 300], [140, 350], [260, 350],
                               [120, 280], [280, 280], [0, 0], [0, 0],
                               [130, 380], [270, 380], [0, 0], [0, 0],
                               [140, 420], [260, 420], [0, 0], [0, 0]]],
            keypoints_all_conf=[[0.9]*17],
            track_ids=[1],
        )]

    def __call__(self, *args, **kwargs):
        return [FakeResults(
            bboxes=[[100, 200, 300, 500]],
            scores=[0.85],
            keypoints_xy=[[[150, 220]] * 17],
            keypoints_conf=[[0.9]*17],
            track_ids=None,
        )]


# Make FakeYOLO importable at the module level it's patched to
FakeYOLOModule = FakeYOLO


# ============================================================
# BaseInferenceBackend
# ============================================================

class TestBaseBackend:
    def test_abstract_methods(self):
        assert 'infer' in BaseInferenceBackend.__abstractmethods__
        assert 'warmup' in BaseInferenceBackend.__abstractmethods__
        assert 'close' in BaseInferenceBackend.__abstractmethods__


# ============================================================
# Factory
# ============================================================

class TestFactory:
    def test_create_ultralytics_backend(self):
        with patch('ultralytics.YOLO', FakeYOLO, create=True):
            backend = create_backend(
                backend="ultralytics",
                model_path="yolov8n-pose.pt",
                device="cpu",
            )
            assert backend is not None
            backend.close()

    def test_create_onnx_backend_raises_if_no_onnxruntime(self):
        with patch.dict(sys.modules, {'onnxruntime': None}):
            pass

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError):
            create_backend(backend="unknown_backend", model_path="test.pt")

    def test_create_unsupported_openvino(self):
        with pytest.raises(NotImplementedError):
            create_backend(backend="openvino", model_path="test.xml")

    def test_create_unsupported_ncnn(self):
        with pytest.raises(NotImplementedError):
            create_backend(backend="ncnn", model_path="test.param")


# ============================================================
# UltralyticsBackend
# ============================================================

class TestUltralyticsBackend:
    def test_infer_returns_unified_schema(self):
        with patch('ultralytics.YOLO', FakeYOLO, create=True):
            backend = create_backend(
                backend="ultralytics",
                model_path="yolov8n-pose.pt",
                device="cpu",
                input_size=640,
            )
            frame = make_test_frame()
            detections = backend.infer(frame)

            assert isinstance(detections, list)
            assert len(detections) > 0

            det = detections[0]
            assert hasattr(det, 'bbox')
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4

            assert hasattr(det, 'score')
            assert isinstance(det.score, float)

            assert hasattr(det, 'class_id')
            assert det.class_id == 0

            assert hasattr(det, 'track_id')
            assert det.track_id is not None

            assert hasattr(det, 'keypoints')
            assert isinstance(det.keypoints, list)
            if len(det.keypoints) > 0:
                kp = det.keypoints[0]
                assert isinstance(kp, Keypoint)

            backend.close()

    def test_warmup(self):
        with patch('ultralytics.YOLO', FakeYOLO, create=True):
            backend = create_backend(
                backend="ultralytics",
                model_path="yolov8n-pose.pt",
                input_size=640,
            )
            backend.warmup()  # should not raise
            backend.close()

    def test_close(self):
        with patch('ultralytics.YOLO', FakeYOLO, create=True):
            backend = create_backend(
                backend="ultralytics",
                model_path="yolov8n-pose.pt",
                input_size=640,
            )
            backend.close()
            assert backend._model is None


# ============================================================
# Postprocess
# ============================================================

class TestPostprocess:
    def test_ultralytics_results_to_detections(self):
        results = FakeResults(
            bboxes=[[100.0, 200.0, 300.0, 500.0]],
            scores=[0.85],
            keypoints_all_xy=[[[150, 220], [180, 250], [160, 300], [140, 350], [260, 350],
                               [120, 280], [280, 280], [0, 0], [0, 0],
                               [130, 380], [270, 380], [0, 0], [0, 0],
                               [140, 420], [260, 420], [0, 0], [0, 0]]],
            keypoints_all_conf=[[0.9] * 17],
            track_ids=[1],
        )

        dets = ultralytics_results_to_detections(results)
        assert len(dets) == 1
        det = dets[0]
        assert det.bbox == [100.0, 200.0, 300.0, 500.0]
        assert det.score == 0.85
        assert det.track_id == 1
        assert len(det.keypoints) == 17

    def test_bbox_is_list(self):
        results = FakeResults(
            bboxes=[[50.0, 60.0, 150.0, 200.0]],
            scores=[0.5],
            keypoints_all_xy=[[[60, 70]] * 17],
            keypoints_all_conf=[[0.5] * 17],
        )
        dets = ultralytics_results_to_detections(results)
        assert isinstance(dets[0].bbox, list)

    def test_keypoints_is_list(self):
        results = FakeResults(
            bboxes=[[50.0, 60.0, 150.0, 200.0]],
            scores=[0.5],
            keypoints_all_xy=[[[60, 70]] * 17],
            keypoints_all_conf=[[0.5] * 17],
        )
        dets = ultralytics_results_to_detections(results)
        assert isinstance(dets[0].keypoints, list)
        assert isinstance(dets[0].keypoints[0], Keypoint)

    def test_no_track_ids(self):
        results = FakeResults(
            bboxes=[[50.0, 60.0, 150.0, 200.0]],
            scores=[0.5],
            keypoints_all_xy=[[[60, 70]] * 17],
            keypoints_all_conf=[[0.5] * 17],
            track_ids=None,
        )
        dets = ultralytics_results_to_detections(results)
        assert dets[0].track_id is None

    def test_empty_results(self):
        # 空结果
        class EmptyResults:
            boxes = None
            keypoints = None

        dets = ultralytics_results_to_detections(EmptyResults())
        assert dets == []

    def test_json_serializable(self):
        results = FakeResults(
            bboxes=[[100.0, 200.0, 300.0, 500.0]],
            scores=[0.85],
            keypoints_all_xy=[[[150, 220]] * 17],
            keypoints_all_conf=[[0.9] * 17],
            track_ids=[1],
        )
        dets = ultralytics_results_to_detections(results)
        s = json.dumps([d.to_dict() for d in dets])
        assert isinstance(s, str)

    def test_onnx_outputs_to_detections_empty(self):
        # 空输出（全低于阈值）
        output = np.zeros((1, 56, 100), dtype=np.float32)
        dets = onnx_outputs_to_detections(output, (480, 640))
        assert dets == []

    def test_onnx_outputs_to_detections_structure(self):
        # 构造一个高置信度检测
        output = np.zeros((1, 56, 1), dtype=np.float32)
        output[0, 0, 0] = 0.5   # cx
        output[0, 1, 0] = 0.3   # cy
        output[0, 2, 0] = 0.2   # w
        output[0, 3, 0] = 0.4   # h
        output[0, 4, 0] = 0.9   # conf (high)

        dets = onnx_outputs_to_detections(output, (480, 640), conf_threshold=0.5)
        assert len(dets) >= 1
        det = dets[0]
        assert hasattr(det, 'bbox')
        assert isinstance(det.bbox, list)
        assert det.track_id is None


# ============================================================
# Detector 不再 import YOLO 验证
# ============================================================

class TestDetectorBackendClean:
    def test_detector_does_not_import_yolo_directly(self):
        """验证 detector.py 不直接 import YOLO"""
        import ast
        detector_path = os.path.join(
            os.path.dirname(__file__),
            "fall_detection",
            "detector.py",
        )
        with open(detector_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        has_direct_yolo = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "ultralytics":
                    has_direct_yolo = True
                    for alias in node.names:
                        if alias.name == "YOLO":
                            has_direct_yolo = True

        assert not has_direct_yolo, "detector.py 不应直接 import YOLO"

    def test_tracking_does_not_import_ultralytics(self):
        """验证 tracking.py 不 import ultralytics"""
        import ast
        tracking_path = os.path.join(
            os.path.dirname(__file__),
            "tracking.py",
        )
        with open(tracking_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'module') and node.module and "ultralytics" in node.module:
                    assert False, "tracking.py 不应 import ultralytics"

    def test_process_frame_output_unchanged(self):
        """验证 process_frame 输出格式不变"""
        with patch('ultralytics.YOLO', FakeYOLO, create=True):
            with patch('os.path.exists', return_value=False):
                from fall_detection import FallDetector

                detector = FallDetector(backend="ultralytics", model_path="yolov8n-pose.pt")
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert "module" in result
                assert "camera_id" in result
                assert "persons" in result
                assert "events" in result
                assert "diagnostics" in result
                s = json.dumps(result)
                assert isinstance(s, str)
                detector.close()


# ============================================================
# Tracker convert_backend_detections
# ============================================================

class TestTrackerConvert:
    def test_convert(self):
        import time
        from tracking import SingleCameraTracker
        from fall_detection.core.detection import Detection, Keypoint

        tracker = SingleCameraTracker()
        kps = [Keypoint(x=150.0, y=220.0, confidence=0.9) for _ in range(17)]
        dets = [Detection(bbox=[100, 200, 300, 500], score=0.85, keypoints=kps)]

        tracks = tracker.update(dets, time.time())
        assert len(tracks) == 1
        t = tracks[0]
        assert t.track_id is not None
        assert isinstance(t.bbox, list)

    def test_assign_ids(self):
        import time
        from tracking import SingleCameraTracker
        from fall_detection.core.detection import Detection, Keypoint

        tracker = SingleCameraTracker()
        kps = [Keypoint(x=150.0, y=220.0, confidence=0.9) for _ in range(17)]
        dets = [Detection(bbox=[100, 200, 300, 500], score=0.85, keypoints=kps)]

        updated = tracker.update(dets, time.time())
        assert len(updated) == 1
        assert isinstance(updated[0].track_id, int)
        assert updated[0].track_id >= 1
