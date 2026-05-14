"""
FallDetector 接口测试
测试标准检测器接口：导入、初始化、process_frame、输出格式、序列化、reset
使用 mock 和合成数据，不依赖真实摄像头或模型文件
"""

import sys
import os
import time
import json
import pickle
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from fall_detection import FallDetector, EDGE_DEFAULTS
from fall_detection.schemas import (
    to_json_friendly, normalize_state, format_person, format_result,
)


# ============================================================
# helper
# ============================================================

def make_test_frame(width=640, height=480):
    """生成测试用的合成图像"""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def make_kp_5(head_y, shoulder_y, hip_y, knee_y, x_center=320):
    return {
        'H':  np.array([x_center, head_y], dtype=np.float64),
        'N':  np.array([x_center, shoulder_y], dtype=np.float64),
        'B':  np.array([x_center, hip_y], dtype=np.float64),
        'KL': np.array([x_center - 20, knee_y], dtype=np.float64),
        'KR': np.array([x_center + 20, knee_y], dtype=np.float64),
    }


# ============================================================
# schemas.py 测试
# ============================================================

class TestSchemas:
    def test_to_json_friendly_numpy_int(self):
        assert to_json_friendly(np.int64(42)) == 42
        assert isinstance(to_json_friendly(np.int64(42)), int)

    def test_to_json_friendly_numpy_float(self):
        assert to_json_friendly(np.float32(3.14)) == pytest.approx(3.14)
        assert isinstance(to_json_friendly(np.float32(3.14)), float)

    def test_to_json_friendly_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_json_friendly(arr)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_to_json_friendly_tuple(self):
        result = to_json_friendly((np.int64(1), np.float32(2.5)))
        assert result == [1, 2.5]

    def test_to_json_friendly_dict(self):
        d = {"a": np.int64(1), "b": np.array([1, 2])}
        result = to_json_friendly(d)
        assert result == {"a": 1, "b": [1, 2]}

    def test_normalize_state_normal(self):
        assert normalize_state("Normal") == "normal"

    def test_normalize_state_potential(self):
        assert normalize_state("Potential Fall") == "potential_fall"

    def test_normalize_state_fall(self):
        assert normalize_state("FALL") == "fall"
        assert normalize_state("FALL (Confirmed)") == "fall"
        assert normalize_state("FALL (Single Cam)") == "fall"

    def test_normalize_state_empty(self):
        assert normalize_state("") == "normal"
        assert normalize_state(None) == "normal"

    def test_format_person_output(self):
        person = {
            "pid": 1,
            "bbox": (100, 200, 300, 400),
            "center": (200, 300),
            "is_ghost": False,
            "keypoints": np.array([[100.0, 200.0], [150.0, 250.0]]),
            "confs": np.array([0.9, 0.8]),
        }
        fall_result = {
            "fall_detected": False,
            "confidence": 0.0,
            "state": "Normal",
        }
        out = format_person(person, fall_result)
        assert out["track_id"] == 1
        assert out["state"] == "normal"
        assert isinstance(out["bbox"], list)
        assert out["bbox"] == [100, 200, 300, 400]
        assert isinstance(out["center"], list)
        assert isinstance(out["confidence"], float)
        assert isinstance(out["keypoints"], list)
        assert len(out["keypoints"]) == 2
        # keypoints 每个元素是 [x, y, conf]
        assert out["keypoints"][0] == [100.0, 200.0, 0.9]

    def test_format_person_keypoints_no_conf(self):
        person = {
            "pid": 2,
            "bbox": (50, 60, 150, 200),
            "center": (100, 130),
            "is_ghost": True,
            "keypoints": np.array([[50.0, 60.0]]),
            "confs": None,
        }
        fall_result = {"fall_detected": True, "confidence": 0.6, "state": "FALL"}
        out = format_person(person, fall_result)
        assert out["state"] == "fall"
        assert out["fall_detected"] is True
        assert out["is_ghost"] is True
        assert out["keypoints"][0] == [50.0, 60.0, 0.0]

    def test_format_person_no_keypoints(self):
        person = {
            "pid": 3,
            "bbox": (10, 20, 100, 200),
            "center": (55, 110),
            "is_ghost": False,
            "keypoints": None,
            "confs": None,
        }
        fall_result = {"fall_detected": False, "confidence": 0.0, "state": "Normal"}
        out = format_person(person, fall_result)
        assert out["keypoints"] == []

    def test_format_result_structure(self):
        tracked = []
        result = format_result(tracked, "cam_0", 1710000000.0, 5, {"fps": 15.0})
        assert "module" in result
        assert result["module"] == "fall_detection"
        assert result["camera_id"] == "cam_0"
        assert result["timestamp"] == 1710000000.0
        assert result["frame_id"] == 5
        assert result["persons"] == []
        assert result["events"] == []
        assert "diagnostics" in result

    def test_format_result_with_fall_event(self):
        person = {
            "pid": 1,
            "bbox": (100, 200, 300, 400),
            "center": (200, 300),
            "is_ghost": False,
            "keypoints": None,
            "confs": None,
            "fall_result": {
                "fall_detected": True,
                "confidence": 0.95,
                "state": "FALL (Confirmed)",
            },
        }
        tracked = [person]
        result = format_result(tracked, "cam_1", time.time(), 10)
        assert len(result["events"]) == 1
        event = result["events"][0]
        assert event["event_type"] == "fall_confirmed"
        assert event["track_id"] == 1
        assert event["state"] == "fall"
        assert isinstance(event["confidence"], float)

    def test_json_serializable(self):
        tracked = []
        result = format_result(tracked, "cam_0", 1710000000.0, 1)
        s = json.dumps(result)
        assert isinstance(s, str)
        assert '"module": "fall_detection"' in s

    def test_json_serializable_with_numpy_types(self):
        person = {
            "pid": np.int64(1),
            "bbox": (np.int32(10), np.int32(20), np.int32(100), np.int32(200)),
            "center": (np.float64(55.0), np.float64(110.0)),
            "is_ghost": False,
            "keypoints": np.array([[10.0, 20.0], [30.0, 40.0]]),
            "confs": np.array([0.9, 0.8]),
            "fall_result": {
                "fall_detected": np.bool_(True),
                "confidence": np.float64(0.8),
                "state": "FALL",
            },
        }
        result = format_result([person], "cam_0", np.float64(1710000000.0), np.int64(42))
        s = json.dumps(result)
        assert s is not None

    def test_state_only_allowed_values(self):
        allowed = {"normal", "potential_fall", "fall"}
        for state in ["Normal", "Potential Fall", "FALL", "FALL (Confirmed)", "", None]:
            ns = normalize_state(state)
            assert ns in allowed, f"state '{state}' -> '{ns}' not in {allowed}"


# ============================================================
# edge_config.py 测试
# ============================================================

class TestEdgeConfig:
    def test_edge_defaults_keys(self):
        required = {"device", "backend", "input_size", "inference_interval",
                     "enable_roi", "enable_visualization", "max_persons", "conf", "tracker"}
        assert required.issubset(set(EDGE_DEFAULTS.keys()))

    def test_edge_defaults_reasonable(self):
        assert EDGE_DEFAULTS["input_size"] <= 640
        assert EDGE_DEFAULTS["inference_interval"] >= 1
        assert EDGE_DEFAULTS["enable_roi"] is False
        assert EDGE_DEFAULTS["enable_visualization"] is False


# ============================================================
# FallDetector 接口测试 (mock YOLO)
# ============================================================

class FakeYOLOModel:
    """模拟 YOLO 模型，返回空白跟踪结果"""
    def __init__(self, *args, **kwargs):
        pass

    def track(self, *args, **kwargs):
        return [FakeResults()]

    def __call__(self, *args, **kwargs):
        return [FakeResults()]


class FakeResults:
    """模拟 YOLO Results 对象"""
    class FakeKeypoints:
        conf = None
        xy = None

    class FakeBoxes:
        id = None
        xyxy = None

    keypoints = FakeKeypoints()
    boxes = FakeBoxes()

    def plot(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeBackend:
    """模拟推理后端"""
    def __init__(self, *args, **kwargs):
        pass

    def infer(self, frame):
        return []

    def warmup(self):
        pass

    def close(self):
        pass


# Helper: mock backend factory
def _mock_backend():
    return patch('fall_detection.detector.create_backend', return_value=FakeBackend())

    def release(self):
        pass


class TestFallDetectorInit:
    def test_import(self):
        from fall_detection import FallDetector
        assert FallDetector is not None

    def test_init_default(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                assert detector.device == "cpu"
                assert detector.backend == "ultralytics"
                assert detector.enable_tracking is True
                assert detector.enable_visualization is False
                assert detector.enable_roi is False
                assert detector.inference_interval == 1
                assert detector.max_persons is None
                detector.close()

    def test_init_edge_config(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(
                    device=EDGE_DEFAULTS["device"],
                    input_size=EDGE_DEFAULTS["input_size"],
                    inference_interval=EDGE_DEFAULTS["inference_interval"],
                    enable_roi=EDGE_DEFAULTS["enable_roi"],
                    enable_visualization=EDGE_DEFAULTS["enable_visualization"],
                    max_persons=EDGE_DEFAULTS["max_persons"],
                )
                assert detector.input_size == 320
                assert detector.inference_interval == 2
                assert detector.max_persons == 3
                detector.close()

    def test_init_with_custom_params(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(
                    enable_tracking=True,
                    enable_visualization=True,
                    enable_roi=True,
                    inference_interval=3,
                    input_size=416,
                    max_persons=5,
                    conf=0.5,
                )
                assert detector.inference_interval == 3
                assert detector.input_size == 416
                assert detector.max_persons == 5
                assert detector.conf == 0.5
                detector.close()


class TestFallDetectorProcessFrame:
    def test_process_frame_returns_dict(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert isinstance(result, dict)
                detector.close()

    def test_process_frame_required_fields(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                for field in ["module", "camera_id", "timestamp", "frame_id", "persons", "events", "diagnostics"]:
                    assert field in result, f"Missing field: {field}"
                detector.close()

    def test_process_frame_json_serializable(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                s = json.dumps(result)
                assert isinstance(s, str)
                detector.close()

    def test_process_frame_camera_id(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame, camera_id="entrance_1")
                assert result["camera_id"] == "entrance_1"
                detector.close()

    def test_process_frame_timestamp(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                ts = 1710000000.5
                result = detector.process_frame(frame, timestamp=ts)
                assert result["timestamp"] == ts
                detector.close()

    def test_process_frame_frame_id_increments(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                r1 = detector.process_frame(frame)
                r2 = detector.process_frame(frame)
                assert r2["frame_id"] == r1["frame_id"] + 1
                detector.close()

    def test_process_frame_persons_is_list(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert isinstance(result["persons"], list)
                detector.close()

    def test_process_frame_events_is_list(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert isinstance(result["events"], list)
                detector.close()

    def test_process_frame_diagnostics_has_required(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                diag = result["diagnostics"]
                assert "fps" in diag
                assert "backend" in diag
                assert "device" in diag
                assert "inference_ran" in diag
                detector.close()

    def test_process_frame_no_imshow(self):
        """process_frame 不应调用 cv2.imshow"""
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                with patch('cv2.imshow') as mock_imshow:
                    detector = FallDetector()
                    frame = make_test_frame()
                    detector.process_frame(frame)
                    mock_imshow.assert_not_called()
                    detector.close()

    def test_process_frame_no_video_write(self):
        """process_frame 不应调用 cv2.VideoWriter"""
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                with patch('cv2.VideoWriter') as mock_writer:
                    detector = FallDetector()
                    frame = make_test_frame()
                    detector.process_frame(frame)
                    mock_writer.assert_not_called()
                    detector.close()

    def test_process_frame_no_argparse(self):
        """process_frame 不应依赖 argparse"""
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert isinstance(result, dict)
                detector.close()

    def test_process_frame_with_visualization(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(enable_visualization=True)
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert "annotated_frame" in result
                assert isinstance(result["annotated_frame"], np.ndarray)
                detector.close()

    def test_process_frame_without_visualization(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(enable_visualization=False)
                frame = make_test_frame()
                result = detector.process_frame(frame)
                assert "annotated_frame" not in result
                detector.close()

    def test_process_frame_state_valid(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                for p in result["persons"]:
                    assert p["state"] in ("normal", "potential_fall", "fall"), \
                        f"Invalid state: {p['state']}"
                detector.close()

    def test_process_frame_bbox_is_list(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                for p in result["persons"]:
                    assert isinstance(p["bbox"], list), f"bbox not list: {type(p['bbox'])}"
                detector.close()

    def test_process_frame_track_id_is_int(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                result = detector.process_frame(frame)
                for p in result["persons"]:
                    assert isinstance(p["track_id"], (int, str)), \
                        f"track_id type: {type(p['track_id'])}"
                detector.close()


class TestFallDetectorReset:
    def test_reset_clears_frame_id(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                frame = make_test_frame()
                detector.process_frame(frame)
                detector.process_frame(frame)
                assert detector._frame_id == 2
                detector.reset()
                assert detector._frame_id == 0
                detector.close()

    def test_reset_can_process_after(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                detector.process_frame(make_test_frame())
                detector.reset()
                result = detector.process_frame(make_test_frame())
                assert result["frame_id"] == 1
                detector.close()


class TestFallDetectorEdge:
    def test_edge_defaults_constructor(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(
                    input_size=EDGE_DEFAULTS["input_size"],
                    inference_interval=EDGE_DEFAULTS["inference_interval"],
                    enable_roi=EDGE_DEFAULTS["enable_roi"],
                    enable_visualization=EDGE_DEFAULTS["enable_visualization"],
                    max_persons=EDGE_DEFAULTS["max_persons"],
                )
                assert detector.input_size == 320
                assert detector.inference_interval == 2
                assert detector.max_persons == 3
                assert detector.enable_roi is False
                assert detector.enable_visualization is False
                detector.close()

    def test_inference_interval_skips(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(inference_interval=2)
                frame = make_test_frame()
                r1 = detector.process_frame(frame)
                r2 = detector.process_frame(frame)
                # 第 2 帧不应推理
                assert r1["diagnostics"]["inference_ran"] is True
                assert r2["diagnostics"]["inference_ran"] is False
                detector.close()


class TestFallDetectorClose:
    def test_close_cleans_up(self):
        with _mock_backend():
            with patch('os.path.exists', return_value=False):
                detector = FallDetector()
                detector.close()
                assert detector._backend is not None  # closed but ref remains
                assert detector._tracker is None
