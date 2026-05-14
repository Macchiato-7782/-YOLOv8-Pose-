"""
Runtime Pipeline 测试
测试 Detection/TrackState/Event 数据模型、序列化、Pipeline、detector 兼容性
"""

import sys
import os
import json
import time
import pytest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fall_detection.core.detection import Detection, Keypoint
from fall_detection.core.track import TrackState
from fall_detection.core.event import Event
from fall_detection.core.frame import FrameContext
from fall_detection.core.serializers import (
    serialize_detection, serialize_track, serialize_event,
    serialize_frame_context, serialize_result,
)
from fall_detection.core.validators import (
    validate_detection, validate_track, validate_event, validate_state,
)


def make_frame(w=640, h=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ============================================================
# Data Models
# ============================================================

class TestDataModels:
    def test_keypoint_create(self):
        kp = Keypoint(x=100.0, y=200.0, confidence=0.95)
        assert kp.x == 100.0
        assert kp.confidence == 0.95

    def test_keypoint_to_dict(self):
        kp = Keypoint(x=100.0, y=200.0, confidence=0.95)
        d = kp.to_dict()
        assert d == {"x": 100.0, "y": 200.0, "confidence": 0.95}

    def test_keypoint_from_dict(self):
        kp = Keypoint.from_dict({"x": 1.0, "y": 2.0, "confidence": 0.5})
        assert kp.x == 1.0

    def test_keypoint_to_list(self):
        kp = Keypoint(x=1.0, y=2.0, confidence=0.5)
        assert kp.to_list() == [1.0, 2.0, 0.5]

    def test_detection_create(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85)
        assert len(det.bbox) == 4
        assert det.bbox[0] == 10.0
        assert det.score == 0.85

    def test_detection_bbox_is_float_list(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85)
        assert isinstance(det.bbox, list)
        assert all(isinstance(v, float) for v in det.bbox)

    def test_detection_to_dict(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85,
                        keypoints=[Keypoint(50, 60, 0.9)])
        d = det.to_dict()
        assert d["score"] == 0.85
        assert len(d["keypoints"]) == 1

    def test_detection_from_dict(self):
        d = {"bbox": [1, 2, 3, 4], "score": 0.5, "keypoints": []}
        det = Detection.from_dict(d)
        assert det.bbox == [1, 2, 3, 4]

    def test_detection_json_serializable(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85)
        s = json.dumps(det.to_dict())
        assert isinstance(s, str)

    def test_trackstate_create(self):
        t = TrackState(track_id=1, bbox=[10, 20, 100, 200], center=[55, 110],
                       confidence=0.6, state="normal")
        assert t.track_id == 1
        assert t.state == "normal"

    def test_trackstate_to_dict(self):
        t = TrackState(track_id=1, bbox=[10, 20, 100, 200], center=[55, 110],
                       confidence=0.6, state="fall")
        d = t.to_dict()
        assert d["state"] == "fall"
        assert d["fall_detected"] is False

    def test_trackstate_from_detection(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85)
        t = TrackState.from_detection(det, track_id=42)
        assert t.track_id == 42
        assert t.center == [55.0, 110.0]

    def test_event_create(self):
        e = Event(event_type="fall_confirmed", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[10, 20, 100, 200])
        assert e.event_type == "fall_confirmed"

    def test_event_to_dict(self):
        e = Event(event_type="fall_confirmed", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[10, 20, 100, 200])
        d = e.to_dict()
        assert "timestamp" in d
        assert d["state"] == "fall"

    def test_event_json_serializable(self):
        e = Event(event_type="fall_confirmed", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[10, 20, 100, 200])
        s = json.dumps(e.to_dict())
        assert isinstance(s, str)

    def test_framecontext_create(self):
        ctx = FrameContext(frame_id=1, timestamp=100.0, camera_id="cam_0",
                          width=640, height=480)
        assert ctx.frame_id == 1
        assert ctx.width == 640

    def test_framecontext_from_frame(self):
        frame = make_frame()
        ctx = FrameContext.from_frame(frame, 1, "cam_0")
        assert ctx.width == 640
        assert ctx.height == 480


# ============================================================
# Serializers
# ============================================================

class TestSerializers:
    def test_serialize_detection(self):
        det = Detection(bbox=[1, 2, 3, 4], score=0.8,
                        keypoints=[Keypoint(50, 60, 0.9)])
        d = serialize_detection(det)
        assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]
        assert d["score"] == 0.8

    def test_serialize_track(self):
        t = TrackState(track_id=1, bbox=[10, 20, 100, 200], center=[55, 110],
                       confidence=0.6, state="fall", fall_detected=True,
                       keypoints=[[50, 60, 0.9]])
        d = serialize_track(t)
        assert d["state"] == "fall"
        assert d["fall_detected"] is True

    def test_serialize_event(self):
        e = Event(event_type="fall_confirmed", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[10, 20, 100, 200])
        d = serialize_event(e)
        assert d["event_type"] == "fall_confirmed"

    def test_serialize_result_json(self):
        result = serialize_result([], [], "cam_0", 100.0, 1, {"fps": 10})
        s = json.dumps(result)
        assert "module" in s

    def test_no_numpy_in_serialized(self):
        det = Detection(bbox=[1, 2, 3, 4], score=np.float32(0.5))
        d = serialize_detection(det)
        assert isinstance(d["score"], float)
        assert not isinstance(d["score"], np.floating)

    def test_no_numpy_in_track(self):
        t = TrackState(track_id=1, bbox=np.array([1, 2, 3, 4]), center=[55, 110],
                       confidence=np.float64(0.5), state="normal")
        d = serialize_track(t)
        assert isinstance(d["bbox"][0], float)
        assert isinstance(d["confidence"], float)


# ============================================================
# Validators
# ============================================================

class TestValidators:
    def test_validate_detection_ok(self):
        det = Detection(bbox=[1, 2, 3, 4], score=0.5)
        ok, _ = validate_detection(det)
        assert ok

    def test_validate_detection_bad_bbox(self):
        det = Detection(bbox=[1, 2], score=0.5)
        ok, msg = validate_detection(det)
        assert not ok

    def test_validate_detection_bad_score(self):
        det = Detection(bbox=[1, 2, 3, 4], score=1.5)
        ok, msg = validate_detection(det)
        assert not ok

    def test_validate_track_ok(self):
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3], confidence=0.5)
        ok, _ = validate_track(t)
        assert ok

    def test_validate_track_bad_state(self):
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3],
                       confidence=0.5, state="unknown")
        ok, _ = validate_track(t)
        assert not ok

    def test_validate_event_ok(self):
        e = Event(event_type="fall_confirmed", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[1, 2, 3, 4])
        ok, _ = validate_event(e)
        assert ok

    def test_validate_event_bad_type(self):
        e = Event(event_type="bad_type", timestamp=123.0, track_id=1,
                  camera_id="cam_0", confidence=0.6, bbox=[1, 2, 3, 4])
        ok, _ = validate_event(e)
        assert not ok

    def test_validate_state(self):
        assert validate_state("normal")
        assert validate_state("potential_fall")
        assert validate_state("fall")
        assert not validate_state("unknown")


# ============================================================
# Pipeline Compatibility
# ============================================================

class FakeDetectionBackend:
    def __init__(self):
        self.backend = "test"
        self.device = "cpu"

    def infer(self, frame):
        return []

    def close(self):
        pass


class TestPipelineCompatibility:
    def test_pipeline_import(self):
        from fall_detection.core.pipeline import DetectionPipeline
        assert DetectionPipeline is not None

    def test_runtime_import(self):
        from fall_detection.core.runtime import EventRuntime
        assert EventRuntime is not None

    def test_event_runtime_cooldown(self):
        from fall_detection.core.runtime import EventRuntime
        rt = EventRuntime(cooldown_seconds=5.0)
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3],
                       confidence=0.6, state="fall", fall_detected=True)
        ctx = FrameContext(frame_id=1, timestamp=100.0, camera_id="cam_0",
                          width=640, height=480)

        events = rt.process_tracks([t], ctx)
        assert len(events) >= 1
        assert events[0].event_type == "fall_confirmed"

        # 立即再调用应该不生成事件（cooldown）
        events2 = rt.process_tracks([t], ctx)
        assert len(events2) == 0

    def test_event_runtime_warning(self):
        from fall_detection.core.runtime import EventRuntime
        rt = EventRuntime(cooldown_seconds=0.1)
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3],
                       confidence=0.4, state="potential_fall")
        ctx = FrameContext(frame_id=1, timestamp=100.0, camera_id="cam_0",
                          width=640, height=480)
        events = rt.process_tracks([t], ctx)
        has_warning = any(e.event_type == "fall_warning" for e in events)
        assert has_warning

    def test_process_frame_compatible(self):
        """验证 process_frame 输出格式兼容"""
        from fall_detection import FallDetector
        from fall_detection.backends.ultralytics_backend import UltralyticsBackend

        # Mock _init_model to avoid real YOLO loading
        def fake_init(self):
            self._model = FakeYOLOInstance()

        with patch.object(UltralyticsBackend, '_init_model', fake_init):
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(backend="ultralytics", model_path="test.pt")
                frame = make_frame()
                result = detector.process_frame(frame)
                assert "module" in result
                assert result["module"] == "fall_detection"
                assert "camera_id" in result
                assert "persons" in result
                assert "events" in result
                assert "diagnostics" in result
                s = json.dumps(result)
                assert isinstance(s, str)
                detector.close()


class FakeYOLOInstance:
    def track(self, *args, **kwargs):
        return [FakeEmptyResults()]


class FakeEmptyResults:
    class FakeBoxes:
        xyxy = None
        conf = None
        id = None

    class FakeKeypoints:
        xy = None
        conf = None

    boxes = FakeBoxes()
    keypoints = None

    def plot(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
