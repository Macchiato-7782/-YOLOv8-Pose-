"""
Runtime Integrity Tests
验证 Runtime 内部数据类型完整性：无 dict leak, 无 numpy leak, 纯对象流水线
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fall_detection.core.detection import Detection, Keypoint
from fall_detection.core.track import TrackState
from fall_detection.core.event import Event
from fall_detection.core.frame import FrameContext
from fall_detection.core.types import DetectionList, TrackList, EventList


def make_frame(w=640, h=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ============================================================
# Backend Return Type Tests
# ============================================================

class TestBackendReturnsDetection:
    def test_ultralytics_results_to_detections_returns_detection_list(self):
        from fall_detection.backends.postprocess import ultralytics_results_to_detections

        class FakeBoxes:
            xyxy = np.array([])
            conf = np.array([])
            id = None

        class FakeResults:
            boxes = FakeBoxes()
            keypoints = None

        dets = ultralytics_results_to_detections(FakeResults())
        assert isinstance(dets, list)
        # empty results → empty list
        assert dets == []

    def test_onnx_outputs_to_detections_returns_detection_list(self):
        from fall_detection.backends.postprocess import onnx_outputs_to_detections
        output = np.zeros((1, 56, 100), dtype=np.float32)
        dets = onnx_outputs_to_detections(output, (480, 640))
        assert isinstance(dets, list)

    def test_detection_bbox_is_float_list(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85)
        assert isinstance(det.bbox, list)
        assert all(isinstance(v, float) for v in det.bbox)

    def test_detection_keypoints_are_keypoint_objects(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85,
                        keypoints=[Keypoint(50, 60, 0.9), Keypoint(70, 80, 0.8)])
        assert len(det.keypoints) == 2
        for kp in det.keypoints:
            assert isinstance(kp, Keypoint)
            assert isinstance(kp.x, float)
            assert isinstance(kp.y, float)
            assert isinstance(kp.confidence, float)

    def test_detection_no_numpy_in_output(self):
        det = Detection(bbox=[10, 20, 100, 200], score=0.85,
                        keypoints=[Keypoint(50, 60, 0.9)])
        d = det.to_dict()
        for val in d["bbox"]:
            assert not isinstance(val, (np.integer, np.floating, np.ndarray))
        for kp in d["keypoints"]:
            for val in [kp["x"], kp["y"], kp["confidence"]]:
                assert not isinstance(val, (np.integer, np.floating, np.ndarray))


# ============================================================
# Tracking Returns TrackState
# ============================================================

class TestTrackerReturnsTrackState:
    def test_tracker_update_returns_trackstate_list(self):
        from tracking import SingleCameraTracker
        import time

        tracker = SingleCameraTracker()
        det = Detection(bbox=[100, 200, 300, 500], score=0.85,
                        keypoints=[Keypoint(150, 220, 0.9)] * 17, track_id=None)
        tracks = tracker.update([det], time.time())
        assert isinstance(tracks, list)
        if len(tracks) > 0:
            assert isinstance(tracks[0], TrackState)

    def test_trackstate_has_required_fields(self):
        from tracking import SingleCameraTracker
        import time

        tracker = SingleCameraTracker()
        det = Detection(bbox=[100, 200, 300, 500], score=0.85,
                        keypoints=[Keypoint(150, 220, 0.9)] * 17)
        tracks = tracker.update([det], time.time())
        if len(tracks) > 0:
            t = tracks[0]
            assert hasattr(t, 'track_id')
            assert hasattr(t, 'bbox')
            assert hasattr(t, 'center')
            assert hasattr(t, 'state')
            assert hasattr(t, 'is_ghost')
            assert hasattr(t, 'keypoints')
            assert hasattr(t, 'fall_detected')

    def test_trackstate_no_dict_access(self):
        """TrackState should be accessed via attributes, not dict keys"""
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3], confidence=0.5)
        # These should work (attribute access)
        assert t.track_id == 1
        assert t.state == "normal"
        # bbox is a list of floats
        assert isinstance(t.bbox, list)


# ============================================================
# Event Tests
# ============================================================

class TestEventObject:
    def test_event_from_track(self):
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3], confidence=0.6)
        e = Event.from_track("fall_confirmed", t, "cam_0", 100.0)
        assert e.event_type == "fall_confirmed"
        assert e.track_id == 1

    def test_event_no_numpy(self):
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3], confidence=0.6)
        e = Event.from_track("fall_confirmed", t, "cam_0", 100.0)
        d = e.to_dict()
        for v in d["bbox"]:
            assert not isinstance(v, (np.integer, np.floating, np.ndarray))


# ============================================================
# Keypoint Object Tests
# ============================================================

class TestKeypointObjects:
    def test_keypoints_are_objects_not_lists(self):
        kp = Keypoint(x=1.0, y=2.0, confidence=0.5)
        assert hasattr(kp, 'x')
        assert hasattr(kp, 'y')
        assert hasattr(kp, 'confidence')

    def test_keypoint_to_list_for_compat(self):
        kp = Keypoint(x=1.0, y=2.0, confidence=0.5)
        lst = kp.to_list()
        assert lst == [1.0, 2.0, 0.5]


# ============================================================
# Pipeline Integrity
# ============================================================

class TestPipelineIntegrity:
    def test_pipeline_exists(self):
        from fall_detection.core.pipeline import DetectionPipeline
        assert DetectionPipeline is not None

    def test_process_frame_compatible(self):
        from fall_detection import FallDetector
        from fall_detection.backends.ultralytics_backend import UltralyticsBackend

        def fake_init(self):
            class FakeM:
                def track(self, *a, **kw):
                    class R:
                        class B:
                            xyxy = np.array([])
                            conf = np.array([])
                            id = None
                        boxes = B()
                        keypoints = None
                    return [R()]
            self._model = FakeM()

        with patch.object(UltralyticsBackend, '_init_model', fake_init):
            with patch('os.path.exists', return_value=False):
                detector = FallDetector(backend="ultralytics", model_path="test.pt")
                frame = make_frame()
                result = detector.process_frame(frame)
                assert "module" in result
                assert "persons" in result
                assert "events" in result
                s = json.dumps(result)
                assert isinstance(s, str)
                detector.close()

    def test_serialization_only_at_output(self):
        """序列化结果可 json.dumps"""
        from fall_detection.core.serializers import serialize_result
        result = serialize_result([], [], "cam_0", 100.0, 1, {"fps": 10})
        s = json.dumps(result)
        assert "module" in s
        assert "fall_detection" in s


# ============================================================
# Validators
# ============================================================

class TestValidatorsRuntime:
    def test_validate_detection_passes(self):
        from fall_detection.core.validators import validate_detection
        det = Detection(bbox=[1, 2, 3, 4], score=0.5)
        ok, _ = validate_detection(det)
        assert ok

    def test_validate_track_passes(self):
        from fall_detection.core.validators import validate_track
        t = TrackState(track_id=1, bbox=[1, 2, 3, 4], center=[2, 3], confidence=0.5)
        ok, _ = validate_track(t)
        assert ok

    def test_validate_event_rejects_bad_type(self):
        from fall_detection.core.validators import validate_event
        e = Event(event_type="bad", timestamp=1, track_id=1, camera_id="c", confidence=0.5, bbox=[1, 2, 3, 4])
        ok, _ = validate_event(e)
        assert not ok
