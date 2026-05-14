"""
tracking.py 单元测试
测试 Pure Runtime 跟踪器：Detection → TrackState
"""

import sys
import time
import numpy as np
import pytest

sys.path.insert(0, '.')
from tracking import SingleCameraTracker
from fall_detection.core.detection import Detection, Keypoint


_track_id_counter = 0


@pytest.fixture(autouse=True)
def reset_track_id_counter():
    global _track_id_counter
    _track_id_counter = 0


def make_detection(center, bbox=None, with_keypoints=True, track_id=None):
    """构造 Detection 对象"""
    global _track_id_counter
    cx, cy = center
    if bbox is None:
        bbox = [cx - 30, cy - 60, cx + 30, cy + 60]
    if track_id is None:
        _track_id_counter += 1
        track_id = _track_id_counter
    kps = []
    if with_keypoints:
        kps = [Keypoint(x=float(cx), y=float(cy - 50 + i * 10), confidence=0.5) for i in range(17)]
    return Detection(bbox=[float(v) for v in bbox], score=0.85, keypoints=kps, track_id=track_id)


class TestExtractDetections:
    def test_empty_converted(self):
        tracker = SingleCameraTracker()
        # empty Detection list
        result = tracker.update([], time.time())
        assert result == []


class TestUpdate:
    def test_new_detection_creates_id(self):
        tracker = SingleCameraTracker()
        t = time.time()
        det = make_detection((320, 240))
        tracked = tracker.update([det], t)
        assert len(tracked) == 1
        assert tracked[0].track_id == 1

    def test_sequential_tracking(self):
        tracker = SingleCameraTracker()
        t = time.time()

        det1 = make_detection((320, 240))
        tracked1 = tracker.update([det1], t)
        pid1 = tracked1[0].track_id

        det2 = make_detection((325, 242), track_id=det1.track_id)
        tracked2 = tracker.update([det2], t)
        assert len(tracked2) == 1
        assert tracked2[0].track_id == pid1

    def test_two_persons(self):
        tracker = SingleCameraTracker()
        t = time.time()

        det_a = make_detection((200, 240))
        det_b = make_detection((500, 240))
        tracked = tracker.update([det_a, det_b], t)
        assert len(tracked) == 2
        pids = {p.track_id for p in tracked}
        assert len(pids) == 2

    def test_partial_body_tracked(self):
        tracker = SingleCameraTracker()
        t = time.time()

        det = make_detection((320, 240), with_keypoints=False)
        tracked = tracker.update([det], t)
        assert len(tracked) == 1


class TestCleanup:
    def test_ghost_timeout(self):
        tracker = SingleCameraTracker(ghost_timeout=0.5, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracker.update([det], t)
        assert len(tracker.person_history) == 1

        t += 0.2
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert tracker.person_history[1].get('is_ghost') is True

        t += 0.6
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert len(tracker.person_history) == 0

    def test_ghost_extended_for_fallen(self):
        tracker = SingleCameraTracker(ghost_timeout=0.5, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracker.update([det], t)
        tracker.person_history[1]['fall_state']['fall_detected'] = True

        t += 0.2
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)

        t += 0.6
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert len(tracker.person_history) == 1

    def test_reactivate_ghost(self):
        tracker = SingleCameraTracker(ghost_timeout=1.0, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracked = tracker.update([det], t)
        pid = tracked[0].track_id

        t += 0.2
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert tracker.person_history[pid].get('is_ghost') is True

        t += 0.1
        det2 = make_detection((325, 242), track_id=det.track_id)
        tracked2 = tracker.update([det2], t)
        assert len(tracked2) == 1
        assert tracked2[0].track_id == pid
