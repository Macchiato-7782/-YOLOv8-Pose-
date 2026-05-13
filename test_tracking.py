"""
tracking.py 单元测试
测试单摄像头跟踪器的检测提取、跟踪更新、幽灵机制
"""

import sys
import time
import numpy as np
import pytest

sys.path.insert(0, '.')
from tracking import SingleCameraTracker


_track_id_counter = 0

@pytest.fixture(autouse=True)
def reset_track_id_counter():
    global _track_id_counter
    _track_id_counter = 0

def make_detection(center, bbox=None, is_full_body=True, kp_5=None):
    """构造一个模拟检测结果（模拟 ByteTracker 分配的 track_id）"""
    global _track_id_counter
    _track_id_counter += 1
    cx, cy = center
    if bbox is None:
        bbox = (cx - 30, cy - 60, cx + 30, cy + 60)
    if kp_5 is None:
        kp_5 = {
            'H':  np.array([cx, cy - 50], dtype=np.float64),
            'N':  np.array([cx, cy - 30], dtype=np.float64),
            'B':  np.array([cx, cy + 10], dtype=np.float64),
            'KL': np.array([cx - 15, cy + 50], dtype=np.float64),
            'KR': np.array([cx + 15, cy + 50], dtype=np.float64),
        }
    return {
        'track_id': _track_id_counter,
        'center': center,
        'bbox': bbox,
        'aspect_ratio': 2.0,
        'kp_5': kp_5,
        'keypoints': np.zeros((17, 2)),
        'confs': np.ones(17) * 0.5,
        'is_full_body': is_full_body,
    }


# ============================================================
# extract_detections
# ============================================================

class TestExtractDetections:
    def test_empty_results(self):
        tracker = SingleCameraTracker()

        class MockResults:
            keypoints = None
            boxes = None

        result = tracker.extract_detections(MockResults())
        assert result == []


# ============================================================
# update (Hungarian matching + motion model)
# ============================================================

class TestUpdate:
    def test_new_detection_creates_id(self):
        tracker = SingleCameraTracker()
        t = time.time()
        det = make_detection((320, 240))
        tracked = tracker.update([det], t)
        assert len(tracked) == 1
        assert tracked[0]['pid'] == 1

    def test_sequential_tracking(self):
        """ByteTracker 保证同一目标跨帧 track_id 不变"""
        tracker = SingleCameraTracker()
        t = time.time()

        # 第一帧：ByteTracker 分配 track_id=1
        det1 = make_detection((320, 240))
        tracked1 = tracker.update([det1], t)
        pid1 = tracked1[0]['pid']

        # 第二帧：ByteTracker 匹配同一目标，track_id 不变
        det2 = make_detection((325, 242))
        det2['track_id'] = det1['track_id']  # 模拟 ByteTracker 匹配
        tracked2 = tracker.update([det2], t)
        assert len(tracked2) == 1
        assert tracked2[0]['pid'] == pid1  # 同一个人

    def test_two_persons(self):
        tracker = SingleCameraTracker()
        t = time.time()

        det_a = make_detection((200, 240))
        det_b = make_detection((500, 240))
        tracked = tracker.update([det_a, det_b], t)
        assert len(tracked) == 2
        pids = {p['pid'] for p in tracked}
        assert len(pids) == 2  # 两个不同的 ID

    def test_partial_body_tracked(self):
        """ByteTracker 跟踪所有检测（含非全身），但不更新关键点历史"""
        tracker = SingleCameraTracker()
        t = time.time()

        det = make_detection((320, 240), is_full_body=False)
        tracked = tracker.update([det], t)
        assert len(tracked) == 1  # ByteTracker 仍然跟踪
        assert tracked[0]['is_full_body'] is False


# ============================================================
# cleanup (ghost mechanism)
# ============================================================

class TestCleanup:
    def test_ghost_timeout(self):
        tracker = SingleCameraTracker(ghost_timeout=0.5, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracker.update([det], t)
        assert len(tracker.person_history) == 1

        # 触发 cleanup → 标记为幽灵
        t += 0.2
        tracker.last_cleanup_time = 0  # 强制触发
        tracker.cleanup(t)
        assert tracker.person_history[1].get('is_ghost') is True

        # 超过幽灵超时 → 删除
        t += 0.6
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert len(tracker.person_history) == 0

    def test_ghost_extended_for_fallen(self):
        """已确认跌倒的幽灵目标超时更长"""
        tracker = SingleCameraTracker(ghost_timeout=0.5, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracker.update([det], t)

        # 模拟已确认跌倒
        tracker.person_history[1]['fall_state']['fall_detected'] = True

        # 标记为幽灵
        t += 0.2
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)

        # 普通超时后仍在（跌倒目标超时更长）
        t += 0.6
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert len(tracker.person_history) == 1  # 还在

    def test_reactivate_ghost(self):
        """幽灵目标重新出现时应恢复（ByteTracker 分配相同 track_id）"""
        tracker = SingleCameraTracker(ghost_timeout=1.0, cleanup_interval=0.1)
        t = time.time()

        det = make_detection((320, 240))
        tracked = tracker.update([det], t)
        pid = tracked[0]['pid']

        # 标记为幽灵
        t += 0.2
        tracker.last_cleanup_time = 0
        tracker.cleanup(t)
        assert tracker.person_history[pid].get('is_ghost') is True

        # 重新出现（ByteTracker 匹配到同一 track_id）
        t += 0.1
        det2 = make_detection((325, 242))
        det2['track_id'] = det['track_id']  # 模拟 ByteTracker 匹配
        tracked2 = tracker.update([det2], t)
        assert len(tracked2) == 1
        assert tracked2[0]['pid'] == pid
        assert tracked2[0]['recovered_from_ghost'] is True
