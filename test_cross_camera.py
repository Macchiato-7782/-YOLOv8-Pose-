"""
cross_camera.py 集成测试
测试跨摄像头人物匹配模块
"""

import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, '.')
from cross_camera import get_color_histogram, match_cross_camera, remove_wrongly_matched, merge_tracking_ids


def make_person(pid, center, hist=None):
    """构造一个模拟跟踪结果"""
    cx, cy = center
    return {
        'pid': pid,
        'center': center,
        'bbox': (cx - 30, cy - 60, cx + 30, cy + 60),
        'hist': hist,
    }


def make_histogram(hue_mean, saturation_mean=128):
    """构造一个指定色调的 HSV 直方图"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = hue_mean  # H
    img[:, :, 1] = saturation_mean  # S
    img[:, :, 2] = 200  # V
    bbox = (10, 10, 90, 90)
    return get_color_histogram(img, bbox)


# ============================================================
# get_color_histogram
# ============================================================

class TestColorHistogram:
    def test_returns_array(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        hist = get_color_histogram(img, (20, 20, 100, 100))
        assert hist is not None
        assert hist.ndim == 2

    def test_normalized(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        hist = get_color_histogram(img, (20, 20, 100, 100))
        # L1 归一化后总和约等于 1
        assert abs(hist.sum() - 1.0) < 0.1

    def test_empty_bbox(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        hist = get_color_histogram(img, (50, 50, 50, 50))  # 面积为 0
        assert hist is None


# ============================================================
# match_cross_camera
# ============================================================

class TestMatchCrossCamera:
    def test_same_color_matches(self):
        """相同颜色的人应该匹配"""
        hist_a = make_histogram(30)
        hist_b = make_histogram(30)
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        pairs = match_cross_camera(tracked_a, tracked_b)
        assert len(pairs) == 1
        assert pairs[0] == (0, 0)

    def test_different_color_no_match(self):
        """不同颜色的人不应匹配"""
        hist_a = make_histogram(10)   # 红色
        hist_b = make_histogram(120)  # 蓝色
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        pairs = match_cross_camera(tracked_a, tracked_b)
        assert len(pairs) == 0

    def test_empty_input(self):
        assert match_cross_camera([], []) == []
        assert match_cross_camera([make_person(1, (200, 200))], []) == []
        assert match_cross_camera([], [make_person(1, (200, 200))]) == []

    def test_none_histogram_skipped(self):
        """没有直方图的不应参与匹配"""
        tracked_a = [make_person(1, (200, 200), hist=None)]
        tracked_b = [make_person(10, (400, 200), hist=None)]
        pairs = match_cross_camera(tracked_a, tracked_b)
        assert len(pairs) == 0


# ============================================================
# remove_wrongly_matched
# ============================================================

class TestRemoveWronglyMatched:
    def test_removes_low_correlation(self):
        hist_a = make_histogram(10)
        hist_b = make_histogram(120)
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        pairs = remove_wrongly_matched(tracked_a, tracked_b, [(0, 0)])
        assert len(pairs) == 0

    def test_keeps_high_correlation(self):
        hist_a = make_histogram(30)
        hist_b = make_histogram(30)
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        pairs = remove_wrongly_matched(tracked_a, tracked_b, [(0, 0)])
        assert len(pairs) == 1


# ============================================================
# merge_tracking_ids
# ============================================================

class TestMergeTrackingIds:
    def test_new_match_assigns_global_id(self):
        hist_a = make_histogram(30)
        hist_b = make_histogram(30)
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        global_id_map = {}
        pid_to_global = merge_tracking_ids(tracked_a, tracked_b, [(0, 0)], global_id_map)
        assert len(global_id_map) == 1
        assert pid_to_global[1] == pid_to_global[10]

    def test_existing_match_preserves_id(self):
        """同一对再次匹配应保持相同 global_id"""
        hist_a = make_histogram(30)
        hist_b = make_histogram(30)
        tracked_a = [make_person(1, (200, 200), hist_a)]
        tracked_b = [make_person(10, (400, 200), hist_b)]
        global_id_map = {}

        pid_to_global = merge_tracking_ids(tracked_a, tracked_b, [(0, 0)], global_id_map)
        first_gid = pid_to_global[1]

        # 第二次匹配
        pid_to_global2 = merge_tracking_ids(tracked_a, tracked_b, [(0, 0)], global_id_map, pid_to_global)
        assert pid_to_global2[1] == first_gid

    def test_multiple_persons(self):
        hist1 = make_histogram(30)
        hist2 = make_histogram(90)
        tracked_a = [make_person(1, (200, 200), hist1), make_person(2, (400, 200), hist2)]
        tracked_b = [make_person(10, (200, 300), hist2), make_person(11, (400, 300), hist1)]

        global_id_map = {}
        pid_to_global = merge_tracking_ids(tracked_a, tracked_b, [(0, 1), (1, 0)], global_id_map)
        # 两个匹配对应两个不同的 global_id
        assert pid_to_global[1] != pid_to_global[2]
        assert pid_to_global[1] == pid_to_global[11]  # 相同色调
        assert pid_to_global[2] == pid_to_global[10]  # 相同色调
