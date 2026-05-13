"""
跌倒检测逻辑模拟测试
用合成数据测试 evaluate_fall 的各种场景，不依赖摄像头
"""

import sys
import time
import pickle
import pytest
import numpy as np

sys.path.insert(0, '.')
from fall_logic import evaluate_fall, calculate_angle
from features import compute_all_features, yolo_to_5keypoints


def make_kp_5(head_y, shoulder_y, hip_y, knee_y, x_center=320):
    return {
        'H':  np.array([x_center, head_y], dtype=np.float64),
        'N':  np.array([x_center, shoulder_y], dtype=np.float64),
        'B':  np.array([x_center, hip_y], dtype=np.float64),
        'KL': np.array([x_center - 20, knee_y], dtype=np.float64),
        'KR': np.array([x_center + 20, knee_y], dtype=np.float64),
    }


def make_angle_keypoints(shoulder_y, hip_y, knee_y, x=320):
    return {
        'shoulder': (float(x), float(shoulder_y)),
        'hip':      (float(x), float(hip_y)),
        'knee':     (float(x), float(knee_y)),
    }


# 水平躺卧的 kp_5（头在左，膝在右，y 坐标相近）
FALLEN_KP_5 = {
    'H':  np.array([450, 350], dtype=np.float64),
    'N':  np.array([420, 340], dtype=np.float64),
    'B':  np.array([320, 350], dtype=np.float64),
    'KL': np.array([300, 420], dtype=np.float64),
    'KR': np.array([340, 420], dtype=np.float64),
}
FALLEN_AK = {
    'shoulder': (420.0, 340.0),
    'hip':      (320.0, 350.0),
    'knee':     (300.0, 420.0),
}


def make_person(pid, kp_5, aspect_ratio, angle_keypoints=None, history=None):
    all_pts = np.array([kp_5['H'], kp_5['N'], kp_5['B'], kp_5['KL'], kp_5['KR']])
    x1, y1 = all_pts.min(axis=0).astype(int)
    x2, y2 = all_pts.max(axis=0).astype(int)
    return {
        'pid': pid,
        'kp_5': kp_5,
        'aspect_ratio': aspect_ratio,
        'angle_keypoints': angle_keypoints,
        'history': history or [],
        'bbox': (int(x1), int(y1), int(x2), int(y2)),
        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
        'fall_state': {
            'is_potential_fall': False,
            'fall_start_time': None,
            'fall_detected': False,
            'trigger_history': [],
            'min_head_y': 0,
            'angle_history': [],
        },
        'keypoints': np.zeros((17, 2)),
        'confs': np.ones(17) * 0.5,
        'is_full_body': True,
    }


def run_scenario(name, frames, expected_final_state, expected_fall_detected=None):
    """运行场景并返回最终 fall_result"""
    current_time = time.time()
    person = None
    fall_result = None

    for i, (kp_5, ar, ak) in enumerate(frames):
        if person is None:
            person = make_person(1, kp_5, ar, ak)
        else:
            person['kp_5'] = kp_5
            person['aspect_ratio'] = ar
            person['angle_keypoints'] = ak
            all_pts = np.array([kp_5['H'], kp_5['N'], kp_5['B'], kp_5['KL'], kp_5['KR']])
            x1, y1 = all_pts.min(axis=0).astype(int)
            x2, y2 = all_pts.max(axis=0).astype(int)
            person['bbox'] = (int(x1), int(y1), int(x2), int(y2))

        person['history'].append(kp_5)
        if len(person['history']) > 36:
            person['history'] = person['history'][-36:]

        current_time += 0.05
        fall_result = evaluate_fall(person, current_time)

    assert expected_final_state in fall_result['state'], \
        f"expected '{expected_final_state}' got '{fall_result['state']}'"
    if expected_fall_detected is not None:
        assert fall_result['fall_detected'] == expected_fall_detected, \
            f"expected fall_detected={expected_fall_detected} got {fall_result['fall_detected']}"
    return fall_result


# ============================================================
# calculate_angle 测试
# ============================================================

class TestCalculateAngle:
    def test_standing(self):
        angle = calculate_angle((320, 160), (320, 300), (320, 420))
        assert abs(angle - 180) < 5

    def test_horizontal(self):
        angle = calculate_angle((160, 300), (320, 300), (440, 300))
        assert abs(angle - 180) < 5

    def test_sitting(self):
        angle = calculate_angle((320, 185), (320, 280), (400, 350))
        assert 100 < angle < 160


# ============================================================
# 跌倒场景测试
# ============================================================

class TestFallScenarios:
    def test_standing_still(self):
        frames = []
        for _ in range(30):
            kp_5 = make_kp_5(100, 160, 300, 420)
            frames.append((kp_5, 2.5, make_angle_keypoints(160, 300, 420)))
        run_scenario("站立不动", frames, "Normal", False)

    def test_sitting(self):
        frames = []
        for _ in range(30):
            kp_5 = make_kp_5(130, 185, 280, 350)
            frames.append((kp_5, 1.4, make_angle_keypoints(185, 280, 350)))
        run_scenario("坐着不动", frames, "Normal", False)

    def test_forward_fall(self):
        frames = []
        for i in range(15):
            t = min(i / 10.0, 1.0)
            kp_5 = make_kp_5(100 + t*200, 160 + t*140, 300 + t*50, 420)
            ar = 2.5 * (1 - t) + 0.4 * t
            ak = make_angle_keypoints(160 + t*140, 300 + t*50, 420)
            frames.append((kp_5, ar, ak))
        for _ in range(80):
            kp_5 = make_kp_5(300, 300, 350, 420)
            frames.append((kp_5, 0.4, make_angle_keypoints(300, 350, 420)))
        run_scenario("前倒", frames, "FALL", True)

    def test_side_fall(self):
        frames = []
        for i in range(15):
            t = min(i / 10.0, 1.0)
            kp_5 = make_kp_5(100 + t*250, 160 + t*140, 300 + t*50, 420)
            ar = 2.5 * (1 - t) + 0.3 * t
            ak = make_angle_keypoints(160 + t*140, 300 + t*50, 420)
            frames.append((kp_5, ar, ak))
        for _ in range(80):
            kp_5 = make_kp_5(350, 350, 380, 420)
            frames.append((kp_5, 0.3, make_angle_keypoints(350, 380, 420)))
        run_scenario("侧倒", frames, "FALL", True)

    def test_fast_fall(self):
        frames = []
        for _ in range(5):
            kp_5 = make_kp_5(100, 160, 300, 420)
            frames.append((kp_5, 2.5, make_angle_keypoints(160, 300, 420)))
        for _ in range(5):
            frames.append((FALLEN_KP_5.copy(), 0.4, FALLEN_AK))
        for _ in range(80):
            frames.append((FALLEN_KP_5.copy(), 0.4, FALLEN_AK))
        run_scenario("快速摔倒", frames, "FALL", True)

    def test_bend_and_recover(self):
        frames = []
        for _ in range(5):
            kp_5 = make_kp_5(100, 160, 300, 420)
            frames.append((kp_5, 2.5, make_angle_keypoints(160, 300, 420)))
        for _ in range(10):
            kp_5 = make_kp_5(250, 250, 300, 420)
            frames.append((kp_5, 0.8, make_angle_keypoints(250, 300, 420)))
        for _ in range(15):
            kp_5 = make_kp_5(100, 160, 300, 420)
            frames.append((kp_5, 2.5, make_angle_keypoints(160, 300, 420)))
        run_scenario("弯腰后站起", frames, "Normal", False)

    def test_already_lying(self):
        frames = []
        for _ in range(30):
            frames.append((FALLEN_KP_5.copy(), 0.2, FALLEN_AK))
        run_scenario("已躺在地上", frames, "Potential Fall")


# ============================================================
# 边界条件测试
# ============================================================

class TestEdgeCases:
    def test_cross_process_serialization(self):
        kp_5 = make_kp_5(100, 160, 300, 420)
        ak = make_angle_keypoints(160, 300, 420)

        kp_5_p = pickle.loads(pickle.dumps(kp_5))
        ak_p = pickle.loads(pickle.dumps(ak))

        for key in ['H', 'N', 'B', 'KL', 'KR']:
            assert np.allclose(kp_5[key], kp_5_p[key]), f"kp_5['{key}'] corrupted after pickle"
        for key in ['shoulder', 'hip', 'knee']:
            assert ak[key] == ak_p[key], f"ak['{key}'] corrupted after pickle"

        person = make_person(1, kp_5_p, 2.5, ak_p)
        for _ in range(10):
            person['history'].append(kp_5_p)
        result = evaluate_fall(person, time.time())
        assert result is not None and 'state' in result

    def test_angle_keypoints_fallback(self):
        kp_5 = make_kp_5(300, 300, 350, 420)
        person = make_person(1, kp_5, 0.4, angle_keypoints=None)
        for _ in range(10):
            person['history'].append(kp_5)
        result = evaluate_fall(person, time.time())
        assert result is not None and 'state' in result

    def test_fall_state_persistence(self):
        """模拟双摄模式：camera_process 每帧传来新 person，主进程用 fall_state_store 持久化"""
        fall_state_store = {}
        history_store = {}

        frames_data = []
        for _ in range(5):
            frames_data.append((make_kp_5(100, 160, 300, 420), 2.5, make_angle_keypoints(160, 300, 420)))
        for i in range(15):
            t = min(i / 10.0, 1.0)
            frames_data.append((FALLEN_KP_5.copy(), 2.5 * (1-t) + 0.4 * t, FALLEN_AK))
        for _ in range(80):
            frames_data.append((FALLEN_KP_5.copy(), 0.4, FALLEN_AK))

        current_time = time.time()
        fall_result = None

        for i, (kp_5, ar, ak) in enumerate(frames_data):
            person = make_person(1, kp_5, ar, ak)

            store_key = ('A', 1)
            if store_key not in history_store:
                history_store[store_key] = []
            history_store[store_key].append(kp_5)
            if len(history_store[store_key]) > 36:
                history_store[store_key] = history_store[store_key][-36:]
            person['history'] = list(history_store[store_key])

            if store_key in fall_state_store:
                person['fall_state'] = fall_state_store[store_key]

            current_time += 0.05
            fall_result = evaluate_fall(person, current_time)
            fall_state_store[store_key] = person['fall_state']

        assert fall_result['fall_detected'], f"fall not detected (state={fall_result['state']})"
