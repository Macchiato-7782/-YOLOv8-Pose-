"""
features.py 单元测试
测试物理特征计算模块的各种函数
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '.')
from features import (
    yolo_to_5keypoints, get_ratio_bbox, get_height_bbox,
    get_angle_vertical, get_torso_inclination,
    get_rot_energy, get_ratio_derivative, get_gf, get_head_descent,
    compute_all_features,
)


def make_kp_5(head_y, shoulder_y, hip_y, knee_y, x_center=320):
    return {
        'H':  np.array([x_center, head_y], dtype=np.float64),
        'N':  np.array([x_center, shoulder_y], dtype=np.float64),
        'B':  np.array([x_center, hip_y], dtype=np.float64),
        'KL': np.array([x_center - 20, knee_y], dtype=np.float64),
        'KR': np.array([x_center + 20, knee_y], dtype=np.float64),
    }


# ============================================================
# yolo_to_5keypoints
# ============================================================

class TestYoloTo5Keypoints:
    def test_valid_keypoints(self):
        kpts = np.zeros((17, 2), dtype=np.float64)
        confs = np.ones(17, dtype=np.float64) * 0.9
        # 设置关键点位置
        kpts[0] = [320, 100]   # Nose
        kpts[5] = [300, 160]   # LShoulder
        kpts[6] = [340, 160]   # RShoulder
        kpts[11] = [300, 300]  # LHip
        kpts[12] = [340, 300]  # RHip
        kpts[13] = [290, 420]  # LKnee
        kpts[14] = [350, 420]  # RKnee

        result = yolo_to_5keypoints(kpts, confs)
        assert result is not None
        assert 'H' in result and 'N' in result and 'B' in result
        assert 'KL' in result and 'KR' in result
        # H = Nose
        assert np.allclose(result['H'], [320, 100])
        # N = 双肩中点
        assert np.allclose(result['N'], [320, 160])
        # B = 双髋中点
        assert np.allclose(result['B'], [320, 300])

    def test_low_confidence_returns_none(self):
        kpts = np.zeros((17, 2), dtype=np.float64)
        confs = np.ones(17, dtype=np.float64) * 0.1  # 低于阈值 0.2
        kpts[0] = [320, 100]
        result = yolo_to_5keypoints(kpts, confs)
        assert result is None

    def test_zero_position_returns_none(self):
        kpts = np.zeros((17, 2), dtype=np.float64)
        confs = np.ones(17, dtype=np.float64) * 0.9
        # 关键点位置为 (0, 0) → 无效
        result = yolo_to_5keypoints(kpts, confs)
        assert result is None


# ============================================================
# get_ratio_bbox / get_height_bbox
# ============================================================

class TestBBoxMetrics:
    def test_standing_ratio(self):
        kp = make_kp_5(100, 160, 300, 420)
        ratio = get_ratio_bbox(kp)
        # 高 > 宽，ratio > 1
        assert ratio > 1.0

    def test_fallen_ratio(self):
        # 躺卧：x 范围大（宽），y 范围小（高）
        kp = {
            'H':  np.array([200, 300], dtype=np.float64),
            'N':  np.array([280, 305], dtype=np.float64),
            'B':  np.array([360, 310], dtype=np.float64),
            'KL': np.array([420, 315], dtype=np.float64),
            'KR': np.array([440, 315], dtype=np.float64),
        }
        ratio = get_ratio_bbox(kp)
        assert ratio < 1.0

    def test_height(self):
        kp = make_kp_5(100, 160, 300, 420)
        h = get_height_bbox(kp)
        assert h == pytest.approx(320, abs=5)


# ============================================================
# get_torso_inclination
# ============================================================

class TestTorsoInclination:
    def test_standing_near_zero(self):
        kp = make_kp_5(100, 160, 300, 420)
        angle = get_torso_inclination(kp)
        # 站立时躯干接近垂直，倾斜角接近 0°
        assert angle < 15

    def test_fallen_large_angle(self):
        # 躯干水平：髋(320,350) → 肩(420,340)，接近水平
        kp = {
            'H':  np.array([450, 350], dtype=np.float64),
            'N':  np.array([420, 340], dtype=np.float64),
            'B':  np.array([320, 350], dtype=np.float64),
            'KL': np.array([300, 420], dtype=np.float64),
            'KR': np.array([340, 420], dtype=np.float64),
        }
        angle = get_torso_inclination(kp)
        assert angle > 50


# ============================================================
# get_rot_energy
# ============================================================

class TestRotEnergy:
    def test_no_change(self):
        kp = make_kp_5(100, 160, 300, 420)
        re = get_rot_energy(kp, kp)
        assert re == pytest.approx(0.0, abs=1e-6)

    def test_with_dt_normalization(self):
        prev = make_kp_5(100, 160, 300, 420)
        curr = make_kp_5(300, 300, 350, 420)
        re_no_dt = get_rot_energy(prev, curr)
        re_with_dt = get_rot_energy(prev, curr, dt=0.2)
        # dt=0.2 → 归一化后值应为 5 倍
        assert re_with_dt == pytest.approx(re_no_dt / 0.2, rel=0.01)

    def test_symmetry(self):
        prev = make_kp_5(100, 160, 300, 420)
        curr = make_kp_5(300, 300, 350, 420)
        re1 = get_rot_energy(prev, curr)
        re2 = get_rot_energy(curr, prev)
        # 绝对值应该相同（只是方向不同）
        assert re1 == pytest.approx(re2, rel=0.01)


# ============================================================
# get_gf
# ============================================================

class TestGravityFactor:
    def test_no_movement(self):
        kp = make_kp_5(100, 160, 300, 420)
        gf = get_gf(kp, kp, kp)
        assert gf == pytest.approx(0.0, abs=1e-6)

    def test_downward_acceleration(self):
        kp0 = make_kp_5(100, 160, 300, 420)
        kp1 = make_kp_5(150, 200, 320, 420)
        kp2 = make_kp_5(250, 280, 340, 420)
        gf = get_gf(kp0, kp1, kp2)
        # 向下加速 → 正值
        assert gf > 0

    def test_with_dt_normalization(self):
        kp0 = make_kp_5(100, 160, 300, 420)
        kp1 = make_kp_5(150, 200, 320, 420)
        kp2 = make_kp_5(250, 280, 340, 420)
        gf_no_dt = get_gf(kp0, kp1, kp2)
        gf_with_dt = get_gf(kp0, kp1, kp2, dt=0.2)
        assert gf_with_dt == pytest.approx(gf_no_dt / (0.2 * 0.2), rel=0.01)


# ============================================================
# get_head_descent
# ============================================================

class TestHeadDescent:
    def test_no_descent(self):
        history = [make_kp_5(100, 160, 300, 420) for _ in range(5)]
        curr = make_kp_5(100, 160, 300, 420)
        result = get_head_descent(history, curr)
        assert result == 0

    def test_significant_descent(self):
        # 逐步下降（避免单帧噪声滤波器拦截）
        # 噪声阈值 = max(30, 0.1*body_height)，body_height=320 → 阈值=32
        # 单帧跳变必须 < 32
        history = []
        for i in range(20):
            head_y = 100 + i * 8  # 100 → 252，每帧 8px（< 32）
            history.append(make_kp_5(head_y, head_y + 60, head_y + 200, head_y + 320))
        # 当前帧：头部在 y=252 附近
        curr = make_kp_5(252, 312, 452, 572)
        result = get_head_descent(history, curr, initial_body_height=320)
        assert result > 0.2  # 超过 20% 身高

    def test_insufficient_history(self):
        history = [make_kp_5(100, 160, 300, 420)]
        curr = make_kp_5(200, 250, 340, 420)
        result = get_head_descent(history, curr)
        assert result == 0


# ============================================================
# compute_all_features
# ============================================================

class TestComputeAllFeatures:
    def test_no_kp5(self):
        result = compute_all_features({'history': []})
        assert result is None

    def test_basic_features(self):
        kp = make_kp_5(100, 160, 300, 420)
        person = {
            'kp_5': kp,
            'history': [kp] * 5,
        }
        features = compute_all_features(person)
        assert features is not None
        assert 're' in features
        assert 'gf' in features
        assert 'head_descent' in features

    def test_with_dt(self):
        kp = make_kp_5(100, 160, 300, 420)
        person = {
            'kp_5': kp,
            'history': [kp] * 5,
            'dt': 0.1,
        }
        features = compute_all_features(person)
        assert features is not None
