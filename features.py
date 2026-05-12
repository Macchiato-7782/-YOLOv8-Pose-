"""
物理特征计算模块
从 HumanFallDetection 项目移植并适配 YOLOv8 的 17 关键点格式

特征列表:
- ratio_bbox: 边界框宽高比
- log_angle: 身体与垂直方向夹角的对数
- re: 旋转能量 (Rotational Energy) — 倒立摆模型
- ratio_derivative: 宽高比变化率
- gf: 重力因子 (Gravity Factor)
"""

import numpy as np

FEATURE_LIST = ["ratio_bbox", "log_angle", "re", "ratio_derivative", "gf", "head_descent"]
FRAME_FEATURES = 2  # ratio_bbox 和 log_angle 只需要当前帧

# 头部下降趋势参数
HEAD_DESCENT_WINDOW = 20   # 检查最近 N 帧（约 1 秒 @20fps）
HEAD_DESCENT_MIN_PIXELS = 15  # 至少下降 15px 才算（过滤关键点抖动）


def yolo_to_5keypoints(keypoints, confs):
    """
    将 YOLOv8 的 17 关键点映射为 HumanFallDetection 的 5 关键点格式

    YOLO 索引:
    0=Nose, 5=LShoulder, 6=RShoulder, 11=LHip, 12=RHip, 13=LKnee, 14=RKnee

    映射:
    H (头部) = keypoint[0] (Nose)
    N (颈部) = (keypoint[5] + keypoint[6]) / 2 (双肩中点)
    B (臀部) = (keypoint[11] + keypoint[12]) / 2 (双髋中点)
    KL (左膝) = keypoint[13]
    KR (右膝) = keypoint[14]

    返回: dict {'H': [x,y], 'N': [x,y], 'B': [x,y], 'KL': [x,y], 'KR': [x,y]}
    或 None 如果关键点不足
    """
    required_indices = [0, 5, 6, 11, 12, 13, 14]
    for idx in required_indices:
        if idx >= len(keypoints) or confs[idx] < 0.2:
            return None
        if keypoints[idx][0] <= 0 and keypoints[idx][1] <= 0:
            return None

    H = keypoints[0]
    N = (keypoints[5] + keypoints[6]) / 2  # 双肩中点
    B = (keypoints[11] + keypoints[12]) / 2  # 双髋中点
    KL = keypoints[13]
    KR = keypoints[14]

    return {
        'H': H,
        'N': N,
        'B': B,
        'KL': KL,
        'KR': KR,
    }


def get_height_bbox(kp):
    """计算关键点 bounding box 的高度"""
    points = np.array([kp['H'], kp['N'], kp['B'], kp['KL'], kp['KR']])
    return np.max(points[:, 1]) - np.min(points[:, 1])


def get_ratio_bbox(kp):
    """宽高比：bbox 高/宽"""
    points = np.array([kp['H'], kp['N'], kp['B'], kp['KL'], kp['KR']])
    w = np.max(points[:, 0]) - np.min(points[:, 0])
    h = np.max(points[:, 1]) - np.min(points[:, 1])
    if w <= 0:
        return 0
    return h / w


def get_angle_vertical(body_vector):
    """身体与垂直方向的夹角（弧度）"""
    vertical = np.array([0, -1])  # 垂直向上
    cos_angle = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def get_rot_energy(prev_kp, curr_kp):
    """
    旋转能量：基于倒立摆模型
    计算躯干向量（N→B）在两帧之间的角度变化率
    """
    prev_body = prev_kp['N'] - prev_kp['B']
    curr_body = curr_kp['N'] - curr_kp['B']

    prev_angle = np.arctan2(prev_body[1], prev_body[0])
    curr_angle = np.arctan2(curr_body[1], curr_body[0])

    d_angle = curr_angle - prev_angle
    # 归一化到 [-pi, pi]
    if d_angle > np.pi:
        d_angle -= 2 * np.pi
    elif d_angle < -np.pi:
        d_angle += 2 * np.pi

    return abs(d_angle)


def get_ratio_derivative(prev_kp, curr_kp):
    """宽高比变化率"""
    prev_ratio = get_ratio_bbox(prev_kp)
    curr_ratio = get_ratio_bbox(curr_kp)
    return curr_ratio - prev_ratio


def get_gf(kp_t2, kp_t1, kp_t0):
    """
    重力因子：连续 3 帧重心加速度方向与重力方向的一致性
    kp_t2 最早, kp_t1 中间, kp_t0 最新
    """
    def center_of_gravity(kp):
        """估算重心位置"""
        return (kp['N'] + kp['B']) / 2

    p2 = center_of_gravity(kp_t2)
    p1 = center_of_gravity(kp_t1)
    p0 = center_of_gravity(kp_t0)

    # 二阶差分 = 加速度
    accel = p0 - 2 * p1 + p2

    # 重力方向（向下为正）
    gravity = np.array([0, 1])

    # 用原始投影值，不归一化（避免小噪声被放大到 ±1.0）
    # 站立静止时噪声约 ±10，真实摔倒时远大于此
    return float(np.dot(accel, gravity))


def get_head_descent(history, curr_kp, window=HEAD_DESCENT_WINDOW, initial_body_height=None):
    """
    头部下降趋势：检测头部在短时间内是否持续下降
    用于捕捉慢速滑倒（旋转能量和重力因子可能不够的场景）

    返回: float, 头部下降量 / 身高，值越大说明下降越多
    """
    if len(history) < 3 or curr_kp is None:
        return 0

    # 与上一帧对比，过滤单帧突变（YOLO 关键点噪声）
    prev_kp = history[-2]
    if prev_kp is not None and prev_kp['H'][1] > 0:
        single_frame_jump = abs(curr_kp['H'][1] - prev_kp['H'][1])
        if single_frame_jump > 30:  # 单帧跳变 > 30px，判定为噪声
            return 0

    # 取窗口内的历史帧
    recent = history[-window:] if len(history) >= window else history

    # 找到最早的有头部关键点的帧
    first_kp = None
    for kp in recent:
        if kp is not None and kp['H'][1] > 0:
            first_kp = kp
            break

    if first_kp is None:
        return 0

    # 头部 y 坐标变化（y 轴向下为正，所以下降是正值）
    head_drop = curr_kp['H'][1] - first_kp['H'][1]

    # 过滤微小抖动
    if head_drop < HEAD_DESCENT_MIN_PIXELS:
        return 0

    # 归一化：优先用初始站立时的 body_height（稳定），否则用当前帧
    if initial_body_height is not None and initial_body_height > 10:
        body_height = initial_body_height
    else:
        body_height = (curr_kp['KL'][1] + curr_kp['KR'][1]) / 2 - curr_kp['H'][1]
    if body_height < 10:
        return 0

    return max(0, head_drop / body_height)


def compute_all_features(person_data):
    """
    为一个跟踪目标计算所有物理特征

    person_data: dict 包含:
        - 'kp_5': 当前帧的 5 关键点
        - 'history': 最近 N 帧的 5 关键点历史列表

    返回: dict 特征值，或 None 如果数据不足
    """
    history = person_data.get('history', [])
    curr_kp = person_data.get('kp_5')

    if curr_kp is None:
        return None

    features = {}

    # 当前帧特征
    body_vector = curr_kp['N'] - curr_kp['B']
    features['ratio_bbox'] = get_ratio_bbox(curr_kp)
    features['angle_vertical'] = get_angle_vertical(body_vector)
    features['log_angle'] = np.log(1 + abs(features['angle_vertical']))

    # 需要前 1 帧的特征
    if len(history) >= 2:
        prev_kp = history[-2]
        if prev_kp is not None:
            features['re'] = get_rot_energy(prev_kp, curr_kp)
            features['ratio_derivative'] = get_ratio_derivative(prev_kp, curr_kp)

            # 需要前 2 帧的特征
            if len(history) >= 3 and history[-3] is not None:
                features['gf'] = get_gf(history[-3], prev_kp, curr_kp)
            else:
                features['gf'] = 0
        else:
            features['re'] = 0
            features['ratio_derivative'] = 0
            features['gf'] = 0
    else:
        features['re'] = 0
        features['ratio_derivative'] = 0
        features['gf'] = 0

    # 头部下降趋势（捕捉慢速滑倒）
    features['head_descent'] = get_head_descent(
        history, curr_kp,
        initial_body_height=person_data.get('initial_body_height')
    )

    return features
