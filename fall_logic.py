"""
融合跌倒判断逻辑
结合规则判断（项目2）和物理特征（项目1），支持双摄像头交叉验证
"""

import time
import numpy as np
from features import compute_all_features, FEATURE_LIST

# 规则阈值（来自 Real-Time-Fall-Detection）
HORIZONTAL_AR_THRESHOLD = 0.6
ANGLE_THRESHOLD = 130          # 站立约180°，跌倒时躯干倾斜>50°
MIN_FALL_POSE_DURATION = 1.0   # 必须持续 1.0 秒才确认跌倒

# 物理特征阈值（来自 HumanFallDetection）
RE_THRESHOLD = 1.0             # 旋转能量（坐姿约 0.3-0.6，跌倒 > 1.0）
GF_THRESHOLD = 20              # 重力因子（站立噪声 ±10，跌倒 > 20）
HEAD_DESCENT_THRESHOLD = 0.2   # 头部下降阈值（身高的 20%，坐姿约 15-25%）

# 高置信度快速通道阈值（四个条件同时满足时跳过持续时间要求）
FAST_RE_THRESHOLD = 1.0
FAST_GF_THRESHOLD = 20
FAST_ANGLE_THRESHOLD = 120

# 滑动时间窗口
WINDOW_SIZE = 10           # 窗口大小（帧数）
WINDOW_TRIGGER_RATIO = 0.5 # 窗口内 50% 帧触发即可

# 初始站立检测：AR 必须大于此值才开始跌倒检测
MIN_STANDING_AR = 1.2

# 小目标面积阈值（低于此值的 bbox 宽高比不可靠）
MIN_BBOX_AREA = 3000

# 双摄确认
DUAL_CAM_CONFIRM = True
SINGLE_CAM_FALL_CONFIDENCE = 0.6
DUAL_CAM_FALL_CONFIDENCE = 0.95


def calculate_angle(a, b, c):
    """计算三点角度（b 为顶点）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.array_equal(a, b) or np.array_equal(b, c):
        return 180.0

    vec_ba = a - b
    vec_bc = c - b
    radians = np.arctan2(vec_bc[1], vec_bc[0]) - np.arctan2(vec_ba[1], vec_ba[0])
    angle = abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle


def check_rule_based(kp_5, aspect_ratio, angle_keypoints, bbox_area=None, initial_ar=None, smoothed_angle=None):
    """
    规则判断：宽高比 AND 角度同时触发才算（减少坐姿误报）

    返回: bool 是否触发"可能跌倒"
    """
    ar_triggered = False
    angle_triggered = False

    # 规则 1: 宽高比（小目标跳过，用变化量代替绝对值）
    if aspect_ratio is not None:
        if bbox_area is not None and bbox_area < MIN_BBOX_AREA:
            pass  # 小目标宽高比不可靠，跳过
        elif initial_ar is not None and initial_ar > 0:
            ar_ratio = aspect_ratio / initial_ar
            if ar_ratio < 0.35:  # 比站姿扁了 65%
                ar_triggered = True
        elif aspect_ratio < HORIZONTAL_AR_THRESHOLD:
            ar_triggered = True

    # 规则 2: 髋部角度（优先用平滑后的角度，更稳定）
    if smoothed_angle is not None and smoothed_angle > 0:
        if smoothed_angle < ANGLE_THRESHOLD:
            angle_triggered = True
    elif angle_keypoints is not None:
        angle = calculate_angle(
            angle_keypoints['shoulder'],
            angle_keypoints['hip'],
            angle_keypoints['knee']
        )
        if angle < ANGLE_THRESHOLD:
            angle_triggered = True

    # AND 逻辑：两个条件都满足才触发
    return ar_triggered and angle_triggered


def check_physical_features(features):
    """
    物理特征判断（来自 HumanFallDetection）

    返回: bool 是否触发"可能跌倒"
    """
    if features is None:
        return False

    triggered = False

    # 旋转能量突然增大
    if features.get('re', 0) > RE_THRESHOLD:
        triggered = True

    # 重力因子异常（向下加速）
    if features.get('gf', 0) > GF_THRESHOLD:
        triggered = True

    # 头部持续下降（捕捉慢速滑倒）
    if features.get('head_descent', 0) > HEAD_DESCENT_THRESHOLD:
        triggered = True

    return triggered


def evaluate_fall(person_data, current_time, dual_cam_fall=None):
    """
    综合判断跌倒

    person_data: dict 包含:
        - 'kp_5': 当前 5 关键点
        - 'aspect_ratio': 当前宽高比
        - 'angle_keypoints': 用于角度计算的关键点坐标 (可选)
        - 'history': 历史 5 关键点列表
        - 'fall_state': 跌倒状态 dict

    dual_cam_fall: 另一个摄像头对该人的跌倒判断结果 (可选)

    返回: dict 包含:
        - 'fall_detected': bool
        - 'confidence': float
        - 'state': 'Normal' / 'Potential Fall' / 'Fall'
    """
    fall_state = person_data['fall_state']

    # 计算物理特征
    features = compute_all_features(person_data)

    # 记录初始站立宽高比（如果当前 AR 更大，说明更可能是站立状态，更新基线）
    current_ar = person_data.get('aspect_ratio')
    if 'initial_ar' not in fall_state:
        fall_state['initial_ar'] = current_ar
    elif current_ar is not None and current_ar > fall_state.get('initial_ar', 0) and not fall_state.get('is_potential_fall'):
        # 不在跌倒检测中，且 AR 比记录值更大 → 更新为更站立的状态
        fall_state['initial_ar'] = current_ar

    # 记录初始 body_height（头到膝中点的距离）
    kp_5 = person_data.get('kp_5')
    if 'initial_body_height' not in fall_state and kp_5 is not None:
        bh = (kp_5['KL'][1] + kp_5['KR'][1]) / 2 - kp_5['H'][1]
        if bh > 10:
            fall_state['initial_body_height'] = bh
    person_data['initial_body_height'] = fall_state.get('initial_body_height')

    # 计算 bbox 面积
    bbox = person_data.get('bbox')
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else None

    # 髋部角度计算（两种来源）
    ak = person_data.get('angle_keypoints')
    kp_5 = person_data.get('kp_5')
    hip_angle = 0

    if ak is not None:
        hip_angle = calculate_angle(ak['shoulder'], ak['hip'], ak['knee'])
    elif kp_5 is not None:
        # fallback: 用 5 关键点的 B(髋)/KL(膝)/H(头) 计算
        hip_angle = calculate_angle(kp_5['B'], kp_5['KL'], kp_5['H'])

    if hip_angle > 0:
        if 'angle_history' not in fall_state:
            fall_state['angle_history'] = []
        fall_state['angle_history'].append(hip_angle)
        if len(fall_state['angle_history']) > 3:
            fall_state['angle_history'] = fall_state['angle_history'][-3:]
    hip_angle_smoothed = float(np.mean(fall_state['angle_history'])) if fall_state.get('angle_history') else 0.0

    # Debug 打印
    hd = features.get('head_descent', 0) if features else 0
    hd_info = ""
    if kp_5 is not None and hd > 0:
        head_y = kp_5['H'][1]
        kl_y = kp_5['KL'][1]
        kr_y = kp_5['KR'][1]
        body_h = (kl_y + kr_y) / 2 - head_y
        hd_info = f" head_y:{head_y:.0f} knees_avg:{(kl_y+kr_y)/2:.0f} body_h:{body_h:.0f}"
    initial_ar = fall_state.get('initial_ar')
    print(f"[DEBUG] PID:{person_data.get('pid')} "
          f"AR:{person_data.get('aspect_ratio', 0):.2f} init_AR:{initial_ar or 0:.2f} "
          f"angle:{hip_angle:.1f}(s:{hip_angle_smoothed:.1f}) src:{'ak' if ak else 'kp5' if kp_5 else 'none'} "
          f"HD:{hd:.2f}{hd_info} "
          f"RE:{features.get('re', 0) if features else 0:.2f} "
          f"GF:{features.get('gf', 0) if features else 0:.2f} "
          f"hist:{len(fall_state.get('trigger_history', []))} "
          f"guard:{'BLOCK' if initial_ar is not None and initial_ar < MIN_STANDING_AR else 'OK'}")

    # 保护：初始状态不是站立（AR < 1.2）→ 跳过 AR 变化检测，但允许物理特征检测
    initial_ar_low = initial_ar is not None and initial_ar < MIN_STANDING_AR

    # 三路检测（任一路径触发即为"可能跌倒"）
    current_ar = person_data.get('aspect_ratio')

    # 路径 1: 几何检测（AR + 角度同时变化）→ 前倒/后倒/侧倒
    # 初始 AR 太低（非站立）时跳过，因为 AR 变化不可靠
    if initial_ar_low:
        rule_triggered = False
    else:
        rule_triggered = check_rule_based(
            person_data.get('kp_5'), current_ar, ak,
            bbox_area=bbox_area, initial_ar=initial_ar,
            smoothed_angle=hip_angle_smoothed
        )

    # 路径 2: 物理检测（RE 或 GF 强信号）→ 快速摔倒/蜷缩倒
    physics_triggered = check_physical_features(features)

    # 路径 3: AR 剧变 + 头部下降 → 侧倒（角度可能不够低）
    ar_dramatic = False
    if not initial_ar_low and initial_ar is not None and initial_ar > 0 and current_ar is not None:
        ar_ratio = current_ar / initial_ar
        if ar_ratio < 0.35:
            ar_dramatic = True
    # 初始 AR 低时：只要当前 AR 很低且有头部下降，也算触发
    elif initial_ar_low and current_ar is not None and current_ar < HORIZONTAL_AR_THRESHOLD:
        ar_dramatic = True
    head_drop_triggered = features and features.get('head_descent', 0) > HEAD_DESCENT_THRESHOLD
    side_fall_triggered = ar_dramatic and head_drop_triggered

    # 路径 4: 已经在地上（AR 很低且持续多帧）→ 初次检测到已在地面的人
    already_down = False
    if initial_ar_low and current_ar is not None and current_ar < HORIZONTAL_AR_THRESHOLD:
        hist_len = len(fall_state.get('trigger_history', []))
        if hist_len >= 5:
            already_down = True

    # 任一路径触发则标记为"可能跌倒"
    is_potential_fall = bool(rule_triggered or physics_triggered or side_fall_triggered or already_down)

    # 路径诊断（只在有触发时打印，减少噪音）
    if is_potential_fall or fall_state.get('is_potential_fall'):
        ar_ratio_val = current_ar / initial_ar if initial_ar and current_ar else 0
        print(f"  [PATH] P1:{rule_triggered} P2:{physics_triggered} P3:{side_fall_triggered} P4:{already_down} "
              f"ar_ratio:{ar_ratio_val:.2f} guard:{'LOW' if initial_ar_low else 'OK'} "
              f"win:{sum(fall_state.get('trigger_history',[]))}/{len(fall_state.get('trigger_history',[]))} "
              f"pot:{fall_state.get('is_potential_fall')} det:{fall_state.get('fall_detected')}")

    # 滑动时间窗口：记录最近 N 帧的触发状态
    if 'trigger_history' not in fall_state:
        fall_state['trigger_history'] = []

    fall_state['trigger_history'].append(is_potential_fall)
    if len(fall_state['trigger_history']) > WINDOW_SIZE:
        fall_state['trigger_history'] = fall_state['trigger_history'][-WINDOW_SIZE:]

    trigger_ratio = sum(fall_state['trigger_history']) / len(fall_state['trigger_history'])
    window_triggered = trigger_ratio >= WINDOW_TRIGGER_RATIO

    if window_triggered:
        if not fall_state['is_potential_fall']:
            fall_state['is_potential_fall'] = True
            fall_state['fall_start_time'] = current_time
        # 记录头部最低位置（y 越大越低）
        curr_kp = person_data.get('kp_5')
        if curr_kp is not None:
            head_y = curr_kp['H'][1]
            if 'min_head_y' not in fall_state or head_y > fall_state['min_head_y']:
                fall_state['min_head_y'] = head_y
    else:
        # 已确认跌倒后，滑动窗口失效不应重置状态（人躺在地上时物理信号会消失）
        if not fall_state.get('fall_detected', False):
            if fall_state['is_potential_fall']:
                fall_state['is_potential_fall'] = False
                fall_state['fall_start_time'] = None
                fall_state['trigger_history'] = []
                fall_state.pop('min_head_y', None)

    # 持续时间确认 + 头部回弹检测
    fall_detected = fall_state.get('fall_detected', False)  # 保持之前的跌倒状态
    if fall_state['is_potential_fall'] and fall_state['fall_start_time'] is not None:
        duration = current_time - fall_state['fall_start_time']
        if duration >= MIN_FALL_POSE_DURATION:
            # 检查头部是否已回弹（弯腰后站起 vs 真正摔倒）
            curr_kp = person_data.get('kp_5')
            rebound = False
            if curr_kp is not None and 'min_head_y' in fall_state:
                head_drop = fall_state['min_head_y'] - curr_kp['H'][1]
                body_h = abs(curr_kp['H'][1] - (curr_kp['KL'][1] + curr_kp['KR'][1]) / 2)
                if body_h > 10 and head_drop / body_h > 0.1:
                    rebound = True
            if not rebound:
                fall_detected = True

    # 高置信度快速通道：四个条件同时强力触发时，跳过持续时间要求
    if not fall_detected and features is not None:
        ar = person_data.get('aspect_ratio')
        fast_angle = hip_angle_smoothed > 0 and hip_angle_smoothed < FAST_ANGLE_THRESHOLD

        if (ar is not None and ar < HORIZONTAL_AR_THRESHOLD and
            fast_angle and
            features.get('re', 0) > FAST_RE_THRESHOLD and
            features.get('gf', 0) > FAST_GF_THRESHOLD):
            fall_detected = True

    # 恢复检测：如果已确认跌倒但人站起来了，重置状态
    if fall_detected and initial_ar is not None and initial_ar > MIN_STANDING_AR:
        current_ar = person_data.get('aspect_ratio')
        if current_ar is not None and current_ar > initial_ar * 0.7:
            # AR 恢复到站立基线的 70% 以上，认为人已站起
            fall_detected = False
            fall_state['is_potential_fall'] = False
            fall_state['fall_start_time'] = None
            fall_state['trigger_history'] = []
            fall_state.pop('min_head_y', None)

    fall_state['fall_detected'] = fall_detected

    # 确定状态和置信度
    if fall_detected:
        if DUAL_CAM_CONFIRM and dual_cam_fall is not None:
            # 双摄像头交叉验证
            if dual_cam_fall:
                confidence = DUAL_CAM_FALL_CONFIDENCE
                state = "FALL (Confirmed)"
            else:
                confidence = SINGLE_CAM_FALL_CONFIDENCE
                state = "FALL (Single Cam)"
        else:
            confidence = SINGLE_CAM_FALL_CONFIDENCE
            state = "FALL"
    elif fall_state['is_potential_fall']:
        confidence = 0.3
        state = "Potential Fall"
    else:
        confidence = 0.0
        state = "Normal"

    return {
        'fall_detected': fall_detected,
        'confidence': confidence,
        'state': state,
        'features': features,
    }
