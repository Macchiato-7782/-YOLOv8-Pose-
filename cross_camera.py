"""
跨摄像头人物匹配模块
从 HumanFallDetection 项目移植: HSV 直方图 + 稳定婚姻算法
"""

import cv2
import numpy as np

# 直方图匹配阈值
HIST_THRESHOLD = 0.5       # > 此值认为是同一人
HIST_MISMATCH_THRESHOLD = 0.2  # < 此值拆开已匹配对


def get_color_histogram(img, bbox, nbins=3):
    """
    计算人体上半身区域的 HSV 颜色直方图
    来自 HumanFallDetection helpers.py: get_hist

    只取上半身（bbox 的上半部分），避免腿部遮挡和裤子颜色干扰
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # 只取上半身
    upper_y2 = y1 + h // 2
    upper_bbox = (x1, y1, x2, upper_y2)

    # 创建掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:upper_y2, x1:x2] = 1

    if mask.sum() == 0:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [nbins, 2 * nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)

    return hist


def match_cross_camera(tracked_a, tracked_b, threshold=HIST_THRESHOLD):
    """
    跨摄像头人物匹配

    使用稳定婚姻算法（Gale-Shapley）做全局最优匹配

    tracked_a: 摄像头 A 的跟踪结果列表
    tracked_b: 摄像头 B 的跟踪结果列表

    返回: list of (idx_a, idx_b) 匹配对
    """
    if not tracked_a or not tracked_b:
        return []

    # 计算相关性矩阵
    n_a = len(tracked_a)
    n_b = len(tracked_b)
    corr_matrix = np.zeros((n_a, n_b))

    for i in range(n_a):
        hist_a = tracked_a[i].get('hist')
        if hist_a is None:
            continue
        for j in range(n_b):
            hist_b = tracked_b[j].get('hist')
            if hist_b is None:
                continue
            corr_matrix[i][j] = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    # 稳定婚姻算法
    # tracked_a 的人按偏好排序（相关性从高到低）
    preferences = np.argsort(-corr_matrix, axis=1)

    freelist = list(range(n_a))
    next_proposal = [0] * n_a  # 每个人下一个要"求婚"的对象索引
    engaged_b = [-1] * n_b  # tracked_b 中每个人当前的配对
    finish = [False] * n_a

    while freelist:
        a_idx = freelist[-1]

        if finish[a_idx]:
            freelist.pop()
            continue

        if next_proposal[a_idx] >= n_b:
            finish[a_idx] = True
            freelist.pop()
            continue

        b_idx = preferences[a_idx][next_proposal[a_idx]]
        next_proposal[a_idx] += 1

        if engaged_b[b_idx] == -1:
            # 对方单身，直接配对
            engaged_b[b_idx] = a_idx
            freelist.pop()
        else:
            # 对方已有配对，比谁更匹配
            current_a = engaged_b[b_idx]
            if corr_matrix[a_idx][b_idx] > corr_matrix[current_a][b_idx]:
                engaged_b[b_idx] = a_idx
                freelist.pop()
                if not finish[current_a]:
                    freelist.append(current_a)
        # else: a_idx 继续找下一个

    # 提取匹配结果（只保留相关性高于阈值的）
    matched_pairs = []
    for b_idx, a_idx in enumerate(engaged_b):
        if a_idx >= 0 and corr_matrix[a_idx][b_idx] > threshold:
            matched_pairs.append((a_idx, b_idx))

    return matched_pairs


def remove_wrongly_matched(tracked_a, tracked_b, matched_pairs, threshold=HIST_MISMATCH_THRESHOLD):
    """
    修正错误匹配：检查已匹配对的直方图相关性，低于阈值则拆开

    返回: 修正后的匹配对列表
    """
    valid_pairs = []
    for a_idx, b_idx in matched_pairs:
        hist_a = tracked_a[a_idx].get('hist')
        hist_b = tracked_b[b_idx].get('hist')

        if hist_a is None or hist_b is None:
            continue

        corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        if corr > threshold:
            valid_pairs.append((a_idx, b_idx))

    return valid_pairs


def merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map):
    """
    合并两个摄像头中同一个人的跟踪 ID

    global_id_map: dict, key=(cam_a_pid, cam_b_pid), value=global_id
    """
    next_global_id = max(global_id_map.values(), default=0) + 1

    for a_idx, b_idx in matched_pairs:
        pid_a = tracked_a[a_idx]['pid']
        pid_b = tracked_b[b_idx]['pid']

        key = (pid_a, pid_b)
        if key not in global_id_map:
            # 检查是否有一方已经有全局 ID
            existing = None
            for (pa, pb), gid in global_id_map.items():
                if pa == pid_a or pb == pid_b:
                    existing = gid
                    break

            if existing is not None:
                global_id_map[key] = existing
            else:
                global_id_map[key] = next_global_id
                next_global_id += 1
