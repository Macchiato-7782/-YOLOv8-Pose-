"""
跨摄像头人物匹配模块
HSV 直方图 + 匈牙利算法（scipy.optimize.linear_sum_assignment）
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import CROSS_CAM as _CFG

# 从配置文件读取参数
HIST_THRESHOLD = _CFG.get('hist_threshold', 0.5)
HIST_MISMATCH_THRESHOLD = _CFG.get('hist_mismatch_threshold', 0.2)


def get_color_histogram(img, bbox, nbins=None):
    """
    计算人体上半身区域的 HSV 颜色直方图

    只取上半身（bbox 的上半部分），避免腿部遮挡和裤子颜色干扰
    """
    if nbins is None:
        nbins = _CFG.get('hist_nbins', 8)

    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # 只取上半身
    upper_y2 = y1 + h // 2

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

    使用匈牙利算法（linear_sum_assignment）做对称全局最优匹配

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

    # 匈牙利算法：求最大匹配（转为最小化问题，取负相关性）
    # 用大值填充无效位置，防止匹配到无直方图的对
    cost_matrix = np.where(corr_matrix > 0, -corr_matrix, 1.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 只保留相关性高于阈值的匹配
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if corr_matrix[r][c] > threshold:
            matched_pairs.append((r, c))

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


def merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map, pid_to_global=None):
    """
    合并两个摄像头中同一个人的跟踪 ID

    global_id_map: dict, key=(cam_a_pid, cam_b_pid), value=global_id
    pid_to_global: dict, key=pid, value=global_id（反向索引，O(1)查找）
    """
    if pid_to_global is None:
        # 从 global_id_map 构建反向索引
        pid_to_global = {}
        for (pa, pb), gid in global_id_map.items():
            pid_to_global[pa] = gid
            pid_to_global[pb] = gid

    next_global_id = max(global_id_map.values(), default=0) + 1

    for a_idx, b_idx in matched_pairs:
        pid_a = tracked_a[a_idx]['pid']
        pid_b = tracked_b[b_idx]['pid']

        key = (pid_a, pid_b)
        if key not in global_id_map:
            # 用反向索引 O(1) 查找已有 ID
            existing = pid_to_global.get(pid_a) or pid_to_global.get(pid_b)

            if existing is not None:
                global_id_map[key] = existing
                pid_to_global[pid_a] = existing
                pid_to_global[pid_b] = existing
            else:
                global_id_map[key] = next_global_id
                pid_to_global[pid_a] = next_global_id
                pid_to_global[pid_b] = next_global_id
                next_global_id += 1

    return pid_to_global
