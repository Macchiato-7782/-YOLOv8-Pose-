"""
单摄像头内的质心跟踪模块
从 main.py 提取并封装为类
"""

import numpy as np
from features import yolo_to_5keypoints


# YOLOv8 COCO 关键点索引
REQUIRED_KPTS_FOR_FULL_BODY = [0, 5, 6, 11, 12, 15, 16]
MIN_KPTS_FOR_FULL_BODY = 4
MIN_CONF_KPT = 0.2
GHOST_TIMEOUT = 3.0
GHOST_TIMEOUT_FALLEN = 30.0  # 已确认跌倒的目标保持更久
GHOST_DISTANCE_MULTIPLIER = 1.5


class SingleCameraTracker:
    """单摄像头内的多人质心跟踪器"""

    def __init__(self, distance_threshold=150, cleanup_interval=5, history_length=36,
                 ghost_timeout=GHOST_TIMEOUT):
        self.distance_threshold = distance_threshold
        self.cleanup_interval = cleanup_interval
        self.history_length = history_length
        self.ghost_timeout = ghost_timeout
        self.person_history = {}
        self.next_person_id = 1
        self.last_cleanup_time = 0

    def extract_detections(self, results):
        """
        从 YOLO 结果中提取检测信息

        返回: list of dict, 每个 dict 包含:
            - 'keypoints': 17 关键点坐标 (17, 2)
            - 'confs': 17 关键点置信度 (17,)
            - 'kp_5': 映射后的 5 关键点 dict
            - 'bbox': (x1, y1, x2, y2)
            - 'center': (cx, cy)
            - 'aspect_ratio': h/w
            - 'is_full_body': bool
        """
        detections = []

        if results.keypoints is None:
            return detections

        for pose in results.keypoints:
            if pose.conf is None or len(pose.conf) == 0:
                continue
            if pose.xy is None or len(pose.xy) == 0 or len(pose.xy[0]) < 17:
                continue

            keypoints = pose.xy[0].cpu().numpy().copy()
            confs = pose.conf[0].cpu().numpy().copy()

            # 检查全身可见性
            visible_kpts = []
            for idx in REQUIRED_KPTS_FOR_FULL_BODY:
                if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_KPT:
                    visible_kpts.append(keypoints[idx])

            is_full_body = len(visible_kpts) >= MIN_KPTS_FOR_FULL_BODY

            # 计算 bounding box
            if len(visible_kpts) < 2:
                continue

            visible_kpts = np.array(visible_kpts)
            x1, y1 = visible_kpts.min(axis=0).astype(int)
            x2, y2 = visible_kpts.max(axis=0).astype(int)
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            aspect_ratio = h / w

            # 映射到 5 关键点（深拷贝，避免 numpy view 跨进程序列化问题）
            kp_5 = yolo_to_5keypoints(keypoints, confs)
            if kp_5 is not None:
                kp_5 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in kp_5.items()}

            detections.append({
                'keypoints': keypoints,
                'confs': confs,
                'kp_5': kp_5,
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'aspect_ratio': aspect_ratio,
                'is_full_body': is_full_body,
            })

        return detections

    def update(self, detections, current_time):
        """
        用当前帧的检测结果更新跟踪状态

        返回: list of dict, 每个 dict 包含:
            - 'pid': 跟踪 ID
            - 'center': 当前中心点
            - 'bbox': 当前 bbox
            - 'aspect_ratio': 当前宽高比
            - 'kp_5': 当前 5 关键点
            - 'is_full_body': 是否全身可见
            - 'history': 历史 5 关键点列表
            - 'fall_state': dict 包含跌倒状态信息
        """
        matched_pids = set()
        tracked_results = []

        for det in detections:
            center = det['center']
            matched_id = None
            was_ghost = False

            # 匹配已有跟踪目标（包括幽灵目标）
            for pid, data in self.person_history.items():
                if pid in matched_pids:
                    continue
                dist = np.sqrt((center[0] - data['center'][0])**2 +
                               (center[1] - data['center'][1])**2)
                threshold = self.distance_threshold
                if data.get('is_ghost', False):
                    threshold *= GHOST_DISTANCE_MULTIPLIER
                if dist < threshold:
                    matched_id = pid
                    break

            # 新目标（必须全身可见）
            if matched_id is None and det['is_full_body']:
                matched_id = self.next_person_id
                self.next_person_id += 1
                self.person_history[matched_id] = {
                    'center': center,
                    'bbox': det['bbox'],
                    'kp_5': det['kp_5'],
                    'history': [],
                    'fall_state': {
                        'is_potential_fall': False,
                        'fall_start_time': None,
                        'fall_detected': False,
                        'trigger_history': [],
                        'min_head_y': 0,
                        'angle_history': [],
                    },
                    'last_seen': current_time,
                }
                matched_pids.add(matched_id)

            if matched_id is not None:
                data = self.person_history[matched_id]
                was_ghost = data.get('is_ghost', False)
                data['is_ghost'] = False
                data['center'] = center
                data['bbox'] = det['bbox']
                data['last_seen'] = current_time

                if det['is_full_body'] and det['kp_5'] is not None:
                    matched_pids.add(matched_id)
                    data['kp_5'] = det['kp_5']
                    data['history'].append(det['kp_5'])
                    # 保留最近 N 帧
                    if len(data['history']) > self.history_length:
                        data['history'] = data['history'][-self.history_length:]

                tracked_results.append({
                    'pid': matched_id,
                    'center': center,
                    'bbox': det['bbox'],
                    'aspect_ratio': det['aspect_ratio'],
                    'kp_5': det['kp_5'],
                    'keypoints': det['keypoints'],
                    'confs': det['confs'],
                    'is_full_body': det['is_full_body'],
                    'history': data['history'],
                    'fall_state': data['fall_state'],
                    'recovered_from_ghost': was_ghost,
                })

        return tracked_results

    def cleanup(self, current_time):
        """清理长时间未出现的跟踪目标（幽灵机制）"""
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return

        ids_to_remove = []
        for pid, data in self.person_history.items():
            elapsed = current_time - data['last_seen']

            if elapsed > self.cleanup_interval and not data.get('is_ghost', False):
                # 超时 → 标记为幽灵
                data['is_ghost'] = True
                data['ghost_start_time'] = current_time

            elif data.get('is_ghost', False):
                # 已是幽灵 → 检查是否超过幽灵超时（跌倒目标延长）
                ghost_elapsed = current_time - data.get('ghost_start_time', data['last_seen'])
                timeout = self.ghost_timeout
                if data.get('fall_state', {}).get('fall_detected', False):
                    timeout = GHOST_TIMEOUT_FALLEN
                if ghost_elapsed > timeout:
                    ids_to_remove.append(pid)

        for pid in ids_to_remove:
            del self.person_history[pid]

        self.last_cleanup_time = current_time
