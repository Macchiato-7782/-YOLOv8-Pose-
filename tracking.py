"""
单摄像头内的跟踪模块
使用 Ultralytics 内置 ByteTracker（卡尔曼滤波 + 级联匹配）
保留跌倒状态管理和幽灵目标机制
"""

import numpy as np
from features import yolo_to_5keypoints
from config import TRACKING as _CFG

# 从配置文件读取参数
REQUIRED_KPTS_FOR_FULL_BODY = _CFG.get('required_kpts_for_full_body', [0, 5, 6, 11, 12, 15, 16])
MIN_KPTS_FOR_FULL_BODY = _CFG.get('min_kpts_for_full_body', 4)
MIN_CONF_KPT = _CFG.get('min_conf_kpt', 0.2)
GHOST_TIMEOUT = _CFG.get('ghost_timeout', 3.0)
GHOST_TIMEOUT_FALLEN = _CFG.get('ghost_timeout_fallen', 30.0)


class SingleCameraTracker:
    """基于 Ultralytics ByteTracker 的跟踪器，管理跌倒状态和幽灵目标"""

    def __init__(self, cleanup_interval=None, history_length=None, ghost_timeout=GHOST_TIMEOUT):
        self.cleanup_interval = cleanup_interval or _CFG.get('cleanup_interval', 5)
        self.history_length = history_length or _CFG.get('history_length', 36)
        self.ghost_timeout = ghost_timeout
        self.person_history = {}
        self.last_cleanup_time = 0

    def extract_detections(self, results):
        """
        从 Ultralytics ByteTracker 结果中提取检测信息

        results: model.track() 返回的 Results 对象，包含 boxes.id 和 keypoints

        返回: list of dict, 每个 dict 包含:
            - 'track_id': ByteTracker 分配的跟踪 ID (int)
            - 'keypoints': 17 关键点坐标 (17, 2)
            - 'confs': 17 关键点置信度 (17,)
            - 'kp_5': 映射后的 5 关键点 dict
            - 'bbox': (x1, y1, x2, y2) 来自 YOLO 检测框
            - 'center': (cx, cy)
            - 'aspect_ratio': h/w
            - 'is_full_body': bool
        """
        detections = []

        if results.keypoints is None or results.boxes is None:
            return detections
        if results.boxes.id is None:
            return detections

        track_ids = results.boxes.id.cpu().numpy().astype(int)
        bboxes = results.boxes.xyxy.cpu().numpy()

        for i, pose in enumerate(results.keypoints):
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

            # 使用 YOLO 检测框（比关键点算的 bbox 更准确）
            x1, y1, x2, y2 = bboxes[i].astype(int)
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            aspect_ratio = h / w

            # 映射到 5 关键点
            kp_5 = yolo_to_5keypoints(keypoints, confs)
            if kp_5 is not None:
                kp_5 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in kp_5.items()}

            detections.append({
                'track_id': int(track_ids[i]),
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
        用 ByteTracker 的检测结果更新跟踪状态

        ByteTracker 已完成匹配，这里只需：
        1. 更新 person_history 中已有 track 的状态
        2. 为新 track 创建初始状态
        3. 标记消失的 track 为幽灵

        返回: list of dict，格式与之前一致
        """
        current_track_ids = set()
        tracked_results = []

        for det in detections:
            tid = det['track_id']
            current_track_ids.add(tid)
            center = det['center']
            was_ghost = False

            if tid in self.person_history:
                # 已有 track → 更新状态
                data = self.person_history[tid]
                was_ghost = data.get('is_ghost', False)
                data['is_ghost'] = False
                data['center'] = center
                data['bbox'] = det['bbox']
                data['last_seen'] = current_time

                if det['is_full_body'] and det['kp_5'] is not None:
                    data['kp_5'] = det['kp_5']
                    data['history'].append(det['kp_5'])
                    if len(data['history']) > self.history_length:
                        data['history'] = data['history'][-self.history_length:]
            else:
                # 新 track → 检查是否与幽灵目标重叠（ByteTracker 分配了新 ID）
                inherited_fall_state = self._find_ghost_fall_state(det['bbox'])
                if inherited_fall_state is not None:
                    # 继承幽灵目标的跌倒状态
                    fall_state = inherited_fall_state
                else:
                    fall_state = {
                        'is_potential_fall': False,
                        'fall_start_time': None,
                        'fall_detected': False,
                        'trigger_history': [],
                        'min_head_y': 0,
                        'angle_history': [],
                    }
                self.person_history[tid] = {
                    'center': center,
                    'bbox': det['bbox'],
                    'kp_5': det['kp_5'],
                    'history': [],
                    'fall_state': fall_state,
                    'last_seen': current_time,
                }
                data = self.person_history[tid]

            tracked_results.append({
                'pid': tid,
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

        # 未匹配但有活跃跌倒状态的目标（幽灵），继续传递给跌倒检测
        for tid, data in self.person_history.items():
            if tid not in current_track_ids:
                fs = data.get('fall_state', {})
                if fs.get('is_potential_fall') or fs.get('fall_detected'):
                    data['is_ghost'] = True
                    tracked_results.append({
                        'pid': tid,
                        'center': data['center'],
                        'bbox': data['bbox'],
                        'aspect_ratio': 0,
                        'kp_5': data.get('kp_5'),
                        'keypoints': None,
                        'confs': None,
                        'is_full_body': False,
                        'history': data.get('history', []),
                        'fall_state': fs,
                        'is_ghost': True,
                    })

        return tracked_results

    def update_roi(self, roi_detections, current_time):
        """
        将 ROI 二次推理结果关联到已有 track（通过 IoU 匹配）

        roi_detections: 来自 run_roi_inference() 的检测列表
        current_time: 当前时间

        返回: 已关联到 track 的检测列表（补充到 tracked_results 中）
        """
        added = []
        for det in roi_detections:
            best_tid = None
            best_iou = 0
            for tid, data in self.person_history.items():
                iou = self._compute_iou(det['bbox'], data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou > 0.3:
                data = self.person_history[best_tid]
                # 更新关键点（如果 ROI 检测到更好的）
                if det.get('kp_5') is not None:
                    data['kp_5'] = det['kp_5']
                    data['history'].append(det['kp_5'])
                    if len(data['history']) > self.history_length:
                        data['history'] = data['history'][-self.history_length:]
                data['bbox'] = det['bbox']
                data['center'] = det['center']
                data['last_seen'] = current_time

        return added

    def _find_ghost_fall_state(self, new_bbox):
        """检查新检测是否与幽灵目标重叠，如果是则返回其 fall_state（用于继承跌倒状态）"""
        for tid, data in self.person_history.items():
            if not data.get('is_ghost', False):
                continue
            fs = data.get('fall_state', {})
            if not fs.get('fall_detected', False):
                continue
            iou = self._compute_iou(new_bbox, data['bbox'])
            if iou > 0.2:
                # 继承跌倒状态，清理幽灵条目
                del self.person_history[tid]
                return fs
        return None

    @staticmethod
    def _compute_iou(b1, b2):
        """计算两个 bbox 的 IoU"""
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _predict_position(self, data):
        """获取目标中心位置（兼容旧接口，ROI 推理用）"""
        return data['center']

    def cleanup(self, current_time):
        """清理长时间未出现的跟踪目标（幽灵机制）"""
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return

        ids_to_remove = []
        for tid, data in self.person_history.items():
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
                    ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.person_history[tid]

        self.last_cleanup_time = current_time
