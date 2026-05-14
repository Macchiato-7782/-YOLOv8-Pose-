"""
单摄像头内的跟踪模块
Backend 无关：接收统一 detection schema，管理跌倒状态和幽灵目标
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
IOU_MATCH_THRESHOLD = _CFG.get('distance_threshold', 150)  # 复用配置中的距离阈值做 IoU 匹配


class SingleCameraTracker:
    """Backend 无关的跟踪器，通过 IoU 匹配管理目标 ID 和跌倒状态"""

    def __init__(self, cleanup_interval=None, history_length=None, ghost_timeout=GHOST_TIMEOUT):
        self.cleanup_interval = cleanup_interval or _CFG.get('cleanup_interval', 5)
        self.history_length = history_length or _CFG.get('history_length', 36)
        self.ghost_timeout = ghost_timeout
        self.person_history = {}
        self.last_cleanup_time = 0
        self._next_assign_id = 1

    def convert_backend_detections(self, detections):
        """
        将后端统一 detection schema 转换为跟踪器内部格式

        Args:
            detections: list[dict], 后端 inference() 返回的统一检测列表

        Returns:
            list[dict]: 跟踪器内部格式（兼容原有 update 逻辑）
        """
        converted = []

        for det in detections:
            # 提取关键点
            kps_raw = det.get("keypoints", [])
            if not kps_raw or len(kps_raw) < 17:
                continue

            keypoints = np.array([[k[0], k[1]] for k in kps_raw[:17]], dtype=np.float64)
            confs = np.array([k[2] for k in kps_raw[:17]], dtype=np.float64)

            # 检查全身可见性
            visible_kpts = []
            for idx in REQUIRED_KPTS_FOR_FULL_BODY:
                if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_KPT:
                    visible_kpts.append(keypoints[idx])

            is_full_body = len(visible_kpts) >= MIN_KPTS_FOR_FULL_BODY

            # bbox
            bbox_raw = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3])
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            aspect_ratio = h / w if w > 0 else 0

            # 映射到 5 关键点
            kp_5 = yolo_to_5keypoints(keypoints, confs)
            if kp_5 is not None:
                kp_5 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in kp_5.items()}

            converted.append({
                'track_id': det.get("track_id"),
                'keypoints': keypoints,
                'confs': confs,
                'kp_5': kp_5,
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'aspect_ratio': aspect_ratio,
                'is_full_body': is_full_body,
            })

        return converted

    def _assign_ids(self, detections, current_time):
        """
        为没有 track_id 的检测分配 ID（IoU 匹配）

        已有 track_id 的检测直接使用；没有的通过 IoU 匹配已有 track。
        """
        unmatched_dets = []
        unmatched_tids = set(self.person_history.keys())

        # 先处理已有 track_id 的检测
        for det in detections:
            tid = det.get('track_id')
            if tid is not None and tid in self.person_history:
                det['track_id'] = int(tid)
                unmatched_tids.discard(int(tid))
            elif tid is not None:
                det['track_id'] = int(tid)
            else:
                unmatched_dets.append(det)

        if not unmatched_dets:
            return detections

        # 对没有 track_id 的检测做 IoU 匹配
        # 收集未匹配 track 的最近 bbox
        unmatched_track_bboxes = {}
        for tid in unmatched_tids:
            data = self.person_history[tid]
            if not data.get('is_ghost', False):
                unmatched_track_bboxes[tid] = data['bbox']

        # IoU 匹配
        matched_tids = set()
        for det in unmatched_dets:
            best_tid = None
            best_iou = 0.0
            for tid, tbbox in unmatched_track_bboxes.items():
                if tid in matched_tids:
                    continue
                iou = self._compute_iou(det['bbox'], tbbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou > 0.2:
                det['track_id'] = int(best_tid)
                matched_tids.add(best_tid)
            else:
                # 新目标
                det['track_id'] = self._next_assign_id
                self._next_assign_id += 1

        return detections

    def update(self, detections, current_time):
        """
        用检测结果更新跟踪状态

        Args:
            detections: list[dict], 跟踪器内部格式（来自 convert_backend_detections）
            current_time: 当前时间戳

        Returns:
            list[dict]: 带完整状态的跟踪目标列表
        """
        detections = self._assign_ids(detections, current_time)

        current_track_ids = set()
        tracked_results = []

        for det in detections:
            tid = det['track_id']
            current_track_ids.add(tid)
            center = det['center']
            was_ghost = False

            if tid in self.person_history:
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
                inherited_fall_state = self._find_ghost_fall_state(det['bbox'])
                if inherited_fall_state is not None:
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

        # 幽灵目标
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

        Args:
            roi_detections: 来自 run_roi_inference() 的检测列表（跟踪器内部格式）
            current_time: 当前时间
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
        """检查新检测是否与幽灵目标重叠，如果是则返回其 fall_state"""
        for tid, data in self.person_history.items():
            if not data.get('is_ghost', False):
                continue
            fs = data.get('fall_state', {})
            if not fs.get('fall_detected', False):
                continue
            iou = self._compute_iou(new_bbox, data['bbox'])
            if iou > 0.2:
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
        """获取目标中心位置"""
        return data['center']

    def cleanup(self, current_time):
        """清理长时间未出现的跟踪目标"""
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return

        ids_to_remove = []
        for tid, data in self.person_history.items():
            elapsed = current_time - data['last_seen']

            if elapsed > self.cleanup_interval and not data.get('is_ghost', False):
                data['is_ghost'] = True
                data['ghost_start_time'] = current_time

            elif data.get('is_ghost', False):
                ghost_elapsed = current_time - data.get('ghost_start_time', data['last_seen'])
                timeout = self.ghost_timeout
                if data.get('fall_state', {}).get('fall_detected', False):
                    timeout = GHOST_TIMEOUT_FALLEN
                if ghost_elapsed > timeout:
                    ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.person_history[tid]

        self.last_cleanup_time = current_time
