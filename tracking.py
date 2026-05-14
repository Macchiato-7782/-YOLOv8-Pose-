"""
单摄像头内的跟踪模块
Pure Runtime: 输入 list[Detection]，输出 list[TrackState]
Backend 无关，内部无 dict bridge
"""

import numpy as np
from features import yolo_to_5keypoints
from config import TRACKING as _CFG

REQUIRED_KPTS_FOR_FULL_BODY = _CFG.get('required_kpts_for_full_body', [0, 5, 6, 11, 12, 15, 16])
MIN_KPTS_FOR_FULL_BODY = _CFG.get('min_kpts_for_full_body', 4)
MIN_CONF_KPT = _CFG.get('min_conf_kpt', 0.2)
GHOST_TIMEOUT = _CFG.get('ghost_timeout', 3.0)
GHOST_TIMEOUT_FALLEN = _CFG.get('ghost_timeout_fallen', 30.0)


class SingleCameraTracker:
    """Pure Runtime 跟踪器：Detection → TrackState，无 dict"""

    def __init__(self, cleanup_interval=None, history_length=None, ghost_timeout=GHOST_TIMEOUT):
        self.cleanup_interval = cleanup_interval or _CFG.get('cleanup_interval', 5)
        self.history_length = history_length or _CFG.get('history_length', 36)
        self.ghost_timeout = ghost_timeout
        self.person_history: dict = {}
        self.last_cleanup_time = 0
        self._next_assign_id = 1

    def update(self, detections, current_time: float):
        """
        用 Detection 列表更新跟踪状态

        Args:
            detections: list[Detection]
            current_time: 当前时间戳

        Returns:
            list[TrackState]
        """
        from fall_detection.core.track import TrackState
        # 分配 ID
        assigned = self._assign_ids(detections, current_time)

        current_track_ids = set()
        tracked_results = []

        for det in assigned:
            tid = det.track_id
            current_track_ids.add(tid)

            # 计算内部字段
            bbox = det.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            w, h = x2 - x1, y2 - y1
            aspect_ratio = h / w if w > 0 else 0.0

            # 检查全身可见性 & 5 关键点
            keypoints = None
            confs = None
            kp_5 = None
            is_full_body = False

            if det.keypoints and len(det.keypoints) >= 17:
                kps_arr = np.array([[k.x, k.y] for k in det.keypoints[:17]], dtype=np.float64)
                confs_arr = np.array([k.confidence for k in det.keypoints[:17]], dtype=np.float64)

                visible = []
                for idx in REQUIRED_KPTS_FOR_FULL_BODY:
                    if idx < len(kps_arr) and kps_arr[idx][0] > 0 and kps_arr[idx][1] > 0 and confs_arr[idx] > MIN_CONF_KPT:
                        visible.append(kps_arr[idx])
                is_full_body = len(visible) >= MIN_KPTS_FOR_FULL_BODY

                kp_5 = yolo_to_5keypoints(kps_arr, confs_arr)
                if kp_5 is not None:
                    kp_5 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in kp_5.items()}
                keypoints = kps_arr
                confs = confs_arr

            was_ghost = False
            bbox_tuple = (x1, y1, x2, y2)

            if tid in self.person_history:
                data = self.person_history[tid]
                was_ghost = data.get('is_ghost', False)
                data['is_ghost'] = False
                data['center'] = center
                data['bbox'] = bbox_tuple
                data['last_seen'] = current_time
                data['keypoints'] = keypoints
                data['confs'] = confs

                if is_full_body and kp_5 is not None:
                    data['kp_5'] = kp_5
                    data['history'].append(kp_5)
                    if len(data['history']) > self.history_length:
                        data['history'] = data['history'][-self.history_length:]
            else:
                inherited_fs = self._find_ghost_fall_state(bbox_tuple)
                fall_state = inherited_fs or {
                    'is_potential_fall': False,
                    'fall_start_time': None,
                    'fall_detected': False,
                    'trigger_history': [],
                    'min_head_y': 0,
                    'angle_history': [],
                }
                self.person_history[tid] = {
                    'center': center,
                    'bbox': bbox_tuple,
                    'kp_5': kp_5,
                    'history': [],
                    'fall_state': fall_state,
                    'last_seen': current_time,
                    'keypoints': keypoints,
                    'confs': confs,
                }
                data = self.person_history[tid]

            # 构建 TrackState
            state = "normal"
            fs = data.get('fall_state', {})
            if fs.get('fall_detected'):
                state = "fall"
            elif fs.get('is_potential_fall'):
                state = "potential_fall"

            kps_obj = det.keypoints  # Keypoint objects
            ts = TrackState(
                track_id=tid,
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                center=[float(center[0]), float(center[1])],
                confidence=float(fs.get('confidence', det.score)),
                state=state,
                keypoints=kps_obj,
                is_ghost=False,
                aspect_ratio=float(aspect_ratio),
                fall_detected=bool(fs.get('fall_detected', False)),
                history=data.get('history', []),
                fall_state=fs,
            )
            # attach raw keypoints for evaluate_fall bridge
            ts._raw_keypoints = keypoints
            ts._raw_confs = confs
            ts._kp_5 = kp_5
            ts._bbox_tuple = bbox_tuple
            tracked_results.append(ts)

        # 幽灵目标
        for tid, data in self.person_history.items():
            if tid not in current_track_ids:
                fs = data.get('fall_state', {})
                if fs.get('is_potential_fall') or fs.get('fall_detected'):
                    data['is_ghost'] = True
                    state = "fall" if fs.get('fall_detected') else "potential_fall"
                    ts = TrackState(
                        track_id=tid,
                        bbox=[float(v) for v in data['bbox']],
                        center=[float(data['center'][0]), float(data['center'][1])],
                        confidence=float(fs.get('confidence', 0)),
                        state=state,
                        keypoints=[],
                        is_ghost=True,
                        aspect_ratio=0.0,
                        fall_detected=bool(fs.get('fall_detected', False)),
                        fall_state=fs,
                    )
                    ts._raw_keypoints = data.get('keypoints')
                    ts._raw_confs = data.get('confs')
                    ts._kp_5 = data.get('kp_5')
                    ts._bbox_tuple = data['bbox']
                    tracked_results.append(ts)

        return tracked_results

    def _assign_ids(self, detections: list, current_time: float) -> list:
        """为没有 track_id 的 Detection 分配 ID"""
        unmatched_dets = []
        unmatched_tids = set(self.person_history.keys())

        for det in detections:
            tid = det.track_id
            if tid is not None and tid in self.person_history:
                det.track_id = int(tid)
                unmatched_tids.discard(int(tid))
            elif tid is not None:
                det.track_id = int(tid)
            else:
                unmatched_dets.append(det)

        if not unmatched_dets:
            return detections

        # IoU 匹配
        unmatched_bboxes = {}
        for tid in unmatched_tids:
            data = self.person_history[tid]
            if not data.get('is_ghost', False):
                unmatched_bboxes[tid] = data['bbox']

        matched_tids = set()
        for det in unmatched_dets:
            best_tid = None
            best_iou = 0.0
            det_bbox = tuple(int(v) for v in det.bbox)
            for tid, tbbox in unmatched_bboxes.items():
                if tid in matched_tids:
                    continue
                iou = self._compute_iou(det_bbox, tbbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou > 0.2:
                det.track_id = int(best_tid)
                matched_tids.add(best_tid)
            else:
                det.track_id = self._next_assign_id
                self._next_assign_id += 1

        return detections

    def update_roi(self, roi_detections, current_time):
        """ROI 二次推理结果关联"""
        for det in roi_detections:
            best_tid = None
            best_iou = 0
            det_bbox = tuple(int(v) for v in det.bbox) if hasattr(det, 'bbox') else None
            if det_bbox is None:
                continue
            for tid, data in self.person_history.items():
                iou = self._compute_iou(det_bbox, data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou > 0.3:
                data = self.person_history[best_tid]
                kp_5 = getattr(det, '_kp_5', None)
                if kp_5 is not None:
                    data['kp_5'] = kp_5
                    data['history'].append(kp_5)
                    if len(data['history']) > self.history_length:
                        data['history'] = data['history'][-self.history_length:]
                data['bbox'] = det_bbox
                data['last_seen'] = current_time

        return []

    def _find_ghost_fall_state(self, new_bbox):
        for tid, data in self.person_history.items():
            if not data.get('is_ghost', False):
                continue
            fs = data.get('fall_state', {})
            if not fs.get('fall_detected', False):
                continue
            if self._compute_iou(new_bbox, data['bbox']) > 0.2:
                del self.person_history[tid]
                return fs
        return None

    @staticmethod
    def _compute_iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _predict_position(self, data):
        return data.get('center', (0, 0))

    def cleanup(self, current_time):
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
