"""
Pure Runtime Detection Pipeline
infer → Detection → tracking → TrackState → fall logic → Event → serialize
内部完全使用 dataclass 对象，无 dict，无 numpy 泄露
"""

import time
import logging
import traceback
from typing import Optional
import numpy as np

import cv2

from fall_detection.core.detection import Detection, Keypoint
from fall_detection.core.track import TrackState
from fall_detection.core.event import Event
from fall_detection.core.frame import FrameContext
from fall_detection.core.serializers import serialize_result
from fall_detection.core.runtime import EventRuntime
from fall_detection.core.validators import validate_detection, validate_track, validate_event

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Pure Runtime Pipeline: object-in, object-out, object-throughout"""

    def __init__(
        self,
        backend,
        tracker,
        enable_roi: bool = False,
        max_persons: Optional[int] = None,
        enable_visualization: bool = False,
    ):
        self._backend = backend
        self._tracker = tracker
        self._enable_roi = enable_roi
        self._max_persons = max_persons
        self._enable_visualization = enable_visualization
        self._event_runtime = EventRuntime(cooldown_seconds=5.0)

    def run(self, frame, frame_context: FrameContext) -> dict:
        """执行完整流水线，返回 JSON-friendly dict"""

        # Stage 1: Backend → Detection
        detections = self._backend.infer(frame)
        for d in detections:
            ok, msg = validate_detection(d)
            if not ok:
                raise TypeError(f"Backend returned invalid Detection: {msg}")

        # Stage 2: Tracking → TrackState
        tracks = self._tracker.update(detections, frame_context.timestamp)
        for t in tracks:
            ok, msg = validate_track(t)
            if not ok:
                raise TypeError(f"Tracker returned invalid TrackState: {msg}")

        # 限制人数
        if self._max_persons is not None and len(tracks) > self._max_persons:
            tracks.sort(key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]), reverse=True)
            keep_ids = {t.track_id for t in tracks[:self._max_persons]}
            tracks = [t for t in tracks if t.track_id in keep_ids]

        self._tracker.cleanup(frame_context.timestamp)

        # Stage 3: Fall Logic → updated TrackState
        tracks = self._run_fall_logic(tracks, frame_context)

        # Stage 4: Event Generation → Event
        events = self._event_runtime.process_tracks(tracks, frame_context)
        for e in events:
            validate_event(e)

        # Stage 5: Visualization (optional)
        annotated = None
        if self._enable_visualization:
            annotated = self._visualize(frame, tracks, frame_context)

        # Stage 6: Serialization (only here, final output)
        diagnostics = {
            "fps": frame_context.fps or 0,
            "backend": getattr(self._backend, 'backend', getattr(self._backend, '__class__.__name__', 'unknown')),
            "device": getattr(self._backend, 'device', 'cpu'),
            "inference_ran": True,
        }
        result = serialize_result(
            tracked=tracks,
            events=events,
            camera_id=frame_context.camera_id,
            timestamp=frame_context.timestamp,
            frame_id=frame_context.frame_id,
            diagnostics=diagnostics,
        )

        if annotated is not None:
            result["annotated_frame"] = annotated

        return result

    def _run_fall_logic(self, tracks: list, ctx: FrameContext) -> list:
        """对每个 TrackState 执行跌倒判断"""
        from fall_logic import evaluate_fall
        from camera_process import extract_angle_keypoints

        prev_time = getattr(self, '_last_fall_time', None)
        dt = (ctx.timestamp - prev_time) if prev_time else None
        self._last_fall_time = ctx.timestamp

        for t in tracks:
            if t.is_ghost:
                continue
            try:
                # 构建 evaluate_fall 兼容的 person_data（唯一合法 dict bridge，算法要求）
                kps_arr = getattr(t, '_raw_keypoints', None)
                confs_arr = getattr(t, '_raw_confs', None)
                if kps_arr is None and t.keypoints:
                    kps_list = []
                    confs_list = []
                    for k in t.keypoints:
                        kps_list.append([k.x, k.y])
                        confs_list.append(k.confidence)
                    kps_arr = np.array(kps_list, dtype=np.float64)
                    confs_arr = np.array(confs_list, dtype=np.float64)

                person_data = {
                    'pid': t.track_id,
                    'kp_5': getattr(t, '_kp_5', None),
                    'aspect_ratio': t.aspect_ratio,
                    'bbox': getattr(t, '_bbox_tuple', tuple(int(v) for v in t.bbox)),
                    'fall_state': t.fall_state or {},
                    'history': t.history if isinstance(t.history, list) else [],
                    'keypoints': kps_arr,
                    'confs': confs_arr,
                    'angle_keypoints': extract_angle_keypoints(kps_arr, confs_arr)
                    if kps_arr is not None and confs_arr is not None else None,
                    'dt': dt,
                }

                fall_result = evaluate_fall(person_data, ctx.timestamp)
                t.fall_state = person_data['fall_state']
                t.fall_detected = bool(fall_result.get('fall_detected', False))
                t.confidence = float(fall_result.get('confidence', 0))
                state = fall_result.get('state', 'Normal').lower().replace(" ", "_")
                t.state = "fall" if "fall" in state and "potential" not in state else \
                          "potential_fall" if "potential" in state else "normal"
            except Exception:
                logger.debug(f"Fall logic error: {traceback.format_exc()}")

        return tracks

    def _visualize(self, frame, tracks: list, ctx: FrameContext):
        from fall_detection.visualizer import draw_person_info, draw_fall_alert
        annotated = frame.copy()
        state_map = {"fall": "FALL", "potential_fall": "Potential Fall", "normal": "Normal"}

        for t in tracks:
            person = {
                'pid': t.track_id,
                'bbox': tuple(int(v) for v in t.bbox),
                'is_ghost': t.is_ghost,
                'fall_state': {
                    'is_potential_fall': t.state == 'potential_fall',
                    'fall_start_time': ctx.timestamp if t.state != 'normal' else None,
                },
            }
            fr = {'fall_detected': t.fall_detected, 'confidence': t.confidence, 'state': state_map[t.state]}
            draw_person_info(annotated, person, fr)

        fall_results = [{'state': state_map[t.state], 'is_ghost': t.is_ghost} for t in tracks]
        annotated = draw_fall_alert(annotated, fall_results)

        cv2.putText(annotated, f"FPS: {ctx.fps or 0:.0f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ts = time.strftime("%H:%M:%S", time.localtime(ctx.timestamp))
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return annotated

    def reset(self):
        self._event_runtime.reset()
