"""
统一检测 Pipeline
编排整个 AI 检测流程：infer → postprocess → Detection → tracking → TrackState → fall logic → Event → serialization
"""

import time
import logging
import traceback
from typing import Optional

import cv2

from fall_detection.core.detection import Detection
from fall_detection.core.track import TrackState
from fall_detection.core.event import Event
from fall_detection.core.frame import FrameContext
from fall_detection.core.serializers import serialize_result, serialize_track, serialize_event
from fall_detection.core.runtime import EventRuntime
from fall_detection.core.validators import validate_track, validate_state

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """统一 AI 检测流水线

    编排: infer → Detection → tracking → TrackState → fall logic → Event → serialize
    """

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
        self._low_conf_rois = []

    def run(
        self,
        frame,
        frame_context: FrameContext,
    ) -> dict:
        """
        执行完整的检测流水线

        Args:
            frame: OpenCV BGR frame
            frame_context: 帧上下文

        Returns:
            dict: JSON-friendly 标准化输出
        """
        # Stage 1: Backend Inference → Detection objects
        try:
            raw_detections = self._backend.infer(frame)
            detections = [
                Detection(
                    bbox=d["bbox"],
                    score=d["score"],
                    class_id=d.get("class_id", 0),
                    keypoints=[
                        Detection.Keypoint if hasattr(Detection, 'Keypoint') else
                        type('Keypoint', (), {'x': k[0], 'y': k[1], 'confidence': k[2]})
                        for k in d.get("keypoints", [])
                    ],
                    track_id=d.get("track_id"),
                )
                if isinstance(d, dict) else d
                for d in raw_detections
            ]
            # Convert raw Detection dicts to real Detection objects if needed
            detections = self._ensure_detections(raw_detections)
        except Exception:
            logger.debug(f"Inference error: {traceback.format_exc()}")
            detections = []

        # Stage 2: Tracking → TrackState objects
        tracks = self._run_tracking(detections, frame_context)

        # Stage 3: Fall Logic on each TrackState
        tracks = self._run_fall_logic(tracks, frame_context)

        # Stage 4: Event Generation
        events = self._event_runtime.process_tracks(tracks, frame_context)

        # Stage 5: Visualization (optional)
        annotated = None
        if self._enable_visualization:
            annotated = self._visualize(frame, tracks, frame_context)

        # Stage 6: Serialization
        diagnostics = {
            "fps": frame_context.fps or 0,
            "backend": getattr(self._backend, 'backend', 'unknown'),
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

    def _ensure_detections(self, raw) -> list:
        """确保所有检测都是 Detection 对象"""
        from fall_detection.core.detection import Keypoint
        result = []
        for d in raw:
            if isinstance(d, Detection):
                result.append(d)
            elif isinstance(d, dict):
                kps = []
                for k in d.get("keypoints", []):
                    if isinstance(k, (list, tuple)) and len(k) >= 3:
                        kps.append(Keypoint(x=float(k[0]), y=float(k[1]), confidence=float(k[2])))
                    elif hasattr(k, 'x'):
                        kps.append(k)
                result.append(Detection(
                    bbox=d["bbox"],
                    score=d["score"],
                    class_id=d.get("class_id", 0),
                    keypoints=kps,
                    track_id=d.get("track_id"),
                ))
            else:
                result.append(d)
        return result

    def _run_tracking(self, detections: list, ctx: FrameContext) -> list:
        """Tracking: Detection → TrackState"""
        try:
            # Convert Detection → internal format
            internal_dets = self._tracker.convert_backend_detections(
                [d.to_dict() if hasattr(d, 'to_dict') else self._det_to_dict(d) for d in detections]
            )
            raw_tracked = self._tracker.update(internal_dets, ctx.timestamp)

            # 限制人数
            if self._max_persons is not None and len(raw_tracked) > self._max_persons:
                raw_tracked.sort(
                    key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]),
                    reverse=True,
                )
                keep_pids = {p["pid"] for p in raw_tracked[:self._max_persons]}
                raw_tracked = [p for p in raw_tracked if p["pid"] in keep_pids]

            self._tracker.cleanup(ctx.timestamp)

            # Convert internal dicts to TrackState objects
            tracks = []
            for r in raw_tracked:
                state = r.get("fall_result", {}).get("state", "Normal")
                state = state.lower().replace(" ", "_")
                if "fall" in state and "potential" not in state:
                    state = "fall"
                elif "potential" in state:
                    state = "potential_fall"
                else:
                    state = "normal"

                kps = r.get("keypoints")
                confs = r.get("confs")
                keypoints_list = []
                if kps is not None:
                    if confs is not None and len(kps) == len(confs):
                        keypoints_list = [[float(k[0]), float(k[1]), float(confs[i])] for i, k in enumerate(kps)]
                    else:
                        keypoints_list = [[float(k[0]), float(k[1]), 0.0] for k in kps]

                bbox = r["bbox"]
                center = r["center"]
                t = TrackState(
                    track_id=r["pid"],
                    bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    center=[float(center[0]), float(center[1])],
                    confidence=float(r.get("fall_result", {}).get("confidence", 0)),
                    state=state,
                    keypoints=keypoints_list,
                    is_ghost=bool(r.get("is_ghost", False)),
                    aspect_ratio=float(r.get("aspect_ratio", 0)),
                    fall_detected=bool(r.get("fall_result", {}).get("fall_detected", False)),
                    fall_state=r.get("fall_state", {}),
                )
                tracks.append(t)

            return tracks
        except Exception:
            logger.debug(f"Tracking error: {traceback.format_exc()}")
            return []

    def _det_to_dict(self, det) -> dict:
        """Detection → dict（兼容旧 tracker API）"""
        kps = []
        for k in det.keypoints:
            if hasattr(k, 'x'):
                kps.append([k.x, k.y, k.confidence])
            elif isinstance(k, (list, tuple)):
                kps.append(list(k))
        return {
            "bbox": det.bbox,
            "score": det.score,
            "class_id": det.class_id,
            "track_id": det.track_id,
            "keypoints": kps,
        }

    def _run_fall_logic(self, tracks: list, ctx: FrameContext) -> list:
        """对每个 TrackState 执行跌倒判断（通过内部 dict 桥接）"""
        from fall_logic import evaluate_fall
        from camera_process import extract_angle_keypoints
        import numpy as np

        for t in tracks:
            try:
                if t.is_ghost:
                    continue

                # 构造 evaluate_fall 需要的 person dict 桥接
                kps_arr = np.array([[float(k[0]), float(k[1])] for k in t.keypoints], dtype=np.float64) if t.keypoints else None
                confs_arr = np.array([float(k[2]) for k in t.keypoints], dtype=np.float64) if t.keypoints else None

                person_data = {
                    'pid': t.track_id,
                    'kp_5': self._make_kp_5(kps_arr, confs_arr),
                    'aspect_ratio': t.aspect_ratio,
                    'bbox': tuple(int(v) for v in t.bbox),
                    'fall_state': t.fall_state or {},
                    'history': t.fall_state.get('_history', []) if t.fall_state else [],
                    'keypoints': kps_arr,
                    'confs': confs_arr,
                    'angle_keypoints': extract_angle_keypoints(kps_arr, confs_arr) if kps_arr is not None and confs_arr is not None else None,
                }

                fall_result = evaluate_fall(person_data, ctx.timestamp)
                t.fall_state = person_data['fall_state']
                t.fall_detected = bool(fall_result.get('fall_detected', False))
                t.confidence = float(fall_result.get('confidence', 0))
                state = fall_result.get('state', 'Normal').lower().replace(" ", "_")
                if "fall" in state and "potential" not in state:
                    t.state = "fall"
                elif "potential" in state:
                    t.state = "potential_fall"
                else:
                    t.state = "normal"
            except Exception:
                logger.debug(f"Fall logic error: {traceback.format_exc()}")

        return tracks

    def _make_kp_5(self, keypoints, confs):
        from features import yolo_to_5keypoints
        if keypoints is None or confs is None or len(keypoints) < 17:
            return None
        return yolo_to_5keypoints(keypoints, confs)

    def _visualize(self, frame, tracks: list, ctx: FrameContext):
        from fall_detection.visualizer import draw_person_info, draw_fall_alert
        annotated = frame.copy()
        for t in tracks:
            person = {
                'pid': t.track_id,
                'bbox': tuple(int(v) for v in t.bbox),
                'is_ghost': t.is_ghost,
                'fall_state': {'is_potential_fall': t.state == 'potential_fall',
                               'fall_start_time': ctx.timestamp if t.state != 'normal' else None},
            }
            fr = {'fall_detected': t.fall_detected, 'confidence': t.confidence,
                  'state': {'fall': 'FALL', 'potential_fall': 'Potential Fall', 'normal': 'Normal'}[t.state]}
            draw_person_info(annotated, person, fr)

        fall_results = [{'state': {'fall': 'FALL', 'potential_fall': 'Potential Fall', 'normal': 'Normal'}[t.state],
                         'is_ghost': t.is_ghost} for t in tracks]
        annotated = draw_fall_alert(annotated, fall_results)

        fps = ctx.fps or 0
        cv2.putText(annotated, f"FPS: {fps:.0f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ts = time.strftime("%H:%M:%S", time.localtime(ctx.timestamp))
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return annotated

    def reset(self):
        self._event_runtime.reset()
        self._low_conf_rois = []
