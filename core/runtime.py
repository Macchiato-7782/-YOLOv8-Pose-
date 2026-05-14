"""
事件运行时
统一事件生成、去重、防重复
"""

import time
import logging
from typing import Optional

from fall_detection.core.event import Event
from fall_detection.core.track import TrackState
from fall_detection.core.frame import FrameContext
from fall_detection.core.validators import validate_event

logger = logging.getLogger(__name__)


class EventRuntime:
    """统一事件生成和去重"""

    def __init__(self, cooldown_seconds: float = 5.0):
        self._cooldown = cooldown_seconds
        self._last_event_time: dict = {}  # key=(event_type, track_id) → timestamp

    def process_tracks(
        self,
        tracks: list,
        frame_context: FrameContext,
    ) -> list:
        """
        从跟踪列表生成事件

        Args:
            tracks: list[TrackState]
            frame_context: 帧上下文

        Returns:
            list[Event]: 本帧产生的事件
        """
        events = []

        for track in tracks:
            track_events = self._evaluate_track(track, frame_context)
            events.extend(track_events)

        return events

    def _evaluate_track(self, track, ctx: FrameContext) -> list:
        events = []

        if track.fall_detected and not track.is_ghost:
            event = self._create_event("fall_confirmed", track, ctx)
            if event:
                events.append(event)

        elif track.state == "potential_fall" and not track.is_ghost:
            event = self._create_event("fall_warning", track, ctx)
            if event:
                events.append(event)

        if track.lost_frames == 0 and not track.is_ghost:
            event = self._create_event("person_detected", track, ctx)
            if event:
                events.append(event)

        if track.is_ghost or track.lost_frames > 30:
            event = self._create_event("track_lost", track, ctx)
            if event:
                events.append(event)

        return events

    def _create_event(self, event_type: str, track, ctx: FrameContext) -> Optional[Event]:
        key = (event_type, track.track_id)
        now = ctx.timestamp

        # 冷却检查
        if key in self._last_event_time:
            elapsed = now - self._last_event_time[key]
            if elapsed < self._cooldown:
                return None

        self._last_event_time[key] = now

        event = Event.from_track(event_type, track, ctx.camera_id, ctx.timestamp)
        ok, err = validate_event(event)
        if not ok:
            logger.debug(f"Invalid event: {err}")
            return None

        return event

    def reset(self):
        self._last_event_time.clear()
