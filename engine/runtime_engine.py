"""
Runtime Engine -- 工业级边缘 AI Runtime Engine Entrypoint
统一 Session / Scheduler / Pipeline / EventBus / Workers / Health / Metrics
"""

import time
import logging
from typing import Optional

from fall_detection.engine.event_bus import EventBus
from fall_detection.engine.runtime_session import RuntimeSession
from fall_detection.engine.registry import RuntimeRegistry
from fall_detection.engine.health import RuntimeHealthMonitor
from fall_detection.engine.metrics import RuntimeMetrics
from fall_detection.engine.signals import LifecycleSignal, ErrorSignal

logger = logging.getLogger(__name__)


class RuntimeEngine:
    """工业级边缘 AI Runtime Engine

    管理:
    - RuntimeRegistry: 所有 sessions / cameras / pipelines / workers / backends
    - EventBus: 全局事件总线
    - HealthMonitor: 全局健康监控
    - Metrics: 全局指标收集
    """

    def __init__(self):
        self.registry = RuntimeRegistry()
        self.event_bus = EventBus()
        self.health = RuntimeHealthMonitor(event_bus=self.event_bus)
        self.metrics = RuntimeMetrics()
        self._sessions: dict = {}

    def create_session(
        self,
        session_id: str,
        camera_id: str,
        pipeline,
        backend,
        config: dict = None,
    ) -> RuntimeSession:
        """创建运行时会话"""
        session = RuntimeSession(
            session_id=session_id,
            camera_id=camera_id,
            pipeline=pipeline,
            backend=backend,
            config=config,
        )
        self._sessions[session_id] = session
        self.registry.register_session(session_id, session)
        self.registry.register_pipeline(f"{session_id}_pipeline", pipeline)
        self.registry.register_backend(f"{session_id}_backend", backend)

        logger.info(f"RuntimeEngine: created session '{session_id}'")
        return session

    def get_session(self, session_id: str) -> Optional[RuntimeSession]:
        return self._sessions.get(session_id)

    def start_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.start()
            self.event_bus.publish(LifecycleSignal(
                signal_type="lifecycle",
                lifecycle_event="engine_session_started",
                session_id=session_id,
            ))

    def stop_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.stop()
            self.event_bus.publish(LifecycleSignal(
                signal_type="lifecycle",
                lifecycle_event="engine_session_stopped",
                session_id=session_id,
            ))

    def remove_session(self, session_id: str):
        session = self._sessions.pop(session_id, None)
        if session:
            session.stop()
            self.registry.unregister_session(session_id)

    def restart_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.restart()

    def shutdown(self):
        """关闭引擎"""
        for sid in list(self._sessions.keys()):
            self.stop_session(sid)
            self.remove_session(sid)
        self.registry.clear()
        logger.info("RuntimeEngine: shutdown complete")

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def is_running(self) -> bool:
        return any(s.lifecycle.is_running for s in self._sessions.values())
