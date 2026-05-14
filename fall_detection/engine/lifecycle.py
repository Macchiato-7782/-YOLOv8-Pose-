"""
Runtime Lifecycle Manager
统一管理 startup / shutdown / restart / backend reload / camera reconnect
"""

import time
import logging
from typing import Optional, Callable
from enum import Enum

from fall_detection.engine.signals import LifecycleSignal

logger = logging.getLogger(__name__)


class RuntimeStatus(Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class RuntimeLifecycleManager:
    """统一运行时生命周期管理"""

    def __init__(self, event_bus=None):
        self._status = RuntimeStatus.CREATED
        self._event_bus = event_bus
        self._hooks: dict = {
            "on_start": [],
            "on_stop": [],
            "on_pause": [],
            "on_resume": [],
            "on_restart": [],
            "on_error": [],
        }

    def register_hook(self, event: str, callback: Callable):
        if event in self._hooks:
            self._hooks[event].append(callback)

    def start(self) -> bool:
        if self._status in (RuntimeStatus.RUNNING, RuntimeStatus.STARTING):
            return True
        self._status = RuntimeStatus.STARTING
        self._emit("on_start")
        self._signal("started")
        self._status = RuntimeStatus.RUNNING
        logger.info("Runtime started")
        return True

    def stop(self):
        if self._status == RuntimeStatus.STOPPED:
            return
        self._status = RuntimeStatus.STOPPING
        self._emit("on_stop")
        self._signal("stopped")
        self._status = RuntimeStatus.STOPPED

    def pause(self):
        if self._status != RuntimeStatus.RUNNING:
            return
        self._status = RuntimeStatus.PAUSED
        self._emit("on_pause")
        self._signal("paused")

    def resume(self):
        if self._status != RuntimeStatus.PAUSED:
            return
        self._status = RuntimeStatus.RUNNING
        self._emit("on_resume")
        self._signal("resumed")

    def restart(self) -> bool:
        self._signal("restarted")
        self._emit("on_restart")
        return self.start()

    def set_error(self):
        self._status = RuntimeStatus.ERROR
        self._emit("on_error")
        self._signal("error")

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == RuntimeStatus.RUNNING

    def _emit(self, hook_key: str):
        for cb in self._hooks.get(hook_key, []):
            try:
                cb()
            except Exception as e:
                logger.debug(f"Lifecycle hook error: {e}")

    def _signal(self, event: str):
        if self._event_bus:
            self._event_bus.publish(LifecycleSignal(
                signal_type="lifecycle",
                lifecycle_event=event,
                reason="",
            ))
