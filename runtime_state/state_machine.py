"""
Runtime State Machine
统一 Runtime 状态流转，非法转换必须 reject
"""

from enum import Enum, auto
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class RuntimeState(Enum):
    INITIALIZING = "initializing"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    DEGRADED = "degraded"
    BACKPRESSURE = "backpressure"
    OVERLOADED = "overloaded"
    RECOVERING = "recovering"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


VALID_TRANSITIONS = {
    RuntimeState.INITIALIZING: [RuntimeState.WARMING_UP, RuntimeState.FAILED],
    RuntimeState.WARMING_UP: [RuntimeState.RUNNING, RuntimeState.FAILED],
    RuntimeState.RUNNING: [RuntimeState.DEGRADED, RuntimeState.BACKPRESSURE,
                           RuntimeState.OVERLOADED, RuntimeState.STOPPING,
                           RuntimeState.FAILED, RuntimeState.RECOVERING],
    RuntimeState.DEGRADED: [RuntimeState.RUNNING, RuntimeState.BACKPRESSURE,
                            RuntimeState.OVERLOADED, RuntimeState.STOPPING,
                            RuntimeState.RECOVERING],
    RuntimeState.BACKPRESSURE: [RuntimeState.RUNNING, RuntimeState.OVERLOADED,
                                 RuntimeState.DEGRADED, RuntimeState.STOPPING],
    RuntimeState.OVERLOADED: [RuntimeState.BACKPRESSURE, RuntimeState.DEGRADED,
                               RuntimeState.STOPPING, RuntimeState.FAILED],
    RuntimeState.RECOVERING: [RuntimeState.RUNNING, RuntimeState.FAILED,
                               RuntimeState.STOPPING],
    RuntimeState.RESTARTING: [RuntimeState.INITIALIZING, RuntimeState.FAILED],
    RuntimeState.STOPPING: [RuntimeState.STOPPED, RuntimeState.FAILED],
    RuntimeState.STOPPED: [],
    RuntimeState.FAILED: [RuntimeState.RECOVERING, RuntimeState.RESTARTING],
}


class RuntimeStateMachine:
    """统一运行时状态机"""

    def __init__(self, event_bus=None):
        self._current = RuntimeState.INITIALIZING
        self._previous: Optional[RuntimeState] = None
        self._history: list = []
        self._event_bus = event_bus
        self._auto_recover_enabled = True

    def transition_to(self, target: RuntimeState) -> bool:
        if not self.can_transition(target):
            logger.warning(f"Illegal state transition: {self._current.value} -> {target.value}")
            return False

        self._previous = self._current
        self._current = target
        self._history.append({
            "from": self._previous.value,
            "to": self._current.value,
            "timestamp": time.time(),
        })

        if self._event_bus:
            from fall_detection.engine.signals import LifecycleSignal
            self._event_bus.publish(LifecycleSignal(
                signal_type="state_change",
                lifecycle_event=f"{self._previous.value}->{self._current.value}",
            ))

        logger.info(f"State transition: {self._previous.value} -> {self._current.value}")
        return True

    def can_transition(self, target: RuntimeState) -> bool:
        allowed = VALID_TRANSITIONS.get(self._current, [])
        return target in allowed

    @property
    def current(self) -> RuntimeState:
        return self._current

    @property
    def previous(self) -> Optional[RuntimeState]:
        return self._previous

    @property
    def history(self) -> list:
        return self._history

    @property
    def is_healthy(self) -> bool:
        return self._current in (RuntimeState.RUNNING, RuntimeState.WARMING_UP,
                                  RuntimeState.INITIALIZING, RuntimeState.DEGRADED)

    @property
    def is_running(self) -> bool:
        return self._current == RuntimeState.RUNNING

    def enable_auto_recover(self):
        self._auto_recover_enabled = True

    def disable_auto_recover(self):
        self._auto_recover_enabled = False
