"""
Fault Recovery System
Runtime 异常后自动恢复：backend restart / worker restart / camera reconnect / session recovery
"""

import time
import logging
from typing import Optional
from fall_detection.runtime_state.state_machine import RuntimeState

logger = logging.getLogger(__name__)


class FaultRecovery:
    """故障自动恢复"""

    def __init__(self, state_machine=None, policy=None):
        self._state_machine = state_machine
        self._policy = policy
        self._recovery_count: int = 0
        self._last_recovery_time: float = 0
        self._recovery_stats: dict = {
            "backend_restarts": 0,
            "worker_restarts": 0,
            "camera_reconnects": 0,
            "session_recoveries": 0,
        }

    def attempt_recovery(self, component: str, recover_fn, max_retries: int = None) -> bool:
        """尝试恢复指定组件"""
        max_retries = max_retries or (self._policy.reconnect_max_retries if self._policy else 3)
        interval = self._policy.reconnect_interval_seconds if self._policy else 2.0

        if self._state_machine:
            self._state_machine.transition_to(RuntimeState.RECOVERING)

        self._recovery_count += 1
        self._last_recovery_time = time.time()

        for attempt in range(max_retries):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{max_retries} for {component}")
                result = recover_fn()
                if result:
                    logger.info(f"Recovery successful for {component}")
                    self._record(component)
                    if self._state_machine and self._state_machine.current == RuntimeState.RECOVERING:
                        self._state_machine.transition_to(RuntimeState.RUNNING)
                    return True
            except Exception as e:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(interval)

        logger.error(f"Recovery failed for {component} after {max_retries} attempts")
        if self._state_machine:
            self._state_machine.transition_to(RuntimeState.FAILED)
        return False

    def _record(self, component: str):
        if "backend" in component:
            self._recovery_stats["backend_restarts"] += 1
        elif "worker" in component:
            self._recovery_stats["worker_restarts"] += 1
        elif "camera" in component:
            self._recovery_stats["camera_reconnects"] += 1
        elif "session" in component:
            self._recovery_stats["session_recoveries"] += 1

    @property
    def stats(self) -> dict:
        return {
            "total_recoveries": self._recovery_count,
            "last_recovery_time": self._last_recovery_time,
            **self._recovery_stats,
        }
