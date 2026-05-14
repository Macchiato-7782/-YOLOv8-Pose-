"""
Degradation System
边缘设备 overload 时自动降级：降低分辨率/关闭 visual/减少 FPS
"""

import logging

logger = logging.getLogger(__name__)


class DegradationController:
    """自动降级控制器"""

    def __init__(self, state_machine=None, policy=None):
        self._state_machine = state_machine
        self._policy = policy
        self._level: int = 0
        self._actions: list = []

    LEVELS = {
        0: "normal",
        1: "reduce_fps",
        2: "reduce_resolution",
        3: "disable_visualization",
        4: "minimal",
    }

    def evaluate(self, cpu: float, ram_mb: float, queue_backlog: int = 0) -> int:
        """评估降级级别"""
        previous = self._level

        if self._policy and self._policy.should_overload(cpu, queue_backlog, 500):
            self._level = 4
        elif self._policy and self._policy.should_degrade(cpu, ram_mb):
            if ram_mb > self._policy.degradation_ram_mb * 1.2:
                self._level = min(4, self._level + 2)
            else:
                self._level = min(4, self._level + 1)
        elif self._level > 0 and cpu < 0.5 and ram_mb < self._policy.degradation_ram_mb * 0.8:
            self._level = max(0, self._level - 1)

        if self._level != previous:
            from fall_detection.runtime_state.state_machine import RuntimeState
            if self._state_machine and self._level > 0:
                self._state_machine.transition_to(RuntimeState.DEGRADED)
            elif self._state_machine and self._level == 0:
                self._state_machine.transition_to(RuntimeState.RUNNING)

        return self._level

    def get_actions(self, level: int) -> list:
        """获取对应级别的降级动作"""
        actions = []
        if level >= 1:
            actions.append(("reduce_fps", self._policy.degradation_reduce_fps_to if self._policy else 0.5))
        if level >= 2:
            actions.append(("reduce_input_size", self._policy.degradation_reduce_input_size_to if self._policy else 320))
        if level >= 3:
            actions.append(("disable_visualization", True))
        if level >= 4:
            actions.append(("minimal_mode", True))
        return actions

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_name(self) -> str:
        return self.LEVELS.get(self._level, "unknown")
