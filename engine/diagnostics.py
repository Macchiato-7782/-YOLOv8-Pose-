"""
Runtime Diagnostics
统一 Runtime 诊断信息输出
"""

from fall_detection.runtime_state.state_machine import RuntimeState


class RuntimeDiagnostics:
    """Runtime 诊断器"""

    def __init__(self, state_machine=None, resource_manager=None,
                 health_monitor=None, metrics=None, recovery=None, backpressure=None):
        self._state_machine = state_machine
        self._resource_manager = resource_manager
        self._health_monitor = health_monitor
        self._metrics = metrics
        self._recovery = recovery
        self._backpressure = backpressure

    def snapshot(self) -> dict:
        """获取诊断快照"""
        d = {
            "runtime_state": self._state_machine.current.value if self._state_machine else "unknown",
            "is_healthy": self._state_machine.is_healthy if self._state_machine else True,
        }

        if self._resource_manager:
            d.update(self._resource_manager.snapshot())

        if self._health_monitor:
            d["avg_fps"] = self._health_monitor.avg_fps
            d["health_errors"] = self._health_monitor.snapshot.backend_errors

        if self._metrics:
            d["metrics"] = self._metrics.to_dict()

        if self._recovery:
            d["recovery"] = self._recovery.stats

        if self._backpressure:
            d["backpressure"] = self._backpressure.stats

        return d

    def print(self):
        """打印诊断信息"""
        s = self.snapshot()
        print("=" * 50)
        print("Runtime Diagnostics")
        print("=" * 50)
        print(f"Runtime State:   {s.get('runtime_state', '?').upper()}")
        print(f"CPU Usage:        {s.get('cpu_usage', 0) * 100:.0f}%")
        print(f"RAM Usage:        {s.get('ram_usage_mb', 0):.0f} MB")
        print(f"Avg FPS:          {s.get('avg_fps', 0):.1f}")
        print(f"Queue Backlog:    {s.get('queue_backlog', 0)}")
        print(f"Recovery Count:   {s.get('recovery', {}).get('total_recoveries', 0)}")
        print(f"Dropped Frames:   {s.get('backpressure', {}).get('dropped_frames', 0)}")
        print("=" * 50)
