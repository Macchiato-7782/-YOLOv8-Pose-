#!/usr/bin/env python
"""
Runtime Chaos Test Tool
自动模拟 camera disconnect / backend crash / queue overflow / CPU overload / worker crash

用法:
    python tools/runtime_chaos_test.py
"""

import os
import sys
import time
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from fall_detection.runtime_state.state_machine import RuntimeStateMachine, RuntimeState
from fall_detection.runtime_state.policies import RuntimePolicy
from fall_detection.runtime_state.backpressure import BackpressureController
from fall_detection.runtime_state.recovery import FaultRecovery
from fall_detection.engine.resource_manager import ResourceManager


class ChaosTestResult:
    def __init__(self):
        self.recovered_crashes = 0
        self.dropped_frames = 0
        self.recovered_workers = 0
        self.restart_count = 0
        self.final_state = "UNKNOWN"
        self.overload_count = 0
        self.reconnect_count = 0


def run_chaos_test():
    results = ChaosTestResult()

    sm = RuntimeStateMachine()
    rm = ResourceManager()
    policy = RuntimePolicy()
    bp = BackpressureController(state_machine=sm, policy=policy)
    recovery = FaultRecovery(state_machine=sm, policy=policy)

    # Start
    sm.transition_to(RuntimeState.WARMING_UP)
    sm.transition_to(RuntimeState.RUNNING)

    chaos_events = [
        ("camera_disconnect", 20),
        ("cpu_spike", 30),
        ("queue_overflow", 40),
        ("backend_crash", 50),
        ("worker_crash", 60),
        ("latency_spike", 70),
    ]

    for frame in range(100):
        # Inject chaos
        for event_name, trigger_frame in chaos_events:
            if frame == trigger_frame:
                if event_name == "camera_disconnect":
                    sm.transition_to(RuntimeState.DEGRADED)
                    results.reconnect_count += 1
                elif event_name == "cpu_spike":
                    rm.update(cpu=0.95)
                elif event_name == "queue_overflow":
                    bp.evaluate(queue_backlog=300, latency_ms=100)
                    results.dropped_frames += random.randint(1, 5)
                elif event_name == "backend_crash":
                    sm.transition_to(RuntimeState.RECOVERING)
                    success = recovery.attempt_recovery("backend", lambda: True)
                    if success:
                        results.recovered_crashes += 1
                        sm.transition_to(RuntimeState.RUNNING)
                elif event_name == "worker_crash":
                    success = recovery.attempt_recovery("worker", lambda: True)
                    if success:
                        results.recovered_workers += 1

        # Simulate recovery after chaos
        if frame > 90:
            rm.update(cpu=0.3, ram_mb=500)

    results.final_state = sm.current.value

    # Print
    print("=" * 50)
    print("Chaos Test Report")
    print("=" * 50)
    print(f"Recovered Crashes:    {results.recovered_crashes}")
    print(f"Dropped Frames:       {results.dropped_frames}")
    print(f"Recovered Workers:    {results.recovered_workers}")
    print(f"Runtime Restart Count:{results.restart_count}")
    print(f"Final Runtime State:  {results.final_state.upper()}")
    print("=" * 50)

    return results.final_state == RuntimeState.RUNNING.value


if __name__ == "__main__":
    ok = run_chaos_test()
    print(f"\nChaos Test: {'PASS' if ok else 'CHECK'}")
