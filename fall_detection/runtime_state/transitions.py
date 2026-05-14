"""
State Transitions Helpers
"""

from fall_detection.runtime_state.state_machine import RuntimeState


def is_degraded_state(state: RuntimeState) -> bool:
    return state in (RuntimeState.DEGRADED, RuntimeState.BACKPRESSURE, RuntimeState.OVERLOADED)


def is_healthy_state(state: RuntimeState) -> bool:
    return state in (RuntimeState.INITIALIZING, RuntimeState.WARMING_UP, RuntimeState.RUNNING)


def is_error_state(state: RuntimeState) -> bool:
    return state in (RuntimeState.FAILED, RuntimeState.RECOVERING)
