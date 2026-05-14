"""
Fault-Tolerant Runtime State Management
"""

from fall_detection.runtime_state.state_machine import RuntimeStateMachine, RuntimeState
from fall_detection.runtime_state.policies import RuntimePolicy
from fall_detection.runtime_state.backpressure import BackpressureController
from fall_detection.runtime_state.degradation import DegradationController
from fall_detection.runtime_state.recovery import FaultRecovery
from fall_detection.runtime_state.transitions import is_degraded_state, is_healthy_state, is_error_state

__all__ = [
    "RuntimeStateMachine", "RuntimeState",
    "RuntimePolicy",
    "BackpressureController",
    "DegradationController",
    "FaultRecovery",
    "is_degraded_state", "is_healthy_state", "is_error_state",
]
