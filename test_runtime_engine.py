"""
Runtime Engine Tests
测试 RuntimeSession, CameraSession, EventBus, Scheduler, Worker, Registry, Health, Metrics
"""

import sys
import os
import time
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fall_detection.engine.event_bus import EventBus
from fall_detection.engine.scheduler import RuntimeScheduler
from fall_detection.engine.worker import RuntimeWorker, WorkerStatus
from fall_detection.engine.lifecycle import RuntimeLifecycleManager, RuntimeStatus
from fall_detection.engine.registry import RuntimeRegistry
from fall_detection.engine.metrics import RuntimeMetrics
from fall_detection.engine.health import RuntimeHealthMonitor
from fall_detection.engine.signals import EventSignal, LifecycleSignal
from fall_detection.engine.frame_buffer import SharedFrameBuffer
from fall_detection.core.frame import FrameContext


# ============================================================
# EventBus
# ============================================================

class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        def handler(signal):
            received.append(signal)

        bus.subscribe("test_event", handler)
        bus.publish(EventSignal(signal_type="test_event", confidence=0.85))

        assert len(received) == 1
        assert received[0].signal_type == "test_event"
        assert received[0].confidence == 0.85

    def test_unsubscribe(self):
        bus = EventBus()
        received = []

        def handler(signal):
            received.append(signal)

        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        bus.publish(EventSignal(signal_type="test_event", confidence=0.85))

        assert len(received) == 0

    def test_global_handler(self):
        bus = EventBus()
        received = []

        def handler(signal):
            received.append(signal)

        bus.subscribe_all(handler)
        bus.publish(EventSignal(signal_type="type_a"))
        bus.publish(EventSignal(signal_type="type_b"))

        assert len(received) == 2

    def test_clear(self):
        bus = EventBus()
        received = []

        def handler(signal):
            received.append(signal)

        bus.subscribe("test", handler)
        bus.clear()
        bus.publish(EventSignal(signal_type="test"))
        assert len(received) == 0

    def test_subscriber_count(self):
        bus = EventBus()

        def handler(s):
            pass

        bus.subscribe("a", handler)
        bus.subscribe("b", handler)
        bus.subscribe_all(handler)
        assert bus.subscriber_count() == 3


# ============================================================
# Scheduler
# ============================================================

class TestScheduler:
    def test_register_and_tick(self):
        sched = RuntimeScheduler()
        called = []

        sched.register("test", lambda: called.append(1), interval_frames=1)
        results = sched.tick()
        assert len(results) == 1
        assert len(called) == 1

    def test_interval_skip(self):
        sched = RuntimeScheduler()
        called = []

        sched.register("test", lambda: called.append(1), interval_frames=2)
        sched.tick()  # frame 1: should run
        sched.tick()  # frame 2: should skip

        assert len(called) == 1  # only ran on frame 1

    def test_disable_enable(self):
        sched = RuntimeScheduler()
        called = []

        sched.register("test", lambda: called.append(1))
        sched.disable("test")
        sched.tick()
        assert len(called) == 0

        sched.enable("test")
        sched.tick()
        assert len(called) == 1

    def test_reset(self):
        sched = RuntimeScheduler()
        called = []

        sched.register("test", lambda: called.append(1))
        sched.tick()
        sched.reset()
        sched.tick()
        assert len(called) == 2  # still called because reset doesn't clear tasks

    def test_clear(self):
        sched = RuntimeScheduler()
        sched.register("test", lambda: None)
        sched.clear()
        assert sched.task_count == 0


# ============================================================
# Worker
# ============================================================

class TestWorker:
    def test_start_and_process(self):
        w = RuntimeWorker("test", lambda x: x * 2)
        w.start()
        w.submit(5)
        w.submit(10)
        results = w.process()
        assert results == [10, 20]

    def test_pause_resume(self):
        w = RuntimeWorker("test", lambda x: x)
        w.start()
        w.submit(1)
        w.pause()
        assert w.status == WorkerStatus.PAUSED
        assert not w.submit(2)
        w.resume()
        assert w.status == WorkerStatus.RUNNING

    def test_stop_and_drain(self):
        w = RuntimeWorker("test", lambda x: x)
        w.start()
        w.submit(1)
        w.stop()
        assert w.status == WorkerStatus.STOPPED
        assert w.queue_backlog == 0

    def test_error_isolation(self):
        w = RuntimeWorker("test", lambda x: 1 / 0 if x == 0 else x)
        w.start()
        w.submit(0)
        w.process()
        assert w.status == WorkerStatus.ERROR
        assert w.error_count == 1

    def test_queue_backlog(self):
        w = RuntimeWorker("test", lambda x: x, queue_size=2)
        w.start()
        w.submit(1)
        w.submit(2)
        w.submit(3)  # exceeds queue_size, oldest dropped
        assert w.queue_backlog == 2


# ============================================================
# Lifecycle
# ============================================================

class TestLifecycle:
    def test_start_stop(self):
        lm = RuntimeLifecycleManager()
        assert lm.status == RuntimeStatus.CREATED
        lm.start()
        assert lm.status == RuntimeStatus.RUNNING
        lm.stop()
        assert lm.status == RuntimeStatus.STOPPED

    def test_pause_resume(self):
        lm = RuntimeLifecycleManager()
        lm.start()
        lm.pause()
        assert lm.status == RuntimeStatus.PAUSED
        lm.resume()
        assert lm.status == RuntimeStatus.RUNNING

    def test_is_running(self):
        lm = RuntimeLifecycleManager()
        assert not lm.is_running
        lm.start()
        assert lm.is_running


# ============================================================
# Registry
# ============================================================

class TestRegistry:
    def test_register_session(self):
        reg = RuntimeRegistry()
        reg.register_session("s1", {"name": "test"})
        assert reg.get_session("s1")["name"] == "test"
        assert reg.session_count == 1

    def test_unregister_session(self):
        reg = RuntimeRegistry()
        reg.register_session("s1", {})
        reg.unregister_session("s1")
        assert reg.session_count == 0

    def test_register_camera(self):
        reg = RuntimeRegistry()
        reg.register_camera("c0", "mock_camera")
        assert reg.get_camera("c0") == "mock_camera"


# ============================================================
# HealthMonitor
# ============================================================

class TestHealthMonitor:
    def test_record_fps(self):
        hm = RuntimeHealthMonitor()
        hm.record_fps(30.0)
        hm.record_fps(28.0)
        assert hm.avg_fps == 29.0

    def test_record_error(self):
        hm = RuntimeHealthMonitor(interval_seconds=0)
        hm.record_error()
        hm.record_error()
        hm.check()
        assert hm.snapshot.backend_errors == 2


# ============================================================
# Metrics
# ============================================================

class TestMetrics:
    def test_record_frame(self):
        m = RuntimeMetrics()
        m.record_frame(10.0)
        m.record_frame(20.0)
        s = m.snapshot()
        assert s.total_frames == 2

    def test_record_event(self):
        m = RuntimeMetrics()
        m.record_event("fall_confirmed")
        m.record_event("fall_confirmed")
        m.record_event("person_detected")
        s = m.snapshot()
        assert s.total_events == 3
        assert s.event_counts["fall_confirmed"] == 2

    def test_to_dict(self):
        m = RuntimeMetrics()
        d = m.to_dict()
        assert "avg_fps" in d
        assert "total_frames" in d


# ============================================================
# FrameBuffer
# ============================================================

class TestFrameBuffer:
    def test_push_and_latest(self):
        fb = SharedFrameBuffer(ring_size=4)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = FrameContext(frame_id=1, timestamp=100.0, camera_id="c0", width=640, height=480)
        fb.push(frame, ctx)
        f, c = fb.latest()
        assert c.frame_id == 1

    def test_ring_buffer(self):
        fb = SharedFrameBuffer(ring_size=2)
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ctx = FrameContext(frame_id=i + 1, timestamp=float(i), camera_id="c0", width=640, height=480)
            fb.push(frame, ctx)
        f, c = fb.latest()
        assert c.frame_id == 10  # latest = 10
        snap = fb.snapshot(offset=0)
        assert snap[1].frame_id == 10
