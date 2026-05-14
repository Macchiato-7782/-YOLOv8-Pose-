"""
EventBus -- 统一运行时事件总线
支持 publish / subscribe / broadcast，future websocket / mqtt / kafka 可订阅
"""

import logging
from typing import Callable, Optional

from fall_detection.engine.signals import RuntimeSignal

logger = logging.getLogger(__name__)

EventHandler = Callable[[RuntimeSignal], None]


class EventBus:
    """统一运行时事件总线，Runtime 内所有消息通过此总线广播"""

    def __init__(self):
        self._subscribers: dict = {}  # signal_type → list[EventHandler]
        self._global_subscribers: list = []  # 所有信号

    def subscribe(self, signal_type: str, handler: EventHandler):
        """订阅特定类型的信号"""
        if signal_type not in self._subscribers:
            self._subscribers[signal_type] = []
        self._subscribers[signal_type].append(handler)

    def subscribe_all(self, handler: EventHandler):
        """订阅所有信号"""
        self._global_subscribers.append(handler)

    def unsubscribe(self, signal_type: str, handler: EventHandler):
        """取消订阅"""
        if signal_type in self._subscribers:
            try:
                self._subscribers[signal_type].remove(handler)
            except ValueError:
                pass
        try:
            self._global_subscribers.remove(handler)
        except ValueError:
            pass

    def publish(self, signal: RuntimeSignal):
        """发布信号到特定类型订阅者和全局订阅者"""
        try:
            for handler in self._global_subscribers:
                try:
                    handler(signal)
                except Exception:
                    logger.debug(f"EventBus global handler error for {signal.signal_type}")

            handlers = self._subscribers.get(signal.signal_type, [])
            for handler in handlers:
                try:
                    handler(signal)
                except Exception:
                    logger.debug(f"EventBus handler error for {signal.signal_type}")
        except Exception:
            pass

    def broadcast(self, signal: RuntimeSignal):
        """广播到所有订阅者（同 publish）"""
        self.publish(signal)

    def clear(self):
        """清空所有订阅"""
        self._subscribers.clear()
        self._global_subscribers.clear()

    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values()) + len(self._global_subscribers)
