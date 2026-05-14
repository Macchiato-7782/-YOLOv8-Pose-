"""
CameraSession -- 统一摄像头生命周期管理
支持 USB / RTSP / video file / future WebRTC
"""

import time
import logging
import cv2
from typing import Optional

from fall_detection.core.frame import FrameContext
from fall_detection.engine.frame_buffer import SharedFrameBuffer

logger = logging.getLogger(__name__)


class CameraSession:
    """统一摄像头会话，管理连接、重连、帧读取"""

    def __init__(self, camera_id: str, source, config: dict = None):
        self.camera_id = camera_id
        self.source = source
        self.config = config or {}
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._running = False
        self._frame_id = 0
        self._dropped_frames = 0
        self._fps = 0.0
        self._fps_t0 = time.time()
        self._frame_buffer = SharedFrameBuffer(ring_size=8)

    def connect(self) -> bool:
        """连接摄像头"""
        try:
            if isinstance(self.source, str):
                self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(int(self.source))
            self._connected = self._cap.isOpened()
            if self._connected:
                ret, _ = self._cap.read()
                if not ret:
                    self._connected = False
                    self._cap.release()
            return self._connected
        except Exception as e:
            logger.error(f"Camera {self.camera_id} connect error: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connected = False

    def reconnect(self) -> bool:
        """重新连接"""
        self.disconnect()
        time.sleep(0.5)
        return self.connect()

    def read_frame(self) -> Optional[tuple]:
        """读取一帧，返回 (frame, FrameContext) 或 None"""
        if not self._connected or not self._cap:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self._connected = False
            return None

        self._frame_id += 1
        elapsed = time.time() - self._fps_t0
        self._fps = self._frame_id / (elapsed + 1e-8)

        ctx = FrameContext.from_frame(
            frame, self._frame_id, self.camera_id,
            timestamp=time.time(), fps=self._fps,
        )

        self._frame_buffer.push(frame, ctx)
        return (frame, ctx)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_buffer(self) -> SharedFrameBuffer:
        return self._frame_buffer
