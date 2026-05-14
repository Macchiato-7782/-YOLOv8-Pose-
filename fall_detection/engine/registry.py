"""
Runtime Registry -- 统一注册中心
管理 active sessions / cameras / pipelines / workers / backends
"""

from typing import Optional


class RuntimeRegistry:
    """统一运行时注册中心"""

    def __init__(self):
        self._sessions: dict = {}
        self._cameras: dict = {}
        self._pipelines: dict = {}
        self._workers: dict = {}
        self._backends: dict = {}

    def register_session(self, session_id: str, session):
        self._sessions[session_id] = session

    def unregister_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    def list_sessions(self) -> list:
        return list(self._sessions.keys())

    def register_camera(self, camera_id: str, camera):
        self._cameras[camera_id] = camera

    def unregister_camera(self, camera_id: str):
        self._cameras.pop(camera_id, None)

    def get_camera(self, camera_id: str):
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list:
        return list(self._cameras.keys())

    def register_pipeline(self, name: str, pipeline):
        self._pipelines[name] = pipeline

    def get_pipeline(self, name: str):
        return self._pipelines.get(name)

    def list_pipelines(self) -> list:
        return list(self._pipelines.keys())

    def register_worker(self, name: str, worker):
        self._workers[name] = worker

    def get_worker(self, name: str):
        return self._workers.get(name)

    def register_backend(self, name: str, backend):
        self._backends[name] = backend

    def get_backend(self, name: str):
        return self._backends.get(name)

    def clear(self):
        self._sessions.clear()
        self._cameras.clear()
        self._pipelines.clear()
        self._workers.clear()
        self._backends.clear()

    @property
    def session_count(self) -> int:
        return len(self._sessions)
