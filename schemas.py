"""
标准输出格式定义和序列化辅助函数
确保 FallDetector 返回值可被 json.dumps 序列化
"""

import numpy as np


def to_json_friendly(obj):
    """递归将 numpy 类型和 tuple 转换为 Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_json_friendly(obj.tolist())
    if isinstance(obj, tuple):
        return [to_json_friendly(v) for v in obj]
    if isinstance(obj, list):
        return [to_json_friendly(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_json_friendly(v) for k, v in obj.items()}
    return obj


def normalize_state(state: str) -> str:
    """将内部状态字符串标准化为小写+下划线格式"""
    if not state:
        return "normal"
    s = state.lower().replace(" ", "_")
    if "fall" in s and "potential" not in s:
        return "fall"
    if "potential" in s:
        return "potential_fall"
    return "normal"


def format_person(person: dict, fall_result: dict) -> dict:
    """将内部 person dict 转换为标准化 person 输出"""
    state = normalize_state(fall_result.get("state", "normal"))

    # 关键点序列化：YOLO keypoints 是 (17,2) 或带 conf
    keypoints_raw = person.get("keypoints")
    confs_raw = person.get("confs")
    if keypoints_raw is not None:
        if confs_raw is not None and len(keypoints_raw) == len(confs_raw):
            keypoints = [
                [float(k[0]), float(k[1]), float(confs_raw[i])]
                for i, k in enumerate(keypoints_raw)
            ]
        else:
            keypoints = [
                [float(k[0]), float(k[1]), 0.0]
                for k in keypoints_raw
            ]
    else:
        keypoints = []

    return to_json_friendly({
        "track_id": person.get("pid"),
        "bbox": person.get("bbox"),
        "center": person.get("center"),
        "state": state,
        "fall_detected": bool(fall_result.get("fall_detected", False)),
        "confidence": float(fall_result.get("confidence", 0.0)),
        "is_ghost": bool(person.get("is_ghost", False)),
        "keypoints": keypoints,
    })


def format_result(
    tracked: list,
    camera_id: str,
    timestamp: float,
    frame_id: int,
    diagnostics: dict = None,
    module_name: str = "fall_detection",
) -> dict:
    """构建标准化 process_frame 返回值"""
    persons = []
    events = []

    for person in tracked:
        fr = person.get("fall_result", {
            "fall_detected": False,
            "confidence": 0.0,
            "state": "normal",
        })
        persons.append(format_person(person, fr))

        # 仅在确认跌倒时生成事件
        if fr.get("fall_detected"):
            events.append(to_json_friendly({
                "event_type": "fall_confirmed",
                "track_id": person.get("pid"),
                "camera_id": camera_id,
                "timestamp": timestamp,
                "confidence": float(fr.get("confidence", 0.0)),
                "bbox": person.get("bbox"),
                "state": normalize_state(fr.get("state", "fall")),
            }))

    result = to_json_friendly({
        "module": module_name,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "frame_id": frame_id,
        "persons": persons,
        "events": events,
        "diagnostics": diagnostics or {},
    })

    return result
