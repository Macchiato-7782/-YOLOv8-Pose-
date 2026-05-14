"""
可视化绘制函数
从 main.py 迁移，供 FallDetector 和 main.py 共用
"""

import cv2
import time


def draw_person_info(frame, person, fall_result, global_id=None):
    """在画面上绘制单个人的信息"""
    pid = person.get("pid", "?")
    x1, y1, x2, y2 = person["bbox"]
    state = fall_result["state"]
    is_ghost = person.get("is_ghost", False)

    if is_ghost:
        color = (128, 128, 128)
        for i in range(x1, x2, 10):
            cv2.line(frame, (i, y1), (min(i + 5, x2), y1), color, 2)
            cv2.line(frame, (i, y2), (min(i + 5, x2), y2), color, 2)
        for i in range(y1, y2, 10):
            cv2.line(frame, (x1, i), (x1, min(i + 5, y2)), color, 2)
            cv2.line(frame, (x2, i), (x2, min(i + 5, y2)), color, 2)
        label = f"ID {pid}: LOST"
        text_y = min(y1 + 20, y2 - 5)
        text_x = max(x1 + 3, 5)
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    if "FALL" in state:
        color = (0, 0, 255)
        thickness = 4
    elif state == "Potential Fall":
        color = (0, 165, 255)
        thickness = 3
    else:
        color = (0, 255, 0)
        thickness = 2

    if "FALL" in state:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"G{global_id}: {state}" if global_id is not None else f"ID {pid}: {state}"

    fall_state = person.get("fall_state", {})
    if fall_state.get("is_potential_fall") and fall_state.get("fall_start_time"):
        duration = time.time() - fall_state["fall_start_time"]
        label += f" ({duration:.1f}s)"

    if fall_result.get("confidence", 0) > 0:
        label += f" [{fall_result['confidence']:.0%}]"

    font_scale = 0.8 if "FALL" in state else 0.6
    text_y = min(y1 + 25, y2 - 5)
    text_x = max(x1 + 3, 5)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    bg_x2 = min(text_x + tw + 4, frame.shape[1] - 2)
    bg_y1 = max(text_y - th - 4, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 2, bg_y1), (bg_x2, text_y + baseline + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame


def draw_fall_alert(frame, fall_results):
    """如果检测到跌倒，在画面顶部画警告横幅（排除幽灵目标）"""
    has_fall = any("FALL" in r["state"] and not r.get("is_ghost") for r in fall_results)
    if not has_fall:
        return frame

    h, w = frame.shape[:2]
    blink = int(time.time() * 3) % 2 == 0
    banner_color = (0, 0, 200) if blink else (0, 0, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), banner_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "!! FALL DETECTED !!", (w // 2 - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    return frame


def draw_skeleton(frame, results):
    """使用 YOLO 内置方法绘制骨骼"""
    return results.plot()
