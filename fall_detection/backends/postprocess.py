"""
统一后处理模块
将不同后端的原始输出转换为统一 detection schema
"""

import numpy as np


def ultralytics_results_to_detections(results):
    """
    将 ultralytics Results 对象转换为统一检测列表

    Args:
        results: ultralytics YOLO Results 对象 (单个元素或列表)

    Returns:
        list[dict]: 统一 detection schema
    """
    detections = []

    if hasattr(results, "boxes") and results.boxes is not None:
        raw_xyxy = results.boxes.xyxy
        raw_conf = results.boxes.conf
        if hasattr(raw_xyxy, 'cpu'):
            raw_xyxy = raw_xyxy.cpu().numpy()
        if hasattr(raw_conf, 'cpu'):
            raw_conf = raw_conf.cpu().numpy()
        bboxes = raw_xyxy if isinstance(raw_xyxy, np.ndarray) else np.array(raw_xyxy)
        scores = raw_conf if isinstance(raw_conf, np.ndarray) else np.array(raw_conf)
        track_ids = None
        if hasattr(results.boxes, "id") and results.boxes.id is not None:
            raw_id = results.boxes.id
            if hasattr(raw_id, 'cpu'):
                track_ids = raw_id.cpu().numpy().astype(int)
            else:
                track_ids = np.array(raw_id).astype(int)
        # handle empty
        if bboxes.ndim == 1 and len(bboxes) == 0:
            bboxes = np.zeros((0, 4))
        elif len(bboxes) == 0:
            bboxes = np.zeros((0, 4))

        keypoints = None
        if hasattr(results, "keypoints") and results.keypoints is not None:
            kp_xy = results.keypoints.xy
            kp_conf = results.keypoints.conf
            if kp_xy is not None and kp_conf is not None:
                pass  # will be extracted per-detection below

        for i in range(len(bboxes)):
            kps = []
            if (results.keypoints is not None and
                    results.keypoints.xy is not None and
                    results.keypoints.conf is not None):
                kp_xy_arr = results.keypoints.xy[i]
                kp_conf_arr = results.keypoints.conf[i]
                if hasattr(kp_xy_arr, 'cpu'):
                    kp_xy_arr = kp_xy_arr.cpu().numpy()
                if hasattr(kp_conf_arr, 'cpu'):
                    kp_conf_arr = kp_conf_arr.cpu().numpy()
                kps = [
                    [float(x), float(y), float(c)]
                    for (x, y), c in zip(kp_xy_arr, kp_conf_arr)
                ]

            detections.append({
                "bbox": [float(v) for v in bboxes[i]],
                "score": float(scores[i]),
                "class_id": 0,
                "track_id": int(track_ids[i]) if track_ids is not None else None,
                "keypoints": kps,
            })

    return detections


def onnx_outputs_to_detections(outputs, frame_shape, conf_threshold=0.35, iou_threshold=0.45):
    """
    将 ONNX YOLOv8-Pose 输出转换为统一检测列表

    ONNX 输出格式: [batch, 56, 8400] 或类似结构
    - 前 4 个通道: bbox cx, cy, w, h
    - 第 4 个: 置信度
    - 后续 51 个: 17 * 3 = 51 关键点 (x, y, conf)
    总计: 4 + 1 + 51 = 56

    Args:
        outputs: ONNX session.run() 的输出 (通常是单个 numpy array [1, 56, N])
        frame_shape: (H, W) 原始图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        list[dict]: 统一 detection schema
    """
    if isinstance(outputs, (list, tuple)):
        output = outputs[0]
    else:
        output = outputs

    # 处理不同形状的输出
    if output.ndim == 3:
        output = output[0]  # [56, N]

    if output.shape[0] < 5:
        return []

    # 转置为 [N, 56]
    if output.shape[0] > output.shape[1]:
        output = output.T

    frame_h, frame_w = frame_shape[:2]

    # 提取各分量
    cx = output[:, 0]
    cy = output[:, 1]
    w = output[:, 2]
    h = output[:, 3]
    obj_conf = output[:, 4]

    # 转换为 x1, y1, x2, y2
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 置信度过滤
    mask = obj_conf > conf_threshold
    if not np.any(mask):
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    obj_conf = obj_conf[mask]
    output = output[mask]

    # 缩放到原始图像尺寸
    x1 *= frame_w
    y1 *= frame_h
    x2 *= frame_w
    y2 *= frame_h

    # NMS
    areas = (x2 - x1) * (y2 - y1)
    order = obj_conf.argsort()[::-1]

    keep = []
    indices = np.arange(len(x1))
    while len(order) > 0:
        idx = order[0]
        keep.append(indices[idx])
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[order[1:]], x1[idx])
        yy1 = np.maximum(y1[order[1:]], y1[idx])
        xx2 = np.minimum(x2[order[1:]], x2[idx])
        yy2 = np.minimum(y2[order[1:]], y2[idx])

        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        iou = inter / (areas[order[1:]] + areas[idx] - inter)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    detections = []
    for idx in keep:
        kps = []
        try:
            kp_data = output[idx, 5:56]  # 51 values
            if len(kp_data) >= 51:
                for j in range(17):
                    kx = float(kp_data[j * 3] * frame_w)
                    ky = float(kp_data[j * 3 + 1] * frame_h)
                    kc = float(1.0 / (1.0 + np.exp(-kp_data[j * 3 + 2])))
                    kps.append([kx, ky, kc])
        except (IndexError, ValueError):
            kps = []

        detections.append({
            "bbox": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
            "score": float(obj_conf[idx]),
            "class_id": 0,
            "track_id": None,
            "keypoints": kps,
        })

    return detections
