"""
摄像头处理进程模块
每个摄像头独立运行一个进程，通过 Queue 向主进程传递数据
"""

import cv2
import time
import logging
import torch
import numpy as np
from ultralytics import YOLO
from tracking import SingleCameraTracker
from cross_camera import get_color_histogram
from config import CAM_PROC as _CFG

logger = logging.getLogger(__name__)

# 从配置文件读取参数
ANGLE_KPTS_INDICES = _CFG.get('angle_kpts_indices', {'shoulder': 5, 'hip': 11, 'knee': 13})
MIN_CONF_FOR_ANGLE_KPTS = _CFG.get('min_conf_for_angle_kpts', 0.2)
ROI_ENABLED = _CFG.get('roi_enabled', True)
ROI_CONF = _CFG.get('roi_conf', 0.35)
ROI_EXPAND_RATIO = _CFG.get('roi_expand_ratio', 0.25)
ROI_MATCH_IOU = _CFG.get('roi_match_iou', 0.3)
ROI_MAX_TIME = _CFG.get('roi_max_time', 0.1)
IOU_DEDUP_THRESHOLD = _CFG.get('iou_dedup_threshold', 0.9)


def compute_iou(b1, b2):
    """计算两个 bbox 的 IoU"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def merge_detections(all_detections):
    """合并去重：IoU > 阈值的重复检测只保留一个"""
    merged = []
    for det in all_detections:
        dup = False
        for m in merged:
            if compute_iou(det['bbox'], m['bbox']) > IOU_DEDUP_THRESHOLD:
                dup = True
                break
        if not dup:
            merged.append(det)
    return merged


def run_roi_inference(model, frame, rois, tracker):
    """
    ROI 二次推理：仅在检测到潜在跌倒时，在预测位置附近小范围用低置信度重推理

    model: YOLO 模型
    frame: 当前帧
    rois: [{'bbox': (x1,y1,x2,y2), 'pid': int, 'predicted_center': (cx,cy)}, ...]
    tracker: SingleCameraTracker 实例（用于 extract_detections）

    返回: list of detection dicts
    """
    if not ROI_ENABLED or not rois:
        return []

    roi_detections = []
    roi_start = time.time()
    try:
        frame_h, frame_w = frame.shape[:2]
        for roi in rois:
            if time.time() - roi_start > ROI_MAX_TIME:
                break
            roi_bbox = roi['bbox']
            rx1, ry1, rx2, ry2 = roi_bbox
            w, h = rx2 - rx1, ry2 - ry1

            # 用预测位置做 ROI 中心（如果没有预测位置就用 bbox 中心）
            pred = roi.get('predicted_center')
            if pred is not None:
                pcx, pcy = pred
            else:
                pcx, pcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2

            # 以预测位置为中心，bbox 尺寸为基础，小范围扩展
            half_w = w * (0.5 + ROI_EXPAND_RATIO)
            half_h = h * (0.5 + ROI_EXPAND_RATIO)
            ex1 = max(0, int(pcx - half_w))
            ey1 = max(0, int(pcy - half_h))
            ex2 = min(frame_w, int(pcx + half_w))
            ey2 = min(frame_h, int(pcy + half_h))

            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            crop_results = model(crop, conf=ROI_CONF, verbose=False)[0]

            # 映射回全图坐标
            if crop_results.boxes is not None and len(crop_results.boxes) > 0:
                data = crop_results.boxes.data.clone()
                data[:, 0] += ex1
                data[:, 1] += ey1
                data[:, 2] += ex1
                data[:, 3] += ey1
                crop_results.boxes.data = data
            if crop_results.keypoints is not None:
                for kp in crop_results.keypoints:
                    if kp.xy is not None:
                        kp.data = kp.data.clone()
                        kp.data[..., 0] += ex1
                        kp.data[..., 1] += ey1

            # 只保留与 ROI 源 bbox 重叠的检测
            roi_dets = tracker.extract_detections(crop_results)
            for det in roi_dets:
                if compute_iou(det['bbox'], roi_bbox) >= ROI_MATCH_IOU:
                    roi_detections.append(det)
    except Exception as e:
        logger.warning(f"ROI 推理异常: {e}")
        roi_detections = []

    return roi_detections


def extract_angle_keypoints(keypoints, confs):
    """提取用于角度计算的关键点坐标（返回纯 Python float，避免 numpy 类型跨进程序列化问题）"""
    coords = {}
    for name, idx in ANGLE_KPTS_INDICES.items():
        if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_FOR_ANGLE_KPTS:
            coords[name] = (float(keypoints[idx][0]), float(keypoints[idx][1]))
        else:
            return None
    return coords


def camera_process(camera_id, queue, model_path, stop_event, is_video=False):
    """
    单个摄像头的处理进程

    camera_id: 摄像头索引(int) 或视频路径(str)
    queue: multiprocessing.Queue, 传递检测结果
    model_path: YOLO 模型路径
    stop_event: multiprocessing.Event, 停止信号
    is_video: 是否为视频文件
    """
    try:
        if is_video:
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(int(camera_id))

        if not cap.isOpened():
            logger.error(f"[Camera {camera_id}] 无法打开摄像头/视频")
            queue.put(None)
            return

        # 验证摄像头能实际读取帧
        ret, _ = cap.read()
        if not ret:
            logger.error(f"[Camera {camera_id}] 能打开但无法读取帧")
            cap.release()
            queue.put(None)
            return

        model = YOLO(model_path)
        tracker = SingleCameraTracker()

        # 预热摄像头：丢弃前几帧（有些摄像头需要几帧才稳定）
        for _ in range(5):
            cap.read()
        frame_count = 0
        fps = 0
        t0 = time.time()
        low_conf_rois = []  # ROI 二次推理区域

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # 重试几次再放弃
                for retry in range(3):
                    time.sleep(0.05)
                    ret, frame = cap.read()
                    if ret:
                        break
            if not ret:
                logger.error(f"[Camera {camera_id}] 视频结束或无法读取帧")
                break

            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - t0 + 1e-8)

            try:
                # ByteTracker 跟踪推理
                results = model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

                # 提取跟踪结果
                main_detections = tracker.extract_detections(results)
                tracked = tracker.update(main_detections, current_time)

                # ROI 二次推理：每 N 帧推理一次，关联到已有 track
                roi_interval = _CFG.get('roi_interval', 3)
                if frame_count % roi_interval == 0 and low_conf_rois:
                    roi_detections = run_roi_inference(model, frame, low_conf_rois, tracker)
                    tracker.update_roi(roi_detections, current_time)

                tracker.cleanup(current_time)
            except Exception as e:
                logger.warning(f"[Camera {camera_id}] 检测/跟踪异常: {e}")
                continue

            # 为每个跟踪目标计算直方图和角度关键点
            for person in tracked:
                # HSV 直方图
                person['hist'] = get_color_histogram(frame, person['bbox'])

                # 角度关键点（在 camera_process 中计算，数据有效）
                kpts = person.get('keypoints')
                confs = person.get('confs')
                if kpts is not None and confs is not None:
                    person['angle_keypoints'] = extract_angle_keypoints(kpts, confs)
                else:
                    person['angle_keypoints'] = None

            # 更新 ROI 二次推理区域（AR < 1.5 可能是躺卧，用预测位置做 ROI 中心）
            low_conf_rois = []
            for person in tracked:
                ar = person.get('aspect_ratio')
                if ar is not None and ar < 1.5:
                    pid = person['pid']
                    predicted = tracker._predict_position(tracker.person_history.get(pid, {}))
                    low_conf_rois.append({
                        'bbox': person['bbox'],
                        'pid': pid,
                        'predicted_center': predicted,
                    })

            # 传递给主进程（非阻塞，丢帧也不卡死）
            import queue as _queue
            try:
                while queue.full():
                    try:
                        queue.get_nowait()
                    except _queue.Empty:
                        break
                queue.put({
                    'camera_id': camera_id,
                    'frame': frame,
                    'results': results,
                    'tracked': tracked,
                    'fps': fps,
                    'timestamp': current_time,
                }, timeout=0.5)
            except (_queue.Full, TimeoutError):
                pass  # 队列满就丢帧

        cap.release()
        queue.put(None)

    except Exception as e:
        logger.error(f"[Camera {camera_id}] 进程异常: {e}")
        queue.put(None)
