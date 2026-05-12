"""
摄像头处理进程模块
每个摄像头独立运行一个进程，通过 Queue 向主进程传递数据
"""

import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from tracking import SingleCameraTracker
from cross_camera import get_color_histogram

# YOLO 关键点索引（用于髋部角度计算）
ANGLE_KPTS_INDICES = {
    'shoulder': 5,
    'hip': 11,
    'knee': 13
}
MIN_CONF_FOR_ANGLE_KPTS = 0.2
ROI_ENABLED = True       # ROI 二次推理开关
ROI_CONF = 0.35         # ROI 二次推理置信度（比主推理低，但比 0.2 高）
ROI_EXPAND_RATIO = 0.25 # ROI 向外扩展比例
ROI_MATCH_IOU = 0.3     # ROI 检测结果必须与已有目标 IoU > 此值才保留（过滤假检）
ROI_MAX_TIME = 0.1      # ROI 推理最大耗时（秒），超时跳过
IOU_DEDUP_THRESHOLD = 0.9  # 去重 IoU 阈值


def compute_iou(b1, b2):
    """计算两个 bbox 的 IoU"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


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
            print(f"[Camera {camera_id}] 无法打开摄像头/视频")
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
                print(f"[Camera {camera_id}] 视频结束或无法读取帧")
                break

            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - t0 + 1e-8)

            # YOLO 推理（降低阈值以检测躺卧姿态）
            results = model(frame, conf=0.3, verbose=False)[0]

            # 提取主推理检测结果
            main_detections = tracker.extract_detections(results)

            # ROI 二次推理：每 3 帧推理一次（降低开销）
            roi_detections = []
            roi_start = time.time()
            if ROI_ENABLED and low_conf_rois and frame_count % 3 == 0:
                try:
                    frame_h, frame_w = frame.shape[:2]
                    for roi in low_conf_rois:
                        if time.time() - roi_start > ROI_MAX_TIME:
                            break  # ROI 超时，跳过剩余
                        roi_bbox = roi['bbox']
                        rx1, ry1, rx2, ry2 = roi_bbox
                        w, h = rx2 - rx1, ry2 - ry1
                        ex1 = max(0, int(rx1 - w * ROI_EXPAND_RATIO))
                        ey1 = max(0, int(ry1 - h * ROI_EXPAND_RATIO))
                        ex2 = min(frame_w, int(rx2 + w * ROI_EXPAND_RATIO))
                        ey2 = min(frame_h, int(ry2 + h * ROI_EXPAND_RATIO))

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
                    print(f"[Camera {camera_id}] ROI 推理异常: {e}")
                    roi_detections = []

            # 合并去重：IoU > 0.9 保留高分那个
            all_detections = main_detections + roi_detections
            merged = []
            for det in all_detections:
                dup = False
                for i, m in enumerate(merged):
                    if compute_iou(det['bbox'], m['bbox']) > IOU_DEDUP_THRESHOLD:
                        dup = True
                        break
                if not dup:
                    merged.append(det)

            # 跟踪更新（用合并后的 detections）
            tracked = tracker.update(merged, current_time)
            tracker.cleanup(current_time)

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

            # 更新 ROI 二次推理区域
            low_conf_rois = []
            for person in tracked:
                # 用宽高比判断是否需要 ROI 二次推理（AR < 1.5 可能是躺卧）
                ar = person.get('aspect_ratio')
                if ar is not None and ar < 1.5:
                    low_conf_rois.append({'bbox': person['bbox'], 'pid': person['pid']})

            # 传递给主进程（非阻塞，丢帧也不卡死）
            try:
                while queue.full():
                    try:
                        queue.get_nowait()
                    except Exception:
                        break
                queue.put({
                    'camera_id': camera_id,
                    'frame': frame,
                    'results': results,
                    'tracked': tracked,
                    'fps': fps,
                    'timestamp': current_time,
                }, timeout=0.5)
            except Exception:
                pass  # 队列满就丢帧

        cap.release()
        queue.put(None)

    except Exception as e:
        print(f"[Camera {camera_id}] 进程异常: {e}")
        queue.put(None)
