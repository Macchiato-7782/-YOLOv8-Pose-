"""
实时跌倒检测系统
支持单摄像头和双摄像头模式

单摄像头: python main.py
双摄像头: python main.py --num_cams 2 --cam_ids 0 1
双视频:   python main.py --num_cams 2 --video cam1.mp4 cam2.mp4
Headless: python main.py --headless
边缘模式: python main.py --edge
"""

import cv2
import os
import sys
import time
import json
import logging
import argparse
import traceback
import multiprocessing as mp
from ultralytics import YOLO

logger = logging.getLogger(__name__)

from tracking import SingleCameraTracker
from cross_camera import match_cross_camera, remove_wrongly_matched, merge_tracking_ids
from fall_logic import evaluate_fall
from features import yolo_to_5keypoints
from camera_process import camera_process, extract_angle_keypoints, run_roi_inference
from config import CAM_PROC
from fall_detection.visualizer import draw_person_info, draw_fall_alert, draw_skeleton
from fall_detection.edge_config import EDGE_DEFAULTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, 'yolov8n-pose.pt')

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


# ============================================================
# 单摄像头模式（保持原有功能）
# ============================================================

def run_single_camera(args):
    """单摄像头模式"""
    model = YOLO(args.model)

    if args.video:
        cap = cv2.VideoCapture(args.video[0])
        if not cap.isOpened():
            logger.error(f"无法打开视频: {args.video[0]}")
            return
    else:
        cap = None
        cam_ids = args.cam_ids if args.cam_ids else [0]
        for cam_id in cam_ids:
            logger.info(f"尝试摄像头 {cam_id}...")
            test_cap = cv2.VideoCapture(cam_id)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    cap = test_cap
                    logger.info(f"摄像头 {cam_id} 可用")
                    break
                else:
                    logger.warning(f"摄像头 {cam_id} 能打开但无法读取帧")
                    test_cap.release()
            else:
                logger.warning(f"摄像头 {cam_id} 无法打开")
                test_cap.release()

        if cap is None:
            logger.error("所有摄像头均不可用")
            return

    tracker = SingleCameraTracker()
    output_video = None
    low_conf_rois = []
    frame_count = 0
    fps_t0 = time.time()

    logger.info("单摄像头模式启动，按 ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            for retry in range(3):
                time.sleep(0.05)
                ret, frame = cap.read()
                if ret:
                    break
        if not ret:
            logger.warning("视频结束或无法读取帧")
            break

        current_time = time.time()
        frame_count += 1

        try:
            results = model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            main_detections = tracker.extract_detections(results)
            tracked = tracker.update(main_detections, current_time)

            roi_interval = CAM_PROC.get('roi_interval', 3)
            if frame_count % roi_interval == 0 and low_conf_rois:
                roi_detections = run_roi_inference(model, frame, low_conf_rois, tracker)
                tracker.update_roi(roi_detections, current_time)

            tracker.cleanup(current_time)
        except Exception as e:
            logger.warning(f"检测/跟踪异常: {e}")
            traceback.print_exc()
            continue

        plotted_frame = draw_skeleton(frame, results)

        for person in tracked:
            try:
                if person.get('is_ghost'):
                    if 'fall_result' not in person:
                        person['fall_result'] = {'fall_detected': False, 'confidence': 0, 'state': 'Normal'}
                    plotted_frame = draw_person_info(plotted_frame, person, person['fall_result'])
                    continue

                kpts = person.get('keypoints')
                confs = person.get('confs')
                if kpts is not None and confs is not None:
                    person['angle_keypoints'] = extract_angle_keypoints(kpts, confs)
                else:
                    person['angle_keypoints'] = None

                fall_result = evaluate_fall(person, current_time)
                person['fall_result'] = fall_result
                plotted_frame = draw_person_info(plotted_frame, person, fall_result)
            except Exception as e:
                logger.warning(f"单摄处理异常: {e}")
                traceback.print_exc()

        low_conf_rois = []
        for person in tracked:
            if person.get('fall_result', {}).get('state') in ('Potential Fall', 'FALL'):
                pid = person['pid']
                predicted = tracker._predict_position(tracker.person_history.get(pid, {}))
                low_conf_rois.append({
                    'bbox': person['bbox'],
                    'pid': pid,
                    'predicted_center': predicted,
                })

        fall_results = []
        for p in tracked:
            if 'fall_result' in p:
                fr = dict(p['fall_result'])
                fr['is_ghost'] = p.get('is_ghost', False)
                fall_results.append(fr)
        plotted_frame = draw_fall_alert(plotted_frame, fall_results)

        fps = frame_count / (time.time() - fps_t0 + 1e-8)
        cv2.putText(plotted_frame, f"FPS: {fps:.0f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(plotted_frame, timestamp, (10, plotted_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if args.save_output and output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            output_video = cv2.VideoWriter(
                filename='output.avi', fourcc=fourcc,
                fps=18, frameSize=(plotted_frame.shape[1], plotted_frame.shape[0])
            )
        if output_video is not None:
            output_video.write(plotted_frame)

        cv2.imshow("Fall Detection - Single Camera", plotted_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            logger.info("ESC 退出")
            break

    cap.release()
    if output_video:
        output_video.release()
    cv2.destroyAllWindows()


# ============================================================
# 双摄像头模式
# ============================================================

def run_dual_camera(args):
    """双摄像头模式：多进程 + 跨摄像头匹配"""
    model_path = args.model
    stop_event = mp.Event()
    queue_a = mp.Queue(maxsize=1)
    queue_b = mp.Queue(maxsize=1)

    if args.video:
        cam_a = args.video[0]
        cam_b = args.video[1]
        is_video = True
    else:
        cam_a = args.cam_ids[0] if args.cam_ids else 0
        cam_b = args.cam_ids[1] if len(args.cam_ids) > 1 else 1
        is_video = False

    p1 = mp.Process(target=camera_process, args=(cam_a, queue_a, model_path, stop_event, is_video))
    p1.start()
    time.sleep(2)
    p2 = mp.Process(target=camera_process, args=(cam_b, queue_b, model_path, stop_event, is_video))
    p2.start()

    logger.info(f"双摄像头模式启动: Camera A={cam_a}, Camera B={cam_b}")
    logger.info("按 ESC 退出")

    output_video = None
    global_id_map = {}
    pid_to_global_store = {}
    cam_a_alive = True
    cam_b_alive = True
    data_a = None
    data_b = None
    fall_state_store = {}

    def get_latest(queue):
        import queue as _queue
        data = None
        while not queue.empty():
            try:
                newer = queue.get_nowait()
                if newer is None:
                    return 'STOP'
                data = newer
            except _queue.Empty:
                break
        return data

    while cam_a_alive or cam_b_alive:
        if cam_a_alive:
            new_a = get_latest(queue_a)
            if new_a == 'STOP':
                logger.info("摄像头 A 结束")
                cam_a_alive = False
                data_a = None
            elif new_a is not None:
                data_a = new_a

        if cam_b_alive:
            new_b = get_latest(queue_b)
            if new_b == 'STOP':
                logger.info("摄像头 B 结束")
                cam_b_alive = False
                data_b = None
            elif new_b is not None:
                data_b = new_b

        if data_a is None and data_b is None:
            if not cam_a_alive and not cam_b_alive:
                break
            time.sleep(0.01)
            continue

        current_time = (data_a or data_b)['timestamp']

        if data_a is not None:
            frame_a = data_a['frame'].copy()
            tracked_a = data_a['tracked']
            frame_a = draw_skeleton(frame_a, data_a['results'])
        else:
            frame_a = None
            tracked_a = []

        if data_b is not None:
            frame_b = data_b['frame'].copy()
            tracked_b = data_b['tracked']
            frame_b = draw_skeleton(frame_b, data_b['results'])
        else:
            frame_b = None
            tracked_b = []

        matched_pairs = []
        pid_to_global_a = {}
        pid_to_global_b = {}
        if tracked_a and tracked_b:
            matched_pairs = match_cross_camera(tracked_a, tracked_b)
            matched_pairs = remove_wrongly_matched(tracked_a, tracked_b, matched_pairs)
            pid_to_global = merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map, pid_to_global_store)

            for p in tracked_a:
                gid = pid_to_global.get(p['pid'])
                if gid is not None:
                    pid_to_global_a[p['pid']] = gid
            for p in tracked_b:
                gid = pid_to_global.get(p['pid'])
                if gid is not None:
                    pid_to_global_b[p['pid']] = gid
            pid_to_global_store = pid_to_global

        if frame_a is not None:
            for person in tracked_a:
                try:
                    store_key = ('A', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    gid = pid_to_global_a.get(person['pid'])

                    if person.get('is_ghost'):
                        if 'fall_result' not in person:
                            person['fall_result'] = {'fall_detected': False, 'confidence': 0, 'state': 'Normal'}
                        frame_a = draw_person_info(frame_a, person, person['fall_result'], global_id=gid)
                        continue

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_a[a_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_b[b_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']
                    frame_a = draw_person_info(frame_a, person, fall_result, global_id=gid)
                except Exception as e:
                    logger.warning(f"Camera A 处理异常: {e}")
                    traceback.print_exc()

        if frame_b is not None:
            for person in tracked_b:
                try:
                    store_key = ('B', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    gid = pid_to_global_b.get(person['pid'])

                    if person.get('is_ghost'):
                        if 'fall_result' not in person:
                            person['fall_result'] = {'fall_detected': False, 'confidence': 0, 'state': 'Normal'}
                        frame_b = draw_person_info(frame_b, person, person['fall_result'], global_id=gid)
                        continue

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_b[b_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_a[a_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']
                    frame_b = draw_person_info(frame_b, person, fall_result, global_id=gid)
                except Exception as e:
                    logger.warning(f"Camera B 处理异常: {e}")
                    traceback.print_exc()

        # 清理 fall_state_store
        active_keys = set()
        for p in tracked_a:
            active_keys.add(('A', p['pid']))
        for p in tracked_b:
            active_keys.add(('B', p['pid']))
        stale_keys = [k for k in fall_state_store if k not in active_keys]
        for k in stale_keys:
            del fall_state_store[k]

        # 清理 global_id_map
        active_pids = set()
        for p in tracked_a:
            active_pids.add(p['pid'])
        for p in tracked_b:
            active_pids.add(p['pid'])
        stale_gid_keys = [k for k in global_id_map if k[0] not in active_pids and k[1] not in active_pids]
        for k in stale_gid_keys:
            gid = global_id_map[k]
            del global_id_map[k]
            stale_ptg = [pid for pid, g in pid_to_global_store.items() if g == gid and pid not in active_pids]
            for pid in stale_ptg:
                del pid_to_global_store[pid]

        fall_results_a = []
        for p in tracked_a:
            if 'fall_result' in p:
                fr = dict(p['fall_result'])
                fr['is_ghost'] = p.get('is_ghost', False)
                fall_results_a.append(fr)
        fall_results_b = []
        for p in tracked_b:
            if 'fall_result' in p:
                fr = dict(p['fall_result'])
                fr['is_ghost'] = p.get('is_ghost', False)
                fall_results_b.append(fr)
        if frame_a is not None:
            frame_a = draw_fall_alert(frame_a, fall_results_a)
        if frame_b is not None:
            frame_b = draw_fall_alert(frame_b, fall_results_b)

        if frame_a is not None and data_a is not None:
            fps_a = data_a.get('fps', 0)
            cv2.putText(frame_a, f"FPS: {fps_a:.0f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if frame_b is not None and data_b is not None:
            fps_b = data_b.get('fps', 0)
            cv2.putText(frame_b, f"FPS: {fps_b:.0f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        timestamp = time.strftime("%H:%M:%S", time.localtime())
        if frame_a is not None:
            cv2.putText(frame_a, timestamp, (10, frame_a.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if frame_b is not None:
            cv2.putText(frame_b, timestamp, (10, frame_b.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if frame_a is not None and frame_b is not None:
            h = max(frame_a.shape[0], frame_b.shape[0])
            if frame_a.shape[0] != h:
                frame_a = cv2.resize(frame_a, (int(frame_a.shape[1] * h / frame_a.shape[0]), h))
            if frame_b.shape[0] != h:
                frame_b = cv2.resize(frame_b, (int(frame_b.shape[1] * h / frame_b.shape[0]), h))
            combined = frame_a.shape[1] + frame_b.shape[1]
            combined = np.hstack((frame_a, frame_b))
        elif frame_a is not None:
            combined = frame_a
        else:
            combined = frame_b

        if args.save_output and output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            output_video = cv2.VideoWriter(
                filename='output_dual.avi', fourcc=fourcc,
                fps=18, frameSize=(combined.shape[1], combined.shape[0])
            )
        if output_video is not None:
            output_video.write(combined)

        cv2.imshow("Dual Camera Fall Detection", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            logger.info("ESC 退出")
            stop_event.set()
            break

    p1.join(timeout=5)
    p2.join(timeout=5)
    if output_video:
        output_video.release()
    cv2.destroyAllWindows()


# ============================================================
# Headless / Edge 模式（使用 FallDetector 标准接口）
# ============================================================

def run_headless(args):
    """
    Headless 模式：使用 FallDetector 标准接口处理帧
    不显示 cv2.imshow，打印结构化结果摘要
    """
    import numpy as np
    from fall_detection import FallDetector

    # 根据参数构造 FallDetector
    detector_kwargs = {
        "model_path": args.model,
        "config_path": os.path.join(SCRIPT_DIR, "config.yaml"),
        "enable_visualization": False,
        "enable_roi": False,
    }

    if args.edge:
        detector_kwargs.update({
            "device": EDGE_DEFAULTS["device"],
            "backend": EDGE_DEFAULTS["backend"],
            "input_size": EDGE_DEFAULTS["input_size"],
            "inference_interval": EDGE_DEFAULTS["inference_interval"],
            "enable_roi": EDGE_DEFAULTS["enable_roi"],
            "max_persons": EDGE_DEFAULTS["max_persons"],
            "conf": EDGE_DEFAULTS["conf"],
        })
        logger.info("使用边缘设备默认配置")
    else:
        detector_kwargs.update({
            "input_size": 640,
            "inference_interval": 1,
            "max_persons": None,
        })

    # 打开视频源
    if args.video:
        cap = cv2.VideoCapture(args.video[0])
        video_src = args.video[0]
        if not cap.isOpened():
            logger.error(f"无法打开视频: {args.video[0]}")
            return
    else:
        cap = None
        cam_ids = args.cam_ids if args.cam_ids else [0]
        cam_id = cam_ids[0]
        logger.info(f"尝试摄像头 {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            logger.error(f"摄像头 {cam_id} 无法打开")
            return
        ret, _ = cap.read()
        if not ret:
            logger.error(f"摄像头 {cam_id} 能打开但无法读取帧")
            cap.release()
            return
        video_src = f"camera_{cam_id}"

    detector = FallDetector(**detector_kwargs)
    output_video = None
    frame_count = 0
    camera_id = f"cam_{args.cam_ids[0]}" if args.cam_ids else "cam_0"

    logger.info(f"Headless 模式启动，源: {video_src}，按 Ctrl+C 退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                for retry in range(3):
                    time.sleep(0.05)
                    ret, frame = cap.read()
                    if ret:
                        break
            if not ret:
                logger.warning("视频结束或无法读取帧")
                break

            frame_count += 1
            result = detector.process_frame(frame, camera_id=camera_id)

            # 打印结构化摘要
            persons_summary = []
            for p in result.get("persons", []):
                persons_summary.append(
                    f"ID:{p['track_id']} state:{p['state']} conf:{p['confidence']:.2f}"
                )

            status_line = (
                f"Frame {result['frame_id']:>5d} | "
                f"FPS {result['diagnostics'].get('fps', 0):.1f} | "
                f"Persons: {len(result['persons'])} | "
                f"Inference: {result['diagnostics'].get('inference_ran', False)}"
            )
            if persons_summary:
                status_line += f" | [{', '.join(persons_summary)}]"

            print(f"\r{status_line}", end="", flush=True)

            # 事件日志
            for event in result.get("events", []):
                print(f"\n[EVENT] {event['event_type']} | "
                      f"cam={event['camera_id']} track={event['track_id']} "
                      f"conf={event['confidence']:.2f}")

            # 视频保存
            if args.save_output:
                annotated = result.get("annotated_frame")
                if annotated is None:
                    annotated = frame
                if output_video is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MP42')
                    output_video = cv2.VideoWriter(
                        filename='output_headless.avi', fourcc=fourcc,
                        fps=18, frameSize=(annotated.shape[1], annotated.shape[0])
                    )
                if output_video is not None:
                    output_video.write(annotated)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        print()  # 换行
        cap.release()
        if output_video:
            output_video.release()
        detector.close()


def run_dual_headless(args):
    """
    双摄像头 Headless 模式：使用双进程推理，主进程通过 FallDetector 逻辑汇总结果
    不显示 cv2.imshow
    """
    import numpy as np
    from fall_detection import FallDetector
    from fall_detection.schemas import format_result

    model_path = args.model
    stop_event = mp.Event()
    queue_a = mp.Queue(maxsize=1)
    queue_b = mp.Queue(maxsize=1)

    if args.video:
        cam_a = args.video[0]
        cam_b = args.video[1]
        is_video = True
    else:
        cam_a = args.cam_ids[0] if args.cam_ids else 0
        cam_b = args.cam_ids[1] if len(args.cam_ids) > 1 else 1
        is_video = False

    p1 = mp.Process(target=camera_process, args=(cam_a, queue_a, model_path, stop_event, is_video))
    p1.start()
    time.sleep(2)
    p2 = mp.Process(target=camera_process, args=(cam_b, queue_b, model_path, stop_event, is_video))
    p2.start()

    logger.info(f"双摄像头 Headless 模式启动: Camera A={cam_a}, Camera B={cam_b}")
    logger.info("按 Ctrl+C 退出")

    output_video = None
    global_id_map = {}
    pid_to_global_store = {}
    cam_a_alive = True
    cam_b_alive = True
    data_a = None
    data_b = None
    fall_state_store = {}
    frame_count = 0

    def get_latest(queue):
        import queue as _queue
        data = None
        while not queue.empty():
            try:
                newer = queue.get_nowait()
                if newer is None:
                    return 'STOP'
                data = newer
            except _queue.Empty:
                break
        return data

    try:
        while cam_a_alive or cam_b_alive:
            if cam_a_alive:
                new_a = get_latest(queue_a)
                if new_a == 'STOP':
                    logger.info("摄像头 A 结束")
                    cam_a_alive = False
                    data_a = None
                elif new_a is not None:
                    data_a = new_a

            if cam_b_alive:
                new_b = get_latest(queue_b)
                if new_b == 'STOP':
                    logger.info("摄像头 B 结束")
                    cam_b_alive = False
                    data_b = None
                elif new_b is not None:
                    data_b = new_b

            if data_a is None and data_b is None:
                if not cam_a_alive and not cam_b_alive:
                    break
                time.sleep(0.01)
                continue

            current_time = (data_a or data_b)['timestamp']
            frame_count += 1

            if data_a is not None:
                tracked_a = data_a['tracked']
            else:
                tracked_a = []

            if data_b is not None:
                tracked_b = data_b['tracked']
            else:
                tracked_b = []

            matched_pairs = []
            pid_to_global_a = {}
            pid_to_global_b = {}
            if tracked_a and tracked_b:
                matched_pairs = match_cross_camera(tracked_a, tracked_b)
                matched_pairs = remove_wrongly_matched(tracked_a, tracked_b, matched_pairs)
                pid_to_global = merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map, pid_to_global_store)

                for p in tracked_a:
                    gid = pid_to_global.get(p['pid'])
                    if gid is not None:
                        pid_to_global_a[p['pid']] = gid
                for p in tracked_b:
                    gid = pid_to_global.get(p['pid'])
                    if gid is not None:
                        pid_to_global_b[p['pid']] = gid
                pid_to_global_store = pid_to_global

            # Camera A 跌倒判断
            for person in tracked_a:
                try:
                    store_key = ('A', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    if person.get('is_ghost'):
                        if 'fall_result' not in person:
                            person['fall_result'] = {'fall_detected': False, 'confidence': 0, 'state': 'Normal'}
                        continue

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_a[a_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_b[b_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']
                except Exception as e:
                    logger.warning(f"Camera A 处理异常: {e}")

            # Camera B 跌倒判断
            for person in tracked_b:
                try:
                    store_key = ('B', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    if person.get('is_ghost'):
                        if 'fall_result' not in person:
                            person['fall_result'] = {'fall_detected': False, 'confidence': 0, 'state': 'Normal'}
                        continue

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_b[b_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_a[a_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']
                except Exception as e:
                    logger.warning(f"Camera B 处理异常: {e}")

            # 清理
            active_keys = set()
            for p in tracked_a:
                active_keys.add(('A', p['pid']))
            for p in tracked_b:
                active_keys.add(('B', p['pid']))
            stale_keys = [k for k in fall_state_store if k not in active_keys]
            for k in stale_keys:
                del fall_state_store[k]

            active_pids = set()
            for p in tracked_a:
                active_pids.add(p['pid'])
            for p in tracked_b:
                active_pids.add(p['pid'])
            stale_gid_keys = [k for k in global_id_map if k[0] not in active_pids and k[1] not in active_pids]
            for k in stale_gid_keys:
                del global_id_map[k]

            # 汇总 headless 输出
            all_tracked = tracked_a + tracked_b  # 简化：合并两个摄像头的结果
            # 为 Camera B 的 person 加上 camera 标识
            for p in tracked_b:
                p['_camera'] = 'B'

            # 打印状态摘要
            status_parts = []
            events_this_frame = []
            for camera_label, tracked_list in [('A', tracked_a), ('B', tracked_b)]:
                for p in tracked_list:
                    fr = p.get('fall_result', {})
                    state = fr.get('state', 'Normal')
                    pid = p['pid']
                    status_parts.append(f"{camera_label}[ID:{pid} {state}]")
                    if fr.get('fall_detected'):
                        events_this_frame.append(
                            f"[EVENT] cam_{camera_label} track={pid} conf={fr.get('confidence', 0):.2f}"
                        )

            fps_val = data_a.get('fps', 0) if data_a else data_b.get('fps', 0) if data_b else 0
            print(f"\rFrame {frame_count:>5d} | FPS {fps_val:.1f} | "
                  f"{' '.join(status_parts)}", end="", flush=True)

            for evt in events_this_frame:
                print(f"\n{evt}")

            # 视频保存
            if args.save_output:
                frame_a_vis = data_a['frame'].copy() if data_a else None
                frame_b_vis = data_b['frame'].copy() if data_b else None

                if frame_a_vis is not None and frame_b_vis is not None:
                    h = max(frame_a_vis.shape[0], frame_b_vis.shape[0])
                    if frame_a_vis.shape[0] != h:
                        frame_a_vis = cv2.resize(frame_a_vis, (int(frame_a_vis.shape[1] * h / frame_a_vis.shape[0]), h))
                    if frame_b_vis.shape[0] != h:
                        frame_b_vis = cv2.resize(frame_b_vis, (int(frame_b_vis.shape[1] * h / frame_b_vis.shape[0]), h))
                    combined = np.hstack((frame_a_vis, frame_b_vis))
                elif frame_a_vis is not None:
                    combined = frame_a_vis
                else:
                    combined = frame_b_vis

                if output_video is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MP42')
                    output_video = cv2.VideoWriter(
                        filename='output_dual_headless.avi', fourcc=fourcc,
                        fps=18, frameSize=(combined.shape[1], combined.shape[0])
                    )
                if output_video is not None:
                    output_video.write(combined)

    except KeyboardInterrupt:
        logger.info("用户中断")
        stop_event.set()
    finally:
        print()
        p1.join(timeout=5)
        p2.join(timeout=5)
        if output_video:
            output_video.release()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实时跌倒检测系统 - 支持单/双摄像头",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='YOLOv8 姿态估计模型路径')
    parser.add_argument('--num_cams', type=int, default=1,
                        help='摄像头数量 (1 或 2)')
    parser.add_argument('--cam_ids', type=int, nargs='+', default=[0],
                        help='摄像头 ID 列表')
    parser.add_argument('--video', type=str, nargs='+', default=None,
                        help='视频文件路径（替代摄像头）')
    parser.add_argument('--save_output', action='store_true',
                        help='保存输出视频')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试日志')
    parser.add_argument('--headless', action='store_true',
                        help='Headless 模式：不显示窗口，打印结构化结果')
    parser.add_argument('--edge', action='store_true',
                        help='边缘设备模式：使用低功耗默认配置')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.headless:
        if args.num_cams == 1:
            run_headless(args)
        elif args.num_cams == 2:
            run_dual_headless(args)
        else:
            logger.error(f"不支持 {args.num_cams} 个摄像头")
    elif args.num_cams == 1:
        run_single_camera(args)
    elif args.num_cams == 2:
        run_dual_camera(args)
    else:
        logger.error(f"不支持 {args.num_cams} 个摄像头，目前只支持 1 或 2")


if __name__ == "__main__":
    main()
