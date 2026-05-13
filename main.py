"""
实时跌倒检测系统
支持单摄像头和双摄像头模式

单摄像头: python main.py
双摄像头: python main.py --num_cams 2 --cam_ids 0 1
双视频:   python main.py --num_cams 2 --video cam1.mp4 cam2.mp4
"""

import cv2
import os
import time
import logging
import torch
import argparse
import numpy as np
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, 'yolov8n-pose.pt')

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


# ============================================================
# 绘图工具
# ============================================================

def draw_person_info(frame, person, fall_result, global_id=None):
    """在画面上绘制单个人的信息"""
    pid = person.get('pid', '?')
    x1, y1, x2, y2 = person['bbox']
    state = fall_result['state']
    is_ghost = person.get('is_ghost', False)

    # 幽灵目标：灰色虚线框，不画红色警报
    if is_ghost:
        color = (128, 128, 128)  # 灰色
        # 虚线效果：每隔 10px 画一段
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

    # 颜色和边框粗细
    if 'FALL' in state:
        color = (0, 0, 255)  # 红色
        thickness = 4
    elif state == 'Potential Fall':
        color = (0, 165, 255)  # 橙色
        thickness = 3
    else:
        color = (0, 255, 0)  # 绿色
        thickness = 2

    # 确认跌倒：半透明红色遮罩
    if 'FALL' in state:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # 边框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 标签
    label = f"G{global_id}: {state}" if global_id is not None else f"ID {pid}: {state}"

    fall_state = person.get('fall_state', {})
    if fall_state.get('is_potential_fall') and fall_state.get('fall_start_time'):
        duration = time.time() - fall_state['fall_start_time']
        label += f" ({duration:.1f}s)"

    if fall_result.get('confidence', 0) > 0:
        label += f" [{fall_result['confidence']:.0%}]"

    # 标签文字（在 bbox 内部顶部，避免与相邻框重叠）
    font_scale = 0.8 if 'FALL' in state else 0.6
    text_y = min(y1 + 25, y2 - 5)
    text_x = max(x1 + 3, 5)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    # 半透明背景矩形
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
    has_fall = any('FALL' in r['state'] and not r.get('is_ghost') for r in fall_results)
    if not has_fall:
        return frame

    h, w = frame.shape[:2]
    # 闪烁效果：根据时间取反色
    blink = int(time.time() * 3) % 2 == 0
    banner_color = (0, 0, 200) if blink else (0, 0, 255)

    # 红色横幅
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), banner_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 警告文字
    cv2.putText(frame, "!! FALL DETECTED !!", (w // 2 - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    return frame


def draw_skeleton(frame, results):
    """使用 YOLO 内置方法绘制骨骼"""
    return results.plot()


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
            # 摄像头偶尔会读取失败，重试3次再放弃
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
            # ByteTracker 跟踪推理（卡尔曼滤波 + 级联匹配）
            results = model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

            # 提取跟踪结果
            main_detections = tracker.extract_detections(results)
            tracked = tracker.update(main_detections, current_time)

            # ROI 二次推理：每 N 帧推理一次，关联到已有 track
            roi_interval = CAM_PROC.get('roi_interval', 3)
            if frame_count % roi_interval == 0 and low_conf_rois:
                roi_detections = run_roi_inference(model, frame, low_conf_rois, tracker)
                tracker.update_roi(roi_detections, current_time)

            tracker.cleanup(current_time)
        except Exception as e:
            logger.warning(f"检测/跟踪异常: {e}")
            traceback.print_exc()
            continue

        # 绘制骨骼
        plotted_frame = draw_skeleton(frame, results)

        # 跌倒判断 + 绘制
        for person in tracked:
            try:
                if person.get('is_ghost'):
                    # 幽灵目标：保留之前的 fall_result，跳过重新计算
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

        # 更新 ROI 二次推理区域（仅潜在跌倒目标，用预测位置做 ROI 中心）
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

        # 跌倒警告横幅（传入 is_ghost 标记）
        fall_results = []
        for p in tracked:
            if 'fall_result' in p:
                fr = dict(p['fall_result'])
                fr['is_ghost'] = p.get('is_ghost', False)
                fall_results.append(fr)
        plotted_frame = draw_fall_alert(plotted_frame, fall_results)

        # FPS 显示
        fps = frame_count / (time.time() - fps_t0 + 1e-8)
        cv2.putText(plotted_frame, f"FPS: {fps:.0f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 时间戳水印
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(plotted_frame, timestamp, (10, plotted_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 保存输出
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

    # 确定摄像头来源
    if args.video:
        cam_a = args.video[0]
        cam_b = args.video[1]
        is_video = True
    else:
        cam_a = args.cam_ids[0] if args.cam_ids else 0
        cam_b = args.cam_ids[1] if len(args.cam_ids) > 1 else 1
        is_video = False

    # 启动两个摄像头进程（错开启动避免资源冲突）
    p1 = mp.Process(target=camera_process, args=(cam_a, queue_a, model_path, stop_event, is_video))
    p1.start()
    time.sleep(2)  # 等第一个摄像头稳定后再启动第二个
    p2 = mp.Process(target=camera_process, args=(cam_b, queue_b, model_path, stop_event, is_video))
    p2.start()

    logger.info(f"双摄像头模式启动: Camera A={cam_a}, Camera B={cam_b}")
    logger.info("按 ESC 退出")

    output_video = None
    global_id_map = {}
    pid_to_global_store = {}  # 反向索引：pid → global_id
    cam_a_alive = True
    cam_b_alive = True
    data_a = None
    data_b = None
    # 主进程持久化 fall_state（camera_process 每帧传来的是新 dict，必须在主进程跨帧保存）
    fall_state_store = {}  # key: (camera_label, pid), value: fall_state dict

    def get_latest(queue):
        """排空队列旧帧，只取最新一帧（非阻塞，无数据返回 None）"""
        import queue as _queue
        data = None
        while not queue.empty():
            try:
                newer = queue.get_nowait()
                if newer is None:
                    return 'STOP'  # 摄像头进程结束信号
                data = newer
            except _queue.Empty:
                break
        return data

    while cam_a_alive or cam_b_alive:
        # 非阻塞获取最新帧
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

        # 两边都没数据，等一下再试
        if data_a is None and data_b is None:
            if not cam_a_alive and not cam_b_alive:
                break
            time.sleep(0.01)
            continue

        # 至少有一边有数据，取 current_time
        current_time = (data_a or data_b)['timestamp']

        # 处理摄像头 A
        if data_a is not None:
            frame_a = data_a['frame'].copy()
            tracked_a = data_a['tracked']
            frame_a = draw_skeleton(frame_a, data_a['results'])
        else:
            frame_a = None
            tracked_a = []

        # 处理摄像头 B
        if data_b is not None:
            frame_b = data_b['frame'].copy()
            tracked_b = data_b['tracked']
            frame_b = draw_skeleton(frame_b, data_b['results'])
        else:
            frame_b = None
            tracked_b = []

        # 跨摄像头匹配（两边都有数据时才做）
        matched_pairs = []
        pid_to_global_a = {}
        pid_to_global_b = {}
        if tracked_a and tracked_b:
            matched_pairs = match_cross_camera(tracked_a, tracked_b)
            matched_pairs = remove_wrongly_matched(tracked_a, tracked_b, matched_pairs)
            pid_to_global = merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map, pid_to_global_store)

            # 从 pid_to_global 反向索引直接构建 pid→global 映射
            for p in tracked_a:
                gid = pid_to_global.get(p['pid'])
                if gid is not None:
                    pid_to_global_a[p['pid']] = gid
            for p in tracked_b:
                gid = pid_to_global.get(p['pid'])
                if gid is not None:
                    pid_to_global_b[p['pid']] = gid
            pid_to_global_store = pid_to_global

        # 跌倒判断 + 绘制（摄像头 A）
        if frame_a is not None:
            for person in tracked_a:
                try:
                    # 从主进程持久化存储恢复 fall_state（camera_process 每帧传来新 dict）
                    store_key = ('A', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    gid = pid_to_global_a.get(person['pid'])

                    if person.get('is_ghost'):
                        # 幽灵目标：保留之前的 fall_result，跳过重新计算
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
                    fall_state_store[store_key] = person['fall_state']  # 持久化
                    frame_a = draw_person_info(frame_a, person, fall_result, global_id=gid)
                except Exception as e:
                    logger.warning(f"Camera A 处理异常: {e}")
                    traceback.print_exc()

        # 跌倒判断 + 绘制（摄像头 B）
        if frame_b is not None:
            for person in tracked_b:
                try:
                    # 从主进程持久化存储恢复 fall_state
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
                    fall_state_store[store_key] = person['fall_state']  # 持久化
                    frame_b = draw_person_info(frame_b, person, fall_result, global_id=gid)
                except Exception as e:
                    logger.warning(f"Camera B 处理异常: {e}")
                    traceback.print_exc()

        # 清理 fall_state_store 中消失的 PID（防止内存泄漏）
        active_keys = set()
        for p in tracked_a:
            active_keys.add(('A', p['pid']))
        for p in tracked_b:
            active_keys.add(('B', p['pid']))
        stale_keys = [k for k in fall_state_store if k not in active_keys]
        for k in stale_keys:
            del fall_state_store[k]

        # 清理 global_id_map 中长时间未出现的条目（防止内存泄漏）
        active_pids = set()
        for p in tracked_a:
            active_pids.add(p['pid'])
        for p in tracked_b:
            active_pids.add(p['pid'])
        stale_gid_keys = [k for k in global_id_map if k[0] not in active_pids and k[1] not in active_pids]
        for k in stale_gid_keys:
            gid = global_id_map[k]
            del global_id_map[k]
            # 清理反向索引中对应的条目
            stale_ptg = [pid for pid, g in pid_to_global_store.items() if g == gid and pid not in active_pids]
            for pid in stale_ptg:
                del pid_to_global_store[pid]

        # 跌倒警告横幅（传入 is_ghost 标记）
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

        # FPS 信息
        if frame_a is not None and data_a is not None:
            fps_a = data_a.get('fps', 0)
            cv2.putText(frame_a, f"FPS: {fps_a:.0f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if frame_b is not None and data_b is not None:
            fps_b = data_b.get('fps', 0)
            cv2.putText(frame_b, f"FPS: {fps_b:.0f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 时间戳水印
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        if frame_a is not None:
            cv2.putText(frame_a, timestamp, (10, frame_a.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if frame_b is not None:
            cv2.putText(frame_b, timestamp, (10, frame_b.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 组合画面（支持单边或双边）
        if frame_a is not None and frame_b is not None:
            h = max(frame_a.shape[0], frame_b.shape[0])
            if frame_a.shape[0] != h:
                frame_a = cv2.resize(frame_a, (int(frame_a.shape[1] * h / frame_a.shape[0]), h))
            if frame_b.shape[0] != h:
                frame_b = cv2.resize(frame_b, (int(frame_b.shape[1] * h / frame_b.shape[0]), h))
            combined = np.hstack((frame_a, frame_b))
        elif frame_a is not None:
            combined = frame_a
        else:
            combined = frame_b

        # 保存输出
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
                        help='摄像头 ID 列表 (macOS Continuity Camera 用户可能需要 --cam_ids 1)')
    parser.add_argument('--video', type=str, nargs='+', default=None,
                        help='视频文件路径（替代摄像头）')
    parser.add_argument('--save_output', action='store_true',
                        help='保存输出视频')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试日志（显示每帧检测细节）')

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.num_cams == 1:
        run_single_camera(args)
    elif args.num_cams == 2:
        run_dual_camera(args)
    else:
        logger.error(f"不支持 {args.num_cams} 个摄像头，目前只支持 1 或 2")


if __name__ == "__main__":
    main()
