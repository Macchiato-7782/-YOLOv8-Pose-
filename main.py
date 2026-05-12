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
import torch
import argparse
import numpy as np
import traceback
import multiprocessing as mp
from ultralytics import YOLO

from tracking import SingleCameraTracker
from cross_camera import match_cross_camera, remove_wrongly_matched, get_color_histogram, merge_tracking_ids
from fall_logic import evaluate_fall, calculate_angle
from features import yolo_to_5keypoints
from camera_process import camera_process, extract_angle_keypoints, compute_iou, ROI_ENABLED, ROI_CONF, ROI_EXPAND_RATIO, ROI_MATCH_IOU, ROI_MAX_TIME, IOU_DEDUP_THRESHOLD

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

    # 标签文字（跌倒时更大）
    font_scale = 0.8 if 'FALL' in state else 0.6
    text_y = max(y1 - 10, 20)
    cv2.putText(frame, label, (max(x1, 5), text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame


def draw_fall_alert(frame, fall_results):
    """如果检测到跌倒，在画面顶部画警告横幅"""
    has_fall = any('FALL' in r['state'] for r in fall_results)
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
    elif args.cam_ids:
        cap = cv2.VideoCapture(args.cam_ids[0])
    else:
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头/视频")
        return

    tracker = SingleCameraTracker()
    output_video = None
    low_conf_rois = []

    print("单摄像头模式启动，按 ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或无法读取帧")
            break

        current_time = time.time()

        # YOLO 推理（降低阈值以检测躺卧姿态）
        results = model(frame, conf=0.3, verbose=False)[0]

        # 提取主推理检测结果
        main_detections = tracker.extract_detections(results)

        # ROI 二次推理：只保留与已有目标重叠的检测（过滤假检）
        roi_detections = []
        roi_start = time.time()
        if ROI_ENABLED and low_conf_rois:
            try:
                frame_h, frame_w = frame.shape[:2]
                for roi in low_conf_rois:
                    if time.time() - roi_start > ROI_MAX_TIME:
                        break
                    roi_bbox = roi['bbox']
                    rx1, ry1, rx2, ry2 = roi_bbox
                    w, h_roi = rx2 - rx1, ry2 - ry1
                    ex1 = max(0, int(rx1 - w * ROI_EXPAND_RATIO))
                    ey1 = max(0, int(ry1 - h_roi * ROI_EXPAND_RATIO))
                    ex2 = min(frame_w, int(rx2 + w * ROI_EXPAND_RATIO))
                    ey2 = min(frame_h, int(ry2 + h_roi * ROI_EXPAND_RATIO))

                    crop = frame[ey1:ey2, ex1:ex2]
                    if crop.size == 0:
                        continue
                    crop_results = model(crop, conf=ROI_CONF, verbose=False)[0]

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

                    roi_dets = tracker.extract_detections(crop_results)
                    for det in roi_dets:
                        if compute_iou(det['bbox'], roi_bbox) >= ROI_MATCH_IOU:
                            roi_detections.append(det)
            except Exception as e:
                print(f"ROI 推理异常: {e}")
                roi_detections = []

        # 合并去重
        all_detections = main_detections + roi_detections
        merged = []
        for det in all_detections:
            dup = False
            for m in merged:
                if compute_iou(det['bbox'], m['bbox']) > IOU_DEDUP_THRESHOLD:
                    dup = True
                    break
            if not dup:
                merged.append(det)

        # 跟踪更新
        tracked = tracker.update(merged, current_time)
        tracker.cleanup(current_time)

        # 绘制骨骼
        plotted_frame = draw_skeleton(frame, results)

        # 跌倒判断 + 绘制
        for person in tracked:
            try:
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
                print(f"单摄处理异常: {e}")
                traceback.print_exc()

        # 更新 ROI 二次推理区域
        low_conf_rois = []
        for person in tracked:
            if person.get('fall_result', {}).get('state') in ('Potential Fall', 'FALL'):
                low_conf_rois.append({'bbox': person['bbox'], 'pid': person['pid']})

        # 跌倒警告横幅
        fall_results = [p['fall_result'] for p in tracked if 'fall_result' in p]
        plotted_frame = draw_fall_alert(plotted_frame, fall_results)

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
            print("ESC 退出")
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

    print(f"双摄像头模式启动: Camera A={cam_a}, Camera B={cam_b}")
    print("按 ESC 退出")

    output_video = None
    global_id_map = {}
    cam_a_alive = True
    cam_b_alive = True
    data_a = None
    data_b = None
    # 主进程持久化 fall_state（camera_process 每帧传来的是新 dict，必须在主进程跨帧保存）
    fall_state_store = {}  # key: (camera_label, pid), value: fall_state dict

    def get_latest(queue):
        """排空队列旧帧，只取最新一帧（非阻塞，无数据返回 None）"""
        data = None
        while not queue.empty():
            try:
                newer = queue.get_nowait()
                if newer is None:
                    return 'STOP'  # 摄像头进程结束信号
                data = newer
            except Exception:
                break
        return data

    while cam_a_alive or cam_b_alive:
        # 非阻塞获取最新帧
        if cam_a_alive:
            new_a = get_latest(queue_a)
            if new_a == 'STOP':
                print("摄像头 A 结束")
                cam_a_alive = False
                data_a = None
            elif new_a is not None:
                data_a = new_a

        if cam_b_alive:
            new_b = get_latest(queue_b)
            if new_b == 'STOP':
                print("摄像头 B 结束")
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
            merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map)

            for a_idx, b_idx in matched_pairs:
                pid_a = tracked_a[a_idx]['pid']
                pid_b = tracked_b[b_idx]['pid']
                global_id = global_id_map.get((pid_a, pid_b))
                if global_id is not None:
                    pid_to_global_a[pid_a] = global_id
                    pid_to_global_b[pid_b] = global_id

        # 跌倒判断 + 绘制（摄像头 A）
        if frame_a is not None:
            for person in tracked_a:
                try:
                    # 从主进程持久化存储恢复 fall_state（camera_process 每帧传来新 dict）
                    store_key = ('A', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_a[a_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_b[b_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']  # 持久化
                    gid = pid_to_global_a.get(person['pid'])
                    frame_a = draw_person_info(frame_a, person, fall_result, global_id=gid)
                except Exception as e:
                    print(f"Camera A 处理异常: {e}")
                    traceback.print_exc()

        # 跌倒判断 + 绘制（摄像头 B）
        if frame_b is not None:
            for person in tracked_b:
                try:
                    # 从主进程持久化存储恢复 fall_state
                    store_key = ('B', person['pid'])
                    if store_key in fall_state_store:
                        person['fall_state'] = fall_state_store[store_key]

                    dual_cam_fall = None
                    for a_idx, b_idx in matched_pairs:
                        if tracked_b[b_idx]['pid'] == person['pid']:
                            dual_cam_fall = tracked_a[a_idx].get('fall_result', {}).get('fall_detected', False)
                            break

                    fall_result = evaluate_fall(person, current_time, dual_cam_fall)
                    person['fall_result'] = fall_result
                    fall_state_store[store_key] = person['fall_state']  # 持久化
                    gid = pid_to_global_b.get(person['pid'])
                    frame_b = draw_person_info(frame_b, person, fall_result, global_id=gid)
                except Exception as e:
                    print(f"Camera B 处理异常: {e}")
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

        # 跌倒警告横幅
        fall_results_a = [p['fall_result'] for p in tracked_a if 'fall_result' in p]
        fall_results_b = [p['fall_result'] for p in tracked_b if 'fall_result' in p]
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
            print("ESC 退出")
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
    parser.add_argument('--cam_ids', type=int, nargs='+', default=[1],
                        help='摄像头 ID 列表')
    parser.add_argument('--video', type=str, nargs='+', default=None,
                        help='视频文件路径（替代摄像头）')
    parser.add_argument('--save_output', action='store_true',
                        help='保存输出视频')

    args = parser.parse_args()

    if args.num_cams == 1:
        run_single_camera(args)
    elif args.num_cams == 2:
        run_dual_camera(args)
    else:
        print(f"不支持 {args.num_cams} 个摄像头，目前只支持 1 或 2")


if __name__ == "__main__":
    main()
