#!/usr/bin/env python
"""
推理后端 Benchmark 工具
比较 ultralytics 和 onnx 后端的性能

用法:
    python tools/benchmark_backend.py
    python tools/benchmark_backend.py --onnx yolov8n-pose.onnx --frames 100
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)


def benchmark_backend(backend_name, model_path, device, input_size, conf, frames=100):
    """测试单个后端性能"""
    from fall_detection.backends.factory import create_backend
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # 创建后端
    t0 = time.time()
    if backend_name == "ultralytics":
        backend = create_backend(
            backend="ultralytics",
            model_path=model_path,
            device=device,
            conf=conf,
            input_size=input_size,
            enable_tracking=False,  # benchmark 不测跟踪
        )
    else:
        backend = create_backend(
            backend=backend_name,
            model_path=model_path,
            device=device,
            conf=conf,
            input_size=input_size,
            enable_tracking=False,
        )
    init_time = time.time() - t0

    # 预热
    backend.warmup()

    # 生成测试帧
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Benchmark
    mem_before = process.memory_info().rss / (1024 * 1024)
    latencies = []

    t_start = time.time()
    for i in range(frames):
        t_frame = time.time()
        _ = backend.infer(test_frame)
        latencies.append((time.time() - t_frame) * 1000)

    total_time = time.time() - t_start
    fps = frames / total_time
    mem_after = process.memory_info().rss / (1024 * 1024)

    backend.close()

    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)

    return {
        "backend": backend_name,
        "init_time": init_time,
        "fps": fps,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "memory_mb": mem_after - mem_before,
        "total_memory_mb": mem_after,
        "frames": frames,
        "total_time": total_time,
        "input_size": input_size,
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="推理后端 Benchmark")
    parser.add_argument("--onnx", type=str, default=None,
                        help="ONNX 模型路径（默认用 yolov8n-pose.onnx）")
    parser.add_argument("--pt", type=str, default=None,
                        help="PyTorch 模型路径（默认用 yolov8n-pose.pt）")
    parser.add_argument("--frames", type=int, default=100,
                        help="测试帧数")
    parser.add_argument("--input_size", type=int, default=640,
                        help="输入分辨率")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="置信度阈值")

    args = parser.parse_args()

    pt_path = args.pt or os.path.join(ROOT_DIR, "yolov8n-pose.pt")
    onnx_path = args.onnx or os.path.join(ROOT_DIR, "yolov8n-pose.onnx")

    results = []

    # Test ultralytics backend
    if os.path.exists(pt_path):
        print(f"Testing Ultralytics backend with {pt_path} ...")
        try:
            r = benchmark_backend("ultralytics", pt_path, args.device, args.input_size, args.conf, args.frames)
            results.append(r)
        except Exception as e:
            print(f"  Ultralytics backend failed: {e}")
    else:
        print(f"PT model not found: {pt_path}")

    # Test ONNX backend
    if os.path.exists(onnx_path):
        print(f"Testing ONNX backend with {onnx_path} ...")
        try:
            r = benchmark_backend("onnx", onnx_path, args.device, args.input_size, args.conf, args.frames)
            results.append(r)
        except Exception as e:
            print(f"  ONNX backend failed: {e}")
    else:
        print(f"ONNX model not found: {onnx_path}")

    # Print results
    print()
    print("=" * 50)
    print("Backend Benchmark Results")
    print("=" * 50)

    for r in results:
        print(f"\nBackend:           {r['backend']}")
        print(f"Device:            {r['device']}")
        print(f"Input Size:        {r['input_size']}")
        print(f"Init Time:         {r['init_time']:.2f}s")
        print(f"Frames Tested:     {r['frames']}")
        print(f"Total Time:        {r['total_time']:.2f}s")
        print(f"FPS:               {r['fps']:.1f}")
        print(f"Avg Latency:       {r['avg_latency_ms']:.1f}ms")
        print(f"P50 Latency:       {r['p50_latency_ms']:.1f}ms")
        print(f"P95 Latency:       {r['p95_latency_ms']:.1f}ms")
        print(f"Memory Delta:      {r['memory_mb']:.0f}MB")
        print(f"Total Memory:      {r['total_memory_mb']:.0f}MB")

    print()
    print("=" * 50)

    if len(results) == 2:
        speedup = results[1]["fps"] / results[0]["fps"]
        mem_reduction = results[0]["total_memory_mb"] - results[1]["total_memory_mb"]
        print(f"\nONNX vs Ultralytics:")
        print(f"  FPS:         {results[0]['fps']:.1f} -> {results[1]['fps']:.1f}  ({speedup:.1f}x)")
        print(f"  Latency:     {results[0]['avg_latency_ms']:.1f}ms -> {results[1]['avg_latency_ms']:.1f}ms")
        print(f"  Memory:      {results[0]['total_memory_mb']:.0f}MB -> {results[1]['total_memory_mb']:.0f}MB  ({mem_reduction:.0f}MB saved)")
        print("=" * 50)


if __name__ == "__main__":
    main()
