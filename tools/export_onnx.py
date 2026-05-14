#!/usr/bin/env python
"""
YOLOv8-Pose 模型导出脚本
将 .pt 模型导出为 ONNX 格式，供 ONNXBackend 使用

用法:
    python tools/export_onnx.py
    python tools/export_onnx.py --model yolov8n-pose.pt --imgsz 640 --opset 12
    python tools/export_onnx.py --model yolov8n-pose.pt --dynamic --simplify
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="导出 YOLOv8-Pose 模型为 ONNX")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt",
                        help="输入 .pt 模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 .onnx 路径（默认同目录同名）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="导出图像尺寸")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset 版本")
    parser.add_argument("--dynamic", action="store_true",
                        help="导出动态 batch 维度")
    parser.add_argument("--simplify", action="store_true",
                        help="使用 onnx-simplifier 简化模型")
    parser.add_argument("--half", action="store_true",
                        help="FP16 半精度导出")

    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先下载模型或指定正确路径")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(model_path))[0]
        args.output = os.path.join(os.path.dirname(model_path), f"{base}.onnx")

    print(f"=" * 50)
    print(f"YOLOv8-Pose ONNX Export")
    print(f"=" * 50)
    print(f"Input:   {model_path}")
    print(f"Output:  {args.output}")
    print(f"imgsz:   {args.imgsz}")
    print(f"opset:   {args.opset}")
    print(f"dynamic: {args.dynamic}")
    print(f"simplify:{args.simplify}")
    print(f"half:    {args.half}")
    print(f"=" * 50)

    from ultralytics import YOLO

    print(f"\n加载模型 {model_path} ...")
    model = YOLO(model_path)

    print(f"导出 ONNX ...")
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "opset": args.opset,
        "half": args.half,
    }
    if args.dynamic:
        export_kwargs["dynamic"] = True

    success = model.export(**export_kwargs)

    if success and args.simplify:
        print("简化 ONNX 模型 ...")
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(args.output)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, args.output)
                print("ONNX 简化完成")
            else:
                print("ONNX 简化验证失败，保留原始模型")
        except ImportError:
            print("onnx-simplifier 未安装，跳过简化")
            print("安装: pip install onnx-simplifier")

    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n导出成功: {args.output} ({size_mb:.1f} MB)")
    else:
        print(f"\n导出可能失败，检查 '{os.path.splitext(os.path.basename(model_path))[0]}.onnx'")
        sys.exit(1)


if __name__ == "__main__":
    main()
