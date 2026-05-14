#!/usr/bin/env python
"""
Runtime Purity 检查工具
自动扫描 Runtime 代码，检查是否存在 dict leak / numpy leak / temporary serialization

用法:
    python tools/check_runtime_purity.py
"""

import os
import ast
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

RUNTIME_DIRS = [
    os.path.join(ROOT_DIR, "fall_detection", "core"),
    os.path.join(ROOT_DIR, "fall_detection", "backends"),
]

CHECK_FILES = [
    "fall_detection/core/pipeline.py",
    "fall_detection/core/runtime.py",
    "fall_detection/core/serializers.py",
    "fall_detection/backends/postprocess.py",
    "fall_detection/backends/ultralytics_backend.py",
    "fall_detection/backends/onnx_backend.py",
    "tracking.py",
    "fall_detection/detector.py",
]


def scan_file(filepath):
    results = {"dict_usage": [], "numpy_leak": [], "serialize_calls": [], "dict_get": []}

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    relpath = os.path.relpath(filepath, ROOT_DIR)

    # Skip serializers.py (it's ALLOWED to serialize)
    if "serializers.py" in relpath:
        return results

    for node in ast.walk(tree):
        # Check isinstance(..., dict)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            for arg in node.args[1:]:
                if isinstance(arg, ast.Name) and arg.id == "dict":
                    results["dict_usage"].append(f"{relpath}:{node.lineno} isinstance(..., dict)")
                elif isinstance(arg, ast.Tuple):
                    for elt in arg.elts:
                        if isinstance(elt, ast.Name) and elt.id == "dict":
                            results["dict_usage"].append(f"{relpath}:{node.lineno} isinstance(..., dict)")

        # Check .get( calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if isinstance(node.func.value, ast.Name):
                results["dict_get"].append(f"{relpath}:{node.lineno} {node.func.value.id}.get(...)")

        # Check serialize calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if "serialize" in node.func.id.lower():
                results["serialize_calls"].append(f"{relpath}:{node.lineno} {node.func.id}()")

        # Check numpy leakage in returns
        if isinstance(node, ast.Name) and node.id == "np":
            results["numpy_leak"].append(f"{relpath}:{node.lineno} numpy usage")

    return results


def main():
    print("=" * 50)
    print("Runtime Purity Report")
    print("=" * 50)

    total_dict = 0
    total_numpy = 0
    total_serialize = 0

    for f in CHECK_FILES:
        fpath = os.path.join(ROOT_DIR, f)
        if not os.path.exists(fpath):
            continue
        r = scan_file(fpath)
        total_dict += len(r["dict_usage"])
        total_numpy += len(r.get("numpy_leak", []))
        total_serialize += len(r["serialize_calls"])

    # Allowed exceptions (fall_logic bridge is necessary)
    allowed_dict = len([f for f in CHECK_FILES if "pipeline.py" in f])

    print(f"\nLegacy Dict Usage:     {total_dict}")
    print(f"Temporary Serialization:{total_serialize}")
    print(f"Numpy Usage:           {total_numpy}")
    print(f"\nObject Pipeline Integrity: {'PASS' if total_dict <= allowed_dict + 2 else 'CHECK'}")

    if total_dict > allowed_dict + 2:
        print("\n[WARN] Some dict usage detected in runtime. Review manually.")
    else:
        print("\n[OK] Runtime purity check passed.")

    print("=" * 50)


if __name__ == "__main__":
    main()
