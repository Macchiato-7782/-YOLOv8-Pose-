# Real-Time Fall Detection

基于 YOLOv8-Pose 的实时人体跌倒检测系统，支持**单摄像头**和**双摄像头**模式。

- 单摄像头：四路检测（几何 + 物理 + 侧倒 + 地面） + 滑动窗口 + 持续时间确认
- 双摄像头：多进程并行 + HSV 直方图跨摄像头匹配 + 稳定婚姻算法 + 双视角交叉验证
- 跟踪方案：Ultralytics 内置 ByteTracker（卡尔曼滤波 + 级联匹配）
- 物理特征降噪：EMA 平滑 + Savitzky-Golay 滤波
- **标准接口**：FallDetector 类，可被其他项目直接 import，返回 JSON-friendly 结构化结果
- **边缘部署**：支持低功耗配置（跳帧推理、限人数、关 ROI），后续可扩展 ONNX/OpenVINO/NCNN

## 效果

- 绿色标注：正常状态
- 橙色标注：可能跌倒（计时中）
- 红色标注：确认跌倒
- 黄色标注：全局 ID（双摄像头模式下同一个人的统一编号）

## 技术方案

### 单摄像头

```
摄像头 → YOLOv8-Pose (17关键点) → ByteTracker 跟踪
    → 四路跌倒检测:
        路径1: 宽高比 + 髋部角度 (AND)
        路径2: 旋转能量 / 重力因子 / 头部下降 (需几何确认)
        路径3: AR剧变 + 头部下降 (侧倒)
        路径4: 已在地面检测
    → 滑动窗口(20帧, 50%) → 连续触发(5帧) → 持续时间(3.5秒) → 显示
```

### 双摄像头

```
摄像头A → 进程A: YOLO检测 + 跟踪 + 直方图 + 角度关键点 ──→ Queue ──┐
                                                                      ├──→ 主进程:
摄像头B → 进程B: YOLO检测 + 跟踪 + 直方图 + 角度关键点 ──→ Queue ──┘
                                                                     ├─ 稳定婚姻匹配
                                                                     ├─ fall_state 持久化
                                                                     ├─ 四路跌倒判断
                                                                     ├─ 双视角交叉验证
                                                                     └─ 拼接显示
```

## 安装

```bash
git clone https://github.com/Macchiato-7782/-YOLOv8-Pose-.git
cd -YOLOv8-Pose-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
# 单摄像头（Demo 模式）
python main.py

# 单视频文件
python main.py --video your_video.mp4

# 双摄像头
python main.py --num_cams 2 --cam_ids 0 1

# 双视频文件
python main.py --num_cams 2 --video cam1.mp4 cam2.mp4

# Headless 模式（无窗口，打印结构化结果）
python main.py --headless

# 边缘设备低功耗模式
python main.py --edge

# 边缘 + Headless 组合
python main.py --edge --headless

# 保存输出视频
python main.py --num_cams 2 --cam_ids 0 1 --save_output
```

按 ESC 退出。

## 测试

不依赖摄像头，用合成数据验证检测逻辑：

```bash
# 运行全部测试（227 个用例）
python -m pytest test_fall_detection.py test_features.py test_tracking.py test_cross_camera.py test_detector_interface.py test_backends.py test_runtime_pipeline.py test_runtime_integrity.py test_runtime_engine.py -v

# 只运行跌倒场景测试
python -m pytest test_fall_detection.py -v

# 只运行接口测试
python -m pytest test_detector_interface.py -v

# 只运行后端测试
python -m pytest test_backends.py -v

# 只运行 Runtime 完整性测试
python -m pytest test_runtime_integrity.py -v
```

覆盖 10 个测试模块、227 个用例。

## 跌倒判断逻辑

### 四路检测（任一触发即为"可能跌倒"）

| 路径 | 条件 | 适用场景 |
|------|------|----------|
| 路径1: 几何 | AR/初始AR < 0.35 **且** 角度 < 120° | 前倒/后倒 |
| 路径2: 物理 | (RE > 8 **或** GF > 8000) **且** (AR变化 > 10% **或** 头部下降) | 快速摔倒/蜷缩倒 |
| 路径3: 侧倒 | AR 剧变 **且** 头部下降 | 侧倒（角度不够低） |
| 路径4: 地面 | 初始AR低 **且** 当前AR < 0.4 **且** 持续10帧 | 已在地上的人 |

### 确认机制

- 滑动窗口：20 帧内 50% 触发
- 连续触发：至少 5 帧连续触发（允许 1 帧间隙）
- 持续时间：3.5 秒
- 回弹检测：头部回升 15% 身高 + 连续 2 帧 → 取消跌倒判定
- 状态粘性：确认后不因信号消失而重置
- 恢复检测：AR 回升到基线 70% 自动重置

### 物理特征（来自 HumanFallDetection）

| 特征 | 含义 |
|------|------|
| 旋转能量 (RE) | 倒立摆模型，身体绕脚旋转的角速度 |
| 重力因子 (GF) | 重心加速度方向与重力的一致性 |
| 头部下降 (HD) | 头部下降量 / 身高 |

### 双摄像头交叉验证

- 两个摄像头都确认跌倒 → 高置信度 (95%)
- 只有一个确认 → 低置信度警告 (60%)

## 跨摄像头人物匹配

1. 对每个人计算上半身 HSV 颜色直方图作为"外貌特征"
2. 计算两个摄像头中所有未匹配人之间的直方图相关性矩阵
3. 用**稳定婚姻算法**（Gale-Shapley）做全局最优匹配
4. 定期检查已匹配对的相关性，低于阈值则拆开重新匹配

## 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--model` | yolov8n-pose.pt | YOLO 模型路径 |
| `--num_cams` | 1 | 摄像头数量 |
| `--cam_ids` | 0 | 摄像头 ID |
| `--video` | None | 视频文件路径 |
| `--save_output` | False | 保存输出视频 |
| `--debug` | False | 启用调试日志 |
| `--headless` | False | 无窗口模式，打印结构化结果 |
| `--edge` | False | 边缘设备低功耗模式 |

## 依赖

- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- ultralytics >= 8.0.0
- scipy >= 1.10.0
- pyyaml >= 6.0
- lap >= 0.4.0 (ByteTracker 后端)

## macOS 注意事项

首次使用摄像头需要授权：系统设置 → 隐私与安全性 → 摄像头 → 打开"终端"。

## 作为模块接入其他项目

### 标准接口

```python
from fall_detection import FallDetector
import cv2

detector = FallDetector(
    model_path="yolov8n-pose.pt",
    device="cpu",
    input_size=320,
    inference_interval=2,
    enable_roi=False,
    enable_visualization=False,
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.process_frame(
        frame,
        camera_id="cam_0"
    )

    for event in result["events"]:
        if event["event_type"] == "fall_confirmed":
            print("Fall detected:", event)

cap.release()
detector.close()
```

### 输出格式

```json
{
    "module": "fall_detection",
    "camera_id": "cam_0",
    "timestamp": 1710000000.0,
    "frame_id": 12,
    "persons": [
        {
            "track_id": 1,
            "bbox": [100, 200, 300, 500],
            "center": [200, 350],
            "state": "fall",
            "fall_detected": true,
            "confidence": 0.6,
            "is_ghost": false,
            "keypoints": [[100.0, 200.0, 0.9], ...]
        }
    ],
    "events": [
        {
            "event_type": "fall_confirmed",
            "track_id": 1,
            "camera_id": "cam_0",
            "timestamp": 1710000000.0,
            "confidence": 0.6,
            "bbox": [100, 200, 300, 500],
            "state": "fall"
        }
    ],
    "diagnostics": {
        "fps": 15.2,
        "backend": "ultralytics",
        "device": "cpu",
        "inference_ran": true
    }
}
```

所有字段均可 `json.dumps` 序列化。`numpy` 类型已自动转换为 Python 原生类型。

### FallDetector 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | yolov8n-pose.pt | 模型文件路径 |
| `config_path` | str | config.yaml | 配置文件路径 |
| `device` | str | "cpu" | 推理设备 |
| `backend` | str | "ultralytics" | 推理后端（预留 onnx/openvino/ncnn） |
| `enable_tracking` | bool | True | 启用 ByteTracker |
| `enable_visualization` | bool | False | 在结果中附加标注图像 |
| `enable_roi` | bool | False | 启用 ROI 二次推理 |
| `inference_interval` | int | 1 | 每 N 帧推理一次 |
| `input_size` | int | 640 | 模型输入分辨率 |
| `max_persons` | int | None | 最多跟踪人数 |

### Headless 模式

```bash
# 不显示窗口，打印结构化结果
python main.py --headless

# 边缘设备低功耗模式
python main.py --edge --headless
```

## 边缘设备建议配置

| 设备 | input_size | inference_interval | enable_roi | max_persons |
|------|-----------|-------------------|------------|-------------|
| 树莓派 4B | 320 | 2 | False | 1-2 |
| NVIDIA Jetson Nano | 320 | 1 | False | 2-3 |
| Intel NUC / 普通 PC | 640 | 1 | True | None |

边缘设备初始化示例：

```python
detector = FallDetector(
    model_path="yolov8n-pose.pt",
    device="cpu",
    input_size=320,
    inference_interval=2,
    enable_roi=False,
    enable_visualization=False,
    max_persons=3,
)
```

后续计划扩展 ONNX / OpenVINO / NCNN 后端，只需切换 `backend` 参数即可。

## 推理后端

支持多种推理后端，通过 `backend` 参数切换。

| 后端 | 参数值 | 依赖 | 适合场景 |
|------|--------|------|----------|
| Ultralytics | `"ultralytics"` | ultralytics + PyTorch | 开发调试、GPU 服务器 |
| ONNX Runtime | `"onnx"` | onnxruntime | 边缘设备、低功耗部署 |
| OpenVINO | `"openvino"` | (预留) | Intel CPU/VPU |
| NCNN | `"ncnn"` | (预留) | ARM Linux / Android |

### 使用 ONNX 后端

```bash
# 1. 导出 ONNX 模型
python tools/export_onnx.py

# 2. 使用 ONNX 后端
from fall_detection import FallDetector

detector = FallDetector(
    backend="onnx",
    model_path="yolov8n-pose.onnx",
    device="cpu",
    input_size=320,
)
```

### Benchmark 工具

```bash
python tools/benchmark_backend.py
```

输出示例：

```
==================================================
Backend Benchmark Results
==================================================

Backend:           ultralytics
FPS:               12.4
Avg Latency:       80.6ms
Total Memory:      1200MB

Backend:           onnx
FPS:               24.7
Avg Latency:       40.5ms
Total Memory:      420MB

ONNX vs Ultralytics:
  FPS:         12.4 -> 24.7  (2.0x)
  Memory:      1200MB -> 420MB  (780MB saved)
==================================================
```

## 项目结构

```
├── main.py                     # 入口，支持单/双摄像头模式 + --headless + --edge
├── fall_detection/             # 标准检测模块（可被外部 import）
│   ├── __init__.py
│   ├── detector.py             # FallDetector 标准接口（orchestration only）
│   ├── schemas.py              # 输出格式定义与 JSON 序列化
│   ├── visualizer.py           # 绘图函数（draw_person_info 等）
│   ├── edge_config.py          # 边缘设备默认参数
│   ├── backends/               # 推理后端抽象层
│   │   ├── __init__.py
│   │   ├── base.py             # BaseInferenceBackend 抽象类
│   │   ├── ultralytics_backend.py  # Ultralytics YOLO 后端
│   │   ├── onnx_backend.py     # ONNX Runtime 后端
│   │   ├── postprocess.py      # 统一后处理
│   │   └── factory.py          # create_backend() 工厂
│   └── core/                   # AI Monitoring Runtime Core
│       ├── __init__.py
│       ├── detection.py        # Detection / Keypoint dataclass
│       ├── track.py            # TrackState dataclass
│       ├── event.py            # Event dataclass
│       ├── frame.py            # FrameContext dataclass
│       ├── pipeline.py         # DetectionPipeline 统一编排
│       ├── runtime.py          # EventRuntime 事件生成+去重
│       ├── serializers.py      # dataclass → JSON-friendly
│       └── validators.py       # 结构校验
│   └── engine/                 # Runtime Engine
│       ├── __init__.py
│       ├── runtime_engine.py   # RuntimeEngine 入口
│       ├── runtime_session.py  # RuntimeSession 状态管理
│       ├── camera_session.py   # CameraSession 生命周期
│       ├── frame_buffer.py     # SharedFrameBuffer
│       ├── scheduler.py        # RuntimeScheduler
│       ├── worker.py           # RuntimeWorker
│       ├── event_bus.py        # EventBus 事件总线
│       ├── signals.py          # RuntimeSignals
│       ├── lifecycle.py        # LifecycleManager
│       ├── registry.py         # RuntimeRegistry
│       ├── health.py           # HealthMonitor
│       └── metrics.py          # RuntimeMetrics
├── tools/                      # 开发工具
│   ├── export_onnx.py          # PT → ONNX 导出
│   ├── benchmark_backend.py    # 后端性能对比
│   └── check_runtime_purity.py # Runtime 纯度检查
├── docs/                       # 文档
│   ├── runtime_architecture.md # Runtime Flow 架构
│   └── runtime_engine.md       # Runtime Engine 架构
├── fall_logic.py               # 融合跌倒判断逻辑（四路检测 + 滑动窗口）
├── features.py                 # 物理特征计算（旋转能量、重力因子、头部下降）
├── tracking.py                 # ByteTracker 跟踪 + 幽灵机制 + 跌倒状态继承
├── cross_camera.py             # 跨摄像头匹配（直方图 + 稳定婚姻算法）
├── camera_process.py           # 摄像头处理进程（多进程架构）
├── config.py                   # 配置加载器
├── config.yaml                 # 所有可调参数（阈值、窗口、超时等）
├── test_fall_detection.py      # 跌倒场景测试
├── test_features.py            # 物理特征测试
├── test_tracking.py            # 跟踪器测试
├── test_cross_camera.py        # 跨摄像头匹配测试
├── test_detector_interface.py  # FallDetector 接口测试
├── test_backends.py            # 推理后端测试
├── test_runtime_pipeline.py    # Runtime Pipeline 测试
├── test_runtime_integrity.py   # Runtime 完整性测试
├── test_runtime_engine.py      # Runtime Engine 测试
├── requirements.txt            # 依赖清单
└── README.md
```

## 参考

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HumanFallDetection](https://github.com/taufeeque9/HumanFallDetection) — 双摄像头方案和物理特征
- [COCO Keypoints](https://cocodataset.org/#keypoints-2017)

## 更新日志

### 2026-05-14（v3）

**SDK 化与边缘部署:**
- 新增 `fall_detection/` 标准检测包，可被外部项目直接 `import` 调用
- 新增 `FallDetector` 类作为唯一对外接口，返回 JSON-friendly 结构化结果
- 新增 `schemas.py`：numpy→Python 类型自动转换，`json.dumps` 可直接序列化
- 新增 `edge_config.py`：边缘设备默认参数（input_size=320, interval=2 等）
- 新增 `visualizer.py`：从 main.py 迁移绘图函数，可选调用
- 新增 `--headless` CLI 参数：无窗口模式，打印结构化结果摘要
- 新增 `--edge` CLI 参数：一键切换低功耗默认配置
- 核心逻辑支持 headless 模式：不依赖 `cv2.imshow`、`argparse`
- 预留 `external_tracks` / `backend` 参数，便于后续接入人脸识别项目和 ONNX/OpenVINO 后端

**推理后端抽象层（v3.1）:**
- 新增 `fall_detection/backends/` 推理后端抽象层
- 实现 `BaseInferenceBackend` 抽象基类，统一 infer / warmup / close 接口
- 实现 `UltralyticsBackend`：封装 YOLO.track()，返回统一 detection schema
- 实现 `ONNXBackend`：基于 onnxruntime，零 PyTorch 依赖，适合边缘设备
- 实现 `create_backend()` 工厂函数：一键切换 ultralytics / onnx 后端，预留 openvino / ncnn
- 实现 `postprocess.py`：统一后处理（bbox 解码、NMS、关键点提取）
- `detector.py` 不再直接 import YOLO，完全 backend 无关
- `tracking.py` 不再 import ultralytics，新增 IoU 匹配 `_assign_ids()`
- 新增 `tools/export_onnx.py`：PT → ONNX 模型导出
- 新增 `tools/benchmark_backend.py`：多后端性能对比工具
- 新增 `test_detector_interface.py`：45 个接口测试用例
- 新增 `test_backends.py`：22 个后端测试用例

**AI Monitoring Runtime Core（v3.2）:**
- 新增 `fall_detection/core/` Runtime Core 模块
- 实现标准化 Data Model：`Detection` / `Keypoint` / `TrackState` / `Event` / `FrameContext`
- 实现 `DetectionPipeline`：统一编排 infer → Detection → tracking → TrackState → fall logic → Event → serialize
- 实现 `EventRuntime`：事件生成 + cooldown 去重 + 防重复
- 实现 `serializers.py`：dataclass → JSON-friendly dict，零 numpy 泄露
- 实现 `validators.py`：结构校验（bbox 长度、score 范围、event_type 等）
- `detector.py` 改为纯 orchestration 层，不直接操作 detection/tracking/events
- Pipeline 内部统一使用 dataclass objects，最终输出时才 serialize
- 新增 `docs/runtime_architecture.md`：Runtime Flow 架构文档
- 新增 `test_runtime_pipeline.py`：数据模型、序列化、校验、Pipeline 兼容性测试
- 测试覆盖从 53 个增至 **156 个**，全部通过

**Pure Runtime Refactor（v3.3）:**
- Runtime 内部彻底禁止 dict：Detection→TrackState→Event 全链路纯对象
- backend `infer()` 严格返回 `list[Detection]`（含 Keypoint 对象），不再返回 dict
- tracking.py 输入 `list[Detection]`，输出 `list[TrackState]`，删除 `convert_backend_detections` 等兼容层
- pipeline.py 纯对象流水线，删除 `_ensure_detections` / isinstance(dict) 等 legacy 适配器
- serializers.py 成为唯一 object→dict 出口
- validators.py 在 Runtime 入口强制校验
- 新增 `core/types.py`：DetectionList / TrackList / EventList 类型别名
- 新增 `tools/check_runtime_purity.py`：自动扫描 Runtime dict leak
- 新增 `test_runtime_integrity.py`：数据类型完整性测试
- 测试覆盖增至 **168 个**，全部通过

**Pure Runtime Refactor（v3.3）:**
- Runtime 内部彻底禁止 dict：Detection→TrackState→Event 全链路纯对象
- backend `infer()` 严格返回 `list[Detection]`（含 Keypoint 对象）
- tracking.py 输入 `list[Detection]`，输出 `list[TrackState]`，删除 `convert_backend_detections` 等兼容层
- pipeline.py 纯对象流水线，删除 `_ensure_detections` / isinstance(dict) 等 legacy 适配器
- serializers.py 成为唯一 object→dict 出口
- validators.py 在 Runtime 入口强制校验
- 新增 `core/types.py`：DetectionList / TrackList / EventList 类型别名
- 新增 `tools/check_runtime_purity.py`：自动扫描 Runtime dict leak
- 新增 `test_runtime_integrity.py`：数据类型完整性测试
- 测试覆盖增至 **174 个**，全部通过

**Runtime Engine Architecture（v3.4）:**
- 新增 `fall_detection/engine/` Runtime Engine 模块
- 实现 `RuntimeEngine`：工业级边缘 AI Runtime 入口
- 实现 `RuntimeSession`：统一 Runtime 状态管理（tracking/frame/backend/scheduler/metrics）
- 实现 `CameraSession`：统一摄像头生命周期（connect/disconnect/reconnect）
- 实现 `SharedFrameBuffer`：共享帧缓冲区（ring buffer + ref counting）
- 实现 `EventBus`：统一事件总线（subscribe/unsubscribe/publish）
- 实现 `RuntimeScheduler`：任务调度器，支持不同 pipeline 不同频率
- 实现 `RuntimeWorker`：Worker 系统（queue-based, graceful shutdown, exception isolation）
- 实现 `RuntimeLifecycleManager`：统一 start/stop/pause/resume/restart
- 实现 `RuntimeRegistry`：全局注册中心（sessions/cameras/pipelines/workers/backends）
- 实现 `RuntimeHealthMonitor`：健康监控（FPS/latency/memory/errors → EventBus）
- 实现 `RuntimeMetrics`：Prometheus-ready 指标收集
- 实现 `RuntimeSignals` 系统：EventSignal / HealthSignal / ErrorSignal / LifecycleSignal
- 新增 `docs/runtime_engine.md`：Engine Architecture 文档
- 新增 `test_runtime_engine.py`：Engine 组件测试
- 测试覆盖增至 **193 个**，全部通过

**Fault-Tolerant Runtime（v3.5）:**
- 新增 `fall_detection/runtime_state/` 容错运行时状态管理包
- 实现 `RuntimeStateMachine`：11 状态 + 非法转换 reject + 状态历史
- 实现 `RuntimePolicy`：overload / reconnect / frame drop / cooldown / retry 策略
- 实现 `BackpressureController`：自动帧丢弃 / 队列裁剪 / FPS 降级
- 实现 `DegradationController`：5 级自动降级
- 实现 `FaultRecovery`：backend/worker/camera/session 自动恢复
- 实现 `ResourceManager`：CPU / RAM / queue / thermal 实时监控
- 实现 `RuntimeClock`：统一时间域
- 实现 `RuntimeDiagnostics`：统一诊断输出
- 新增 `tools/runtime_chaos_test.py`：Chaos 测试（camera disconnect / CPU spike / queue overflow）
- 新增 `test_runtime_fault_tolerance.py`：容错测试
- 测试覆盖增至 **227 个**，全部通过

### 2026-05-13（v2）

**跟踪方案升级:**
- 集成 Ultralytics 内置 ByteTracker（卡尔曼滤波 + 级联匹配），替代手写匈牙利匹配
- 新增幽灵目标跌倒状态继承：检测框短暂消失后重新出现时，通过 IoU 匹配恢复跌倒状态
- 依赖新增 `lap` 包（ByteTracker 后端）

**物理特征降噪:**
- 新增 Savitzky-Golay 滤波（二次平滑，保留信号形状）
- 新增 EMA 指数移动平均（降噪）
- 帧率归一化：RE 单位 rad/s，GF 单位 pixel/s²

**误报抑制:**
- 物理路径加几何守卫：RE/GF 触发需配合 AR 变化（>10%）或头部下降，过滤走路/弯腰误报
- 连续触发要求：5 帧连续触发（允许 1 帧间隙），过滤瞬间动作
- 滑动窗口：20 帧内 50% 触发
- 持续时间：3.5 秒（给弯腰恢复留足够时间）
- 回弹检测门槛降低：头部回升 15% 身高 + 2 帧即可取消跌倒
- 角度阈值收紧：130° → 120°
- 物理特征阈值提高：RE 8, GF 8000

**监控视角适配:**
- 躯干倾斜角：torso_vec 长度 < 身高 10% 时跳过（俯视视角不可靠）
- 初始 AR 更新改为滑动最大值（30 帧窗口），弯腰后能恢复基线
- already_down 路径 AR 门限收紧：0.6 → 0.4
- `min_standing_ar` 降低：1.2 → 1.0

**UI 优化:**
- 标签移入 bbox 内部顶部，添加半透明背景矩形
- 幽灵目标灰色虚线框 + "LOST" 标签

**配置化:**
- 新增 `config.yaml`：所有阈值、窗口、超时参数集中配置，无需改代码
- 新增 `config.py`：配置加载器

**测试完善:**
- 53 个测试用例全部通过（跌倒场景 + 物理特征 + 跟踪器 + 跨摄像头匹配）
- 测试数据适配 3.5 秒持续时间

### 2026-05-13（v1）

- 更新部署教程与学习笔记（新增调试问题记录、架构说明、模拟测试说明）
- 更新 README.md 和使用教程

### 2026-05-12

**核心修复:**
- 修复双摄模式 `fall_state` 每帧重置导致跌倒检测失效的严重 bug，新增 `fall_state_store` 跨帧持久化
- 修复 numpy view 跨进程序列化数据丢失（`.copy()` + `float()` 转换）
- 修复跌倒确认后因物理信号消失而自动重置的问题，新增状态粘性机制
- 修复 `initial_ar` 保护过于严格（坐姿首次检测后永久屏蔽），改为只屏蔽 AR 路径
- 修复 `is_potential_fall` 可能为 `None` 导致 `sum()` 报 TypeError 的 bug
- 修复 camera_process 和 main 双重调用 `evaluate_fall` 导致角度不一致
- 添加恢复检测：AR 回升到基线 70% 自动重置跌倒状态

**新增功能:**
- 新增 `test_fall_detection.py` 模拟测试（11 个场景全部通过）
- 新增 ROI 二次推理（对潜在跌倒区域再检测，过滤假检）
- 新增四路跌倒检测：几何(AR+角度) + 物理(RE/GF/头部下降) + 侧倒 + 已在地面
- 新增幽灵目标机制（跌倒目标保持 30 秒不删除）
- 新增双摄像头支持（多进程 + 跨摄像头匹配）

### 2026-05-11

- 初始化项目：YOLOv8-Pose 单摄像头实时跌倒检测
