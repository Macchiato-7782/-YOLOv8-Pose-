# Real-Time Fall Detection

基于 YOLOv8-Pose 的实时人体跌倒检测系统，支持**单摄像头**和**双摄像头**模式。

- 单摄像头：四路检测（几何 + 物理 + 侧倒 + 地面） + 滑动窗口 + 持续时间确认
- 双摄像头：多进程并行 + HSV 直方图跨摄像头匹配 + 稳定婚姻算法 + 双视角交叉验证

## 效果

- 绿色标注：正常状态
- 橙色标注：可能跌倒（计时中）
- 红色标注：确认跌倒
- 黄色标注：全局 ID（双摄像头模式下同一个人的统一编号）

## 技术方案

### 单摄像头

```
摄像头 → YOLOv8-Pose (17关键点) → 全身过滤 → 质心跟踪
    → 四路跌倒检测:
        路径1: 宽高比 + 髋部角度 (AND)
        路径2: 旋转能量 / 重力因子 / 头部下降 (OR)
        路径3: AR剧变 + 头部下降 (侧倒)
        路径4: 已在地面检测
    → 滑动窗口(10帧, 50%) → 持续时间(1秒) → 显示
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
# 单摄像头
python main.py

# 单视频文件
python main.py --video your_video.mp4

# 双摄像头
python main.py --num_cams 2 --cam_ids 0 1

# 双视频文件
python main.py --num_cams 2 --video cam1.mp4 cam2.mp4

# 保存输出视频
python main.py --num_cams 2 --cam_ids 0 1 --save_output
```

按 ESC 退出。

## 模拟测试

不依赖摄像头，用合成数据验证检测逻辑：

```bash
python test_fall_detection.py
```

覆盖 11 个场景：站立、坐、前倒、侧倒、快速摔倒、弯腰站起、已躺地上、序列化、fall_state 持久化等。

## 跌倒判断逻辑

### 四路检测（任一触发即为"可能跌倒"）

| 路径 | 条件 | 适用场景 |
|------|------|----------|
| 路径1: 几何 | AR/初始AR < 0.35 **且** 角度 < 130° | 前倒/后倒 |
| 路径2: 物理 | RE > 1.0 **或** GF > 20 **或** 头部下降 > 0.2 | 快速摔倒/蜷缩倒 |
| 路径3: 侧倒 | AR 剧变 **且** 头部下降 | 侧倒（角度不够低） |
| 路径4: 地面 | 初始AR低 **且** 当前AR < 0.6 **且** 持续5帧 | 已在地上的人 |

### 确认机制

- 滑动窗口：10 帧内 50% 触发
- 持续时间：1 秒
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
| `--cam_ids` | 1 | 摄像头 ID |
| `--video` | None | 视频文件路径 |
| `--save_output` | False | 保存输出视频 |

## 依赖

- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- ultralytics >= 8.0.0

## macOS 注意事项

首次使用摄像头需要授权：系统设置 → 隐私与安全性 → 摄像头 → 打开"终端"。

## 项目结构

```
├── main.py                 # 入口，支持单/双摄像头模式
├── fall_logic.py           # 融合跌倒判断逻辑（四路检测 + 滑动窗口）
├── features.py             # 物理特征计算（旋转能量、重力因子、头部下降）
├── tracking.py             # 单摄像头质心跟踪 + 幽灵机制
├── cross_camera.py         # 跨摄像头匹配（直方图 + 稳定婚姻算法）
├── camera_process.py       # 摄像头处理进程（多进程架构）
├── test_fall_detection.py  # 模拟测试（11 个场景）
├── requirements.txt        # 依赖清单
└── README.md
```

## 参考

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HumanFallDetection](https://github.com/taufeeque9/HumanFallDetection) — 双摄像头方案和物理特征
- [COCO Keypoints](https://cocodataset.org/#keypoints-2017)

## 更新日志

### 2026-05-13

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
