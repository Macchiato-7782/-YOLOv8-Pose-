[部署教程与学习笔记.md](https://github.com/user-attachments/files/27598503/default.md)
# Real-Time-Fall-Detection 部署教程与学习笔记

## 一、项目简介

基于 **YOLOv8-Pose** 的实时人体跌倒检测系统。通过摄像头或视频输入，检测人体姿态，利用规则判断是否发生跌倒。

- GitHub: https://github.com/haashi-r/Real-Time-Fall-Detection
- 技术栈: YOLOv8-Pose + OpenCV + NumPy
- 代码量: 单文件 main.py，约 250 行
- 是否需要训练: 不需要，纯规则判断

---

## 二、完整部署教程（macOS）

### 2.1 环境要求

| 项目 | 要求 |
|------|------|
| 系统 | macOS (本教程) / Windows / Linux |
| Python | 3.8+（本机 3.13） |
| 硬件 | 有摄像头（内置或外接） |
| 网络 | 首次需要下载依赖和模型 |

### 2.2 创建项目目录

```bash
mkdir -p ~/Desktop/学习资料/Real-Time-Fall-Detection
cd ~/Desktop/学习资料/Real-Time-Fall-Detection
```

### 2.3 创建项目文件

需要两个文件：

**requirements.txt** — 依赖清单：
```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
```

**main.py** — 主程序代码（从 GitHub 获取或手动创建）

### 2.4 创建虚拟环境

```bash
python3 -m venv venv
```

虚拟环境的作用：将项目的依赖与系统 Python 隔离，避免版本冲突。

```
项目目录/
├── main.py
├── requirements.txt
└── venv/                  ← 虚拟环境
    ├── bin/python3        ← 独立的 Python 解释器
    ├── bin/pip            ← 独立的包管理器
    └── lib/python3.13/
        └── site-packages/ ← 依赖装在这里
```

### 2.5 激活虚拟环境

```bash
source venv/bin/activate
```

激活后终端提示符会多一个 `(venv)` 前缀，表示当前使用的是虚拟环境。

每次打开新终端都需要重新激活。

### 2.6 安装依赖

```bash
# 使用清华镜像源（国内加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

安装的依赖清单：

| 包名 | 用途 | 大小 |
|------|------|------|
| opencv-python | 图像处理、摄像头读取、窗口显示 | ~46MB |
| numpy | 数值计算 | ~5MB |
| ultralytics | YOLOv8 框架 | ~1.2MB |
| torch (间接) | PyTorch 深度学习框架 | ~80MB |
| torchvision (间接) | 视觉模型工具 | ~2MB |
| scipy (间接) | 科学计算 | ~20MB |
| matplotlib (间接) | 画图 | ~8MB |
| pillow (间接) | 图片处理 | ~5MB |

### 2.7 模型文件

首次运行时，ultralytics 会自动下载 `yolov8n-pose.pt`（约 6MB）。

模型命名规则：
- `yolov8` — YOLO 第 8 版
- `n` — nano 版本（最小最快，还有 s/m/l/x 更大更准的版本）
- `pose` — 姿态估计任务
- `.pt` — PyTorch 格式

### 2.8 macOS 摄像头权限

首次使用摄像头需要授权：

```
系统设置 → 隐私与安全性 → 摄像头 → 打开"终端(Terminal)"的开关
```

如果列表里没有"终端"：
1. 点 `+` 号
2. 按 `Cmd+Shift+G`
3. 输入 `/Applications/Utilities/`
4. 选择 `Terminal.app`
5. 打开开关

### 2.9 关于 Continuity Camera

macOS 的"连续互通相机"功能会让 iPhone 自动成为 Mac 的外接摄像头。如果想用 Mac 内置摄像头：

- 方法 1: 把 iPhone 拿远或关掉蓝牙
- 方法 2: 代码中把 `cv2.VideoCapture(0)` 改成 `cv2.VideoCapture(1)`

### 2.10 运行

```bash
python main.py
```

按 ESC 退出。

### 2.11 使用视频文件代替摄像头

修改 main.py 中的视频源：

```python
cap = cv2.VideoCapture("你的视频路径.mp4")  # 用视频文件
# cap = cv2.VideoCapture(0)                  # 注释掉摄像头
```

---

## 三、项目工作流程

```
摄像头/视频 → 逐帧读取 → YOLOv8-Pose 检测人体+17个关键点
    → 过滤不完整检测（"全身可见"检查）
    → 质心跟踪（跨帧匹配同一人）
    → 规则判断跌倒（宽高比 + 髋部角度 + 持续时间）
    → 在画面上标注结果 → 显示
```

---

## 四、核心知识点

### 4.1 YOLOv8-Pose 姿态估计

YOLOv8 是 Ultralytics 公司开发的目标检测模型，Pose 版本在检测人体的同时输出 17 个关键点：

```
COCO 17 关键点：
0  - Nose (鼻子)
1  - LEye (左眼)
2  - REye (右眼)
3  - LEar (左耳)
4  - REar (右耳)
5  - LShoulder (左肩)
6  - RShoulder (右肩)
7  - LElbow (左肘)
8  - RElbow (右肘)
9  - LWrist (左腕)
10 - RWrist (右腕)
11 - LHip (左髋)
12 - RHip (右髋)
13 - LKnee (左膝)
14 - RKnee (右膝)
15 - LAnkle (左踝)
16 - RAnkle (右踝)
```

每个关键点包含 `(x, y, confidence)` 三个值。

### 4.2 "全身可见"过滤

不是所有检测都可靠——如果一个人只露出上半身，关键点缺失，姿态判断就不准。

代码中定义了 7 个必要关键点 `[0, 5, 6, 11, 12, 15, 16]`（鼻子、双肩、双髋、双踝），至少要有 5 个可见且置信度 > 0.3，才算"全身可见"，才进入跌倒判断逻辑。

**学习点**: 不是所有检测结果都应该用，需要过滤低质量检测。

### 4.3 质心跟踪 (Centroid Tracking)

最简单的多人跟踪方法：

```
1. 每帧检测到的人，计算其 bounding box 的中心点
2. 与上一帧的所有已跟踪目标比较距离
3. 距离 < 阈值（150像素）→ 认为是同一个人，更新位置
4. 没有匹配到的 → 如果是全身检测，创建新跟踪 ID
5. 超过 5 秒没出现 → 删除该跟踪目标
```

**学习点**: 这是最基础的跟踪算法，实际产品中会用 DeepSORT、ByteTrack 等更鲁棒的方法。

### 4.4 跌倒判断的三个规则

#### 规则 1：宽高比 (Aspect Ratio)

```python
aspect_ratio = height / width   # 关键点 bounding box 的高/宽

if aspect_ratio < 0.6:          # 人变得"扁"了
    → 可能跌倒
```

**原理**: 站着的人 bbox 是窄长的（高/宽大），倒下后 bbox 变成宽扁的（高/宽小）。

#### 规则 2：髋部角度 (Hip Angle)

```python
angle = calculate_angle(shoulder, hip, knee)  # 肩-髋-膝 三点夹角

if angle < 140°:                # 身体"折叠"了
    → 可能跌倒
```

**原理**: 站立时肩-髋-膝接近一条直线（~180°），跌倒时身体折叠，角度急剧减小。

#### 规则 3：持续时间 (Duration)

```python
if 可能跌倒状态 持续 >= 1.0秒:
    → 确认跌倒
```

**原理**: 蹲下、弯腰等动作也会触发规则 1 或 2，但持续时间短。真正跌倒后人会倒在地上一段时间，所以需要持续一定时长才确认。

**学习点**: 单一规则容易误报，多规则组合 + 时间窗口可以大幅降低误报率。

### 4.5 三状态模型

```
Normal（正常）
    ↓ 检测到异常姿态（宽高比 < 0.6 或 角度 < 140°）
Potential Fall（可能跌倒）← 橙色标注，开始计时
    ↓ 持续 >= 1 秒
Fall（确认跌倒）← 红色标注
    ↓ 异常姿态消失
Normal（恢复正常）← 绿色标注，计时清零
```

**学习点**: 引入中间状态（Potential Fall）避免了瞬间误判，这是实时检测系统中常见的设计模式。

### 4.6 关键参数及其意义

| 参数 | 值 | 含义 | 调优建议 |
|------|-----|------|----------|
| `MIN_CONF_DETECTION` | 0.5 | 人体检测最低置信度 | 降低会检测到更多人，但增加误检 |
| `MIN_CONF_KPT` | 0.3 | 关键点最低置信度 | 降低会接受更多关键点，但可能不准 |
| `HORIZONTAL_AR_THRESHOLD` | 0.6 | 宽高比阈值 | 降低更严格（更不容易触发） |
| `ANGLE_THRESHOLD` | 140° | 髋部角度阈值 | 降低更严格 |
| `MIN_FALL_POSE_DURATION` | 1.0s | 跌倒确认时间 | 增加减少误报，但检测变慢 |
| `TRACKING_DISTANCE_THRESHOLD` | 150px | 跟踪匹配距离 | 根据摄像头距离调整 |
| `HISTORY_CLEANUP_INTERVAL` | 5s | 跟踪目标过期时间 | 根据场景调整 |

### 4.7 与 HumanFallDetection 项目的对比

| 维度 | HumanFallDetection | Real-Time-Fall-Detection |
|------|-------------------|-------------------------|
| 姿态估计 | OpenPifPaf (已停更) | YOLOv8-Pose (活跃维护) |
| 分类方法 | LSTM 神经网络 | 纯规则判断 |
| 是否需要训练 | 需要 | 不需要 |
| 特征数量 | 5 个物理特征 | 2 个几何特征 + 时间 |
| 多摄像头 | 支持 | 不支持 |
| 代码量 | 多文件，复杂 | 单文件，简单 |
| 准确率 | ~90% | 约 85-90%（规则场景依赖） |
| 部署难度 | 高（Python 版本限制） | 低 |

### 4.8 该领域的技术演进

```
2018-2020: OpenPose + SVM/Random Forest
2020-2022: OpenPifPaf + LSTM / TCN
2022-2024: YOLOv8-Pose + ST-GCN / Transformer
2025-2026: YOLOv8-Pose + 轻量 Transformer / 端到端模型
```

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: cv2` | 没激活虚拟环境 | `source venv/bin/activate` |
| `not authorized to capture video` | macOS 摄像头未授权 | 系统设置 → 隐私与安全性 → 摄像头 |
| 用的是手机摄像头 | Continuity Camera | 把 iPhone 拿远或改 `VideoCapture(1)` |
| `No such file: main.py` | 不在项目目录 | `cd ~/Desktop/学习资料/Real-Time-Fall-Detection` |
| conda-libmamba-solver 报错 | Anaconda 环境损坏 | 忽略或 `conda install conda-libmamba-solver --force-reinstall` |
| 检测不到人 | 距离太远/光线暗 | 降低 `MIN_CONF_DETECTION`，或走近摄像头 |
| 误报太多 | 阈值太松 | 提高 `HORIZONTAL_AR_THRESHOLD` 和 `ANGLE_THRESHOLD` |

---

## 六、下一步学习方向

1. **换用自己的视频测试** — 录一段跌倒视频，用 `VideoCapture("视频路径")` 测试
2. **调参** — 修改阈值参数，观察检测效果变化
3. **加报警功能** — 检测到跌倒时发送通知（邮件/微信/蜂鸣器）
4. **学习 YOLOv8** — 尝试用 YOLOv8 做其他视觉任务（目标检测、分割等）
5. **了解 ST-GCN** — 图卷积网络，比规则判断更准确的时序分类方法
6. **部署到边缘设备** — 树莓派、Jetson Nano 等嵌入式设备上运行
