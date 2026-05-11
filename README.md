[Uploading README.md…]()
# Real-Time Fall Detection

基于 YOLOv8-Pose 的实时人体跌倒检测系统。通过摄像头或视频输入，检测人体姿态，利用多规则判断是否发生跌倒。

## 效果

- 绿色标注：正常状态
- 橙色标注：可能跌倒（计时中）
- 红色标注：确认跌倒

## 技术方案

```
摄像头/视频 → YOLOv8-Pose 检测 17 个关键点 → 全身可见性过滤
    → 质心跟踪（跨帧匹配同一人）
    → 三规则联合判断：宽高比 + 髋部角度 + 持续时间
    → 实时标注显示
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
# 使用摄像头
python main.py

# 使用视频文件（修改 main.py 中的视频源）
# cap = cv2.VideoCapture("your_video.mp4")
```

按 ESC 退出。

## 跌倒判断逻辑

| 规则 | 条件 | 含义 |
|------|------|------|
| 宽高比 | height/width < 0.6 | 人变得"扁"了 |
| 髋部角度 | 肩-髋-膝夹角 < 140° | 身体折叠 |
| 持续时间 | 异常姿态持续 >= 1 秒 | 排除蹲下/弯腰等瞬间动作 |

三个条件组合：宽高比异常 OR 髋部角度异常 → 标记"可能跌倒" → 持续 1 秒 → 确认"跌倒"。

## 依赖

- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- ultralytics >= 8.0.0

## macOS 注意事项

首次使用摄像头需要授权：系统设置 → 隐私与安全性 → 摄像头 → 打开"终端"。

## 项目结构

```
├── main.py              # 主程序
├── requirements.txt     # 依赖清单
├── .gitignore
└── README.md
```

## 参考

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Keypoints](https://cocodataset.org/#keypoints-2017)
