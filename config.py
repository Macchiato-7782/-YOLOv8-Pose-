"""
配置加载模块
从 config.yaml 读取参数，各模块通过此模块获取配置值
"""

import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')


def load_config(path=CONFIG_PATH):
    """加载配置文件，返回嵌套字典"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# 全局配置实例（模块导入时自动加载）
CFG = load_config()

# 快捷访问各模块配置
FALL = CFG.get('fall_logic', {})
FEATURES = CFG.get('features', {})
TRACKING = CFG.get('tracking', {})
CROSS_CAM = CFG.get('cross_camera', {})
CAM_PROC = CFG.get('camera_process', {})
