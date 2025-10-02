# -*- coding: utf-8 -*-
"""
common/utils.py
通用工具模块：
- 动态反射调用函数/类
- 日志工具类
- 配置加载函数
- 网络客户端类（用于环境交互）
- 本地数据加载函数
- 强化学习奖励函数
- 强化学习动作函数
"""

import importlib
import logging
import os
from datetime import datetime
import requests
import pandas as pd
import time
from typing import Any, Dict, Optional
import yaml
import tensorflow as tf
from colorama import Fore, Style, init

# ======================================================
# 日志工具类
# ======================================================
class LoggerManager:
    """
    全局日志管理器：
    - 控制台彩色输出
    - 文件日志纯文本
    - 每天一个文件，按项目存放
    - 每个 Python 文件调用 get_logger(__file__) 获取 logger
    """
    _loggers = {}      # {logger_name: logger实例}
    _file_handlers = {}  # {project: FileHandler实例}
    _LEVEL_COLOR = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    @classmethod
    def get_logger(cls, filename: str, project="LearnAI"):
        init(autoreset=True)
        classname = os.path.splitext(os.path.basename(filename))[0]
        logger_name = f"{project}.{classname}"

        # 已存在直接返回
        if logger_name in cls._loggers:
            return cls._loggers[logger_name]

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # ---------------- 控制台 Handler ----------------
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s][%(classname)s] %(message)s"))
            ch.addFilter(cls.ColorFilter(classname))
            logger.addHandler(ch)

        # ---------------- 文件 Handler ----------------
        today = datetime.now().strftime("%Y%m%d")
        log_home = os.environ.get("LOG_HOME", "/tmp")
        log_dir = os.path.join(log_home, project)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{project}_{today}.log")

        # 每个项目共用一个 FileHandler
        if project not in cls._file_handlers:
            fh = logging.FileHandler(log_file_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s][%(classname)s] %(message)s"))
            fh.addFilter(cls.PlainFilter(classname))  # 文件纯文本
            cls._file_handlers[project] = fh
            print(f"日志文件输出路径: {log_file_path}")

        # 添加 FileHandler 到 logger（避免重复添加）
        if cls._file_handlers[project] not in logger.handlers:
            logger.addHandler(cls._file_handlers[project])

        cls._loggers[logger_name] = logger
        return logger

    # ---------------- Filter ----------------
    class ColorFilter(logging.Filter):
        """控制台彩色输出"""
        def __init__(self, classname):
            super().__init__()
            self.classname = classname

        def filter(self, record):
            record.classname = f"{Fore.BLUE}{self.classname}{Style.RESET_ALL}"
            color = LoggerManager._LEVEL_COLOR.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            if record.levelno >= logging.WARNING:
                record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
            return True

    class PlainFilter(logging.Filter):
        """文件输出纯文本"""
        def __init__(self, classname):
            super().__init__()
            self.classname = classname

        def filter(self, record):
            record.classname = self.classname
            # 恢复原始等级名
            record.levelname = logging.getLevelName(record.levelno)
            # 确保 msg 纯文本
            if hasattr(record, "msg"):
                # 如果 msg 是 ANSI 彩色的，去掉 ESC 序列
                import re
                ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
                record.msg = ansi_escape.sub("", str(record.msg))
            return True



# ======================================================
# 配置加载
# ======================================================
def load_yaml(file: str) -> dict:
    """
    从 YAML 文件加载配置
    """
    project_path = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(project_path, "config")
    config_file = os.path.join(config_dir, file)
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ======================================================
# 动态反射调用
# ======================================================
def call_target(target: str, arguments: Dict[str, Any] = None):
    """
    动态反射调用指定函数/类
    支持两种写法:
        1. 模块路径:函数/类名 -> "tensorflow.keras.layers:Dense"
        2. 完整路径 -> "tensorflow.keras.layers.Dense"

    参数:
        target (str): "模块路径:函数名"
        arguments (dict): 传递给目标函数的参数
    返回:
        调用结果
    """
    if arguments is None:
        arguments = {}

    if ":" in target:
        # 冒号分隔
        module_path, func_name = target.split(":")
        module = importlib.import_module(module_path)
        func_or_class = getattr(module, func_name)
    else:
        # 点分隔，取最后一段为类名，其余为模块
        module_path, func_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func_or_class = getattr(module, func_name)

    if callable(func_or_class):
        return func_or_class(**arguments)
    else:
        raise ValueError(f"{target} is not callable")


# ======================================================
# 网络客户端数据加载器（强化学习环境接口）
# ======================================================
class NetworkClient:
    """
    网络客户端：用于和外部环境交互（强化学习等）
    """
    def __init__(self, host: str, port: int, endpoints: Dict[str, str]):
        self.base_url = f"http://{host}:{port}"
        self.endpoints = endpoints

    def request(self, endpoint_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        向环境发送请求，获取响应
        """
        url = self.base_url + self.endpoints[endpoint_name]
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def build_web_dataset(
    client: NetworkClient,
    endpoint_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    delay: float = 0.0
) -> tf.data.Dataset:
    """
    动态从网络环境获取数据，并生成 tf.data.Dataset
    每次迭代都会发送请求并获取新的状态/奖励

    参数:
        client: NetworkClient 实例
        endpoint_name: 要请求的端点名称
        batch_size: 批大小
        shuffle: 是否打乱（仅对 batch 内有效）
        delay: 每次请求后的等待时间（秒），可控制请求频率
    返回:
        tf.data.Dataset
    """

    def generator():
        while True:
            data = client.request(endpoint_name)
            # 假设返回 dict: {"state": [...], "action": [...], "reward": ...}
            # 可根据你的环境调整
            x = data["state"]
            y = data.get("reward", 0.0)
            yield x, y
            if delay > 0:
                time.sleep(delay)

    # 使用 from_generator 构建 Dataset
    sample_x, sample_y = next(generator())
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=tf.shape(sample_x), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size*10)  # buffer_size 可根据需求调整
    ds = ds.batch(batch_size)
    return ds

# ======================================================
# 本地 CSV 数据加载器
# ======================================================
def build_csv_loader(file_path: str, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    加载 CSV 数据并返回 tf.data.Dataset
    参数:
        file_path: CSV 文件路径
        batch_size: 批大小
        shuffle: 是否打乱
    返回:
        tf.data.Dataset
    """
    df = pd.read_csv(file_path)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size)
    return ds

# ======================================================
# 强化学习奖励函数
# ======================================================
def compute_reward(response: dict, score_delta_weight=1.0, end_bonus=10.0, max_value_bonus=5.0):
    """
    根据环境响应计算奖励

    参数:
        response (dict): 环境返回的响应，包含 reward, done, 等字段
        score_delta_weight (float): 分数变化权重
        end_bonus (float): 完成任务奖励
        max_value_bonus (float): 最大奖励上限

    返回:
        float: 奖励值
    """
    reward = response.get("reward", 0) * score_delta_weight
    if response.get("done", False):
        reward += end_bonus
    reward += min(response.get("reward", 0), max_value_bonus)
    return reward


# ======================================================
# 强化学习动作函数
# ======================================================
def select_action(model_output: int, mapping: dict = None):
    """
    根据模型输出选择动作

    参数:
        model_output (int): 模型输出的动作索引
        mapping (dict): 动作映射表，例如 {0:"up", 1:"down", ...}

    返回:
        str: 动作名称
    """
    if mapping is None:
        mapping = {
            0: "up", 1: "down", 2: "left", 3: "right",
            4: "special1", 5: "special2", 6: "special3", 7: "special4"
        }
    return mapping.get(model_output, "noop")
