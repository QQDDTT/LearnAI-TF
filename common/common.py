# -*- coding: utf-8 -*-
"""
common/common.py
基础公共模块：
- 日志工具类
- 动态反射调用函数/类
- 配置加载函数
"""

import importlib
import logging
import os
from datetime import datetime
import yaml
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

    参数:
        file: YAML 配置文件名（相对于 config 目录）
    返回:
        dict: 配置字典
    """
    project_path = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(project_path, "config")
    config_file = os.path.join(config_dir, file)
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ======================================================
# 动态反射调用
# ======================================================
def call_target(target: str, arguments: dict = None):
    """
    动态反射调用指定函数/类
    支持两种写法:
        1. 模块路径:函数/类名 -> "tensorflow.keras.layers:Dense"
        2. 完整路径 -> "tensorflow.keras.layers.Dense"

    参数:
        target (str): 目标函数/类的路径
            - 冒号分隔: "模块路径:函数名"
            - 点分隔: "完整.模块.路径.函数名"
        arguments (dict): 传递给目标函数的参数

    返回:
        调用结果

    示例:
        >>> call_target("tensorflow.keras.layers:Dense", {"units": 128})
        >>> call_target("tensorflow.keras.layers.Dense", {"units": 128})
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
