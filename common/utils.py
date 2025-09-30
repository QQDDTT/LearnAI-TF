# -*- coding: utf-8 -*-
"""
common/utils.py
通用工具模块：反射调用、日志、配置加载
"""

import importlib
import logging
import yaml


# ---------------- 动态反射调用 ----------------
def call_target(class_path: str, params: dict = None):
    """
    动态导入并实例化对象
    :param class_path: 类路径字符串，例如 'models.model.MyModel'
    :param params: 构造函数参数（dict）
    :return: 类实例
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target_cls = getattr(module, class_name)
    return target_cls(**(params or {}))


# ---------------- 配置文件加载 ----------------
def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    加载 YAML 配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- 日志工具 ----------------
def get_logger(name: str = "LearnAI-TF", log_file: str = "train.log"):
    """
    创建日志对象
    :param name: 日志器名称
    :param log_file: 输出日志文件
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 文件输出
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
