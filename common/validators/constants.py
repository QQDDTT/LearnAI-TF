# -*- coding: utf-8 -*-
# common/validators/constants.py
"""
配置验证器 - 常量定义
"""
from typing import Set, Dict, List

class ConfigStructure:
    """定义配置文件的必须结构"""

    # 顶层必须包含的模块
    REQUIRED_MODULES: Set[str] = {
        "training_mode",
        "models",
        "data_sources",
        "optimizers",
        "losses",
        "training_pipeline"
    }

    # 每个模块内部必须包含的字段
    # 注意：这里假设了结构，原代码中未定义
    # 这里我们只定义原代码中 _validate_constant_layer 隐含的结构
    MODULE_REQUIRED_FIELDS: Dict[str, Set[str]] = {
        "training_mode": {"type"},
        # "models": {}, # models 内部结构由 collection_validator 检查
        # "training_pipeline": {}, # 同上
        "data_sources": {"default"},
        "optimizers": {"default"},
        "losses": {"default"},
    }

class TrainingModeConstants:
    """训练模式常量"""
    SUPPORTED_MODES: Set[str] = {
        "supervised",
        "reinforcement",
        "unsupervised",
        "custom"
    }
