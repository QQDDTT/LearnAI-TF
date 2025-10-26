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
        "global",
        "training_mode",
        "models",
        "data_manager",
        "optimizers",
        "losses",
        "training_pipeline"
    }

    # 每个模块内部必须包含的字段
    # 注意：这里假设了结构，原代码中未定义
    # 这里我们只定义原代码中 _validate_constant_layer 隐含的结构
    MODULE_REQUIRED_FIELDS: Dict[str, Set[str]] = {
        "global": {"name", "version"},
        "training_mode": {"type"},
        "data_manager": {},
        "optimizers": {},
        "losses": {}
        # training_pipeline 的字段由专门验证器检查
    }

class TrainingPipelineConstants:
    """训练流程配置常量"""

    # 支持的循环类型
    SUPPORTED_LOOP_TYPES = {
        "epoch_batch",       # 监督学习：epoch + batch
        "episode_step",      # 强化学习：episode + step
        "iteration",         # 无监督学习：迭代
        "custom"             # 自定义循环
    }

    # 每种循环类型的必需参数
    LOOP_TYPE_REQUIRED_PARAMS = {
        "epoch_batch": {"epochs", "batch_size"},
        "episode_step": {"episodes", "max_steps_per_episode"},
        "iteration": {"max_iterations"},
        "custom": {}  # 自定义循环没有固定参数
    }

    # 终止条件检查类型
    SUPPORTED_TERMINATION_TYPES = {
        "epoch_based",
        "episode_based",
        "convergence_based",
        "time_based",
        "custom"
    }

    # 每个训练模式下必需的字段
    PIPELINE_REQUIRED_FIELDS = {
        "loop_config",      # 循环配置（必须）
        "step_sequence"     # 步骤序列（必须）
    }

    # loop_config 必需的字段
    LOOP_CONFIG_REQUIRED_FIELDS = {
        "type",            # 循环类型
        "parameters",      # 循环参数
        "termination"      # 终止条件
    }


class BridgeConstants:
    """Bridge 控制常量"""

    # 支持的 Bridge 动作类型
    SUPPORTED_ACTIONS = {
        "SKIP",      # 跳过
        "LOOP",      # 循环
        "BRANCH",    # 分支
        "JUMP"       # 跳转（SKIP 的别名）
    }

    # 条件表达式支持的运算符
    COMPARISON_OPERATORS = {"==", "!=", ">", "<", ">=", "<="}
    LOGICAL_OPERATORS = {"AND", "OR", "NOT"}


class TrainingModeConstants:
    """训练模式常量"""

    SUPPORTED_MODES = {
        "supervised",
        "reinforcement",
        "unsupervised",
        "self_supervised",
        "semi_supervised",
        "multi_task",
        "custom"
    }
