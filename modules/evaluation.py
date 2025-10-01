# -*- coding: utf-8 -*-
"""
train/evaluation.py
模型评估模块

该模块定义了不同训练模式的评估步骤函数，
所有函数都遵循统一的参数签名：
    (config, logger, model, **kwargs)

- config: 配置文件 dict（包含 evaluation 部分）
- logger: 日志对象
- model: 构建并训练好的模型对象
- kwargs: 预留扩展参数，例如验证集 dataloader
"""

from modules.utils import Logger, call_target

logger = Logger(__file__)

def evaluate_model(eval_cfg: dict, model_dict: dict):
    logger.warning("building...")
    pass
