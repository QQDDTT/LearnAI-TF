# -*- coding: utf-8 -*-
"""
data/dataloader.py
通过 YAML + call_target 构建 TensorFlow DataLoader
支持：
  1. TensorFlow 内置加载器
  2. CSV 文件加载
  3. Web 数据加载
"""

from typing import Dict
from modules.utils import call_target, Logger
import tensorflow as tf

logger = Logger(__file__)

def build_dataloader(dl_cfg: dict) -> Dict[str, tf.data.Dataset]:
    """
    根据 YAML 配置构建 DataLoader
    dl_cfg 示例：
    """
    dataloaders = {}
    for key, cfg in dl_cfg.items():
        try:
            reflection = cfg["reflection"]
            args = cfg.get("args", {})
            logger.info(f"Building : {key}, target={reflection}")
            dataloaders[key] = call_target(reflection, args)
        except Exception as e:
            logger.error(f"Failed to build dataloader '{key}': {e}")

    return dataloaders
