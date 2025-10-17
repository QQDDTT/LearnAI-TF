# -*- coding: utf-8 -*-
"""
lib/data_utils.py
数据处理工具函数：批次获取、数据加载等
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


def get_batch(
    loader: tf.data.Dataset,
    **kwargs
) -> Dict[str, Any]:
    """
    从数据加载器获取一个批次

    参数:
        loader: TensorFlow Dataset实例

    返回:
        包含x和y的字典
    """
    if not isinstance(loader, tf.data.Dataset):
        raise TypeError(f"loader 必须是 tf.data.Dataset 实例，得到 {type(loader)}")

    try:
        # 获取一个批次
        iterator = iter(loader)
        batch = next(iterator)

        # 解析批次
        if isinstance(batch, tuple):
            if len(batch) == 2:
                x, y = batch
                result = {"x": x, "y": y}
            else:
                x = batch[0]
                result = {"x": x}
        else:
            result = {"x": batch}

        logger.debug(f"批次获取完成，x形状: {result['x'].shape}")

        return result

    except StopIteration:
        logger.warning("数据加载器已耗尽")
        raise
    except Exception as e:
        logger.error(f"批次获取失败: {e}")
        raise


def get_all_data(
    loader: tf.data.Dataset,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    从数据加载器获取所有数据

    参数:
        loader: TensorFlow Dataset实例

    返回:
        包含x和可选y的字典
    """
    if not isinstance(loader, tf.data.Dataset):
        raise TypeError(f"loader 必须是 tf.data.Dataset 实例，得到 {type(loader)}")

    try:
        x_list = []
        y_list = []
        has_labels = False

        # 遍历所有批次
        for batch in loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                x, y = batch
                x_list.append(x.numpy() if isinstance(x, tf.Tensor) else x)
                y_list.append(y.numpy() if isinstance(y, tf.Tensor) else y)
                has_labels = True
            else:
                if isinstance(batch, tuple):
                    x = batch[0]
                else:
                    x = batch
                x_list.append(x.numpy() if isinstance(x, tf.Tensor) else x)

        # 合并所有批次
        x_all = np.concatenate(x_list, axis=0)

        result = {"x": x_all}

        if has_labels:
            y_all = np.concatenate(y_list, axis=0)
            result["y"] = y_all

        logger.info(f"所有数据加载完成，x形状: {x_all.shape}")

        return result

    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


def prepare_batch(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    **kwargs
) -> tf.data.Dataset:
    """
    准备批次数据

    参数:
        x: 特征数组
        y: 标签数组（可选）
        batch_size: 批次大小

    返回:
        TensorFlow Dataset
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0: {batch_size}")

    try:
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(x)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.debug(f"批次数据准备完成，batch_size={batch_size}")

        return dataset

    except Exception as e:
        logger.error(f"批次数据准备失败: {e}")
        raise
