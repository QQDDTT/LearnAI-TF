# -*- coding: utf-8 -*-
"""
lib/data_loader.py
数据加载工具：CSV、NumPy、批次生成等
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


def build_csv_dataset(
    file_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_split: float = 0.0,
    normalize: bool = False,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    **kwargs
) -> tf.data.Dataset:
    """
    构建CSV数据集

    参数:
        file_path: CSV文件路径
        batch_size: 批次大小
        shuffle: 是否打乱
        validation_split: 验证集分割比例
        normalize: 是否归一化
        target_column: 目标列名
        feature_columns: 特征列名列表

    返回:
        tf.data.Dataset
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV文件为空: {file_path}")

    if len(df) == 0:
        raise ValueError(f"数据集为空: {file_path}")

    # 分离特征和标签
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
    elif feature_columns:
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"特征列不存在: {missing_cols}")
        y = df.drop(columns=feature_columns).values
        X = df[feature_columns].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # 归一化
    if normalize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + 1e-8)

    # 验证集分割
    if validation_split > 0:
        if not 0 < validation_split < 1:
            raise ValueError(f"validation_split 必须在 (0, 1) 范围内: {validation_split}")
        split_idx = int(len(X) * (1 - validation_split))
        X, y = X[:split_idx], y[:split_idx]

    # 构建Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_numpy_dataset(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_split: float = 0.0,
    **kwargs
) -> tf.data.Dataset:
    """
    从NumPy数组构建数据集

    参数:
        X: 特征数组
        y: 标签数组
        batch_size: 批次大小
        shuffle: 是否打乱
        validation_split: 验证集分割比例

    返回:
        tf.data.Dataset
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X 必须是 numpy.ndarray, 得到 {type(X)}")

    if len(X) == 0:
        raise ValueError("输入数组为空")

    # 验证集分割
    if validation_split > 0:
        if not 0 < validation_split < 1:
            raise ValueError(f"validation_split 必须在 (0, 1) 范围内: {validation_split}")
        split_idx = int(len(X) * (1 - validation_split))
        X = X[:split_idx]
        if y is not None:
            y = y[:split_idx]

    # 构建Dataset
    if y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def split_dataset(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    分割数据集为训练集、验证集和测试集

    参数:
        X: 特征数组
        y: 标签数组
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        shuffle: 是否打乱
        seed: 随机种子

    返回:
        包含分割后数据的字典
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 必须在 (0, 1) 范围内: {train_ratio}")

    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio 必须在 [0, 1) 范围内: {val_ratio}")

    if train_ratio + val_ratio >= 1:
        raise ValueError(f"train_ratio + val_ratio 必须小于 1: {train_ratio + val_ratio}")

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    result = {
        'X_train': X[train_idx],
        'X_val': X[val_idx] if len(val_idx) > 0 else None,
        'X_test': X[test_idx] if len(test_idx) > 0 else None
    }

    if y is not None:
        result['y_train'] = y[train_idx]
        result['y_val'] = y[val_idx] if len(val_idx) > 0 else None
        result['y_test'] = y[test_idx] if len(test_idx) > 0 else None

    return result


def normalize_data(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准化数据

    参数:
        X: 输入数组
        mean: 均值（如果为None则计算）
        std: 标准差（如果为None则计算）

    返回:
        (标准化后的数据, 均值, 标准差)
    """
    if mean is None:
        mean = np.mean(X, axis=0)

    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / (std + 1e-8)

    return X_normalized, mean, std


def create_batches(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> List[Tuple]:
    """
    创建批次

    参数:
        X: 特征数组
        y: 标签数组
        batch_size: 批次大小
        shuffle: 是否打乱

    返回:
        批次列表
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0: {batch_size}")

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]

        if y is not None:
            batches.append((X[batch_idx], y[batch_idx]))
        else:
            batches.append((X[batch_idx],))

    return batches
