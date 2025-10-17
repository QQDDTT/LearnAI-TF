# -*- coding: utf-8 -*-
"""
lib/training_utils.py
训练工具函数：前向传播、损失计算、梯度计算、参数更新、指标计算等
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional, Union
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


def forward(
    model: tf.keras.Model,
    inputs: Union[tf.Tensor, np.ndarray],
    training: bool = False,
    **kwargs
) -> tf.Tensor:
    """
    前向传播

    参数:
        model: Keras模型
        inputs: 输入数据
        training: 是否为训练模式

    返回:
        模型输出
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model 必须是 tf.keras.Model 实例，得到 {type(model)}")

    try:
        # 确保输入是张量
        if isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs)

        # 前向传播
        outputs = model(inputs, training=training)

        logger.debug(f"前向传播完成，输入形状: {inputs.shape}, 输出形状: {outputs.shape}")

        return outputs

    except Exception as e:
        logger.error(f"前向传播失败: {e}")
        raise


def compute_loss(
    predictions: tf.Tensor,
    targets: Union[tf.Tensor, np.ndarray],
    loss_fn: tf.keras.losses.Loss,
    **kwargs
) -> tf.Tensor:
    """
    计算损失

    参数:
        predictions: 模型预测
        targets: 真实标签
        loss_fn: 损失函数

    返回:
        损失值
    """
    if not isinstance(loss_fn, tf.keras.losses.Loss):
        raise TypeError(f"loss_fn 必须是 tf.keras.losses.Loss 实例，得到 {type(loss_fn)}")

    try:
        # 确保目标是张量
        if isinstance(targets, np.ndarray):
            targets = tf.convert_to_tensor(targets)

        # 计算损失
        loss = loss_fn(targets, predictions)

        logger.debug(f"损失计算完成: {float(loss):.6f}")

        return loss

    except Exception as e:
        logger.error(f"损失计算失败: {e}")
        raise


def compute_gradients(
    loss: tf.Tensor,
    model: tf.keras.Model,
    **kwargs
) -> List[tf.Tensor]:
    """
    计算梯度

    参数:
        loss: 损失值
        model: Keras模型

    返回:
        梯度列表
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model 必须是 tf.keras.Model 实例，得到 {type(model)}")

    try:
        # 计算梯度
        gradients = tf.gradients(loss, model.trainable_variables)

        # 检查是否有None梯度
        if any(g is None for g in gradients):
            logger.warning("存在 None 梯度")

        logger.debug(f"梯度计算完成，{len(gradients)} 个参数")

        return gradients

    except Exception as e:
        logger.error(f"梯度计算失败: {e}")
        raise


def apply_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: List[tf.Tensor],
    model: tf.keras.Model,
    **kwargs
):
    """
    应用梯度更新参数

    参数:
        optimizer: 优化器
        gradients: 梯度列表
        model: Keras模型
    """
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise TypeError(f"optimizer 必须是 tf.keras.optimizers.Optimizer 实例，得到 {type(optimizer)}")

    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model 必须是 tf.keras.Model 实例，得到 {type(model)}")

    try:
        # 应用梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        logger.debug("参数更新完成")

    except Exception as e:
        logger.error(f"参数更新失败: {e}")
        raise


def compute_metrics(
    predictions: Union[tf.Tensor, np.ndarray],
    targets: Union[tf.Tensor, np.ndarray],
    metrics: List[str],
    **kwargs
) -> Dict[str, float]:
    """
    计算评估指标

    参数:
        predictions: 模型预测
        targets: 真实标签
        metrics: 指标名称列表

    返回:
        指标字典
    """
    # 转换为numpy数组
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()

    # 如果是概率分布，取argmax
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        pred_classes = np.argmax(predictions, axis=-1)
    else:
        pred_classes = predictions

    if len(targets.shape) > 1 and targets.shape[-1] > 1:
        true_classes = np.argmax(targets, axis=-1)
    else:
        true_classes = targets

    results = {}

    try:
        for metric in metrics:
            if metric == "accuracy":
                results["accuracy"] = float(np.mean(pred_classes == true_classes))

            elif metric == "precision":
                # 简化的精确率计算（二分类）
                tp = np.sum((pred_classes == 1) & (true_classes == 1))
                fp = np.sum((pred_classes == 1) & (true_classes == 0))
                results["precision"] = float(tp / (tp + fp + 1e-8))

            elif metric == "recall":
                # 简化的召回率计算（二分类）
                tp = np.sum((pred_classes == 1) & (true_classes == 1))
                fn = np.sum((pred_classes == 0) & (true_classes == 1))
                results["recall"] = float(tp / (tp + fn + 1e-8))

            elif metric == "f1":
                # F1分数
                precision = results.get("precision")
                recall = results.get("recall")

                if precision is None or recall is None:
                    # 重新计算
                    tp = np.sum((pred_classes == 1) & (true_classes == 1))
                    fp = np.sum((pred_classes == 1) & (true_classes == 0))
                    fn = np.sum((pred_classes == 0) & (true_classes == 1))
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)

                results["f1"] = float(2 * precision * recall / (precision + recall + 1e-8))

            else:
                logger.warning(f"不支持的指标: {metric}")

        logger.debug(f"指标计算完成: {results}")

        return results

    except Exception as e:
        logger.error(f"指标计算失败: {e}")
        raise
