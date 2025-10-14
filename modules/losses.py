# -*- coding: utf-8 -*-
"""
modules/losses.py
损失函数模块：
- 构建TensorFlow内置损失
- 自定义损失函数
- 多任务损失聚合
"""

import tensorflow as tf
from typing import Dict, List, Any
from common.common import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主构建函数
# ======================================================
def build_all_losses(config: Dict) -> Dict[str, tf.keras.losses.Loss]:
    """
    构建所有损失函数

    参数:
        config: losses配置

    返回:
        dict: 损失函数字典
    """
    loss_dict = {}

    for loss_name, loss_config in config.items():
        logger.info(f"构建损失函数: {loss_name}")

        loss_fn = call_target(
            loss_config["reflection"],
            loss_config.get("args", {})
        )

        loss_dict[loss_name] = loss_fn
        logger.debug(f"  类型: {type(loss_fn).__name__}")

    return loss_dict


# ======================================================
# 自定义损失函数：对比学习
# ======================================================
class ContrastiveLoss(tf.keras.losses.Loss):
    """
    NT-Xent对比损失（用于SimCLR等自监督学习）

    参数:
        temperature: 温度参数
        reduction: 损失聚合方式
    """
    def __init__(self, temperature: float = 0.07, reduction: str = "mean", name: str = "contrastive_loss"):
        super().__init__(reduction=reduction, name=name)
        self.temperature = temperature

    def call(self, z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
        """
        计算对比损失

        参数:
            z1: 第一个视图的特征 [batch_size, embedding_dim]
            z2: 第二个视图的特征 [batch_size, embedding_dim]

        返回:
            损失值
        """
        batch_size = tf.shape(z1)[0]

        # L2归一化
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        # 拼接所有样本
        z = tf.concat([z1, z2], axis=0)

        # 计算相似度矩阵
        similarity_matrix = tf.matmul(z, z, transpose_b=True) / self.temperature

        # 创建标签：对角线偏移batch_size的位置为正样本
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)

        # 排除自身相似度
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -1e9, similarity_matrix)

        # 计算交叉熵损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=similarity_matrix
        )

        return tf.reduce_mean(loss)


# ======================================================
# 多任务损失聚合
# ======================================================
class MultiTaskLoss(tf.keras.losses.Loss):
    """
    多任务学习损失聚合
    使用固定权重或不确定性加权

    参数:
        task_weights: 任务权重字典 {"task1": 0.5, "task2": 0.3, ...}
        uncertainty_weighted: 是否使用不确定性加权
    """
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        uncertainty_weighted: bool = False,
        name: str = "multi_task_loss"
    ):
        super().__init__(name=name)
        self.task_weights = task_weights or {}
        self.uncertainty_weighted = uncertainty_weighted

        if uncertainty_weighted:
            num_tasks = len(task_weights)
            self.log_vars = tf.Variable(
                tf.zeros(num_tasks),
                trainable=True,
                name="log_vars"
            )

    def call(self, task_losses: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        聚合多任务损失

        参数:
            task_losses: 任务损失字典 {"task1": loss1, "task2": loss2, ...}

        返回:
            加权总损失
        """
        if self.uncertainty_weighted:
            return self._uncertainty_weighted_loss(task_losses)
        else:
            return self._fixed_weighted_loss(task_losses)

    def _fixed_weighted_loss(self, task_losses: Dict[str, tf.Tensor]) -> tf.Tensor:
        """固定权重加权"""
        total_loss = 0.0
        for task_name, loss_value in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss_value
        return total_loss

    def _uncertainty_weighted_loss(self, task_losses: Dict[str, tf.Tensor]) -> tf.Tensor:
        """不确定性自动加权"""
        total_loss = 0.0
        for i, (task_name, loss_value) in enumerate(task_losses.items()):
            precision = tf.exp(-self.log_vars[i])
            total_loss += precision * loss_value + self.log_vars[i]
        return total_loss


def aggregate_multi_task_losses(
    losses: Dict[str, tf.Tensor],
    weights: Dict[str, float]
) -> Dict[str, tf.Tensor]:
    """
    聚合多任务损失（函数式接口）

    参数:
        losses: 任务损失字典
        weights: 任务权重字典

    返回:
        dict: {
            "total_loss": 总损失,
            "task1_loss": 任务1损失,
            ...
        }
    """
    total_loss = 0.0
    result = {}

    for task_name, loss_value in losses.items():
        weight = weights.get(task_name, 1.0)
        total_loss += weight * loss_value
        result[f"{task_name}_loss"] = loss_value

    result["total_loss"] = total_loss
    return result


# ======================================================
# 其他自定义损失函数
# ======================================================
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss - 用于类别不平衡

    参数:
        alpha: 类别权重
        gamma: 聚焦参数
        from_logits: 输入是否为logits
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        name: str = "focal_loss"
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """计算Focal Loss"""
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


class TripletLoss(tf.keras.losses.Loss):
    """
    三元组损失 - 用于度量学习

    参数:
        margin: 边界值
        distance_metric: 距离度量
    """
    def __init__(
        self,
        margin: float = 0.5,
        distance_metric: str = "euclidean",
        name: str = "triplet_loss"
    ):
        super().__init__(name=name)
        self.margin = margin
        self.distance_metric = distance_metric

    def call(self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor) -> tf.Tensor:
        """计算三元组损失"""
        if self.distance_metric == "euclidean":
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        elif self.distance_metric == "cosine":
            anchor_norm = tf.math.l2_normalize(anchor, axis=-1)
            positive_norm = tf.math.l2_normalize(positive, axis=-1)
            negative_norm = tf.math.l2_normalize(negative, axis=-1)

            pos_dist = 1 - tf.reduce_sum(anchor_norm * positive_norm, axis=-1)
            neg_dist = 1 - tf.reduce_sum(anchor_norm * negative_norm, axis=-1)
        else:
            raise ValueError(f"未知的距离度量: {self.distance_metric}")

        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)


# ======================================================
# 强化学习损失函数
# ======================================================
def compute_policy_gradient_loss(
    log_probs: tf.Tensor,
    advantages: tf.Tensor,
    entropy_coef: float = 0.01
) -> tf.Tensor:
    """
    策略梯度损失（REINFORCE）

    参数:
        log_probs: 动作的对数概率
        advantages: 优势值
        entropy_coef: 熵正则化系数

    返回:
        损失值
    """
    policy_loss = -tf.reduce_mean(log_probs * advantages)
    entropy = -tf.reduce_mean(tf.exp(log_probs) * log_probs)
    return policy_loss - entropy_coef * entropy


def compute_ppo_loss(
    log_probs: tf.Tensor,
    old_log_probs: tf.Tensor,
    advantages: tf.Tensor,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01
) -> tf.Tensor:
    """
    PPO（近端策略优化）损失

    参数:
        log_probs: 新策略的对数概率
        old_log_probs: 旧策略的对数概率
        advantages: 优势值
        clip_ratio: 裁剪比率
        entropy_coef: 熵正则化系数

    返回:
        损失值
    """
    ratio = tf.exp(log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)

    policy_loss = -tf.reduce_mean(
        tf.minimum(ratio * advantages, clipped_ratio * advantages)
    )

    entropy = -tf.reduce_mean(tf.exp(log_probs) * log_probs)
    return policy_loss - entropy_coef * entropy


def compute_value_loss(
    values: tf.Tensor,
    returns: tf.Tensor,
    value_coef: float = 0.5
) -> tf.Tensor:
    """
    价值函数损失

    参数:
        values: 预测的状态价值
        returns: 实际回报
        value_coef: 价值损失系数

    返回:
        损失值
    """
    return value_coef * tf.reduce_mean(tf.square(values - returns))
