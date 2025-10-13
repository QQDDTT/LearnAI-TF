# -*- coding: utf-8 -*-
"""
losses.py - 损失函数定义和生成
功能：
  - 通过反射调用从配置文件生成损失函数
  - 支持任意损失函数类型
  - 计算损失值
"""

from typing import Dict, Any
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class LossBuilder:
    """
    损失函数构建器：
    - 根据配置文件动态生成损失函数
    - 使用反射调用TensorFlow的Loss API
    - 支持任意损失函数类型
    """

    def build_all(self, losses_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建所有配置中的损失函数
        参数：
            losses_config: 配置文件中的losses部分
        返回：
            {损失函数名: 损失函数实例}
        """
        losses = {}

        for loss_name, loss_cfg in losses_config.items():
            logger.info(f"构建损失函数: {loss_name}")
            try:
                loss = self.build_single(loss_name, loss_cfg)
                losses[loss_name] = loss
                logger.info(f"损失函数 {loss_name} 构建成功")
            except Exception as e:
                logger.error(f"损失函数 {loss_name} 构建失败: {str(e)}", exc_info=True)
                raise

        return losses

    def build_single(self, loss_name: str, loss_cfg: Dict[str, Any]) -> Any:
        """
        构建单个损失函数

        参数：
            loss_name: 损失函数名称
            loss_cfg: 损失函数配置
                {
                    "reflection": "tensorflow.keras.losses.BinaryCrossentropy",
                    "args": {}
                }
        返回：
            损失函数实例
        """
        reflection = loss_cfg.get("reflection")
        args = loss_cfg.get("args", {})

        if not reflection:
            raise ValueError(f"损失函数 {loss_name} 缺少reflection字段")

        logger.debug(f"创建损失函数: {reflection}, 参数: {args}")

        # 通过反射调用损失函数的构造函数
        loss = call_target(reflection, args)

        return loss


class LossComputer:
    """
    损失计算器：
    - 计算预测值和目标值之间的损失
    - 支持多种损失函数
    - 计算总损失（加权求和）
    """

    def __init__(self):
        pass

    def compute_loss(self,
                    loss_fn: Any,
                    predictions: Any,
                    targets: Any) -> float:
        """
        计算单个损失
        参数：
            loss_fn: 损失函数实例
            predictions: 预测值
            targets: 目标值
        返回：
            损失值
        """
        try:
            # 直接调用损失函数
            loss = loss_fn(targets, predictions)

            # 如果是tensor，转换为标量
            if hasattr(loss, 'numpy'):
                loss = float(loss.numpy())
            else:
                loss = float(loss)

            return loss
        except Exception as e:
            logger.error(f"计算损失失败: {str(e)}", exc_info=True)
            raise

    def compute_multi_task_loss(self,
                               predictions: Dict[str, Any],
                               targets: Dict[str, Any],
                               loss_fns: Dict[str, Any],
                               weights: Dict[str, float] = None) -> tuple:
        """
        计算多任务的加权损失
        参数：
            predictions: {任务名: 预测值}
            targets: {任务名: 目标值}
            loss_fns: {任务名: 损失函数}
            weights: {任务名: 权重}
        返回：
            (总损失, {任务名: 单个损失})
        """
        if weights is None:
            weights = {task: 1.0 for task in predictions.keys()}

        task_losses = {}
        total_loss = 0.0
        weight_sum = sum(weights.values())

        for task_name in predictions.keys():
            pred = predictions[task_name]
            target = targets.get(task_name)
            loss_fn = loss_fns.get(task_name)
            weight = weights.get(task_name, 1.0)

            if target is None or loss_fn is None:
                logger.warning(f"任务 {task_name} 缺少目标值或损失函数")
                continue

            try:
                task_loss = self.compute_loss(loss_fn, pred, target)
                task_losses[task_name] = task_loss
                total_loss += task_loss * weight

                logger.debug(f"任务 {task_name} 损失: {task_loss:.6f} (权重: {weight})")
            except Exception as e:
                logger.error(f"任务 {task_name} 损失计算失败: {str(e)}")
                raise

        # 归一化：总损失 = 加权损失 / 权重和
        if weight_sum > 0:
            total_loss = total_loss / weight_sum

        logger.debug(f"总损失: {total_loss:.6f}")

        return total_loss, task_losses

    def compute_gradient(self,
                        model: Any,
                        loss: float) -> Any:
        """
        计算梯度
        参数：
            model: 模型实例
            loss: 损失值
        返回：
            梯度
        """
        # 使用TensorFlow的梯度计算
        # 需要在tf.GradientTape上下文中执行
        try:
            # 这是一个占位符，实际梯度计算在training_pipeline中进行
            logger.debug("梯度计算")
            return loss
        except Exception as e:
            logger.error(f"梯度计算失败: {str(e)}", exc_info=True)
            raise


class RegularizationLoss:
    """
    正则化损失：
    - L1正则化
    - L2正则化
    - 用于防止过拟合
    """

    @staticmethod
    def l1_loss(weights: Any, lambda_l1: float = 0.0001) -> float:
        """
        计算L1正则化损失
        参数：
            weights: 模型权重
            lambda_l1: L1正则化系数
        返回：
            L1损失
        """
        import tensorflow as tf
        l1_loss = lambda_l1 * tf.reduce_sum(tf.abs(weights))
        if hasattr(l1_loss, 'numpy'):
            return float(l1_loss.numpy())
        return float(l1_loss)

    @staticmethod
    def l2_loss(weights: Any, lambda_l2: float = 0.0001) -> float:
        """
        计算L2正则化损失（权重衰减）
        参数：
            weights: 模型权重
            lambda_l2: L2正则化系数
        返回：
            L2损失
        """
        import tensorflow as tf
        l2_loss = lambda_l2 * tf.reduce_sum(tf.square(weights))
        if hasattr(l2_loss, 'numpy'):
            return float(l2_loss.numpy())
        return float(l2_loss)
