# -*- coding: utf-8 -*-
"""
modules/optimizers.py
优化器模块：
- 构建优化器
- 学习率调度
- 优化器管理
"""

import tensorflow as tf
from typing import Dict, Any
from common.common import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主构建函数
# ======================================================
def build_all_optimizers(config: Dict) -> Dict[str, tf.keras.optimizers.Optimizer]:
    """
    构建所有优化器

    参数:
        config: optimizers配置

    返回:
        dict: 优化器字典 {optimizer_name: optimizer}

    示例:
        >>> config = {
        >>>     "adam": {
        >>>         "reflection": "tensorflow.keras.optimizers.Adam",
        >>>         "args": {"learning_rate": 0.001}
        >>>     }
        >>> }
        >>> opts = build_all_optimizers(config)
    """
    optimizer_dict = {}

    for optimizer_name, optimizer_config in config.items():
        logger.info(f"构建优化器: {optimizer_name}")

        try:
            optimizer = call_target(
                optimizer_config["reflection"],
                optimizer_config.get("args", {})
            )

            optimizer_dict[optimizer_name] = optimizer

            # 打印优化器信息
            logger.debug(f"  类型: {type(optimizer).__name__}")
            if hasattr(optimizer, 'learning_rate'):
                lr = optimizer.learning_rate
                if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                    logger.debug(f"  学习率: 动态调度")
                else:
                    logger.debug(f"  学习率: {float(lr.numpy())}")

        except Exception as e:
            logger.error(f"构建优化器 {optimizer_name} 失败: {str(e)}")
            raise

    return optimizer_dict


# ======================================================
# 学习率调度器
# ======================================================
class CustomLearningRateScheduler:
    """
    自定义学习率调度器
    支持多种调度策略

    参数:
        schedule_type: 调度类型 (step/exponential/cosine/warmup/polynomial)
        initial_lr: 初始学习率
        **kwargs: 其他参数

    示例:
        >>> scheduler = CustomLearningRateScheduler(
        >>>     schedule_type="cosine",
        >>>     initial_lr=0.001,
        >>>     total_steps=10000
        >>> )
        >>> lr = scheduler(step=1000)
    """
    def __init__(self, schedule_type: str, initial_lr: float, **kwargs):
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.kwargs = kwargs

        logger.info(f"创建学习率调度器: {schedule_type}")

    def __call__(self, step: int) -> float:
        """
        计算当前步的学习率

        参数:
            step: 当前训练步数

        返回:
            学习率
        """
        if self.schedule_type == "step":
            return self._step_decay(step)
        elif self.schedule_type == "exponential":
            return self._exponential_decay(step)
        elif self.schedule_type == "cosine":
            return self._cosine_decay(step)
        elif self.schedule_type == "warmup":
            return self._warmup_cosine_decay(step)
        elif self.schedule_type == "polynomial":
            return self._polynomial_decay(step)
        elif self.schedule_type == "constant":
            return self.initial_lr
        else:
            logger.warning(f"未知的调度类型: {self.schedule_type}，使用常数学习率")
            return self.initial_lr

    def _step_decay(self, step: int) -> float:
        """
        阶梯式衰减
        每隔一定步数，学习率乘以衰减率

        参数:
            step: 当前步数

        返回:
            学习率
        """
        drop_rate = self.kwargs.get("drop_rate", 0.5)
        drop_every = self.kwargs.get("drop_every", 1000)

        drops = step // drop_every
        return self.initial_lr * (drop_rate ** drops)

    def _exponential_decay(self, step: int) -> float:
        """
        指数衰减
        学习率按指数函数衰减

        参数:
            step: 当前步数

        返回:
            学习率
        """
        decay_rate = self.kwargs.get("decay_rate", 0.96)
        decay_steps = self.kwargs.get("decay_steps", 1000)
        staircase = self.kwargs.get("staircase", False)

        if staircase:
            return self.initial_lr * (decay_rate ** (step // decay_steps))
        else:
            return self.initial_lr * (decay_rate ** (step / decay_steps))

    def _cosine_decay(self, step: int) -> float:
        """
        余弦衰减
        使用余弦函数平滑衰减学习率

        参数:
            step: 当前步数

        返回:
            学习率
        """
        import math

        total_steps = self.kwargs.get("total_steps", 10000)
        min_lr = self.kwargs.get("min_lr", 0.0)

        progress = min(step / total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr + (self.initial_lr - min_lr) * cosine_decay

    def _warmup_cosine_decay(self, step: int) -> float:
        """
        预热 + 余弦衰减
        前期线性增长，后期余弦衰减

        参数:
            step: 当前步数

        返回:
            学习率
        """
        import math

        warmup_steps = self.kwargs.get("warmup_steps", 1000)
        total_steps = self.kwargs.get("total_steps", 10000)
        min_lr = self.kwargs.get("min_lr", 0.0)

        if step < warmup_steps:
            # 线性预热
            return self.initial_lr * (step / warmup_steps)
        else:
            # 余弦衰减
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (self.initial_lr - min_lr) * cosine_decay

    def _polynomial_decay(self, step: int) -> float:
        """
        多项式衰减
        学习率按多项式函数衰减

        参数:
            step: 当前步数

        返回:
            学习率
        """
        decay_steps = self.kwargs.get("decay_steps", 10000)
        end_lr = self.kwargs.get("end_lr", 0.0001)
        power = self.kwargs.get("power", 1.0)

        step = min(step, decay_steps)
        return (self.initial_lr - end_lr) * ((1 - step / decay_steps) ** power) + end_lr


def create_lr_schedule(
    schedule_type: str,
    initial_lr: float,
    **kwargs
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    创建TensorFlow原生学习率调度器

    参数:
        schedule_type: 调度类型
        initial_lr: 初始学习率
        **kwargs: 其他参数

    返回:
        学习率调度器

    示例:
        >>> schedule = create_lr_schedule(
        >>>     "exponential",
        >>>     initial_lr=0.001,
        >>>     decay_steps=1000,
        >>>     decay_rate=0.96
        >>> )
    """
    if schedule_type == "exponential":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=kwargs.get("decay_steps", 1000),
            decay_rate=kwargs.get("decay_rate", 0.96),
            staircase=kwargs.get("staircase", False)
        )

    elif schedule_type == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=kwargs.get("decay_steps", 10000),
            alpha=kwargs.get("alpha", 0.0)
        )

    elif schedule_type == "cosine_restart":
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=kwargs.get("first_decay_steps", 1000),
            t_mul=kwargs.get("t_mul", 2.0),
            m_mul=kwargs.get("m_mul", 1.0),
            alpha=kwargs.get("alpha", 0.0)
        )

    elif schedule_type == "piecewise":
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=kwargs.get("boundaries", [1000, 5000]),
            values=kwargs.get("values", [initial_lr, initial_lr * 0.1, initial_lr * 0.01])
        )

    elif schedule_type == "polynomial":
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=kwargs.get("decay_steps", 10000),
            end_learning_rate=kwargs.get("end_learning_rate", 0.0001),
            power=kwargs.get("power", 1.0)
        )

    elif schedule_type == "inverse_time":
        return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=kwargs.get("decay_steps", 1000),
            decay_rate=kwargs.get("decay_rate", 0.5)
        )

    else:
        raise ValueError(f"未知的学习率调度类型: {schedule_type}")


# ======================================================
# 优化器工具函数
# ======================================================
def get_optimizer_config(optimizer: tf.keras.optimizers.Optimizer) -> Dict:
    """
    获取优化器配置

    参数:
        optimizer: 优化器实例

    返回:
        配置字典

    示例:
        >>> config = get_optimizer_config(optimizer)
        >>> print(config)
    """
    return optimizer.get_config()


def set_learning_rate(optimizer: tf.keras.optimizers.Optimizer, lr: float) -> None:
    """
    设置优化器学习率

    参数:
        optimizer: 优化器实例
        lr: 新的学习率

    示例:
        >>> set_learning_rate(optimizer, 0.0001)
    """
    if hasattr(optimizer, 'learning_rate'):
        if isinstance(optimizer.learning_rate, tf.Variable):
            optimizer.learning_rate.assign(lr)
            logger.info(f"学习率更新为: {lr}")
        else:
            logger.warning("学习率是调度器，无法直接设置")
    else:
        logger.error("优化器没有learning_rate属性")


def get_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    """
    获取当前学习率

    参数:
        optimizer: 优化器实例

    返回:
        当前学习率

    示例:
        >>> lr = get_learning_rate(optimizer)
        >>> print(f"当前学习率: {lr}")
    """
    if hasattr(optimizer, 'learning_rate'):
        lr = optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # 如果是调度器，返回当前步的学习率
            return float(lr(optimizer.iterations).numpy())
        else:
            return float(lr.numpy())
    else:
        logger.error("优化器没有learning_rate属性")
        return 0.0


def apply_gradient_clipping(
    gradients: list,
    clip_type: str = "norm",
    clip_value: float = 1.0
) -> list:
    """
    应用梯度裁剪

    参数:
        gradients: 梯度列表
        clip_type: 裁剪类型 ("norm" 或 "value")
        clip_value: 裁剪值

    返回:
        裁剪后的梯度列表

    示例:
        >>> clipped_grads = apply_gradient_clipping(gradients, "norm", 1.0)
    """
    if clip_type == "norm":
        # 按范数裁剪
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        return clipped_gradients
    elif clip_type == "value":
        # 按值裁剪
        return [tf.clip_by_value(g, -clip_value, clip_value) for g in gradients]
    else:
        logger.warning(f"未知的裁剪类型: {clip_type}，不裁剪")
        return gradients


def compute_gradient_norm(gradients: list) -> float:
    """
    计算梯度的全局范数

    参数:
        gradients: 梯度列表

    返回:
        梯度范数

    示例:
        >>> grad_norm = compute_gradient_norm(gradients)
        >>> print(f"梯度范数: {grad_norm}")
    """
    return float(tf.linalg.global_norm(gradients).numpy())


# ======================================================
# 优化器预设配置
# ======================================================
def get_optimizer_preset(preset_name: str, learning_rate: float = 0.001) -> Dict:
    """
    获取优化器预设配置

    参数:
        preset_name: 预设名称 (adam/sgd/rmsprop/adamw/adagrad)
        learning_rate: 学习率

    返回:
        优化器配置字典

    示例:
        >>> config = get_optimizer_preset("adam", 0.001)
    """
    presets = {
        "adam": {
            "reflection": "tensorflow.keras.optimizers.Adam",
            "args": {
                "learning_rate": learning_rate,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7
            }
        },
        "sgd": {
            "reflection": "tensorflow.keras.optimizers.SGD",
            "args": {
                "learning_rate": learning_rate,
                "momentum": 0.9,
                "nesterov": True
            }
        },
        "rmsprop": {
            "reflection": "tensorflow.keras.optimizers.RMSprop",
            "args": {
                "learning_rate": learning_rate,
                "rho": 0.9,
                "momentum": 0.0,
                "epsilon": 1e-7
            }
        },
        "adamw": {
            "reflection": "tensorflow.keras.optimizers.AdamW",
            "args": {
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "beta_1": 0.9,
                "beta_2": 0.999
            }
        },
        "adagrad": {
            "reflection": "tensorflow.keras.optimizers.Adagrad",
            "args": {
                "learning_rate": learning_rate,
                "initial_accumulator_value": 0.1,
                "epsilon": 1e-7
            }
        }
    }

    if preset_name not in presets:
        raise ValueError(f"未知的预设: {preset_name}，可用: {list(presets.keys())}")

    return presets[preset_name]
