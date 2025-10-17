# -*- coding: utf-8 -*-
"""
modules/optimizers.py
优化器构建器：从配置上下文构建优化器

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from common.train_context import (
    TrainContext,
    OptimizerConfig
)
from common.common import call_target


class OptimizerBuilder:
    """
    优化器构建器

    职责：
    1. 读取 TrainContext 中的优化器配置
    2. 使用 call_target 实例化优化器
    3. 支持学习率调度
    4. 支持梯度裁剪和累积
    5. 支持多优化器策略

    注意：所有优化器通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化优化器构建器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.optimizers: Dict[str, Any] = {}
        self.lr_schedules: Dict[str, Any] = {}

    def build_all_optimizers(self) -> Dict[str, Any]:
        """
        构建所有优化器

        返回:
            优化器字典 {optimizer_name: optimizer_instance}
        """
        if not self.context.optimizers:
            raise ValueError("TrainContext 中没有定义任何优化器")

        optimizers = {}

        for opt_name, opt_config in self.context.optimizers.items():
            try:
                optimizer = self.build_optimizer(opt_name, opt_config)
                optimizers[opt_name] = optimizer
            except Exception as e:
                raise RuntimeError(f"构建优化器 '{opt_name}' 失败: {e}")

        # 缓存到上下文
        self.context.instantiated_optimizers = optimizers

        return optimizers

    def build_optimizer(
        self,
        opt_name: str,
        opt_config: OptimizerConfig
    ) -> Any:
        """
        构建单个优化器

        参数:
            opt_name: 优化器名称
            opt_config: 优化器配置

        返回:
            优化器实例
        """
        if not opt_config.reflection:
            raise ValueError(f"优化器 '{opt_name}' 缺少 reflection 字段")

        # 准备参数
        args = opt_config.args.copy()

        # 处理学习率调度
        if 'learning_rate' in args:
            lr = args['learning_rate']

            # 如果学习率是字典，表示需要创建调度器
            if isinstance(lr, dict):
                lr_schedule = self._build_lr_schedule(opt_name, lr)
                args['learning_rate'] = lr_schedule
                self.lr_schedules[opt_name] = lr_schedule

        # 添加名称（如果优化器支持）
        if 'name' not in args:
            args['name'] = opt_name

        # 使用 call_target 实例化优化器
        optimizer = call_target(
            reflection=opt_config.reflection,
            args=args
        )

        return optimizer

    def _build_lr_schedule(
        self,
        opt_name: str,
        lr_config: Dict[str, Any]
    ) -> Any:
        """
        构建学习率调度器

        参数:
            opt_name: 优化器名称
            lr_config: 学习率配置

        返回:
            学习率调度器实例
        """
        if 'type' not in lr_config:
            raise ValueError(f"学习率配置缺少 'type' 字段")

        schedule_type = lr_config['type']
        schedule_args = lr_config.get('args', {})

        # 根据类型构建不同的调度器
        if schedule_type == "exponential_decay":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:ExponentialDecay",
                args=schedule_args
            )

        elif schedule_type == "polynomial_decay":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:PolynomialDecay",
                args=schedule_args
            )

        elif schedule_type == "piecewise_constant":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:PiecewiseConstantDecay",
                args=schedule_args
            )

        elif schedule_type == "cosine_decay":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:CosineDecay",
                args=schedule_args
            )

        elif schedule_type == "cosine_decay_restarts":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:CosineDecayRestarts",
                args=schedule_args
            )

        elif schedule_type == "inverse_time_decay":
            schedule = call_target(
                reflection="tensorflow.keras.optimizers.schedules:InverseTimeDecay",
                args=schedule_args
            )

        elif schedule_type == "custom":
            # 自定义调度器（通过 reflection）
            if 'reflection' not in lr_config:
                raise ValueError("自定义学习率调度器需要 'reflection' 字段")

            schedule = call_target(
                reflection=lr_config['reflection'],
                args=schedule_args
            )

        else:
            raise ValueError(f"不支持的学习率调度类型: {schedule_type}")

        return schedule

    def get_optimizer(self, opt_name: str) -> Any:
        """
        获取指定的优化器

        参数:
            opt_name: 优化器名称

        返回:
            优化器实例
        """
        if opt_name not in self.optimizers:
            raise ValueError(f"优化器 '{opt_name}' 不存在")

        return self.optimizers[opt_name]

    def get_learning_rate(self, opt_name: str) -> float:
        """
        获取当前学习率

        参数:
            opt_name: 优化器名称

        返回:
            当前学习率
        """
        optimizer = self.get_optimizer(opt_name)

        lr = optimizer.learning_rate

        # 如果是调度器，获取当前值
        if callable(lr):
            current_step = optimizer.iterations
            lr_value = lr(current_step)
        else:
            lr_value = lr

        # 转换为 Python float
        if hasattr(lr_value, 'numpy'):
            return float(lr_value.numpy())
        return float(lr_value)

    def set_learning_rate(self, opt_name: str, learning_rate: float):
        """
        设置学习率

        参数:
            opt_name: 优化器名称
            learning_rate: 新的学习率
        """
        optimizer = self.get_optimizer(opt_name)
        optimizer.learning_rate.assign(learning_rate)

    def create_gradient_clip_optimizer(
        self,
        base_opt_name: str,
        clip_norm: Optional[float] = None,
        clip_value: Optional[float] = None
    ) -> Any:
        """
        创建带梯度裁剪的优化器包装

        参数:
            base_opt_name: 基础优化器名称
            clip_norm: 梯度范数裁剪阈值
            clip_value: 梯度值裁剪阈值

        返回:
            包装后的优化器
        """
        if base_opt_name not in self.optimizers:
            raise ValueError(f"优化器 '{base_opt_name}' 不存在")

        base_optimizer = self.optimizers[base_opt_name]

        # 创建包装优化器
        class GradientClipOptimizer:
            def __init__(self, optimizer, clip_norm, clip_value):
                self.optimizer = optimizer
                self.clip_norm = clip_norm
                self.clip_value = clip_value

            def apply_gradients(self, grads_and_vars):
                # 提取梯度和变量
                gradients = [g for g, v in grads_and_vars]
                variables = [v for g, v in grads_and_vars]

                # 裁剪梯度
                if self.clip_norm is not None:
                    clipped_grads, _ = call_target(
                        reflection="tensorflow:clip_by_global_norm",
                        args={
                            "t_list": gradients,
                            "clip_norm": self.clip_norm
                        }
                    )
                elif self.clip_value is not None:
                    clipped_grads = [
                        call_target(
                            reflection="tensorflow:clip_by_value",
                            args={
                                "t": g,
                                "clip_value_min": -self.clip_value,
                                "clip_value_max": self.clip_value
                            }
                        ) if g is not None else None
                        for g in gradients
                    ]
                else:
                    clipped_grads = gradients

                # 应用梯度
                return self.optimizer.apply_gradients(
                    zip(clipped_grads, variables)
                )

            def __getattr__(self, name):
                # 代理其他属性和方法
                return getattr(self.optimizer, name)

        return GradientClipOptimizer(base_optimizer, clip_norm, clip_value)

    def create_gradient_accumulation_optimizer(
        self,
        base_opt_name: str,
        accumulation_steps: int = 4
    ) -> Any:
        """
        创建梯度累积优化器

        参数:
            base_opt_name: 基础优化器名称
            accumulation_steps: 累积步数

        返回:
            梯度累积优化器
        """
        if base_opt_name not in self.optimizers:
            raise ValueError(f"优化器 '{base_opt_name}' 不存在")

        base_optimizer = self.optimizers[base_opt_name]

        # 创建梯度累积优化器
        class GradientAccumulationOptimizer:
            def __init__(self, optimizer, accumulation_steps):
                self.optimizer = optimizer
                self.accumulation_steps = accumulation_steps
                self.gradient_accumulations = []
                self.step_counter = call_target(
                    reflection="tensorflow:Variable",
                    args={
                        "initial_value": 0,
                        "trainable": False,
                        "dtype": call_target(
                            reflection="tensorflow:int32",
                            args={}
                        )
                    }
                )

            def apply_gradients(self, grads_and_vars):
                # 累积梯度
                if not self.gradient_accumulations:
                    # 第一次，初始化累积器
                    for g, v in grads_and_vars:
                        if g is not None:
                            acc = call_target(
                                reflection="tensorflow:Variable",
                                args={
                                    "initial_value": call_target(
                                        reflection="tensorflow:zeros_like",
                                        args={"tensor": g}
                                    ),
                                    "trainable": False
                                }
                            )
                        else:
                            acc = None
                        self.gradient_accumulations.append(acc)

                # 累加梯度
                for i, (g, v) in enumerate(grads_and_vars):
                    if g is not None and self.gradient_accumulations[i] is not None:
                        self.gradient_accumulations[i].assign_add(g)

                # 增加计数器
                self.step_counter.assign_add(1)

                # 检查是否需要更新
                should_update = call_target(
                    reflection="tensorflow:equal",
                    args={
                        "x": call_target(
                            reflection="tensorflow:math.floormod",
                            args={
                                "x": self.step_counter,
                                "y": self.accumulation_steps
                            }
                        ),
                        "y": 0
                    }
                )

                def update_weights():
                    # 平均梯度
                    averaged_grads = [
                        acc / self.accumulation_steps if acc is not None else None
                        for acc in self.gradient_accumulations
                    ]

                    # 应用梯度
                    variables = [v for g, v in grads_and_vars]
                    result = self.optimizer.apply_gradients(
                        zip(averaged_grads, variables)
                    )

                    # 重置累积器
                    for acc in self.gradient_accumulations:
                        if acc is not None:
                            acc.assign(
                                call_target(
                                    reflection="tensorflow:zeros_like",
                                    args={"tensor": acc}
                                )
                            )

                    return result

                def no_op():
                    return call_target(
                        reflection="tensorflow:no_op",
                        args={}
                    )

                return call_target(
                    reflection="tensorflow:cond",
                    args={
                        "pred": should_update,
                        "true_fn": update_weights,
                        "false_fn": no_op
                    }
                )

            def __getattr__(self, name):
                return getattr(self.optimizer, name)

        return GradientAccumulationOptimizer(base_optimizer, accumulation_steps)

    def create_multi_optimizer_strategy(
        self,
        optimizer_mapping: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        创建多优化器策略（不同参数使用不同优化器）

        参数:
            optimizer_mapping: 优化器到变量名列表的映射
                {optimizer_name: [variable_name_pattern, ...]}

        返回:
            多优化器策略字典
        """
        strategy = {}

        for opt_name, var_patterns in optimizer_mapping.items():
            if opt_name not in self.optimizers:
                raise ValueError(f"优化器 '{opt_name}' 不存在")

            strategy[opt_name] = {
                "optimizer": self.optimizers[opt_name],
                "patterns": var_patterns
            }

        return strategy

    def apply_multi_optimizer_gradients(
        self,
        strategy: Dict[str, Any],
        grads_and_vars: List[Tuple[Any, Any]]
    ):
        """
        应用多优化器梯度更新

        参数:
            strategy: 多优化器策略
            grads_and_vars: 梯度和变量对列表
        """
        # 按优化器分组梯度和变量
        optimizer_groups = {opt_name: [] for opt_name in strategy.keys()}

        for grad, var in grads_and_vars:
            if grad is None:
                continue

            var_name = var.name
            matched = False

            # 匹配变量到优化器
            for opt_name, config in strategy.items():
                for pattern in config["patterns"]:
                    if pattern in var_name:
                        optimizer_groups[opt_name].append((grad, var))
                        matched = True
                        break
                if matched:
                    break

        # 应用各个优化器
        for opt_name, grads_vars in optimizer_groups.items():
            if grads_vars:
                optimizer = strategy[opt_name]["optimizer"]
                optimizer.apply_gradients(grads_vars)

    def get_optimizer_for_mode(
        self,
        training_mode: Optional[str] = None
    ) -> Any:
        """
        根据训练模式获取推荐的优化器

        参数:
            training_mode: 训练模式（可选）

        返回:
            优化器实例
        """
        mode = training_mode or self.context.training_mode

        # 根据训练模式推荐优化器
        mode_optimizer_mapping = {
            "supervised": ["main_optimizer", "adam", "sgd"],
            "unsupervised": ["adam", "main_optimizer"],
            "reinforcement": ["policy_optimizer", "value_optimizer", "adam"],
            "self_supervised": ["adam", "lars", "main_optimizer"],
            "adversarial": ["generator_optimizer", "discriminator_optimizer"]
        }

        # 获取推荐的优化器名称
        recommended_names = mode_optimizer_mapping.get(mode, ["main_optimizer"])

        # 尝试获取第一个存在的优化器
        for opt_name in recommended_names:
            if opt_name in self.optimizers:
                return self.optimizers[opt_name]

        # 如果没有推荐的，返回第一个可用的
        if self.optimizers:
            return list(self.optimizers.values())[0]

        raise ValueError(f"没有为训练模式 '{mode}' 找到合适的优化器")

    def get_optimizer_info(self, opt_name: str) -> Dict[str, Any]:
        """
        获取优化器信息

        参数:
            opt_name: 优化器名称

        返回:
            优化器信息字典
        """
        if opt_name not in self.optimizers:
            raise ValueError(f"优化器 '{opt_name}' 不存在")

        optimizer = self.optimizers[opt_name]

        info = {
            "name": opt_name,
            "type": type(optimizer).__name__,
            "module": type(optimizer).__module__,
            "learning_rate": self.get_learning_rate(opt_name)
        }

        # 尝试获取其他配置参数
        if hasattr(optimizer, 'beta_1'):
            info["beta_1"] = float(optimizer.beta_1)
        if hasattr(optimizer, 'beta_2'):
            info["beta_2"] = float(optimizer.beta_2)
        if hasattr(optimizer, 'epsilon'):
            info["epsilon"] = float(optimizer.epsilon)
        if hasattr(optimizer, 'momentum'):
            info["momentum"] = float(optimizer.momentum)
        if hasattr(optimizer, 'weight_decay'):
            info["weight_decay"] = float(optimizer.weight_decay)

        # 获取迭代次数
        if hasattr(optimizer, 'iterations'):
            info["iterations"] = int(optimizer.iterations.numpy())

        return info

    def create_warmup_schedule(
        self,
        initial_lr: float,
        peak_lr: float,
        warmup_steps: int,
        decay_schedule: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        创建学习率预热调度器

        参数:
            initial_lr: 初始学习率
            peak_lr: 峰值学习率
            warmup_steps: 预热步数
            decay_schedule: 衰减调度配置（可选）

        返回:
            预热调度器
        """
        # 创建预热调度器
        if decay_schedule is None:
            # 仅预热，之后保持峰值
            class WarmupSchedule:
                def __init__(self, initial_lr, peak_lr, warmup_steps):
                    self.initial_lr = initial_lr
                    self.peak_lr = peak_lr
                    self.warmup_steps = warmup_steps

                def __call__(self, step):
                    step = call_target(
                        reflection="tensorflow:cast",
                        args={"x": step, "dtype": call_target(
                            reflection="tensorflow:float32", args={}
                        )}
                    )
                    warmup_steps_float = call_target(
                        reflection="tensorflow:cast",
                        args={"x": self.warmup_steps, "dtype": call_target(
                            reflection="tensorflow:float32", args={}
                        )}
                    )

                    warmup_lr = self.initial_lr + (
                        (self.peak_lr - self.initial_lr) *
                        (step / warmup_steps_float)
                    )

                    return call_target(
                        reflection="tensorflow:minimum",
                        args={"x": warmup_lr, "y": self.peak_lr}
                    )

            return WarmupSchedule(initial_lr, peak_lr, warmup_steps)

        else:
            # 预热 + 衰减
            decay_schedule_obj = self._build_lr_schedule(
                "warmup_decay",
                decay_schedule
            )

            class WarmupDecaySchedule:
                def __init__(self, initial_lr, peak_lr, warmup_steps, decay_schedule):
                    self.initial_lr = initial_lr
                    self.peak_lr = peak_lr
                    self.warmup_steps = warmup_steps
                    self.decay_schedule = decay_schedule

                def __call__(self, step):
                    # 预热阶段
                    step_float = call_target(
                        reflection="tensorflow:cast",
                        args={"x": step, "dtype": call_target(
                            reflection="tensorflow:float32", args={}
                        )}
                    )
                    warmup_steps_float = call_target(
                        reflection="tensorflow:cast",
                        args={"x": self.warmup_steps, "dtype": call_target(
                            reflection="tensorflow:float32", args={}
                        )}
                    )

                    warmup_lr = self.initial_lr + (
                        (self.peak_lr - self.initial_lr) *
                        (step_float / warmup_steps_float)
                    )

                    # 衰减阶段
                    decay_step = call_target(
                        reflection="tensorflow:maximum",
                        args={"x": step - self.warmup_steps, "y": 0}
                    )
                    decay_lr = self.decay_schedule(decay_step)

                    # 选择阶段
                    is_warmup = call_target(
                        reflection="tensorflow:less",
                        args={"x": step, "y": self.warmup_steps}
                    )

                    return call_target(
                        reflection="tensorflow:cond",
                        args={
                            "pred": is_warmup,
                            "true_fn": lambda: warmup_lr,
                            "false_fn": lambda: decay_lr
                        }
                    )

            return WarmupDecaySchedule(
                initial_lr, peak_lr, warmup_steps, decay_schedule_obj
            )

    def validate_optimizer_config(self) -> List[str]:
        """
        验证优化器配置

        返回:
            警告信息列表
        """
        warnings = []

        # 检查是否定义了优化器
        if not self.context.optimizers:
            warnings.append("未定义任何优化器")
            return warnings

        # 检查每个优化器的配置
        for opt_name, opt_config in self.context.optimizers.items():
            if not opt_config.reflection:
                warnings.append(f"优化器 '{opt_name}' 缺少 reflection 字段")

            args = opt_config.args

            # 检查学习率
            if 'learning_rate' not in args:
                warnings.append(f"优化器 '{opt_name}' 未指定学习率")
            else:
                lr = args['learning_rate']
                if isinstance(lr, (int, float)) and lr <= 0:
                    warnings.append(
                        f"优化器 '{opt_name}' 的学习率应大于 0: {lr}"
                    )

            # 检查 Adam 优化器的参数
            if 'Adam' in opt_config.reflection:
                if 'beta_1' in args and not (0 < args['beta_1'] < 1):
                    warnings.append(
                        f"优化器 '{opt_name}' 的 beta_1 应在 (0, 1) 范围内"
                    )
                if 'beta_2' in args and not (0 < args['beta_2'] < 1):
                    warnings.append(
                        f"优化器 '{opt_name}' 的 beta_2 应在 (0, 1) 范围内"
                    )

        return warnings


def build_optimizers_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有优化器的便捷函数

    参数:
        context: 训练上下文

    返回:
        优化器字典
    """
    builder = OptimizerBuilder(context)
    return builder.build_all_optimizers()


def build_single_optimizer(
    context: TrainContext,
    opt_name: str
) -> Any:
    """
    从训练上下文构建单个优化器的便捷函数

    参数:
        context: 训练上下文
        opt_name: 优化器名称

    返回:
        优化器实例
    """
    if opt_name not in context.optimizers:
        raise ValueError(f"优化器 '{opt_name}' 不存在于上下文中")

    builder = OptimizerBuilder(context)
    opt_config = context.optimizers[opt_name]
    return builder.build_optimizer(opt_name, opt_config)


def create_optimizer_builder(context: TrainContext) -> OptimizerBuilder:
    """
    创建并初始化优化器构建器的便捷函数

    参数:
        context: 训练上下文

    返回:
        OptimizerBuilder 实例
    """
    builder = OptimizerBuilder(context)
    builder.build_all_optimizers()
    return builder
