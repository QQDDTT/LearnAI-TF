# -*- coding: utf-8 -*-
"""
modules/losses.py
损失函数构建器：从配置上下文构建损失函数

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable
from common.train_context import (
    TrainContext,
    LossConfig
)
from common.common import call_target


class LossBuilder:
    """
    损失函数构建器

    职责：
    1. 读取 TrainContext 中的损失函数配置
    2. 使用 call_target 实例化损失函数
    3. 支持单个损失和组合损失
    4. 支持加权损失和自定义损失

    注意：所有损失函数通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化损失函数构建器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.loss_functions: Dict[str, Any] = {}

    def build_all_losses(self) -> Dict[str, Any]:
        """
        构建所有损失函数

        返回:
            损失函数字典 {loss_name: loss_instance}
        """
        if not self.context.losses:
            raise ValueError("TrainContext 中没有定义任何损失函数")

        losses = {}

        for loss_name, loss_config in self.context.losses.items():
            try:
                loss = self.build_loss(loss_name, loss_config)
                losses[loss_name] = loss
            except Exception as e:
                raise RuntimeError(f"构建损失函数 '{loss_name}' 失败: {e}")

        # 缓存到上下文
        self.context.instantiated_losses = losses

        return losses

    def build_loss(
        self,
        loss_name: str,
        loss_config: LossConfig
    ) -> Any:
        """
        构建单个损失函数

        参数:
            loss_name: 损失函数名称
            loss_config: 损失函数配置

        返回:
            损失函数实例
        """
        if not loss_config.reflection:
            raise ValueError(f"损失函数 '{loss_name}' 缺少 reflection 字段")

        # 准备参数
        args = loss_config.args.copy()

        # 添加名称（如果损失函数支持）
        if 'name' not in args:
            args['name'] = loss_name

        # 使用 call_target 实例化损失函数
        loss = call_target(
            reflection=loss_config.reflection,
            args=args
        )

        return loss

    def get_loss(self, loss_name: str) -> Any:
        """
        获取指定的损失函数

        参数:
            loss_name: 损失函数名称

        返回:
            损失函数实例
        """
        if loss_name not in self.loss_functions:
            raise ValueError(f"损失函数 '{loss_name}' 不存在")

        return self.loss_functions[loss_name]

    def create_weighted_loss(
        self,
        losses: List[str],
        weights: Optional[List[float]] = None
    ) -> Callable:
        """
        创建加权组合损失函数

        参数:
            losses: 损失函数名称列表
            weights: 权重列表（可选，默认均等权重）

        返回:
            组合损失函数
        """
        if not losses:
            raise ValueError("losses 列表不能为空")

        # 获取所有损失函数
        loss_instances = []
        for loss_name in losses:
            if loss_name not in self.loss_functions:
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_instances.append(self.loss_functions[loss_name])

        # 设置权重
        if weights is None:
            weights = [1.0] * len(losses)
        elif len(weights) != len(losses):
            raise ValueError(
                f"weights 长度 {len(weights)} 与 losses 长度 {len(losses)} 不匹配"
            )

        # 创建组合损失函数
        def combined_loss(y_true, y_pred):
            total_loss = 0.0
            for loss_fn, weight in zip(loss_instances, weights):
                loss_value = loss_fn(y_true, y_pred)
                total_loss += weight * loss_value
            return total_loss

        return combined_loss

    def create_multi_output_loss(
        self,
        output_losses: Dict[str, str],
        output_weights: Optional[Dict[str, float]] = None
    ) -> Callable:
        """
        创建多输出损失函数

        参数:
            output_losses: 输出到损失的映射 {output_name: loss_name}
            output_weights: 输出权重映射（可选）

        返回:
            多输出损失函数
        """
        if not output_losses:
            raise ValueError("output_losses 不能为空")

        # 获取所有损失函数
        loss_mapping = {}
        for output_name, loss_name in output_losses.items():
            if loss_name not in self.loss_functions:
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_mapping[output_name] = self.loss_functions[loss_name]

        # 设置权重
        if output_weights is None:
            output_weights = {name: 1.0 for name in output_losses.keys()}

        # 创建多输出损失函数
        def multi_output_loss(y_true, y_pred):
            """
            y_true: 字典 {output_name: true_values}
            y_pred: 字典 {output_name: pred_values}
            """
            total_loss = 0.0
            for output_name, loss_fn in loss_mapping.items():
                if output_name not in y_true or output_name not in y_pred:
                    raise ValueError(f"输出 '{output_name}' 不存在")

                loss_value = loss_fn(y_true[output_name], y_pred[output_name])
                weight = output_weights.get(output_name, 1.0)
                total_loss += weight * loss_value

            return total_loss

        return multi_output_loss

    def create_task_weighted_loss(
        self,
        task_losses: Dict[str, str],
        task_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = False
    ) -> Callable:
        """
        创建多任务加权损失函数

        参数:
            task_losses: 任务到损失的映射 {task_name: loss_name}
            task_weights: 任务权重（可选）
            adaptive_weighting: 是否使用自适应权重

        返回:
            多任务损失函数
        """
        if not task_losses:
            raise ValueError("task_losses 不能为空")

        # 获取所有损失函数
        loss_mapping = {}
        for task_name, loss_name in task_losses.items():
            if loss_name not in self.loss_functions:
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_mapping[task_name] = self.loss_functions[loss_name]

        # 初始化权重
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_losses.keys()}

        if adaptive_weighting:
            # 自适应权重（使用不确定性加权）
            # 创建可训练的权重参数
            log_vars = {}
            for task_name in task_losses.keys():
                # 通过 call_target 创建 Variable
                log_var = call_target(
                    reflection="tensorflow:Variable",
                    args={
                        "initial_value": 0.0,
                        "trainable": True,
                        "name": f"log_var_{task_name}"
                    }
                )
                log_vars[task_name] = log_var

            def adaptive_loss(y_true, y_pred):
                total_loss = 0.0
                for task_name, loss_fn in loss_mapping.items():
                    task_y_true = y_true[task_name]
                    task_y_pred = y_pred[task_name]

                    loss_value = loss_fn(task_y_true, task_y_pred)

                    # 使用不确定性加权：loss / (2 * sigma^2) + log(sigma)
                    # 其中 sigma^2 = exp(log_var)
                    log_var = log_vars[task_name]
                    precision = call_target(
                        reflection="tensorflow.math:exp",
                        args={"x": -log_var}
                    )
                    total_loss += precision * loss_value + 0.5 * log_var

                return total_loss

            return adaptive_loss

        else:
            # 固定权重
            def weighted_loss(y_true, y_pred):
                total_loss = 0.0
                for task_name, loss_fn in loss_mapping.items():
                    task_y_true = y_true[task_name]
                    task_y_pred = y_pred[task_name]

                    loss_value = loss_fn(task_y_true, task_y_pred)
                    weight = task_weights.get(task_name, 1.0)
                    total_loss += weight * loss_value

                return total_loss

            return weighted_loss

    def create_custom_loss_wrapper(
        self,
        loss_name: str,
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None
    ) -> Callable:
        """
        创建自定义损失包装器

        参数:
            loss_name: 基础损失函数名称
            preprocessing_fn: 预处理函数（应用于 y_true, y_pred）
            postprocessing_fn: 后处理函数（应用于损失值）

        返回:
            包装后的损失函数
        """
        if loss_name not in self.loss_functions:
            raise ValueError(f"损失函数 '{loss_name}' 不存在")

        base_loss = self.loss_functions[loss_name]

        def wrapped_loss(y_true, y_pred):
            # 预处理
            if preprocessing_fn is not None:
                y_true, y_pred = preprocessing_fn(y_true, y_pred)

            # 计算损失
            loss_value = base_loss(y_true, y_pred)

            # 后处理
            if postprocessing_fn is not None:
                loss_value = postprocessing_fn(loss_value)

            return loss_value

        return wrapped_loss

    def get_loss_for_mode(self, training_mode: Optional[str] = None) -> Any:
        """
        根据训练模式获取推荐的损失函数

        参数:
            training_mode: 训练模式（可选，默认使用上下文中的模式）

        返回:
            损失函数实例
        """
        mode = training_mode or self.context.training_mode

        # 根据训练模式推荐损失函数
        mode_loss_mapping = {
            "supervised": ["main_loss", "classification_loss", "regression_loss"],
            "unsupervised": ["reconstruction_loss", "contrastive_loss"],
            "reinforcement": ["policy_loss", "value_loss"],
            "self_supervised": ["contrastive_loss", "reconstruction_loss"],
            "semi_supervised": ["supervised_loss", "unsupervised_loss"],
            "adversarial": ["generator_loss", "discriminator_loss"]
        }

        # 获取推荐的损失函数名称
        recommended_names = mode_loss_mapping.get(mode, ["main_loss"])

        # 尝试获取第一个存在的损失函数
        for loss_name in recommended_names:
            if loss_name in self.loss_functions:
                return self.loss_functions[loss_name]

        # 如果没有推荐的，返回第一个可用的
        if self.loss_functions:
            return list(self.loss_functions.values())[0]

        raise ValueError(f"没有为训练模式 '{mode}' 找到合适的损失函数")

    def compute_loss(
        self,
        loss_name: str,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ) -> Any:
        """
        计算损失值

        参数:
            loss_name: 损失函数名称
            y_true: 真实值
            y_pred: 预测值
            sample_weight: 样本权重（可选）

        返回:
            损失值
        """
        loss_fn = self.get_loss(loss_name)

        if sample_weight is not None:
            # 使用 call_target 调用带权重的损失计算
            # 注意：不同损失函数对 sample_weight 的支持可能不同
            try:
                loss_value = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            except TypeError:
                # 如果不支持 sample_weight，手动应用
                loss_value = loss_fn(y_true, y_pred)
                loss_value = loss_value * sample_weight
        else:
            loss_value = loss_fn(y_true, y_pred)

        return loss_value

    def get_loss_info(self, loss_name: str) -> Dict[str, Any]:
        """
        获取损失函数信息

        参数:
            loss_name: 损失函数名称

        返回:
            损失函数信息字典
        """
        if loss_name not in self.loss_functions:
            raise ValueError(f"损失函数 '{loss_name}' 不存在")

        loss_fn = self.loss_functions[loss_name]

        info = {
            "name": loss_name,
            "type": type(loss_fn).__name__,
            "module": type(loss_fn).__module__
        }

        # 尝试获取额外信息
        if hasattr(loss_fn, 'reduction'):
            info["reduction"] = str(loss_fn.reduction)

        if hasattr(loss_fn, 'name'):
            info["loss_name"] = loss_fn.name

        return info

    def validate_loss_config(self) -> List[str]:
        """
        验证损失函数配置

        返回:
            警告信息列表
        """
        warnings = []

        # 检查是否定义了损失函数
        if not self.context.losses:
            warnings.append("未定义任何损失函数")
            return warnings

        # 检查每个损失函数的配置
        for loss_name, loss_config in self.context.losses.items():
            if not loss_config.reflection:
                warnings.append(f"损失函数 '{loss_name}' 缺少 reflection 字段")

            # 检查常见的参数
            args = loss_config.args

            # 检查 reduction 参数
            if 'reduction' in args:
                valid_reductions = ['auto', 'none', 'sum', 'sum_over_batch_size']
                if args['reduction'] not in valid_reductions:
                    warnings.append(
                        f"损失函数 '{loss_name}' 的 reduction 参数可能无效: "
                        f"{args['reduction']}"
                    )

        return warnings

    def create_focal_loss_wrapper(
        self,
        base_loss_name: str,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> Callable:
        """
        创建 Focal Loss 包装器（用于处理类别不平衡）

        参数:
            base_loss_name: 基础损失函数名称
            alpha: 平衡因子
            gamma: 聚焦参数

        返回:
            Focal Loss 函数
        """
        if base_loss_name not in self.loss_functions:
            raise ValueError(f"损失函数 '{base_loss_name}' 不存在")

        base_loss = self.loss_functions[base_loss_name]

        def focal_loss(y_true, y_pred):
            # 计算基础损失
            ce_loss = base_loss(y_true, y_pred)

            # 计算概率
            # 使用 call_target 调用 TensorFlow 函数
            y_pred_prob = call_target(
                reflection="tensorflow.nn:softmax",
                args={"logits": y_pred}
            )

            # 获取真实类别的概率
            y_true_one_hot = call_target(
                reflection="tensorflow:one_hot",
                args={
                    "indices": y_true,
                    "depth": call_target(
                        reflection="tensorflow:shape",
                        args={"input": y_pred}
                    )[-1]
                }
            )

            pt = call_target(
                reflection="tensorflow:reduce_sum",
                args={
                    "input_tensor": y_true_one_hot * y_pred_prob,
                    "axis": -1
                }
            )

            # Focal Loss: -alpha * (1 - pt)^gamma * log(pt)
            focal_weight = alpha * call_target(
                reflection="tensorflow.math:pow",
                args={
                    "x": 1.0 - pt,
                    "y": gamma
                }
            )

            focal_loss_value = focal_weight * ce_loss

            return call_target(
                reflection="tensorflow:reduce_mean",
                args={"input_tensor": focal_loss_value}
            )

        return focal_loss


def build_losses_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有损失函数的便捷函数

    参数:
        context: 训练上下文

    返回:
        损失函数字典
    """
    builder = LossBuilder(context)
    return builder.build_all_losses()


def build_single_loss(
    context: TrainContext,
    loss_name: str
) -> Any:
    """
    从训练上下文构建单个损失函数的便捷函数

    参数:
        context: 训练上下文
        loss_name: 损失函数名称

    返回:
        损失函数实例
    """
    if loss_name not in context.losses:
        raise ValueError(f"损失函数 '{loss_name}' 不存在于上下文中")

    builder = LossBuilder(context)
    loss_config = context.losses[loss_name]
    return builder.build_loss(loss_name, loss_config)


def create_loss_builder(context: TrainContext) -> LossBuilder:
    """
    创建并初始化损失函数构建器的便捷函数

    参数:
        context: 训练上下文

    返回:
        LossBuilder 实例
    """
    builder = LossBuilder(context)
    builder.build_all_losses()
    return builder
