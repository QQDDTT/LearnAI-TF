# -*- coding: utf-8 -*-
"""
modules/losses.py
损失函数管理器：从配置上下文构建和管理损失函数

设计原则：
1. 支持 TensorFlow 内置损失函数
2. 支持自定义损失函数
3. 支持多任务损失组合
4. 使用 call_target 动态创建

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用 TensorFlow API
"""

from typing import Dict, Any, List, Optional, Callable
from common.train_context import TrainContext
from common.common import call_target
from common.interfaces import LossManagerInterface


class LossManager(LossManagerInterface):
    """
    损失函数管理器

    职责：
    1. 从 TrainContext.losses 读取配置
    2. 使用 call_target 构建 TensorFlow 损失函数
    3. 支持多任务损失组合
    4. 支持加权损失

    支持简化配置：
    - 所有参数可选（使用默认值）
    """

    def __init__(self, context: TrainContext):
        """初始化损失函数管理器"""
        super().__init__(context)
        self.losses: Dict[str, Any] = {}
        self.loss_configs: Dict[str, Any] = {}

    def initialize(self) -> None:
        """从 TrainContext 初始化损失函数管理器"""
        if not self.context.losses:
            raise ValueError("TrainContext 中没有定义任何损失函数")

        # 加载损失函数配置
        for loss_name, loss_config in self.context.losses.items():
            self.loss_configs[loss_name] = loss_config

        self._initialized = True

    def validate_losses(self) -> bool:
        """
        验证损失函数配置

        返回:
            配置是否有效
        """
        for loss_name, loss_config in self.loss_configs.items():
            if not isinstance(loss_config, dict):
                raise ValueError(f"损失函数 '{loss_name}' 配置格式错误")

            # 检查必需字段
            if 'reflection' not in loss_config:
                raise ValueError(
                    f"损失函数 '{loss_name}' 缺少 reflection 字段"
                )

        return True

    # ========================================================================
    # 损失函数构建
    # ========================================================================

    def build_loss(
        self,
        loss_name: str,
        loss_config: Any
    ) -> Any:
        """
        构建单个损失函数

        参数:
            loss_name: 损失函数名称
            loss_config: 损失函数配置

        返回:
            损失函数实例
        """
        # 转换为字典格式
        if not isinstance(loss_config, dict):
            config = {
                'reflection': getattr(loss_config, 'reflection', ''),
                'args': getattr(loss_config, 'args', {})
            }
        else:
            config = loss_config

        reflection = config.get('reflection')
        args = config.get('args', {}).copy()

        if not reflection:
            raise ValueError(f"损失函数 '{loss_name}' 缺少 reflection")

        # 添加名称（如果损失函数支持）
        if 'name' not in args and self._supports_name_param(reflection):
            args['name'] = loss_name

        # 创建损失函数
        loss = call_target(reflection=reflection, args=args)

        return loss

    def _supports_name_param(self, reflection: str) -> bool:
        """
        检查损失函数是否支持 name 参数

        参数:
            reflection: 反射路径

        返回:
            是否支持
        """
        # TensorFlow 内置损失函数通常支持 name 参数
        if 'tensorflow.keras.losses' in reflection:
            return True

        return False

    def build_all_losses(self) -> Dict[str, Any]:
        """
        构建所有损失函数

        返回:
            损失函数字典 {loss_name: loss}
        """
        for loss_name, loss_config in self.loss_configs.items():
            try:
                loss = self.build_loss(loss_name, loss_config)
                self.losses[loss_name] = loss
            except Exception as e:
                raise RuntimeError(
                    f"构建损失函数 '{loss_name}' 失败: {e}"
                ) from e

        return self.losses

    # ========================================================================
    # 多任务损失组合
    # ========================================================================

    def setup_aggregators(self) -> Dict[str, Any]:
        """
        设置多任务损失聚合器（可选）

        返回:
            聚合器字典
        """
        # 可以在这里设置预定义的损失聚合器
        return {}

    def create_weighted_loss(
        self,
        loss_names: List[str],
        weights: Optional[List[float]] = None
    ) -> Callable:
        """
        创建加权组合损失

        参数:
            loss_names: 损失函数名称列表
            weights: 权重列表（可选，默认均等权重）

        返回:
            组合损失函数
        """
        if not loss_names:
            raise ValueError("loss_names 不能为空")

        # 获取所有损失函数
        loss_instances = []
        for loss_name in loss_names:
            if loss_name not in self.losses:
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_instances.append(self.losses[loss_name])

        # 设置权重
        if weights is None:
            weights = [1.0] * len(loss_names)
        elif len(weights) != len(loss_names):
            raise ValueError(
                f"weights 长度与 loss_names 长度不匹配"
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
            if loss_name not in self.losses:
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_mapping[output_name] = self.losses[loss_name]

        # 设置权重
        if output_weights is None:
            output_weights = {name: 1.0 for name in output_losses.keys()}

        # 创建多输出损失函数
        def multi_output_loss(y_true, y_pred):
            """
            y_true: 字典 {output_name: true_values} 或元组
            y_pred: 字典 {output_name: pred_values} 或元组
            """
            total_loss = 0.0

            # 如果是字典格式
            if isinstance(y_true, dict) and isinstance(y_pred, dict):
                for output_name, loss_fn in loss_mapping.items():
                    if output_name not in y_true or output_name not in y_pred:
                        raise ValueError(f"输出 '{output_name}' 不存在")

                    loss_value = loss_fn(y_true[output_name], y_pred[output_name])
                    weight = output_weights.get(output_name, 1.0)
                    total_loss += weight * loss_value

            # 如果是元组格式
            elif isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
                output_names = list(output_losses.keys())
                for i, output_name in enumerate(output_names):
                    loss_fn = loss_mapping[output_name]
                    loss_value = loss_fn(y_true[i], y_pred[i])
                    weight = output_weights.get(output_name, 1.0)
                    total_loss += weight * loss_value

            else:
                raise ValueError("y_true 和 y_pred 格式不支持")

            return total_loss

        return multi_output_loss

    # ========================================================================
    # 接口实现
    # ========================================================================

    def execute(self) -> Dict[str, Any]:
        """执行损失函数构建"""
        return self.build_all_losses()

    def finalize(self) -> None:
        """将损失函数缓存到上下文"""
        self.context.container.losses = self.losses

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_loss(self, loss_name: str) -> Any:
        """
        获取损失函数实例

        参数:
            loss_name: 损失函数名称

        返回:
            损失函数实例
        """
        if loss_name not in self.losses:
            raise ValueError(f"损失函数 '{loss_name}' 不存在")

        return self.losses[loss_name]

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
            try:
                loss_value = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            except TypeError:
                # 如果不支持 sample_weight，手动应用
                loss_value = loss_fn(y_true, y_pred)
                loss_value = loss_value * sample_weight
        else:
            loss_value = loss_fn(y_true, y_pred)

        return loss_value

    def get_all_loss_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有损失函数信息

        返回:
            损失函数信息字典
        """
        info = {}

        for loss_name in self.losses.keys():
            loss_fn = self.losses[loss_name]
            info[loss_name] = {
                'name': loss_name,
                'class': type(loss_fn).__name__,
                'module': type(loss_fn).__module__
            }

        return info


# ============================================================================
# 便捷函数
# ============================================================================

def build_losses_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有损失函数的便捷函数

    参数:
        context: 训练上下文

    返回:
        损失函数字典
    """
    manager = LossManager(context)
    return manager.run()


def get_loss_from_context(
    context: TrainContext,
    loss_name: str
) -> Any:
    """
    从上下文获取单个损失函数

    参数:
        context: 训练上下文
        loss_name: 损失函数名称

    返回:
        损失函数实例
    """
    manager = LossManager(context)
    manager.run()
    return manager.get_loss(loss_name)


def create_loss_manager(context: TrainContext) -> LossManager:
    """
    创建并初始化损失函数管理器

    参数:
        context: 训练上下文

    返回:
        LossManager 实例
    """
    manager = LossManager(context)
    manager.run()
    return manager


def create_simple_loss(loss_type: str = 'mse', **kwargs) -> Any:
    """
    快速创建损失函数的便捷函数

    参数:
        loss_type: 损失函数类型
        **kwargs: 其他参数

    返回:
        损失函数实例
    """
    loss_map = {
        'mse': 'tensorflow.keras.losses:MeanSquaredError',
        'mae': 'tensorflow.keras.losses:MeanAbsoluteError',
        'bce': 'tensorflow.keras.losses:BinaryCrossentropy',
        'cce': 'tensorflow.keras.losses:CategoricalCrossentropy',
        'scce': 'tensorflow.keras.losses:SparseCategoricalCrossentropy',
        'huber': 'tensorflow.keras.losses:Huber',
        'hinge': 'tensorflow.keras.losses:Hinge',
        'kld': 'tensorflow.keras.losses:KLDivergence',
        'cosine': 'tensorflow.keras.losses:CosineSimilarity',
    }

    if loss_type.lower() not in loss_map:
        raise ValueError(
            f"不支持的损失函数类型: {loss_type}。"
            f"支持的类型: {list(loss_map.keys())}"
        )

    reflection = loss_map[loss_type.lower()]
    return call_target(reflection=reflection, args=kwargs)
