# -*- coding: utf-8 -*-
"""
modules/optimizers.py
优化器管理器：从配置上下文构建和管理优化器

设计原则：
1. 支持 TensorFlow 内置优化器（通过 call_target）
2. 支持学习率调度器
3. 支持优化器配置简化（可选步骤）
4. 管理优化器生命周期

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用 TensorFlow 优化器
"""

from typing import Dict, Any, List, Optional, Tuple
from common.train_context import TrainContext, OptimizerConfig
from common.common import call_target
from common.interfaces import OptimizerManagerInterface


class OptimizerManager(OptimizerManagerInterface):
    """
    优化器管理器

    职责：
    1. 从 TrainContext.optimizers 读取配置
    2. 使用 call_target 构建 TensorFlow 优化器
    3. 配置学习率调度器（可选）
    4. 管理优化器状态和参数

    支持简化配置：
    - 学习率调度器可选
    - 梯度裁剪可选
    - 权重衰减可选
    """

    def __init__(self, context: TrainContext):
        """初始化优化器管理器"""
        super().__init__(context)
        self.optimizers: Dict[str, Any] = {}
        self.lr_schedules: Dict[str, Any] = {}
        self.optimizer_configs: Dict[str, OptimizerConfig] = {}

    def initialize(self) -> None:
        """从 TrainContext 初始化优化器管理器"""
        if not self.context.optimizers:
            raise ValueError("TrainContext 中没有定义任何优化器")

        # 加载优化器配置
        for opt_name, opt_config in self.context.optimizers.items():
            self.optimizer_configs[opt_name] = opt_config

        self._initialized = True

    def validate_optimizers(self) -> bool:
        """
        验证优化器配置

        返回:
            配置是否有效
        """
        for opt_name, opt_config in self.optimizer_configs.items():
            if not isinstance(opt_config, (dict, OptimizerConfig)):
                raise ValueError(f"优化器 '{opt_name}' 配置格式错误")

            # 检查必需字段
            if isinstance(opt_config, dict):
                if 'reflection' not in opt_config:
                    raise ValueError(
                        f"优化器 '{opt_name}' 缺少 reflection 字段"
                    )
            else:
                if not opt_config.reflection:
                    raise ValueError(
                        f"优化器 '{opt_name}' 缺少 reflection 字段"
                    )

        return True

    # ========================================================================
    # 优化器构建
    # ========================================================================

    def build_optimizer(
        self,
        optimizer_name: str,
        optimizer_config: Any
    ) -> Any:
        """
        构建单个优化器

        参数:
            optimizer_name: 优化器名称
            optimizer_config: 优化器配置

        返回:
            TensorFlow 优化器实例
        """
        # 转换为字典格式
        if isinstance(optimizer_config, OptimizerConfig):
            config = {
                'reflection': optimizer_config.reflection,
                'args': optimizer_config.args,
                'name': optimizer_config.name
            }
        else:
            config = optimizer_config

        reflection = config.get('reflection')
        args = config.get('args', {}).copy()

        if not reflection:
            raise ValueError(f"优化器 '{optimizer_name}' 缺少 reflection")

        # Step 1: 处理学习率（可能是标量或调度器）
        learning_rate = self._process_learning_rate(
            optimizer_name,
            args.get('learning_rate'),
            config.get('lr_schedule')
        )

        if learning_rate is not None:
            args['learning_rate'] = learning_rate

        # Step 2: 处理梯度裁剪（可选）
        clipnorm = config.get('clipnorm')
        clipvalue = config.get('clipvalue')
        global_clipnorm = config.get('global_clipnorm')

        if clipnorm is not None:
            args['clipnorm'] = clipnorm
        if clipvalue is not None:
            args['clipvalue'] = clipvalue
        if global_clipnorm is not None:
            args['global_clipnorm'] = global_clipnorm

        # Step 3: 处理权重衰减（可选）
        weight_decay = config.get('weight_decay')
        if weight_decay is not None:
            args['weight_decay'] = weight_decay

        # Step 4: 创建优化器
        optimizer = call_target(reflection=reflection, args=args)

        return optimizer

    def _process_learning_rate(
        self,
        optimizer_name: str,
        learning_rate: Any,
        lr_schedule_config: Optional[Dict[str, Any]]
    ) -> Any:
        """
        处理学习率（标量或调度器）

        参数:
            optimizer_name: 优化器名称
            learning_rate: 初始学习率（可能为None）
            lr_schedule_config: 学习率调度器配置（可选）

        返回:
            学习率（标量或调度器对象）
        """
        # 如果没有调度器配置，直接返回学习率
        if not lr_schedule_config or not self._is_config_valid(lr_schedule_config):
            return learning_rate

        # 获取 reflection 和 args
        reflection = lr_schedule_config.get('reflection')
        schedule_args = lr_schedule_config.get('args', {}).copy()

        if not reflection:
            raise ValueError(
                f"优化器 '{optimizer_name}' 的 lr_schedule 缺少 reflection 字段"
            )

        # 设置初始学习率
        if learning_rate is not None and 'initial_learning_rate' not in schedule_args:
            schedule_args['initial_learning_rate'] = learning_rate

        # 创建调度器
        lr_schedule = call_target(reflection=reflection, args=schedule_args)

        # 缓存调度器
        self.lr_schedules[optimizer_name] = lr_schedule

        return lr_schedule

    def build_all_optimizers(self) -> Dict[str, Any]:
        """
        构建所有优化器

        返回:
            优化器字典 {optimizer_name: optimizer}
        """
        for opt_name, opt_config in self.optimizer_configs.items():
            try:
                optimizer = self.build_optimizer(opt_name, opt_config)
                self.optimizers[opt_name] = optimizer
            except Exception as e:
                raise RuntimeError(
                    f"构建优化器 '{opt_name}' 失败: {e}"
                ) from e

        return self.optimizers

    # ========================================================================
    # 学习率调度器管理
    # ========================================================================

    def setup_schedulers(self) -> Dict[str, Any]:
        """
        设置学习率调度器（可选）

        注意：调度器已在 build_optimizer 中创建
        这个方法用于额外的调度器配置

        返回:
            调度器字典
        """
        return self.lr_schedules

    def get_learning_rate(self, optimizer_name: str) -> float:
        """
        获取当前学习率

        参数:
            optimizer_name: 优化器名称

        返回:
            当前学习率
        """
        if optimizer_name not in self.optimizers:
            raise ValueError(f"优化器 '{optimizer_name}' 不存在")

        optimizer = self.optimizers[optimizer_name]

        # 获取学习率
        lr = optimizer.learning_rate

        # 如果是调度器，获取当前值
        if hasattr(lr, 'numpy'):
            return float(lr.numpy())
        elif callable(lr):
            # 调度器对象
            step = optimizer.iterations
            return float(lr(step).numpy())
        else:
            return float(lr)

    def set_learning_rate(
        self,
        optimizer_name: str,
        learning_rate: float
    ) -> None:
        """
        设置学习率

        参数:
            optimizer_name: 优化器名称
            learning_rate: 新的学习率
        """
        if optimizer_name not in self.optimizers:
            raise ValueError(f"优化器 '{optimizer_name}' 不存在")

        optimizer = self.optimizers[optimizer_name]
        optimizer.learning_rate.assign(learning_rate)

    # ========================================================================
    # 接口实现
    # ========================================================================

    def execute(self) -> Dict[str, Any]:
        """执行优化器构建"""
        return self.build_all_optimizers()

    def finalize(self) -> None:
        """将优化器缓存到上下文"""
        self.context.container.optimizers = self.optimizers

        # 缓存学习率调度器
        if self.lr_schedules:
            if not hasattr(self.context.container, 'lr_schedules'):
                self.context.container.lr_schedules = {}
            self.context.container.lr_schedules.update(self.lr_schedules)

    # ========================================================================
    # 工具方法
    # ========================================================================

    def _is_config_valid(self, config: Any) -> bool:
        """
        检查配置是否有效

        参数:
            config: 配置对象

        返回:
            是否有效
        """
        if config is None:
            return False

        if isinstance(config, dict):
            if not config:
                return False

            # 检查关键字段
            if 'type' in config or 'reflection' in config:
                return True

            return len(config) > 0

        return False

    def get_optimizer(self, optimizer_name: str) -> Any:
        """
        获取优化器实例

        参数:
            optimizer_name: 优化器名称

        返回:
            优化器实例
        """
        if optimizer_name not in self.optimizers:
            raise ValueError(f"优化器 '{optimizer_name}' 不存在")

        return self.optimizers[optimizer_name]

    def get_optimizer_config(self, optimizer_name: str) -> Dict[str, Any]:
        """
        获取优化器配置

        参数:
            optimizer_name: 优化器名称

        返回:
            配置字典
        """
        if optimizer_name not in self.optimizers:
            raise ValueError(f"优化器 '{optimizer_name}' 不存在")

        optimizer = self.optimizers[optimizer_name]

        config = {
            'name': optimizer_name,
            'class': type(optimizer).__name__,
            'learning_rate': self.get_learning_rate(optimizer_name),
            'iterations': int(optimizer.iterations.numpy()),
        }

        # 添加其他配置
        if hasattr(optimizer, 'beta_1'):
            config['beta_1'] = float(optimizer.beta_1.numpy())
        if hasattr(optimizer, 'beta_2'):
            config['beta_2'] = float(optimizer.beta_2.numpy())
        if hasattr(optimizer, 'epsilon'):
            config['epsilon'] = float(optimizer.epsilon)
        if hasattr(optimizer, 'rho'):
            config['rho'] = float(optimizer.rho.numpy())
        if hasattr(optimizer, 'momentum'):
            config['momentum'] = float(optimizer.momentum.numpy())

        return config

    def get_all_optimizer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有优化器信息

        返回:
            优化器信息字典
        """
        info = {}

        for opt_name in self.optimizers.keys():
            info[opt_name] = self.get_optimizer_config(opt_name)

        return info

    def reset_optimizer(self, optimizer_name: str) -> None:
        """
        重置优化器状态

        参数:
            optimizer_name: 优化器名称
        """
        if optimizer_name not in self.optimizer_configs:
            raise ValueError(f"优化器 '{optimizer_name}' 不存在")

        # 重新构建优化器
        opt_config = self.optimizer_configs[optimizer_name]
        new_optimizer = self.build_optimizer(optimizer_name, opt_config)
        self.optimizers[optimizer_name] = new_optimizer

    def apply_gradients(
        self,
        optimizer_name: str,
        grads_and_vars: List[Tuple[Any, Any]]
    ) -> None:
        """
        应用梯度

        参数:
            optimizer_name: 优化器名称
            grads_and_vars: 梯度和变量对列表
        """
        optimizer = self.get_optimizer(optimizer_name)
        optimizer.apply_gradients(grads_and_vars)

    def save_optimizer_state(
        self,
        optimizer_name: str,
        filepath: str
    ) -> None:
        """
        保存优化器状态

        参数:
            optimizer_name: 优化器名称
            filepath: 保存路径
        """
        optimizer = self.get_optimizer(optimizer_name)

        # 获取优化器权重
        weights = optimizer.get_weights()

        # 保存
        call_target(
            reflection="numpy:save",
            args={'file': filepath, 'arr': weights}
        )

    def load_optimizer_state(
        self,
        optimizer_name: str,
        filepath: str
    ) -> None:
        """
        加载优化器状态

        参数:
            optimizer_name: 优化器名称
            filepath: 文件路径
        """
        optimizer = self.get_optimizer(optimizer_name)

        # 加载权重
        weights = call_target(
            reflection="numpy:load",
            args={'file': filepath, 'allow_pickle': True}
        )

        # 设置权重
        optimizer.set_weights(weights)


# ============================================================================
# 便捷函数
# ============================================================================

def build_optimizers_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有优化器的便捷函数

    参数:
        context: 训练上下文

    返回:
        优化器字典
    """
    manager = OptimizerManager(context)
    return manager.run()


def get_optimizer_from_context(
    context: TrainContext,
    optimizer_name: str
) -> Any:
    """
    从上下文获取单个优化器

    参数:
        context: 训练上下文
        optimizer_name: 优化器名称

    返回:
        优化器实例
    """
    manager = OptimizerManager(context)
    manager.run()
    return manager.get_optimizer(optimizer_name)


def create_optimizer_manager(context: TrainContext) -> OptimizerManager:
    """
    创建并初始化优化器管理器

    参数:
        context: 训练上下文

    返回:
        OptimizerManager 实例
    """
    manager = OptimizerManager(context)
    manager.run()
    return manager


def create_simple_optimizer(
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    **kwargs
) -> Any:
    """
    快速创建优化器的便捷函数

    参数:
        optimizer_type: 优化器类型 (adam/sgd/rmsprop/adamw)
        learning_rate: 学习率
        **kwargs: 其他参数

    返回:
        TensorFlow 优化器
    """
    optimizer_map = {
        'adam': 'tensorflow.keras.optimizers:Adam',
        'sgd': 'tensorflow.keras.optimizers:SGD',
        'rmsprop': 'tensorflow.keras.optimizers:RMSprop',
        'adamw': 'tensorflow.keras.optimizers:AdamW',
        'adagrad': 'tensorflow.keras.optimizers:Adagrad',
        'adadelta': 'tensorflow.keras.optimizers:Adadelta',
        'adamax': 'tensorflow.keras.optimizers:Adamax',
        'nadam': 'tensorflow.keras.optimizers:Nadam',
        'ftrl': 'tensorflow.keras.optimizers:Ftrl',
    }

    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(
            f"不支持的优化器类型: {optimizer_type}。"
            f"支持的类型: {list(optimizer_map.keys())}"
        )

    reflection = optimizer_map[optimizer_type.lower()]
    args = {'learning_rate': learning_rate, **kwargs}

    return call_target(reflection=reflection, args=args)
