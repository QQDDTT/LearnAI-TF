# -*- coding: utf-8 -*-
"""
modules/losses.py
损失函数管理器：从配置上下文构建和管理损失函数

设计原则：
1. 支持 TensorFlow 内置损失函数（通过 call_target）
2. 支持自定义损失函数类
3. 支持多任务学习的损失聚合
4. 支持损失权重配置

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用 TensorFlow 损失函数
"""

from typing import Dict, Any, List, Optional, Callable
from common.train_context import TrainContext, LossConfig
from common.common import call_target, LoggerManager
from common.interfaces import LossManagerInterface

# 初始化日志
logger = LoggerManager.get_logger(__file__)


class LossManager(LossManagerInterface):
    """
    损失函数管理器

    职责：
    1. 从 TrainContext.losses 读取配置
    2. 使用 call_target 构建 TensorFlow 损失函数
    3. 管理自定义损失函数
    4. 处理多任务损失聚合
    """

    def __init__(self, context: TrainContext):
        """初始化损失函数管理器"""
        super().__init__(context)
        self.losses: Dict[str, Any] = {}
        self.loss_configs: Dict[str, LossConfig] = {}
        self.loss_weights: Dict[str, float] = {}
        logger.info("LossManager 初始化完成")

    def initialize(self) -> None:
        """从 TrainContext 初始化损失函数管理器"""
        logger.info("开始初始化损失函数管理器")

        if not self.context.losses:
            logger.error("TrainContext 中没有定义任何损失函数")
            raise ValueError("TrainContext 中没有定义任何损失函数")

        # 加载损失函数配置
        for loss_name, loss_config in self.context.losses.items():
            self.loss_configs[loss_name] = loss_config
            logger.debug(f"加载损失函数配置: {loss_name}")

        logger.info(f"找到 {len(self.loss_configs)} 个损失函数配置")
        self._initialized = True

    def validate_losses(self) -> bool:
        """验证损失函数配置"""
        logger.info("开始验证损失函数配置")

        for loss_name, loss_config in self.loss_configs.items():
            logger.debug(f"验证损失函数: {loss_name}")

            if not isinstance(loss_config, (dict, LossConfig)):
                logger.error(f"损失函数 '{loss_name}' 配置格式错误")
                raise ValueError(f"损失函数 '{loss_name}' 配置格式错误")

            # 检查必需字段
            if isinstance(loss_config, dict):
                if 'reflection' not in loss_config:
                    logger.error(f"损失函数 '{loss_name}' 缺少 reflection 字段")
                    raise ValueError(f"损失函数 '{loss_name}' 缺少 reflection 字段")
            else:
                if not loss_config.reflection:
                    logger.error(f"损失函数 '{loss_name}' 缺少 reflection 字段")
                    raise ValueError(f"损失函数 '{loss_name}' 缺少 reflection 字段")

        logger.info("损失函数配置验证通过")
        return True

    def build_loss(
        self,
        loss_name: str,
        loss_config: Any
    ) -> Any:
        """构建单个损失函数"""
        logger.info(f"开始构建损失函数: {loss_name}")

        # 转换为字典格式
        if isinstance(loss_config, LossConfig):
            config = {
                'reflection': loss_config.reflection,
                'args': loss_config.args,
                'weight': getattr(loss_config, 'weight', 1.0)
            }
        else:
            config = loss_config

        reflection = config.get('reflection')
        args = config.get('args', {}).copy()
        weight = config.get('weight', 1.0)

        if not reflection:
            logger.error(f"损失函数 '{loss_name}' 缺少 reflection")
            raise ValueError(f"损失函数 '{loss_name}' 缺少 reflection")

        logger.debug(f"损失函数类型: {reflection}")
        logger.debug(f"损失权重: {weight}")

        # 创建损失函数
        loss_fn = call_target(target=reflection, arguments=args)

        # 保存权重
        self.loss_weights[loss_name] = weight

        logger.info(f"损失函数 '{loss_name}' 构建完成")
        return loss_fn

    def build_all_losses(self) -> Dict[str, Any]:
        """构建所有损失函数"""
        logger.info("开始构建所有损失函数")

        for loss_name, loss_config in self.loss_configs.items():
            try:
                loss_fn = self.build_loss(loss_name, loss_config)
                self.losses[loss_name] = loss_fn
                logger.info(f"✓ 损失函数 '{loss_name}' 构建成功")
            except Exception as e:
                logger.error(f"✗ 构建损失函数 '{loss_name}' 失败: {e}", exc_info=True)
                raise RuntimeError(f"构建损失函数 '{loss_name}' 失败: {e}") from e

        logger.info(f"所有损失函数构建完成，共 {len(self.losses)} 个")
        return self.losses

    def create_combined_loss(
        self,
        loss_names: List[str],
        weights: Optional[List[float]] = None
    ) -> Callable:
        """创建组合损失函数"""
        logger.info(f"创建组合损失函数，包含 {len(loss_names)} 个损失")

        if not loss_names:
            logger.error("loss_names 不能为空")
            raise ValueError("loss_names 不能为空")

        # 获取所有损失函数
        loss_instances = []
        for loss_name in loss_names:
            if loss_name not in self.losses:
                logger.error(f"损失函数 '{loss_name}' 不存在")
                raise ValueError(f"损失函数 '{loss_name}' 不存在")
            loss_instances.append(self.losses[loss_name])
            logger.debug(f"添加损失组件: {loss_name}")

        # 设置权重
        if weights is None:
            weights = [1.0] * len(loss_names)
            logger.debug("使用默认权重 (均等)")
        else:
            logger.debug(f"使用自定义权重: {weights}")

        def combined_loss(y_true, y_pred):
            """组合损失函数"""
            total_loss = 0.0
            for loss_fn, weight in zip(loss_instances, weights):
                loss_value = loss_fn(y_true, y_pred)
                total_loss += weight * loss_value
            return total_loss

        logger.info("组合损失函数创建完成")
        return combined_loss

    def execute(self) -> Dict[str, Any]:
        """执行损失函数构建"""
        logger.info("执行损失函数构建流程")
        return self.build_all_losses()

    def finalize(self) -> None:
        """将损失函数缓存到上下文"""
        logger.info("将损失函数保存到上下文")
        self.context.container.losses = self.losses

        if self.loss_weights:
            if not hasattr(self.context.container, 'loss_weights'):
                self.context.container.loss_weights = {}
            self.context.container.loss_weights.update(self.loss_weights)
            logger.debug(f"已保存 {len(self.loss_weights)} 个损失权重")

        logger.debug(f"已保存 {len(self.losses)} 个损失函数到上下文")

    def get_loss(self, loss_name: str) -> Any:
        """获取损失函数实例"""
        if loss_name not in self.losses:
            logger.error(f"损失函数 '{loss_name}' 不存在")
            raise ValueError(f"损失函数 '{loss_name}' 不存在")

        logger.debug(f"获取损失函数: {loss_name}")
        return self.losses[loss_name]

    def compute_loss(
        self,
        loss_name: str,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None
    ) -> Any:
        """计算损失值"""
        logger.debug(f"计算损失: {loss_name}")
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
        """获取所有损失函数信息"""
        logger.debug("获取所有损失函数信息")
        info = {}

        for loss_name in self.losses.keys():
            loss_fn = self.losses[loss_name]
            info[loss_name] = {
                'name': loss_name,
                'class': type(loss_fn).__name__,
                'module': type(loss_fn).__module__,
                'weight': self.loss_weights.get(loss_name, 1.0)
            }

        return info


# 便捷函数
def build_losses_from_context(context: TrainContext) -> Dict[str, Any]:
    """从训练上下文构建所有损失函数的便捷函数"""
    logger.info("使用便捷函数构建损失函数")
    manager = LossManager(context)
    return manager.run()


def create_simple_loss(loss_type: str = 'mse', **kwargs) -> Any:
    """快速创建损失函数的便捷函数"""
    logger.info(f"创建简单损失函数: {loss_type}")

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
        logger.error(f"不支持的损失函数类型: {loss_type}")
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    reflection = loss_map[loss_type.lower()]
    return call_target(target=reflection, arguments=kwargs)
