# -*- coding: utf-8 -*-
"""
modules/models.py
模型构建器：从配置上下文构建 TensorFlow/Keras 模型

设计原则：
1. 支持三种模型类型：Sequential, Functional, Subclass
2. 支持复杂的层连接关系（Connection）
3. 使用 call_target 动态创建层和模型
4. 支持简化配置（可选步骤）

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用 TensorFlow API
"""

from typing import Dict, Any, List, Optional
from common.train_context import TrainContext
from common.common import call_target, LoggerManager
from common.interfaces import ModelBuilderInterface

# 初始化日志
logger = LoggerManager.get_logger(__file__)


class ModelBuilder(ModelBuilderInterface):
    """
    模型构建器

    职责：
    1. 从 TrainContext.models 读取配置
    2. 使用 call_target 实例化层和模型
    3. 处理层之间的连接关系（Connection）
    4. 支持三种模型类型：Sequential、Functional、Subclass

    支持简化配置：
    - connection 可选（默认顺序连接）
    - 层名称可选（自动生成）
    - 输入/输出标记可选
    """

    def __init__(self, context: TrainContext):
        """初始化模型构建器"""
        super().__init__(context)
        self.models: Dict[str, Any] = {}
        self.layer_cache: Dict[str, Any] = {}
        self.layer_outputs: Dict[str, Any] = {}
        logger.info("ModelBuilder 初始化完成")

    def initialize(self) -> None:
        """从 TrainContext 初始化模型构建器"""
        logger.info("开始初始化模型构建器")
        if not self.context.models:
            logger.error("TrainContext 中没有定义任何模型")
            raise ValueError("TrainContext 中没有定义任何模型")
        logger.info(f"找到 {len(self.context.models)} 个模型配置")
        self._initialized = True

    def validate_models(self) -> bool:
        """
        验证模型配置

        返回:
            配置是否有效
        """
        logger.info("开始验证模型配置")
        for model_name, model_config in self.context.models.items():
            logger.debug(f"验证模型: {model_name}")

            if not isinstance(model_config, dict):
                logger.error(f"模型 '{model_name}' 配置格式错误")
                raise ValueError(f"模型 '{model_name}' 配置格式错误")

            # 检查必需字段
            if 'type' not in model_config:
                logger.error(f"模型 '{model_name}' 缺少 type 字段")
                raise ValueError(f"模型 '{model_name}' 缺少 type 字段")

            model_type = model_config['type']
            logger.debug(f"模型 '{model_name}' 类型: {model_type}")

            # Sequential 和 Functional 需要 layers
            if model_type in ['sequential', 'functional']:
                if 'layers' not in model_config or not model_config['layers']:
                    logger.error(f"模型 '{model_name}' (type={model_type}) 必须定义 layers")
                    raise ValueError(
                        f"模型 '{model_name}' (type={model_type}) 必须定义 layers"
                    )
                logger.debug(f"模型 '{model_name}' 包含 {len(model_config['layers'])} 层")

            # Subclass 需要 reflection
            if model_type == 'subclass':
                if 'reflection' not in model_config:
                    logger.error(f"模型 '{model_name}' (type=subclass) 必须定义 reflection")
                    raise ValueError(
                        f"模型 '{model_name}' (type=subclass) 必须定义 reflection"
                    )

        logger.info("模型配置验证通过")
        return True

    def build_model(
        self,
        model_name: str,
        model_config: Any
    ) -> Any:
        """
        构建单个模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Keras 模型实例
        """
        logger.info(f"开始构建模型: {model_name}")

        # 转换为字典格式
        if not isinstance(model_config, dict):
            config = {
                'type': getattr(model_config, 'type', 'sequential'),
                'layers': getattr(model_config, 'layers', []),
                'reflection': getattr(model_config, 'reflection', None),
                'args': getattr(model_config, 'args', {})
            }
        else:
            config = model_config

        model_type = config.get('type', 'sequential').lower()
        logger.debug(f"模型类型: {model_type}")

        # 根据类型构建模型
        if model_type == 'sequential':
            model = self._build_sequential_model(model_name, config)
        elif model_type == 'functional':
            model = self._build_functional_model(model_name, config)
        elif model_type == 'subclass':
            model = self._build_subclass_model(model_name, config)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            raise ValueError(f"不支持的模型类型: {model_type}")

        logger.info(f"模型 '{model_name}' 构建完成")
        return model

    def _build_sequential_model(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """构建顺序模型"""
        logger.debug(f"构建顺序模型: {model_name}")

        layers_config = config.get('layers', [])
        logger.debug(f"包含 {len(layers_config)} 层")

        layers = []
        for i, layer_config in enumerate(layers_config):
            layer_name = layer_config.get('name', f'{model_name}_layer_{i}')
            logger.debug(f"构建层 {i+1}/{len(layers_config)}: {layer_name}")
            layer = self._build_layer(layer_config, layer_name)
            layers.append(layer)

        # 创建 Sequential 模型
        model = call_target(
            reflection="tensorflow.keras:Sequential",
            args={'layers': layers, 'name': model_name}
        )

        logger.info(f"Sequential 模型 '{model_name}' 构建完成，共 {len(layers)} 层")
        return model

    def build_all_models(self) -> Dict[str, Any]:
        """
        构建所有模型

        返回:
            模型字典 {model_name: model}
        """
        logger.info("开始构建所有模型")

        for model_name, model_config in self.context.models.items():
            try:
                model = self.build_model(model_name, model_config)
                self.models[model_name] = model
                logger.info(f"✓ 模型 '{model_name}' 构建成功")
            except Exception as e:
                logger.error(f"✗ 构建模型 '{model_name}' 失败: {e}", exc_info=True)
                raise RuntimeError(
                    f"构建模型 '{model_name}' 失败: {e}"
                ) from e

        logger.info(f"所有模型构建完成，共 {len(self.models)} 个")
        return self.models

    def execute(self) -> Dict[str, Any]:
        """执行模型构建"""
        logger.info("执行模型构建流程")
        models = self.build_all_models()
        self.compile_models()
        return models

    def finalize(self) -> None:
        """将模型缓存到上下文"""
        logger.info("将模型保存到上下文")
        self.context.container.models = self.models
        logger.debug(f"已保存 {len(self.models)} 个模型到上下文")

    def _build_layer(
        self,
        layer_config: Dict[str, Any],
        layer_name: str
    ) -> Any:
        """构建单个层"""
        reflection = layer_config.get('reflection')

        if not reflection:
            logger.error(f"层 '{layer_name}' 缺少 reflection 字段")
            raise ValueError(f"层 '{layer_name}' 缺少 reflection 字段")

        # 准备参数
        args = layer_config.get('args', {}).copy()

        # 设置层名称
        if 'name' not in args:
            args['name'] = layer_name

        logger.debug(f"创建层: {layer_name} ({reflection})")

        # 创建层
        layer = call_target(reflection=reflection, args=args)

        # 缓存层
        self.layer_cache[layer_name] = layer

        return layer

    def compile_models(self) -> None:
        """编译模型（可选）"""
        logger.debug("检查是否需要编译模型")
        # 如果有编译配置，执行编译
        # 否则跳过（使用自定义训练循环）
        pass

    def get_model(self, model_name: str) -> Any:
        """获取模型实例"""
        if model_name not in self.models:
            logger.error(f"模型 '{model_name}' 不存在")
            raise ValueError(f"模型 '{model_name}' 不存在")

        logger.debug(f"获取模型: {model_name}")
        return self.models[model_name]

    def save_model(
        self,
        model_name: str,
        filepath: str,
        save_format: str = 'tf'
    ) -> None:
        """保存模型"""
        logger.info(f"保存模型 '{model_name}' 到 {filepath}")
        model = self.get_model(model_name)
        model.save(filepath, save_format=save_format)
        logger.info(f"模型保存成功: {filepath}")


# 便捷函数
def build_models_from_context(context: TrainContext) -> Dict[str, Any]:
    """从训练上下文构建所有模型的便捷函数"""
    logger.info("使用便捷函数构建模型")
    builder = ModelBuilder(context)
    return builder.run()
