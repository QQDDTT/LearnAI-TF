# -*- coding: utf-8 -*-
"""
modules/data_manager.py
数据管理器：负责数据加载、预处理、增强

设计原则：
1. 支持多种数据源（CSV, NumPy, Web, TFRecord等）
2. 内置数据增强功能
3. 支持监督学习、强化学习、自监督学习的数据格式
4. 使用 call_target 动态调用数据处理函数

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用数据处理功能
"""

from typing import Dict, Any, List, Optional, Tuple
from common.train_context import TrainContext
from common.common import call_target, LoggerManager
from common.interfaces import DataManagerInterface

# 初始化日志
logger = LoggerManager.get_logger(__file__)


class DataManager(DataManagerInterface):
    """
    数据管理器

    职责：
    1. 从 TrainContext 读取数据配置
    2. 加载不同训练模式的数据
    3. 应用数据增强和预处理
    4. 管理数据加载器生命周期
    """

    def __init__(self, context: TrainContext):
        """初始化数据管理器"""
        super().__init__(context)
        self.dataloaders: Dict[str, Any] = {}
        self.data_configs: Dict[str, Any] = {}
        self.augmentation_configs: Dict[str, Any] = {}
        logger.info("DataManager 初始化完成")

    def initialize(self) -> None:
        """从 TrainContext 初始化数据管理器"""
        logger.info("开始初始化数据管理器")

        # 获取训练模式
        training_mode = self.context.training_mode
        logger.info(f"训练模式: {training_mode}")

        # 根据训练模式提取数据配置
        if training_mode in self.context.training_pipelines:
            pipeline = self.context.training_pipelines[training_mode]
            if hasattr(pipeline, 'parameters') and 'data_sources' in pipeline.parameters:
                self.data_configs = pipeline.parameters['data_sources']
                logger.info(f"找到 {len(self.data_configs)} 个数据源配置")
            else:
                logger.warning("未找到数据源配置")

            # 提取增强配置（如果有）
            if hasattr(pipeline, 'parameters') and 'augmentation' in pipeline.parameters:
                self.augmentation_configs = pipeline.parameters['augmentation']
                logger.info(f"找到数据增强配置: {list(self.augmentation_configs.keys())}")

        self._initialized = True
        logger.info("数据管理器初始化完成")

    def validate_data_configs(self) -> bool:
        """验证数据配置"""
        logger.info("开始验证数据配置")

        if not self.data_configs:
            logger.warning("没有数据配置需要验证")
            return True

        for source_name, source_config in self.data_configs.items():
            logger.debug(f"验证数据源: {source_name}")

            if not isinstance(source_config, dict):
                logger.error(f"数据源 '{source_name}' 配置格式错误")
                raise ValueError(f"数据源 '{source_name}' 配置格式错误")

            # 检查必需字段
            if 'type' not in source_config and 'reflection' not in source_config:
                logger.error(f"数据源 '{source_name}' 缺少 type 或 reflection 字段")
                raise ValueError(f"数据源 '{source_name}' 缺少 type 或 reflection 字段")

        logger.info("数据配置验证通过")
        return True

    def load_data_source(
        self,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> Any:
        """加载单个数据源"""
        logger.info(f"开始加载数据源: {source_name}")

        source_type = source_config.get('type', 'csv').lower()
        logger.debug(f"数据源类型: {source_type}")

        # 根据类型选择加载方法
        if source_type == 'csv':
            dataloader = self._load_csv_data(source_name, source_config)
        elif source_type == 'numpy':
            dataloader = self._load_numpy_data(source_name, source_config)
        elif source_type == 'tfrecord':
            dataloader = self._load_tfrecord_data(source_name, source_config)
        elif source_type == 'web':
            dataloader = self._load_web_data(source_name, source_config)
        elif source_type == 'custom':
            dataloader = self._load_custom_data(source_name, source_config)
        else:
            logger.error(f"不支持的数据源类型: {source_type}")
            raise ValueError(f"不支持的数据源类型: {source_type}")

        logger.info(f"数据源 '{source_name}' 加载完成")
        return dataloader

    def _load_csv_data(
        self,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> Any:
        """加载CSV数据"""
        logger.debug(f"加载CSV数据: {source_name}")

        filepath = source_config.get('filepath')
        if not filepath:
            logger.error(f"数据源 '{source_name}' 缺少 filepath")
            raise ValueError(f"数据源 '{source_name}' 缺少 filepath")

        logger.debug(f"CSV文件路径: {filepath}")

        # 使用 call_target 加载
        reflection = source_config.get('reflection', 'common.utils:build_csv_dataset')
        args = source_config.get('args', {})
        args['filepath'] = filepath

        logger.debug(f"使用反射加载: {reflection}")
        dataloader = call_target(reflection=reflection, args=args)

        # 应用数据增强（如果配置）
        if source_name in self.augmentation_configs:
            logger.info(f"应用数据增强: {source_name}")
            dataloader = self._apply_augmentation(dataloader, source_name)

        return dataloader

    def _apply_augmentation(
        self,
        dataloader: Any,
        source_name: str
    ) -> Any:
        """应用数据增强"""
        logger.debug(f"开始应用数据增强: {source_name}")

        aug_config = self.augmentation_configs.get(source_name, {})
        transforms = aug_config.get('transforms', [])

        logger.info(f"应用 {len(transforms)} 个数据增强操作")

        for i, transform in enumerate(transforms):
            transform_type = transform.get('type')
            logger.debug(f"增强 {i+1}/{len(transforms)}: {transform_type}")

            if transform_type == 'random_crop':
                dataloader = self._apply_random_crop(dataloader, transform)
            elif transform_type == 'random_flip':
                dataloader = self._apply_random_flip(dataloader, transform)
            elif transform_type == 'color_jitter':
                dataloader = self._apply_color_jitter(dataloader, transform)
            elif transform_type == 'normalize':
                dataloader = self._apply_normalize(dataloader, transform)
            elif transform_type == 'custom':
                dataloader = self._apply_custom_transform(dataloader, transform)
            else:
                logger.warning(f"未知的增强类型: {transform_type}，跳过")

        logger.info("数据增强应用完成")
        return dataloader

    def load_all_data(self) -> Dict[str, Any]:
        """加载所有数据源"""
        logger.info("开始加载所有数据源")

        for source_name, source_config in self.data_configs.items():
            try:
                dataloader = self.load_data_source(source_name, source_config)
                self.dataloaders[source_name] = dataloader
                logger.info(f"✓ 数据源 '{source_name}' 加载成功")
            except Exception as e:
                logger.error(f"✗ 加载数据源 '{source_name}' 失败: {e}", exc_info=True)
                raise RuntimeError(f"加载数据源 '{source_name}' 失败: {e}") from e

        logger.info(f"所有数据源加载完成，共 {len(self.dataloaders)} 个")
        return self.dataloaders

    def execute(self) -> Dict[str, Any]:
        """执行数据加载流程"""
        logger.info("执行数据加载流程")
        return self.load_all_data()

    def finalize(self) -> None:
        """将数据加载器保存到上下文"""
        logger.info("将数据加载器保存到上下文")
        self.context.container.dataloaders = self.dataloaders
        logger.debug(f"已保存 {len(self.dataloaders)} 个数据加载器到上下文")

    def get_dataloader(self, source_name: str) -> Any:
        """获取数据加载器"""
        if source_name not in self.dataloaders:
            logger.error(f"数据源 '{source_name}' 不存在")
            raise ValueError(f"数据源 '{source_name}' 不存在")

        logger.debug(f"获取数据加载器: {source_name}")
        return self.dataloaders[source_name]

    # 增强方法占位符（需要完整实现）
    def _apply_random_crop(self, dataloader, config):
        logger.debug("应用 random_crop")
        return dataloader

    def _apply_random_flip(self, dataloader, config):
        logger.debug("应用 random_flip")
        return dataloader

    def _apply_color_jitter(self, dataloader, config):
        logger.debug("应用 color_jitter")
        return dataloader

    def _apply_normalize(self, dataloader, config):
        logger.debug("应用 normalize")
        return dataloader

    def _apply_custom_transform(self, dataloader, config):
        logger.debug("应用 custom_transform")
        reflection = config.get('reflection')
        if reflection:
            transform = call_target(reflection=reflection, args=config.get('args', {}))
            return transform(dataloader)
        return dataloader

    def _load_numpy_data(self, source_name, source_config):
        logger.debug(f"加载NumPy数据: {source_name}")
        reflection = source_config.get('reflection', 'common.utils:build_numpy_dataset')
        return call_target(reflection=reflection, args=source_config.get('args', {}))

    def _load_tfrecord_data(self, source_name, source_config):
        logger.debug(f"加载TFRecord数据: {source_name}")
        reflection = source_config.get('reflection', 'common.utils:build_tfrecord_dataset')
        return call_target(reflection=reflection, args=source_config.get('args', {}))

    def _load_web_data(self, source_name, source_config):
        logger.debug(f"加载Web数据: {source_name}")
        reflection = source_config.get('reflection', 'common.utils:build_web_dataset')
        return call_target(reflection=reflection, args=source_config.get('args', {}))

    def _load_custom_data(self, source_name, source_config):
        logger.debug(f"加载自定义数据: {source_name}")
        reflection = source_config.get('reflection')
        if not reflection:
            logger.error(f"自定义数据源 '{source_name}' 缺少 reflection")
            raise ValueError(f"自定义数据源 '{source_name}' 缺少 reflection")
        return call_target(reflection=reflection, args=source_config.get('args', {}))


# 便捷函数
def initialize_data_manager(context: TrainContext) -> Dict[str, Any]:
    """初始化数据管理器的便捷函数"""
    logger.info("使用便捷函数初始化数据管理器")
    manager = DataManager(context)
    return manager.run()
