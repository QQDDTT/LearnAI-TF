# -*- coding: utf-8 -*-
"""
modules/data_manager.py
数据管理器：从配置上下文构建和管理数据加载器

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Tuple
from common.train_context import (
    TrainContext,
    DataSourceConfig,
    DataSourceType
)
from common.common import call_target


class DataManager:
    """
    数据管理器

    职责：
    1. 读取 TrainContext 中的数据源配置
    2. 使用 call_target 实例化数据加载器
    3. 管理不同训练模式的数据源
    4. 支持离线、交互式、流式数据
    5. 处理数据分割（train/val/test）

    注意：所有数据加载器通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化数据管理器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.data_loaders: Dict[str, Any] = {}

    def build_all_data_sources(self) -> Dict[str, Any]:
        """
        构建所有数据源

        返回:
            数据加载器字典 {source_name: {split_name: loader}}
        """
        if not self.context.data_sources:
            raise ValueError("TrainContext 中没有定义任何数据源")

        all_loaders = {}

        for source_name, source_splits in self.context.data_sources.items():
            try:
                loaders = self.build_data_source(source_name, source_splits)
                all_loaders[source_name] = loaders
            except Exception as e:
                raise RuntimeError(f"构建数据源 '{source_name}' 失败: {e}")

        # 缓存到上下文
        self.context.instantiated_dataloaders = all_loaders

        return all_loaders

    def build_data_source(
        self,
        source_name: str,
        source_splits: Dict[str, DataSourceConfig]
    ) -> Dict[str, Any]:
        """
        构建单个数据源的所有分割

        参数:
            source_name: 数据源名称
            source_splits: 数据分割配置字典 {split_name: config}

        返回:
            数据加载器字典 {split_name: loader}
        """
        loaders = {}

        for split_name, split_config in source_splits.items():
            try:
                loader = self._build_data_loader(
                    source_name,
                    split_name,
                    split_config
                )
                loaders[split_name] = loader
            except Exception as e:
                raise RuntimeError(
                    f"构建数据源 '{source_name}.{split_name}' 失败: {e}"
                )

        return loaders

    def _build_data_loader(
        self,
        source_name: str,
        split_name: str,
        split_config: DataSourceConfig
    ) -> Any:
        """
        构建单个数据加载器

        参数:
            source_name: 数据源名称
            split_name: 分割名称
            split_config: 分割配置

        返回:
            数据加载器实例
        """
        if not split_config.reflection:
            raise ValueError(
                f"数据源 '{source_name}.{split_name}' 缺少 reflection 字段"
            )

        # 准备参数
        args = split_config.args.copy()

        # 使用 call_target 创建数据加载器
        loader = call_target(
            reflection=split_config.reflection,
            args=args
        )

        return loader

    def get_data_loader(
        self,
        source_name: str,
        split_name: str = "train"
    ) -> Any:
        """
        获取指定的数据加载器

        参数:
            source_name: 数据源名称
            split_name: 分割名称

        返回:
            数据加载器实例
        """
        if source_name not in self.data_loaders:
            raise ValueError(f"数据源 '{source_name}' 不存在")

        if split_name not in self.data_loaders[source_name]:
            raise ValueError(
                f"数据源 '{source_name}' 中不存在分割 '{split_name}'"
            )

        return self.data_loaders[source_name][split_name]

    def get_training_data(self) -> Tuple[Any, Optional[Any]]:
        """
        获取训练数据

        根据训练模式自动选择合适的数据源

        返回:
            (训练数据加载器, 验证数据加载器)
        """
        training_mode = self.context.training_mode

        # 根据训练模式选择数据源
        if training_mode == "supervised":
            return self._get_supervised_data()

        elif training_mode == "unsupervised":
            return self._get_unsupervised_data()

        elif training_mode == "reinforcement":
            return self._get_reinforcement_data()

        elif training_mode == "semi_supervised":
            return self._get_semi_supervised_data()

        elif training_mode == "self_supervised":
            return self._get_self_supervised_data()

        else:
            # 默认行为：查找通用数据源
            return self._get_default_data()

    def _get_supervised_data(self) -> Tuple[Any, Optional[Any]]:
        """获取监督学习数据"""
        source_name = "supervised_source"

        if source_name not in self.data_loaders:
            raise ValueError(
                f"监督学习模式需要定义 '{source_name}' 数据源"
            )

        train_loader = self.get_data_loader(source_name, "train")

        # 验证集是可选的
        val_loader = None
        if "val" in self.data_loaders[source_name]:
            val_loader = self.get_data_loader(source_name, "val")

        return train_loader, val_loader

    def _get_unsupervised_data(self) -> Tuple[Any, Optional[Any]]:
        """获取无监督学习数据"""
        source_name = "unsupervised_source"

        if source_name not in self.data_loaders:
            raise ValueError(
                f"无监督学习模式需要定义 '{source_name}' 数据源"
            )

        train_loader = self.get_data_loader(source_name, "train")

        return train_loader, None

    def _get_reinforcement_data(self) -> Tuple[Any, Optional[Any]]:
        """获取强化学习数据（环境客户端）"""
        source_name = "reinforcement_source"

        if source_name not in self.data_loaders:
            raise ValueError(
                f"强化学习模式需要定义 '{source_name}' 数据源"
            )

        # 强化学习通常使用 client 而不是 train
        if "client" in self.data_loaders[source_name]:
            client = self.get_data_loader(source_name, "client")
        else:
            client = self.get_data_loader(source_name, "train")

        return client, None

    def _get_semi_supervised_data(self) -> Tuple[Any, Optional[Any]]:
        """获取半监督学习数据"""
        source_name = "semi_supervised_source"

        if source_name not in self.data_loaders:
            raise ValueError(
                f"半监督学习模式需要定义 '{source_name}' 数据源"
            )

        # 半监督需要有标签和无标签数据
        if "labeled" not in self.data_loaders[source_name]:
            raise ValueError(f"半监督学习需要 'labeled' 数据")

        if "unlabeled" not in self.data_loaders[source_name]:
            raise ValueError(f"半监督学习需要 'unlabeled' 数据")

        labeled_loader = self.get_data_loader(source_name, "labeled")
        unlabeled_loader = self.get_data_loader(source_name, "unlabeled")

        # 返回一个组合的加载器
        combined_loader = {
            "labeled": labeled_loader,
            "unlabeled": unlabeled_loader
        }

        return combined_loader, None

    def _get_self_supervised_data(self) -> Tuple[Any, Optional[Any]]:
        """获取自监督学习数据"""
        source_name = "self_supervised_source"

        if source_name not in self.data_loaders:
            raise ValueError(
                f"自监督学习模式需要定义 '{source_name}' 数据源"
            )

        train_loader = self.get_data_loader(source_name, "train")

        return train_loader, None

    def _get_default_data(self) -> Tuple[Any, Optional[Any]]:
        """获取默认数据源"""
        # 尝试查找第一个可用的数据源
        if not self.data_loaders:
            raise ValueError("没有可用的数据源")

        # 获取第一个数据源
        first_source_name = list(self.data_loaders.keys())[0]
        first_source = self.data_loaders[first_source_name]

        # 尝试获取 train 分割
        if "train" in first_source:
            train_loader = first_source["train"]
        else:
            # 使用第一个可用的分割
            train_loader = list(first_source.values())[0]

        # 尝试获取 val 分割
        val_loader = first_source.get("val", None)

        return train_loader, val_loader

    def get_test_data(self) -> Optional[Any]:
        """
        获取测试数据

        返回:
            测试数据加载器或 None
        """
        # 尝试从当前训练模式的数据源获取测试集
        training_mode = self.context.training_mode
        source_name = f"{training_mode}_source"

        if source_name in self.data_loaders:
            if "test" in self.data_loaders[source_name]:
                return self.get_data_loader(source_name, "test")

        # 尝试从第一个数据源获取测试集
        for source_name, source_splits in self.data_loaders.items():
            if "test" in source_splits:
                return source_splits["test"]

        return None

    def validate_data_source(
        self,
        source_name: str,
        required_splits: List[str]
    ) -> bool:
        """
        验证数据源是否包含必需的分割

        参数:
            source_name: 数据源名称
            required_splits: 必需的分割列表

        返回:
            是否满足要求
        """
        if source_name not in self.data_loaders:
            return False

        source_splits = self.data_loaders[source_name]

        for split in required_splits:
            if split not in source_splits:
                return False

        return True

    def get_data_info(self, source_name: str, split_name: str) -> Dict[str, Any]:
        """
        获取数据加载器的信息

        参数:
            source_name: 数据源名称
            split_name: 分割名称

        返回:
            数据信息字典
        """
        loader = self.get_data_loader(source_name, split_name)

        info = {
            "source_name": source_name,
            "split_name": split_name,
            "loader_type": type(loader).__name__
        }

        # 尝试获取数据集大小（如果是 tf.data.Dataset）
        try:
            # 检查是否有 cardinality 方法
            if hasattr(loader, 'cardinality'):
                cardinality = int(loader.cardinality())
                if cardinality >= 0:
                    info["size"] = cardinality
        except:
            pass

        # 尝试获取元素规格（如果是 tf.data.Dataset）
        try:
            if hasattr(loader, 'element_spec'):
                info["element_spec"] = str(loader.element_spec)
        except:
            pass

        return info

    def prepare_batch_iterator(
        self,
        source_name: str,
        split_name: str = "train",
        repeat: bool = False,
        prefetch: bool = True
    ) -> Any:
        """
        准备批次迭代器

        参数:
            source_name: 数据源名称
            split_name: 分割名称
            repeat: 是否重复数据集
            prefetch: 是否预取数据

        返回:
            准备好的数据集迭代器
        """
        loader = self.get_data_loader(source_name, split_name)

        # 如果是 tf.data.Dataset，应用优化
        if hasattr(loader, 'repeat') and hasattr(loader, 'prefetch'):
            if repeat:
                loader = loader.repeat()

            if prefetch:
                # 使用 call_target 调用 AUTOTUNE
                autotune = call_target(
                    reflection="tensorflow.data:AUTOTUNE",
                    args={}
                )
                loader = loader.prefetch(autotune)

        return loader

    def create_combined_loader(
        self,
        loaders: List[Tuple[str, str]],
        strategy: str = "zip"
    ) -> Any:
        """
        创建组合数据加载器

        参数:
            loaders: 数据加载器列表 [(source_name, split_name), ...]
            strategy: 组合策略 ('zip', 'concatenate', 'interleave')

        返回:
            组合后的数据加载器
        """
        if not loaders:
            raise ValueError("loaders 列表不能为空")

        # 获取所有数据加载器
        loader_instances = []
        for source_name, split_name in loaders:
            loader = self.get_data_loader(source_name, split_name)
            loader_instances.append(loader)

        # 根据策略组合
        if strategy == "zip":
            # 使用 call_target 调用 tf.data.Dataset.zip
            combined = call_target(
                reflection="tensorflow.data.Dataset:zip",
                args={"datasets": tuple(loader_instances)}
            )

        elif strategy == "concatenate":
            # 使用第一个数据集的 concatenate 方法
            combined = loader_instances[0]
            for loader in loader_instances[1:]:
                combined = combined.concatenate(loader)

        elif strategy == "interleave":
            # 交错采样（需要更复杂的实现）
            raise NotImplementedError("interleave 策略尚未实现")

        else:
            raise ValueError(f"不支持的组合策略: {strategy}")

        return combined

    def get_batch_sample(
        self,
        source_name: str,
        split_name: str = "train",
        num_batches: int = 1
    ) -> List[Any]:
        """
        获取数据批次样本（用于调试和验证）

        参数:
            source_name: 数据源名称
            split_name: 分割名称
            num_batches: 获取的批次数量

        返回:
            批次样本列表
        """
        loader = self.get_data_loader(source_name, split_name)

        samples = []

        # 如果是 tf.data.Dataset，使用 take
        if hasattr(loader, 'take'):
            for batch in loader.take(num_batches):
                samples.append(batch)
        else:
            # 尝试直接迭代
            iterator = iter(loader)
            for _ in range(num_batches):
                try:
                    batch = next(iterator)
                    samples.append(batch)
                except StopIteration:
                    break

        return samples


def build_data_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有数据源的便捷函数

    参数:
        context: 训练上下文

    返回:
        数据加载器字典
    """
    manager = DataManager(context)
    return manager.build_all_data_sources()


def get_training_data_from_context(
    context: TrainContext
) -> Tuple[Any, Optional[Any]]:
    """
    从训练上下文获取训练数据的便捷函数

    参数:
        context: 训练上下文

    返回:
        (训练数据加载器, 验证数据加载器)
    """
    manager = DataManager(context)
    manager.build_all_data_sources()
    return manager.get_training_data()


def create_data_manager(context: TrainContext) -> DataManager:
    """
    创建并初始化数据管理器的便捷函数

    参数:
        context: 训练上下文

    返回:
        DataManager 实例
    """
    manager = DataManager(context)
    manager.build_all_data_sources()
    return manager
