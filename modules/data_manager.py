# -*- coding: utf-8 -*-
"""
data_manager.py - 数据管理
功能：
  - 根据配置文件生成数据加载器
  - 支持监督、无监督、强化学习等各种数据源
  - 自适应不同的训练方式
"""

from typing import Dict, Any, Optional, Iterator, Tuple
import os
from utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class DataManager:
    """
    数据管理器：
    - 根据训练方式选择对应的数据源
    - 生成数据加载器
    - 管理数据的流向
    """

    def __init__(self, config: Dict[str, Any], training_mode: str):
        """
        初始化数据管理器

        参数：
            config: 完整的配置文件
            training_mode: 训练方式（supervised/reinforcement/unsupervised等）
        """
        self.config = config
        self.training_mode = training_mode
        logger.info(f"初始化数据管理器: {training_mode}")

    def build_all(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建所有数据加载器

        参数：
            data_config: 配置文件中的data_manager部分
        返回：
            {数据集名: 数据加载器实例}
        """
        data_loaders = {}

        if not data_config:
            logger.warning("数据配置为空")
            return data_loaders

        # 根据训练方式选择对应的数据源
        source_key = self._get_source_key()

        if source_key not in data_config:
            logger.warning(f"未找到数据源: {source_key}")
            return data_loaders

        source_cfg = data_config[source_key]

        try:
            data_loaders = self._build_source(source_key, source_cfg)
            logger.info(f"数据加载器构建成功: {list(data_loaders.keys())}")
        except Exception as e:
            logger.error(f"数据加载器构建失败: {str(e)}", exc_info=True)
            raise

        return data_loaders

    def _get_source_key(self) -> str:
        """
        根据训练方式获取数据源键名

        返回：
            数据源键名
        """
        mapping = {
            "supervised": "supervised_source",
            "reinforcement": "rl_source",
            "unsupervised_clustering": "unsupervised_source",
            "unsupervised_autoencoder": "unsupervised_source",
            "self_supervised": "self_supervised_source",
            "semi_supervised_pseudo_labeling": "supervised_source",
            "multi_task": "supervised_source",
            "transfer": "supervised_source",
            "online": "rl_source",
        }

        return mapping.get(self.training_mode, "supervised_source")

    def _build_source(self, source_key: str, source_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建数据源

        参数：
            source_key: 数据源键名
            source_cfg: 数据源配置
        返回：
            {数据集名: 加载器实例}
        """
        if source_key == "supervised_source":
            return self._build_supervised_source(source_cfg)
        elif source_key == "rl_source":
            return self._build_rl_source(source_cfg)
        elif source_key == "unsupervised_source":
            return self._build_unsupervised_source(source_cfg)
        elif source_key == "self_supervised_source":
            return self._build_self_supervised_source(source_cfg)
        else:
            raise ValueError(f"未知的数据源类型: {source_key}")

    def _build_supervised_source(self, source_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建监督学习数据源（train/val/test）

        参数：
            source_cfg: 监督学习数据源配置
        返回：
            {数据集名: 加载器实例}
        """
        data_loaders = {}

        for split_name in ["train", "val", "test"]:
            if split_name not in source_cfg:
                logger.debug(f"未找到{split_name}数据源配置")
                continue

            split_cfg = source_cfg[split_name]
            logger.info(f"构建 {split_name} 数据加载器")

            try:
                loader = self._build_single_loader(split_cfg)
                data_loaders[split_name] = loader
            except Exception as e:
                logger.error(f"构建 {split_name} 加载器失败: {str(e)}", exc_info=True)
                raise

        if not data_loaders:
            logger.warning("未构建任何监督学习数据加载器")

        return data_loaders

    def _build_rl_source(self, source_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建强化学习数据源（环境、请求、反馈）

        参数：
            source_cfg: 强化学习数据源配置
        返回：
            {数据源名: 实例}
        """
        data_loaders = {}

        # 构建环境
        if "environment" in source_cfg:
            env_cfg = source_cfg["environment"]
            logger.info("构建强化学习环境")
            try:
                env = call_target(
                    env_cfg.get("reflection"),
                    env_cfg.get("args", {})
                )
                data_loaders["environment"] = env
                logger.debug("环境构建成功")
            except Exception as e:
                logger.warning(f"构建环境失败: {str(e)}")

        # 构建请求客户端
        if "request" in source_cfg:
            request_cfg = source_cfg["request"]
            logger.info("构建请求客户端")
            try:
                request_client = call_target(
                    request_cfg.get("reflection"),
                    request_cfg.get("args", {})
                )
                data_loaders["request"] = request_client
                logger.debug("请求客户端构建成功")
            except Exception as e:
                logger.warning(f"构建请求客户端失败: {str(e)}")

        # 构建反馈客户端
        if "feedback" in source_cfg:
            feedback_cfg = source_cfg["feedback"]
            logger.info("构建反馈客户端")
            try:
                feedback_client = call_target(
                    feedback_cfg.get("reflection"),
                    feedback_cfg.get("args", {})
                )
                data_loaders["feedback"] = feedback_client
                logger.debug("反馈客户端构建成功")
            except Exception as e:
                logger.warning(f"构建反馈客户端失败: {str(e)}")

        return data_loaders

    def _build_unsupervised_source(self, source_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建无监督学习数据源

        参数：
            source_cfg: 无监督学习数据源配置
        返回：
            {数据源名: 加载器实例}
        """
        logger.info("构建无监督数据加载器")

        try:
            loader = self._build_single_loader(source_cfg)
            return {"unsupervised": loader}
        except Exception as e:
            logger.error(f"构建无监督加载器失败: {str(e)}", exc_info=True)
            raise

    def _build_self_supervised_source(self, source_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建自监督学习数据源

        参数：
            source_cfg: 自监督学习数据源配置
        返回：
            {数据源名: 加载器实例}
        """
        logger.info("构建自监督数据加载器")

        try:
            loader = self._build_single_loader(source_cfg)
            return {"self_supervised": loader}
        except Exception as e:
            logger.error(f"构建自监督加载器失败: {str(e)}", exc_info=True)
            raise

    def _build_single_loader(self, loader_cfg: Dict[str, Any]) -> Any:
        """
        构建单个数据加载器

        参数：
            loader_cfg: 加载器配置
                {
                    "reflection": "modules.utils.build_csv_loader",
                    "args": {...}
                }
        返回：
            数据加载器实例
        """
        reflection = loader_cfg.get("reflection")
        args = loader_cfg.get("args", {})

        if not reflection:
            raise ValueError("加载器配置缺少reflection字段")

        logger.debug(f"创建数据加载器: {reflection}")
        logger.debug(f"加载器参数: {args}")

        try:
            # 通过反射调用数据加载函数
            loader = call_target(reflection, args)
            logger.debug("数据加载器创建成功")
            return loader
        except Exception as e:
            logger.error(f"创建数据加载器失败: {str(e)}", exc_info=True)
            raise


class DataBatchIterator:
    """
    数据批次迭代器：
    - 管理batch的获取
    - 支持shuffle
    - 支持repeat
    """

    def __init__(self, dataset: Any, batch_size: int = 32):
        """
        初始化迭代器

        参数：
            dataset: tf.data.Dataset实例或其他可迭代对象
            batch_size: 批大小
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_batch = 0
        logger.debug(f"数据迭代器初始化: batch_size={batch_size}")

    def __iter__(self) -> Iterator:
        """迭代器协议"""
        try:
            return iter(self.dataset)
        except TypeError:
            logger.error("数据集不可迭代")
            raise

    def __len__(self):
        """获取总batch数"""
        try:
            # 尝试计算总样本数
            total_samples = len(self.dataset)
            return (total_samples + self.batch_size - 1) // self.batch_size
        except:
            logger.debug("无法计算总batch数")
            return None

    def get_next_batch(self):
        """
        获取下一个batch

        返回：
            batch数据，如果数据集结束返回None
        """
        try:
            batch = next(iter(self.dataset))
            self.current_batch += 1
            return batch
        except StopIteration:
            logger.debug("数据集遍历完成")
            return None
        except Exception as e:
            logger.error(f"获取batch失败: {str(e)}")
            return None

    def reset(self):
        """重置迭代器"""
        self.current_batch = 0
        logger.debug("迭代器已重置")


class DataPreprocessor:
    """
    数据预处理器：
    - 数据标准化
    - 数据增强
    - 特征提取
    """

    @staticmethod
    def normalize(data: Any, mean: float = 0.0, std: float = 1.0) -> Any:
        """
        数据标准化（Z-score标准化）

        参数：
            data: 输入数据
            mean: 均值
            std: 标准差
        返回：
            标准化后的数据
        """
        logger.debug(f"执行数据标准化: mean={mean}, std={std}")

        try:
            import tensorflow as tf
            return (data - mean) / (std + 1e-7)
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise

    @staticmethod
    def normalize_to_range(data: Any, min_val: float = 0.0, max_val: float = 1.0) -> Any:
        """
        数据归一化到指定范围

        参数：
            data: 输入数据
            min_val: 最小值
            max_val: 最大值
        返回：
            归一化后的数据
        """
        logger.debug(f"执行数据归一化到 [{min_val}, {max_val}]")

        try:
            import tensorflow as tf
            data_min = tf.reduce_min(data)
            data_max = tf.reduce_max(data)
            normalized = (data - data_min) / (data_max - data_min + 1e-7)
            return normalized * (max_val - min_val) + min_val
        except Exception as e:
            logger.error(f"数据归一化失败: {str(e)}")
            raise

    @staticmethod
    def augment_data(data: Any, transforms: list = None) -> Any:
        """
        数据增强

        参数：
            data: 输入数据
            transforms: 增强操作列表
        返回：
            增强后的数据
        """
        if transforms is None or len(transforms) == 0:
            logger.debug("未指定数据增强操作")
            return data

        logger.debug(f"执行数据增强: {len(transforms)}个操作")

        augmented = data
        for idx, transform in enumerate(transforms):
            try:
                reflection = transform.get("reflection")
                args = transform.get("args", {})

                if not reflection:
                    logger.warning(f"第{idx}个增强操作缺少reflection字段")
                    continue

                logger.debug(f"执行第{idx}个增强操作: {reflection}")

                # 调用增强函数
                augmented = call_target(reflection, {**args, "data": augmented})
                logger.debug(f"第{idx}个增强操作完成")
            except Exception as e:
                logger.warning(f"第{idx}个增强操作失败: {str(e)}")
                continue

        return augmented


class DataCache:
    """
    数据缓存管理器：
    - 缓存已加载的数据
    - 减少重复加载
    """

    def __init__(self, max_cache_size: int = 10):
        """
        初始化缓存管理器

        参数：
            max_cache_size: 最大缓存数量
        """
        self.cache = {}
        self.max_cache_size = max_cache_size
        logger.debug(f"数据缓存初始化: 最大缓存数={max_cache_size}")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        参数：
            key: 缓存键
        返回：
            缓存数据，如果不存在返回None
        """
        return self.cache.get(key)

    def put(self, key: str, data: Any):
        """
        存储缓存数据

        参数：
            key: 缓存键
            data: 数据
        """
        if len(self.cache) >= self.max_cache_size:
            # 移除最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"缓存满，移除最旧的缓存: {oldest_key}")

        self.cache[key] = data
        logger.debug(f"数据已缓存: {key}")

    def clear(self):
        """清空所有缓存"""
        self.cache.clear()
        logger.debug("缓存已清空")
