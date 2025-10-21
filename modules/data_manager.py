# -*- coding: utf-8 -*-
"""
modules/data_manager.py
数据管理器：支持静态加载和动态加载两种方式

设计原则：
1. 静态加载：组织数据处理流程，使用官方库（pandas/sklearn）
2. 动态加载：实时从环境获取数据（强化学习）
3. DataManager仅组织流程，不实现具体功能

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用官方库函数
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from common.train_context import TrainContext, DataSourceConfig
from common.common import call_target, LoggerManager
from common.interfaces import DataManagerInterface

logger = LoggerManager.get_logger(__file__)

class DataManager(DataManagerInterface):
    """
    数据管理器 V3

    支持两种数据加载方式：
    1. 静态加载 (Static Loading)：
       - 一次性加载所有数据
       - 执行完整的数据处理流程
       - 适用于监督学习、无监督学习

    2. 动态加载 (Dynamic Loading)：
       - 实时从环境/API获取数据
       - 不需要预处理流程
       - 适用于强化学习、在线学习

    职责：
    - 组织数据处理流程（不实现具体功能）
    - 使用 call_target 调用官方库函数
    - 管理数据状态和缓存
    """

    def __init__(self, context: TrainContext):
        """初始化数据管理器"""
        super().__init__(context)
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.static_data: Dict[str, Any] = {}
        self.dynamic_clients: Dict[str, Any] = {}
        self.preprocessing_params: Dict[str, Any] = {}

    def initialize(self) -> None:
        """从 TrainContext 初始化数据管理器"""
        if not self.context.data_sources:
            raise ValueError("TrainContext 中没有定义任何数据源")
        self._initialized = True

    def validate_data_sources(self) -> bool:
        """验证数据源配置"""
        for source_name, source_config in self.context.data_sources.items():
            if not isinstance(source_config, dict):
                raise ValueError(f"数据源 '{source_name}' 配置格式错误")

            # 检查是否有 loading_mode
            if 'loading_mode' not in source_config:
                raise ValueError(f"数据源 '{source_name}' 缺少 loading_mode 字段")

            loading_mode = source_config['loading_mode']

            if loading_mode == 'static':
                # 静态加载必须有 load, split, dataset
                if 'load' not in source_config:
                    raise ValueError(f"静态加载数据源 '{source_name}' 缺少 load 配置")
                if 'split' not in source_config:
                    raise ValueError(f"静态加载数据源 '{source_name}' 缺少 split 配置")
                if 'dataset' not in source_config:
                    raise ValueError(f"静态加载数据源 '{source_name}' 缺少 dataset 配置")

            elif loading_mode == 'dynamic':
                # 动态加载必须有 client
                if 'client' not in source_config:
                    raise ValueError(f"动态加载数据源 '{source_name}' 缺少 client 配置")

            else:
                raise ValueError(
                    f"数据源 '{source_name}' 的 loading_mode 必须是 'static' 或 'dynamic'"
                )

        return True

    # ========================================================================
    # 静态加载方式 (Static Loading)
    # ========================================================================

    def build_static_data_source(
        self,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建静态数据源

        执行流程（所有步骤可选，除了 load/split/dataset）：
        1. load - 加载数据集（必须）
        2. clean - 数据清洗（可选，配置为空则跳过）
        3. engineer - 特征工程（可选，配置为空则跳过）
        4. normalize - 归一化（可选，配置为空则跳过）
        5. standardize - 标准化（可选，配置为空则跳过）
        6. split - 数据集分割（必须）
        7. dataset - TensorFlow Dataset 转换（必须）

        参数:
            source_name: 数据源名称
            source_config: 数据源配置

        返回:
            处理后的数据 {"train": tf.data.Dataset, "val": ..., "test": ...}
        """
        # Step 1: 加载数据集（必须）
        data = self._execute_load(source_config['load'])

        # Step 2: 数据清洗（可选）
        clean_config = source_config.get('clean')
        if clean_config and self._is_config_valid(clean_config):
            data = self._execute_clean(data, clean_config)

        # Step 3: 特征工程（可选）
        engineer_config = source_config.get('engineer')
        if engineer_config and self._is_config_valid(engineer_config):
            data = self._execute_engineer(data, engineer_config)

        # Step 4: 归一化（可选）
        normalize_config = source_config.get('normalize')
        if normalize_config and self._is_config_valid(normalize_config):
            data, norm_params = self._execute_normalize(data, normalize_config)
            self.preprocessing_params[f"{source_name}_normalize"] = norm_params

        # Step 5: 标准化（可选）
        standardize_config = source_config.get('standardize')
        if standardize_config and self._is_config_valid(standardize_config):
            data, std_params = self._execute_standardize(data, standardize_config)
            self.preprocessing_params[f"{source_name}_standardize"] = std_params

        # Step 6: 数据集分割（必须）
        splits = self._execute_split(data, source_config['split'])

        # Step 7: 转换为 TensorFlow Dataset（必须）
        dataset_config = source_config.get('dataset', {})
        if not dataset_config:
            # 如果没有配置，使用默认配置
            dataset_config = self._get_default_dataset_config()

        result = self._execute_output(splits, dataset_config)

        # 缓存数据
        self.static_data[source_name] = result

        return result

    def _is_config_valid(self, config: Any) -> bool:
        """
        检查配置是否有效

        参数:
            config: 配置对象

        返回:
            是否有效（非空且有实际内容）
        """
        if config is None:
            return False

        if isinstance(config, dict):
            # 字典必须非空
            if not config:
                return False

            # 如果有 operations 字段，检查是否为空
            if 'operations' in config:
                operations = config.get('operations')
                if not operations or len(operations) == 0:
                    return False

            # 如果有 reflection 字段，检查是否为空
            if 'reflection' in config:
                reflection = config.get('reflection')
                if not reflection:
                    return False

            return True

        return False

    def _get_default_dataset_config(self) -> Dict[str, Any]:
        """
        获取默认的 TensorFlow Dataset 配置

        返回:
            默认配置
        """
        return {
            'label_column': 'label',
            'train': {
                'batch_size': 32,
                'shuffle': True,
                'shuffle_buffer_size': 10000,
                'repeat': False,
                'prefetch_size': 2,
                'cache': False
            },
            'val': {
                'batch_size': 32,
                'shuffle': False,
                'repeat': False,
                'prefetch_size': 2,
                'cache': False
            },
            'test': {
                'batch_size': 32,
                'shuffle': False,
                'repeat': False,
                'prefetch_size': 1,
                'cache': False
            }
        }

    def _execute_load(self, load_config: Dict[str, Any]) -> Any:
        """
        执行数据加载

        示例配置:
        {
            "reflection": "pandas:read_csv",
            "args": {"filepath_or_buffer": "data.csv"}
        }
        """
        reflection = load_config.get('reflection')
        args = load_config.get('args', {})

        if not reflection:
            raise ValueError("load 配置缺少 reflection 字段")

        # 使用 call_target 调用官方库
        data = call_target(reflection=reflection, args=args)

        return data

    def _execute_clean(
        self,
        data: Any,
        clean_config: Dict[str, Any]
    ) -> Any:
        """
        执行数据清洗

        示例配置:
        {
            "operations": [
                {"reflection": "pandas.DataFrame:dropna", "args": {...}},
                {"reflection": "pandas.DataFrame:fillna", "args": {...}}
            ]
        }
        """
        operations = clean_config.get('operations', [])

        current_data = data

        for operation in operations:
            reflection = operation.get('reflection')
            args = operation.get('args', {})

            if not reflection:
                continue

            # 处理实例方法调用（如 DataFrame.dropna）
            if '.' in reflection and ':' in reflection:
                # 格式: "pandas.DataFrame:dropna"
                # 表示调用 data.dropna(**args)
                module_class, method = reflection.split(':')

                # 获取方法
                method_obj = getattr(current_data, method, None)

                if method_obj is None:
                    raise AttributeError(
                        f"对象没有方法 '{method}'"
                    )

                # 调用方法
                current_data = method_obj(**args)

            else:
                # 格式: "module:function"
                # 表示调用 function(data, **args)
                args['data'] = current_data
                current_data = call_target(reflection=reflection, args=args)

        return current_data

    def _execute_engineer(
        self,
        data: Any,
        engineer_config: Dict[str, Any]
    ) -> Any:
        """
        执行特征工程

        示例配置:
        {
            "operations": [
                {
                    "reflection": "pandas:get_dummies",
                    "args": {"columns": ["category"]}
                },
                {
                    "reflection": "sklearn.preprocessing:PolynomialFeatures",
                    "args": {"degree": 2}
                }
            ]
        }
        """
        operations = engineer_config.get('operations', [])

        current_data = data

        for operation in operations:
            reflection = operation.get('reflection')
            args = operation.get('args', {})

            if not reflection:
                continue

            # 特殊处理：sklearn的transformer需要fit_transform
            if 'sklearn.preprocessing' in reflection:
                # 获取列配置
                columns = args.pop('columns', None)

                # 创建transformer实例
                transformer = call_target(reflection=reflection, args=args)

                # 选择要处理的列
                if columns:
                    # 只处理指定列
                    if hasattr(current_data, 'loc'):
                        # pandas DataFrame
                        import pandas as pd
                        transformed = transformer.fit_transform(
                            current_data[columns]
                        )

                        # 替换原始列
                        for i, col in enumerate(columns):
                            if transformed.shape[1] > len(columns):
                                # 多项式特征生成了更多列
                                current_data[f"{col}_poly{i}"] = transformed[:, i]
                            else:
                                current_data[col] = transformed[:, i]
                    else:
                        # numpy array
                        current_data = transformer.fit_transform(current_data)
                else:
                    # 处理所有数值列
                    current_data = transformer.fit_transform(current_data)

                # 保存transformer用于推理
                transformer_name = reflection.split(':')[-1]
                self.preprocessing_params[transformer_name] = transformer

            elif 'pandas:get_dummies' in reflection:
                # pandas独热编码
                args['data'] = current_data
                current_data = call_target(reflection=reflection, args=args)

            elif 'pandas:cut' in reflection:
                # pandas分箱
                # 需要特殊处理返回值
                column_name = args.pop('column', None)
                if column_name and hasattr(current_data, 'loc'):
                    args['x'] = current_data[column_name]
                    binned = call_target(reflection=reflection, args=args)
                    current_data[f"{column_name}_binned"] = binned
                else:
                    current_data = call_target(reflection=reflection, args=args)

            else:
                # 通用调用
                if hasattr(current_data, 'loc'):
                    # DataFrame方法调用
                    method_name = reflection.split(':')[-1]
                    method = getattr(current_data, method_name, None)
                    if method:
                        current_data = method(**args)
                    else:
                        args['data'] = current_data
                        current_data = call_target(reflection=reflection, args=args)
                else:
                    args['data'] = current_data
                    current_data = call_target(reflection=reflection, args=args)

        return current_data

    def _execute_normalize(
        self,
        data: Any,
        normalize_config: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """
        执行归一化

        示例配置:
        {
            "reflection": "sklearn.preprocessing:MinMaxScaler",
            "args": {"feature_range": [0, 1], "columns": ["x1", "x2"]}
        }

        返回:
            (归一化后的数据, scaler对象)
        """
        reflection = normalize_config.get('reflection')
        args = normalize_config.get('args', {})

        # 提取列配置
        columns = args.pop('columns', None)

        # 创建scaler
        scaler = call_target(reflection=reflection, args=args)

        # 应用归一化
        if columns and hasattr(data, 'loc'):
            # pandas DataFrame，只处理指定列
            import pandas as pd
            data_copy = data.copy()
            data_copy[columns] = scaler.fit_transform(data[columns])
            normalized_data = data_copy
        else:
            # numpy array或所有列
            normalized_data = scaler.fit_transform(data)

        return normalized_data, scaler

    def _execute_standardize(
        self,
        data: Any,
        standardize_config: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """
        执行标准化

        示例配置:
        {
            "reflection": "sklearn.preprocessing:StandardScaler",
            "args": {"with_mean": true, "with_std": true, "columns": ["x1"]}
        }

        返回:
            (标准化后的数据, scaler对象)
        """
        reflection = standardize_config.get('reflection')
        args = standardize_config.get('args', {})

        # 提取列配置
        columns = args.pop('columns', None)

        # 创建scaler
        scaler = call_target(reflection=reflection, args=args)

        # 应用标准化
        if columns and hasattr(data, 'loc'):
            # pandas DataFrame，只处理指定列
            import pandas as pd
            data_copy = data.copy()
            data_copy[columns] = scaler.fit_transform(data[columns])
            standardized_data = data_copy
        else:
            # numpy array或所有列
            standardized_data = scaler.fit_transform(data)

        return standardized_data, scaler

    def _execute_split(
        self,
        data: Any,
        split_config: Dict[str, Any]
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        执行数据集分割

        示例配置:
        {
            "reflection": "sklearn.model_selection:train_test_split",
            "args": {
                "test_size": 0.3,
                "random_state": 42,
                "stratify": null
            },
            "validation_split": {
                "enabled": true,
                "reflection": "sklearn.model_selection:train_test_split",
                "args": {"test_size": 0.2}
            }
        }

        返回:
            {"train": (X_train, y_train),
             "val": (X_val, y_val),
             "test": (X_test, y_test)}
        """
        reflection = split_config.get('reflection')
        args = split_config.get('args', {})

        if not reflection:
            raise ValueError("split 配置缺少 reflection 字段")

        # 准备数据和标签
        if hasattr(data, 'loc'):
            # pandas DataFrame
            # 假设最后一列是标签（或通过配置指定）
            feature_cols = list(data.columns[:-1])
            label_col = data.columns[-1]

            X = data[feature_cols].values
            y = data[label_col].values
        else:
            # numpy array
            # 假设最后一列是标签
            X = data[:, :-1]
            y = data[:, -1]

        # 处理stratify参数
        stratify_value = args.get('stratify')
        if stratify_value:
            args['stratify'] = y

        # 第一次分割：train+val vs test
        args_copy = args.copy()
        args_copy.update({'arrays': [X, y]})

        split_result = call_target(reflection=reflection, args=args_copy)

        X_trainval, X_test, y_trainval, y_test = split_result

        # 第二次分割：train vs val (如果配置了)
        validation_split = split_config.get('validation_split', {})

        if validation_split.get('enabled', False):
            val_reflection = validation_split.get('reflection')
            val_args = validation_split.get('args', {})

            # 处理stratify
            val_stratify = val_args.get('stratify')
            if val_stratify:
                val_args['stratify'] = y_trainval

            val_args['arrays'] = [X_trainval, y_trainval]

            val_split_result = call_target(
                reflection=val_reflection,
                args=val_args
            )

            X_train, X_val, y_train, y_val = val_split_result

            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
        else:
            # 没有验证集
            return {
                'train': (X_trainval, y_trainval),
                'test': (X_test, y_test)
            }

    def _execute_output(
        self,
        splits: Dict[str, Tuple[Any, Any]],
        dataset_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行 TensorFlow Dataset 转换

        参数:
            splits: 分割后的数据 {"train": (X, y), "val": (X, y), ...}
            dataset_config: TensorFlow Dataset 配置

        返回:
            {"train": tf.data.Dataset, "val": tf.data.Dataset, ...}
        """
        result = {}

        # 获取标签列配置
        label_column = dataset_config.get('label_column')
        has_labels = label_column is not None

        for split_name, (X, y) in splits.items():
            # 获取该分割的配置
            split_config = dataset_config.get(split_name, {})

            if not split_config:
                # 如果没有配置，使用默认配置
                split_config = {
                    'batch_size': 32,
                    'shuffle': split_name == 'train',
                    'shuffle_buffer_size': 10000,
                    'repeat': False,
                    'prefetch_size': 2,
                    'cache': False
                }

            # 创建 tf.data.Dataset
            if has_labels:
                # 有标签的情况
                dataset = call_target(
                    reflection="tensorflow.data.Dataset:from_tensor_slices",
                    args={'tensors': (X, y)}
                )
            else:
                # 无标签的情况（无监督学习）
                dataset = call_target(
                    reflection="tensorflow.data.Dataset:from_tensor_slices",
                    args={'tensors': X}
                )

            # 应用配置

            # 1. Cache（可选，加速训练）
            if split_config.get('cache', False):
                dataset = dataset.cache()

            # 2. Shuffle（训练集打乱）
            if split_config.get('shuffle', False):
                shuffle_buffer = split_config.get('shuffle_buffer_size', 10000)
                dataset = dataset.shuffle(buffer_size=shuffle_buffer)

            # 3. Repeat（可选，无限重复）
            if split_config.get('repeat', False):
                dataset = dataset.repeat()

            # 4. Batch（批处理）
            batch_size = split_config.get('batch_size', 32)
            dataset = dataset.batch(batch_size)

            # 5. Prefetch（预取，提高性能）
            prefetch_size = split_config.get('prefetch_size', 2)
            if prefetch_size == -1:
                # 自动调优
                dataset = dataset.prefetch(
                    buffer_size=call_target(
                        reflection="tensorflow.data.AUTOTUNE",
                        args={}
                    )
                )
            else:
                dataset = dataset.prefetch(buffer_size=prefetch_size)

            result[split_name] = dataset

        return result

    # ========================================================================
    # 动态加载方式 (Dynamic Loading)
    # ========================================================================

    def build_dynamic_data_source(
        self,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> Any:
        """
        构建动态数据源（强化学习环境）

        参数:
            source_name: 数据源名称
            source_config: 数据源配置

        返回:
            环境客户端实例
        """
        client_config = source_config.get('client')

        if not client_config:
            raise ValueError(f"动态数据源 '{source_name}' 缺少 client 配置")

        reflection = client_config.get('reflection')
        args = client_config.get('args', {})

        if not reflection:
            raise ValueError(f"动态数据源 '{source_name}' 的 client 缺少 reflection")

        # 创建环境客户端
        client = call_target(reflection=reflection, args=args)

        # 缓存客户端
        self.dynamic_clients[source_name] = client

        return client

    # ========================================================================
    # 接口实现
    # ========================================================================

    def build_data_source(
        self,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> Any:
        """
        构建单个数据源

        根据 loading_mode 选择构建方式
        """
        loading_mode = source_config.get('loading_mode', 'static')

        if loading_mode == 'static':
            return self.build_static_data_source(source_name, source_config)
        elif loading_mode == 'dynamic':
            return self.build_dynamic_data_source(source_name, source_config)
        else:
            raise ValueError(f"不支持的加载模式: {loading_mode}")

    def build_all_data_sources(self) -> Dict[str, Any]:
        """
        构建所有数据源

        返回:
            所有数据源 {source_name: data/client}
        """
        all_sources = {}

        for source_name, source_config in self.context.data_sources.items():
            try:
                source_data = self.build_data_source(source_name, source_config)
                all_sources[source_name] = source_data
            except Exception as e:
                raise RuntimeError(
                    f"构建数据源 '{source_name}' 失败: {e}"
                ) from e

        return all_sources

    def finalize(self) -> None:
        """将数据缓存到上下文"""
        # 缓存静态数据
        if self.static_data:
            self.context.container.dataloaders = self.static_data

        # 缓存动态客户端
        if self.dynamic_clients:
            if not hasattr(self.context.container, 'env_clients'):
                self.context.container.env_clients = {}
            self.context.container.env_clients.update(self.dynamic_clients)

        # 缓存预处理参数
        if self.preprocessing_params:
            if 'preprocessing' not in self.context.variables:
                self.context.variables['preprocessing'] = {}
            self.context.variables['preprocessing'].update(
                self.preprocessing_params
            )

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_static_data(
        self,
        source_name: str,
        split_name: str = 'train'
    ) -> Any:
        """
        获取静态数据（TensorFlow Dataset）

        参数:
            source_name: 数据源名称
            split_name: 数据集分割名称 (train/val/test)

        返回:
            tf.data.Dataset
        """
        if source_name not in self.static_data:
            raise ValueError(f"数据源 '{source_name}' 不存在")

        if split_name not in self.static_data[source_name]:
            raise ValueError(
                f"数据源 '{source_name}' 中没有 '{split_name}' 分割"
            )

        return self.static_data[source_name][split_name]

    def get_dynamic_client(self, source_name: str) -> Any:
        """
        获取动态客户端

        参数:
            source_name: 数据源名称

        返回:
            环境客户端
        """
        if source_name not in self.dynamic_clients:
            raise ValueError(f"动态客户端 '{source_name}' 不存在")

        return self.dynamic_clients[source_name]

    def get_preprocessing_params(
        self,
        source_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取预处理参数

        参数:
            source_name: 数据源名称（可选，None返回所有）

        返回:
            预处理参数字典
        """
        if source_name:
            return {
                k: v for k, v in self.preprocessing_params.items()
                if k.startswith(source_name)
            }
        return self.preprocessing_params.copy()

    def save_preprocessing_params(self, file_path: str) -> None:
        """
        保存预处理参数到文件

        参数:
            file_path: 保存路径
        """
        call_target(
            reflection="common.utils:save_pickle",
            args={
                'obj': self.preprocessing_params,
                'file_path': file_path
            }
        )

    def load_preprocessing_params(self, file_path: str) -> None:
        """
        从文件加载预处理参数

        参数:
            file_path: 文件路径
        """
        self.preprocessing_params = call_target(
            reflection="common.utils:load_pickle",
            args={'file_path': file_path}
        )

    def apply_preprocessing(
        self,
        data: Any,
        source_name: str,
        operations: Optional[List[str]] = None
    ) -> Any:
        """
        应用已保存的预处理参数

        参数:
            data: 待处理数据
            source_name: 数据源名称
            operations: 要应用的操作列表 (可选)

        返回:
            处理后的数据
        """
        if operations is None:
            operations = ['normalize', 'standardize']

        processed_data = data

        for operation in operations:
            param_key = f"{source_name}_{operation}"

            if param_key in self.preprocessing_params:
                scaler = self.preprocessing_params[param_key]

                # 应用转换
                if hasattr(scaler, 'transform'):
                    processed_data = scaler.transform(processed_data)

        return processed_data

    def get_data_info(self, source_name: str) -> Dict[str, Any]:
        """
        获取数据源信息

        参数:
            source_name: 数据源名称

        返回:
            数据信息字典
        """
        info = {
            'source_name': source_name,
            'loading_mode': None,
            'splits': [],
            'dataset_type': None,
            'batch_sizes': {},
        }

        source_config = self.context.data_sources.get(source_name)

        if source_config:
            info['loading_mode'] = source_config.get('loading_mode')

        if source_name in self.static_data:
            # 静态数据信息 (tf.data.Dataset)
            data = self.static_data[source_name]
            info['splits'] = list(data.keys())
            info['dataset_type'] = 'tf.data.Dataset'

            for split_name, dataset in data.items():
                # 获取 batch size（从配置）
                dataset_config = source_config.get('dataset', {})
                split_config = dataset_config.get(split_name, {})
                info['batch_sizes'][split_name] = split_config.get('batch_size', 'unknown')

        elif source_name in self.dynamic_clients:
            # 动态客户端信息
            client = self.dynamic_clients[source_name]
            info['client_type'] = type(client).__name__
            info['dataset_type'] = 'dynamic'

            if hasattr(client, 'state_space'):
                info['state_space'] = client.state_space

            if hasattr(client, 'action_space'):
                info['action_space'] = client.action_space

        return info


# ============================================================================
# 便捷函数
# ============================================================================

def build_data_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有数据源的便捷函数

    参数:
        context: 训练上下文

    返回:
        所有数据源
    """
    manager = DataManager(context)
    return manager.run()


def get_training_data(
    context: TrainContext,
    source_name: Optional[str] = None
) -> Tuple[Any, Optional[Any]]:
    """
    从上下文获取训练数据

    参数:
        context: 训练上下文
        source_name: 数据源名称（可选）

    返回:
        (训练数据 tf.data.Dataset, 验证数据 tf.data.Dataset)
    """
    manager = DataManager(context)
    manager.run()

    # 如果未指定数据源，根据训练模式选择
    if source_name is None:
        training_mode = context.training_mode

        if training_mode == 'supervised':
            source_name = 'supervised_source'
        elif training_mode == 'reinforcement':
            source_name = 'reinforcement_source'
        elif training_mode == 'unsupervised':
            source_name = 'unsupervised_source'
        else:
            # 使用第一个数据源
            if manager.static_data:
                source_name = next(iter(manager.static_data.keys()))
            elif manager.dynamic_clients:
                source_name = next(iter(manager.dynamic_clients.keys()))
            else:
                raise ValueError("没有可用的数据源")

    # 静态数据 (tf.data.Dataset)
    if source_name in manager.static_data:
        data = manager.static_data[source_name]
        train_data = data.get('train')
        val_data = data.get('val')
        return train_data, val_data

    # 动态客户端
    if source_name in manager.dynamic_clients:
        client = manager.dynamic_clients[source_name]
        return client, None

    raise ValueError(f"数据源 '{source_name}' 不存在")


def create_data_manager(context: TrainContext) -> DataManager:
    """
    创建并初始化数据管理器

    参数:
        context: 训练上下文

    返回:
        DataManager 实例
    """
    manager = DataManager(context)
    manager.run()
    return manager


def load_and_split_data(
    file_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42,
    context: Optional[TrainContext] = None
) -> Dict[str, Any]:
    """
    快速加载和分割数据的便捷函数

    参数:
        file_path: 数据文件路径
        test_size: 测试集比例
        val_size: 验证集比例
        batch_size: 批次大小
        random_state: 随机种子
        context: 训练上下文（可选）

    返回:
        {"train": tf.data.Dataset, "val": tf.data.Dataset, "test": tf.data.Dataset}
    """
    if context is None:
        from common.train_context import TrainContext
        context = TrainContext()

    # 构建临时配置
    context.data_sources = {
        'temp_source': {
            'loading_mode': 'static',
            'load': {
                'reflection': 'pandas:read_csv',
                'args': {'filepath_or_buffer': file_path}
            },
            'split': {
                'reflection': 'sklearn.model_selection:train_test_split',
                'args': {
                    'test_size': test_size,
                    'random_state': random_state
                },
                'validation_split': {
                    'enabled': True,
                    'reflection': 'sklearn.model_selection:train_test_split',
                    'args': {
                        'test_size': val_size / (1 - test_size),
                        'random_state': random_state
                    }
                }
            },
            'dataset': {
                'label_column': 'label',
                'train': {'batch_size': batch_size, 'shuffle': True},
                'val': {'batch_size': batch_size, 'shuffle': False},
                'test': {'batch_size': batch_size, 'shuffle': False}
            }
        }
    }

    manager = DataManager(context)
    manager.run()

    return manager.static_data['temp_source']
