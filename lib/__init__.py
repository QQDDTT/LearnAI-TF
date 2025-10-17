# -*- coding: utf-8 -*-
"""
lib/__init__.py
lib模块初始化文件
"""

# 数据加载工具
from lib.data_loader import (
    build_csv_dataset,
    build_numpy_dataset,
    split_dataset,
    normalize_data,
    create_batches
)

# 数据处理工具
from lib.data_utils import (
    get_batch,
    get_all_data,
    prepare_batch
)

# 训练工具
from lib.training_utils import (
    forward,
    compute_loss,
    compute_gradients,
    apply_gradients,
    compute_metrics
)

# 条件检查工具
from lib.condition_utils import (
    check_condition,
    check_threshold,
    check_convergence_simple
)

# 强化学习工具
from lib.reinforcement_learning import (
    NetworkClient,
    build_web_dataset,
    observe_state,
    select_action,
    execute_action,
    compute_reward,
    ReplayBuffer,
    store_transition,
    sample_batch,
    compute_gae,
    normalize_rewards,
    discount_rewards,
    eval_episode,
    aggregate_episode_metrics,
    update_actor_critic
)

# 聚类算法
from lib.clustering import (
    assign,
    update_centroids,
    check_convergence,
    KMeans,
    DBSCAN
)

# 检查点管理
from lib.checkpoint import CheckpointManager

# 日志记录器
from lib.logger import (
    FileLogger,
    TensorBoardLogger,
    CSVLogger,
    ConsoleLogger,
    WandbLogger
)

# 模型导出器
from lib.export import (
    ONNXExporter,
    SavedModelExporter,
    TFLiteExporter,
    TorchScriptExporter,
    CoreMLExporter
)

# 模型部署器
from lib.deployment import (
    RestAPIServer,
    GRPCServer,
    TFServingDeployer,
    DockerDeployer,
    LambdaDeployer
)

__all__ = [
    # 数据加载工具
    'build_csv_dataset',
    'build_numpy_dataset',
    'split_dataset',
    'normalize_data',
    'create_batches',

    # 数据处理工具
    'get_batch',
    'get_all_data',
    'prepare_batch',

    # 训练工具
    'forward',
    'compute_loss',
    'compute_gradients',
    'apply_gradients',
    'compute_metrics',

    # 条件检查工具
    'check_condition',
    'check_threshold',
    'check_convergence_simple',

    # 强化学习工具
    'NetworkClient',
    'build_web_dataset',
    'observe_state',
    'select_action',
    'execute_action',
    'compute_reward',
    'ReplayBuffer',
    'store_transition',
    'sample_batch',
    'compute_gae',
    'normalize_rewards',
    'discount_rewards',
    'eval_episode',
    'aggregate_episode_metrics',
    'update_actor_critic',

    # 聚类算法
    'assign',
    'update_centroids',
    'check_convergence',
    'KMeans',
    'DBSCAN',

    # 检查点管理
    'CheckpointManager',

    # 日志记录器
    'FileLogger',
    'TensorBoardLogger',
    'CSVLogger',
    'ConsoleLogger',
    'WandbLogger',

    # 模型导出器
    'ONNXExporter',
    'SavedModelExporter',
    'TFLiteExporter',
    'TorchScriptExporter',
    'CoreMLExporter',

    # 模型部署器
    'RestAPIServer',
    'GRPCServer',
    'TFServingDeployer',
    'DockerDeployer',
    'LambdaDeployer',
]

__version__ = '3.0.0'
