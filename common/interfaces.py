# -*- coding: utf-8 -*-
"""
common/interfaces.py
业务模块接口定义：规定所有模块围绕 TrainContext 展开的标准流程

设计原则：
1. 所有业务模块必须继承对应的接口类
2. 强制围绕 TrainContext 设计，禁止模块间直接依赖
3. 定义标准生命周期：initialize -> setup -> execute -> finalize
4. 使用 call_target 实现动态调用，避免硬编码依赖
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from common.train_context import TrainContext


# ============================================================================
# 基础接口
# ============================================================================

class TrainContextAware(ABC):
    """
    所有业务模块的基类

    强制要求：
    1. 构造函数必须接收 TrainContext
    2. 所有配置从 TrainContext 读取
    3. 所有结果写回 TrainContext

    标准生命周期：
    1. __init__(context) - 接收上下文
    2. initialize() - 从上下文初始化模块
    3. setup() - 配置模块（可选）
    4. execute() - 执行核心逻辑
    5. finalize() - 清理和保存结果
    """

    def __init__(self, context: TrainContext):
        """
        初始化模块

        参数:
            context: 训练上下文，所有模块的数据源
        """
        if not isinstance(context, TrainContext):
            raise TypeError("context 必须是 TrainContext 类型")
        self.context = context
        self._initialized = False
        self._setup_complete = False

    @abstractmethod
    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有必需的配置
        2. 加载模块需要的配置数据
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        pass

    def setup(self) -> None:
        """
        配置模块（可选）

        职责：
        1. 执行额外的配置步骤
        2. 建立模块间的连接
        3. 预处理数据

        注意：不是所有模块都需要 setup，默认为空实现
        """
        self._setup_complete = True

    @abstractmethod
    def execute(self) -> Any:
        """
        执行核心逻辑

        职责：
        1. 执行模块的主要功能
        2. 使用 call_target 调用配置的函数
        3. 生成执行结果

        返回:
            模块执行结果
        """
        pass

    def finalize(self) -> None:
        """
        清理和保存结果（可选）

        职责：
        1. 将执行结果写回 TrainContext
        2. 清理临时资源
        3. 记录日志

        注意：不是所有模块都需要 finalize，默认为空实现
        """
        pass

    def run(self) -> Any:
        """
        运行完整流程：initialize -> setup -> execute -> finalize

        返回:
            execute() 的执行结果
        """
        if not self._initialized:
            self.initialize()
            self._initialized = True

        if not self._setup_complete:
            self.setup()

        result = self.execute()

        self.finalize()

        return result


# ============================================================================
# 数据管理接口
# ============================================================================

class DataManagerInterface(TrainContextAware):
    """
    数据管理器接口

    职责：
    1. 从 TrainContext.data_sources 读取配置
    2. 构建数据加载器（train/val/test splits）
    3. 管理数据预处理和增强
    4. 将实例化的数据加载器写回 TrainContext.container.dataloaders

    标准流程：
    initialize() -> validate_data_sources() -> build_all_data_sources() -> cache_to_context()
    """

    @abstractmethod
    def validate_data_sources(self) -> bool:
        """
        验证数据源配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def build_data_source(
        self,
        source_name: str,
        source_splits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建单个数据源的所有分割

        参数:
            source_name: 数据源名称
            source_splits: 数据分割配置 {split_name: config}

        返回:
            数据加载器字典 {split_name: loader}
        """
        pass

    @abstractmethod
    def build_all_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        构建所有数据源

        返回:
            所有数据加载器 {source_name: {split_name: loader}}
        """
        pass

    def execute(self) -> Dict[str, Dict[str, Any]]:
        """执行数据加载"""
        return self.build_all_data_sources()

    def finalize(self) -> None:
        """将数据加载器缓存到上下文"""
        if hasattr(self, 'data_loaders'):
            self.context.container.dataloaders = self.data_loaders


# ============================================================================
# 优化器管理接口
# ============================================================================

class OptimizerManagerInterface(TrainContextAware):
    """
    优化器管理器接口

    职责：
    1. 从 TrainContext.optimizers 读取配置
    2. 构建优化器实例
    3. 配置学习率调度器
    4. 将实例化的优化器写回 TrainContext.container.optimizers

    标准流程：
    initialize() -> validate_optimizers() -> build_all_optimizers() -> setup_schedulers() -> cache_to_context()
    """

    @abstractmethod
    def validate_optimizers(self) -> bool:
        """
        验证优化器配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
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
            优化器实例
        """
        pass

    @abstractmethod
    def build_all_optimizers(self) -> Dict[str, Any]:
        """
        构建所有优化器

        返回:
            优化器字典 {optimizer_name: optimizer}
        """
        pass

    def setup_schedulers(self) -> Dict[str, Any]:
        """
        设置学习率调度器（可选）

        返回:
            调度器字典 {optimizer_name: scheduler}
        """
        return {}

    def execute(self) -> Dict[str, Any]:
        """执行优化器构建"""
        return self.build_all_optimizers()

    def finalize(self) -> None:
        """将优化器缓存到上下文"""
        if hasattr(self, 'optimizers'):
            self.context.container.optimizers = self.optimizers


# ============================================================================
# 损失函数管理接口
# ============================================================================

class LossManagerInterface(TrainContextAware):
    """
    损失函数管理器接口

    职责：
    1. 从 TrainContext.losses 读取配置
    2. 构建损失函数实例（内置/自定义）
    3. 处理多任务损失聚合
    4. 将实例化的损失函数写回 TrainContext.container.losses

    标准流程：
    initialize() -> validate_losses() -> build_all_losses() -> setup_aggregators() -> cache_to_context()
    """

    @abstractmethod
    def validate_losses(self) -> bool:
        """
        验证损失函数配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def build_all_losses(self) -> Dict[str, Any]:
        """
        构建所有损失函数

        返回:
            损失函数字典 {loss_name: loss}
        """
        pass

    def setup_aggregators(self) -> Dict[str, Any]:
        """
        设置多任务损失聚合器（可选）

        返回:
            聚合器字典
        """
        return {}

    def execute(self) -> Dict[str, Any]:
        """执行损失函数构建"""
        return self.build_all_losses()

    def finalize(self) -> None:
        """将损失函数缓存到上下文"""
        if hasattr(self, 'losses'):
            self.context.container.losses = self.losses


# ============================================================================
# 模型构建接口
# ============================================================================

class ModelBuilderInterface(TrainContextAware):
    """
    模型构建器接口

    职责：
    1. 从 TrainContext.models 读取配置
    2. 构建模型层和连接关系
    3. 支持 Sequential/Functional/Subclass 三种模型类型
    4. 处理预训练模型加载
    5. 将实例化的模型写回 TrainContext.container.models

    标准流程：
    initialize() -> validate_models() -> build_all_models() -> compile_models() -> cache_to_context()
    """

    @abstractmethod
    def validate_models(self) -> bool:
        """
        验证模型配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
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
            模型实例
        """
        pass

    @abstractmethod
    def build_all_models(self) -> Dict[str, Any]:
        """
        构建所有模型

        返回:
            模型字典 {model_name: model}
        """
        pass

    def compile_models(self) -> None:
        """
        编译模型（可选）

        注意：如果使用自定义训练循环，可能不需要编译
        """
        pass

    def execute(self) -> Dict[str, Any]:
        """执行模型构建"""
        models = self.build_all_models()
        self.compile_models()
        return models

    def finalize(self) -> None:
        """将模型缓存到上下文"""
        if hasattr(self, 'models'):
            self.context.container.models = self.models


# ============================================================================
# 奖励函数管理接口（强化学习）
# ============================================================================

class RewardFunctionInterface(TrainContextAware):
    """
    奖励函数管理器接口

    职责：
    1. 从 TrainContext.reward_functions 读取配置
    2. 构建奖励函数实例
    3. 处理奖励塑形（Reward Shaping）
    4. 计算和聚合奖励

    标准流程：
    initialize() -> validate_rewards() -> build_reward_functions() -> setup_shaping() -> compute_reward()
    """

    @abstractmethod
    def validate_rewards(self) -> bool:
        """
        验证奖励函数配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def build_reward_function(
        self,
        reward_name: str,
        reward_config: Any
    ) -> Any:
        """
        构建单个奖励函数

        参数:
            reward_name: 奖励函数名称
            reward_config: 奖励函数配置

        返回:
            奖励函数实例
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励

        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息

        返回:
            奖励值
        """
        pass

    def setup_shaping(self) -> None:
        """
        设置奖励塑形（可选）

        职责：
        1. 配置势函数（Potential Function）
        2. 设置奖励缩放和归一化
        """
        pass

    def execute(self) -> Dict[str, Any]:
        """执行奖励函数构建"""
        if hasattr(self, 'build_all_reward_functions'):
            return self.build_all_reward_functions()
        return {}


# ============================================================================
# 训练流程接口
# ============================================================================

class TrainingPipelineInterface(TrainContextAware):
    """
    训练流程执行器接口

    职责：
    1. 从 TrainContext.training_pipelines 读取配置
    2. 执行训练步骤序列
    3. 处理循环控制（epoch/episode/iteration）
    4. 处理 Bridge 控制流（跳转、分支、循环）
    5. 管理训练状态和检查点

    标准流程：
    initialize() -> validate_pipeline() -> setup_loop() -> execute_training() -> save_checkpoint() -> finalize()
    """

    @abstractmethod
    def validate_pipeline(self) -> bool:
        """
        验证训练流程配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def setup_loop(self) -> None:
        """
        设置训练循环

        职责：
        1. 初始化循环计数器
        2. 配置循环条件
        3. 设置早停条件
        """
        pass

    @abstractmethod
    def execute_step(
        self,
        step_name: str,
        step_config: Any
    ) -> Any:
        """
        执行单个训练步骤

        参数:
            step_name: 步骤名称
            step_config: 步骤配置

        返回:
            步骤执行结果
        """
        pass

    @abstractmethod
    def execute_training(self) -> Dict[str, Any]:
        """
        执行训练循环

        返回:
            训练结果字典
        """
        pass

    def save_checkpoint(self, epoch: int) -> None:
        """
        保存检查点（可选）

        参数:
            epoch: 当前 epoch
        """
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点（可选）

        参数:
            checkpoint_path: 检查点路径
        """
        pass

    def execute(self) -> Dict[str, Any]:
        """执行训练流程"""
        self.setup_loop()
        return self.execute_training()

    def finalize(self) -> None:
        """保存训练结果"""
        if hasattr(self, 'training_results'):
            self.context.execution_results['training'] = self.training_results


# ============================================================================
# 评估接口
# ============================================================================

class EvaluationInterface(TrainContextAware):
    """
    评估器接口

    职责：
    1. 从 TrainContext.evaluation 读取配置
    2. 在验证集/测试集上评估模型
    3. 计算评估指标
    4. 生成评估报告
    5. 将评估结果写回 TrainContext.execution_results

    标准流程：
    initialize() -> validate_evaluation() -> setup_metrics() -> evaluate() -> generate_report() -> finalize()
    """

    @abstractmethod
    def validate_evaluation(self) -> bool:
        """
        验证评估配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def setup_metrics(self) -> Dict[str, Any]:
        """
        设置评估指标

        返回:
            指标字典 {metric_name: metric_fn}
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        执行评估

        参数:
            model: 待评估模型
            dataset: 评估数据集
            metrics: 评估指标

        返回:
            评估结果 {metric_name: value}
        """
        pass

    @abstractmethod
    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        评估所有模型

        返回:
            所有模型的评估结果 {model_name: {metric_name: value}}
        """
        pass

    def generate_report(self) -> Dict[str, Any]:
        """
        生成评估报告（可选）

        返回:
            评估报告字典
        """
        return {}

    def execute(self) -> Dict[str, Any]:
        """执行评估"""
        self.setup_metrics()
        results = self.evaluate_all_models()
        report = self.generate_report()
        return {"results": results, "report": report}

    def finalize(self) -> None:
        """保存评估结果"""
        if hasattr(self, 'evaluation_results'):
            self.context.execution_results['evaluation'] = self.evaluation_results


# ============================================================================
# 模型导出接口
# ============================================================================

class ExportInterface(TrainContextAware):
    """
    模型导出器接口

    职责：
    1. 从 TrainContext.export 读取配置
    2. 导出模型为多种格式（SavedModel/ONNX/TFLite）
    3. 优化和量化模型
    4. 生成模型元数据

    标准流程：
    initialize() -> validate_export() -> prepare_model() -> export_model() -> verify_export() -> finalize()
    """

    @abstractmethod
    def validate_export(self) -> bool:
        """
        验证导出配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        """
        准备导出模型

        参数:
            model: 待导出模型

        返回:
            准备好的模型
        """
        pass

    @abstractmethod
    def export_model(
        self,
        export_name: str,
        export_config: Any
    ) -> str:
        """
        导出单个模型

        参数:
            export_name: 导出配置名称
            export_config: 导出配置

        返回:
            导出路径
        """
        pass

    @abstractmethod
    def export_all(self) -> Dict[str, str]:
        """
        导出所有配置的模型

        返回:
            导出路径字典 {export_name: export_path}
        """
        pass

    def verify_export(self, export_path: str) -> bool:
        """
        验证导出结果（可选）

        参数:
            export_path: 导出路径

        返回:
            验证是否通过
        """
        return True

    def execute(self) -> Dict[str, str]:
        """执行模型导出"""
        return self.export_all()

    def finalize(self) -> None:
        """保存导出路径"""
        if hasattr(self, 'export_paths'):
            self.context.execution_results['export'] = self.export_paths


# ============================================================================
# 模型部署接口
# ============================================================================

class DeploymentInterface(TrainContextAware):
    """
    模型部署器接口

    职责：
    1. 从 TrainContext.deployment 读取配置
    2. 部署模型到目标平台（Local/Cloud/Edge）
    3. 配置服务端点（REST API/gRPC）
    4. 健康检查和监控

    标准流程：
    initialize() -> validate_deployment() -> prepare_deployment() -> deploy() -> health_check() -> finalize()
    """

    @abstractmethod
    def validate_deployment(self) -> bool:
        """
        验证部署配置

        返回:
            配置是否有效
        """
        pass

    @abstractmethod
    def prepare_deployment(self, model_path: str) -> Dict[str, Any]:
        """
        准备部署

        参数:
            model_path: 模型路径

        返回:
            部署准备信息
        """
        pass

    @abstractmethod
    def deploy(
        self,
        deploy_name: str,
        deploy_config: Any
    ) -> Dict[str, Any]:
        """
        部署单个服务

        参数:
            deploy_name: 部署配置名称
            deploy_config: 部署配置

        返回:
            部署信息
        """
        pass

    @abstractmethod
    def deploy_all(self) -> Dict[str, Any]:
        """
        部署所有配置的服务

        返回:
            部署结果字典 {deploy_name: deployment_info}
        """
        pass

    def health_check(self, endpoint: str) -> bool:
        """
        健康检查（可选）

        参数:
            endpoint: 服务端点

        返回:
            服务是否健康
        """
        return True

    def execute(self) -> Dict[str, Any]:
        """执行部署"""
        return self.deploy_all()

    def finalize(self) -> None:
        """保存部署信息"""
        if hasattr(self, 'deployment_info'):
            self.context.execution_results['deployment'] = self.deployment_info


# ============================================================================
# 辅助函数
# ============================================================================

def validate_context_aware(instance: Any) -> bool:
    """
    验证实例是否正确实现了 TrainContextAware 接口

    参数:
        instance: 待验证实例

    返回:
        是否符合接口规范
    """
    if not isinstance(instance, TrainContextAware):
        return False

    required_methods = ['initialize', 'execute', 'run']
    for method in required_methods:
        if not hasattr(instance, method) or not callable(getattr(instance, method)):
            return False

    return True


def get_interface_for_module(module_name: str) -> type:
    """
    根据模块名称获取对应的接口类

    参数:
        module_name: 模块名称

    返回:
        接口类

    抛出:
        ValueError: 如果模块名称不存在
    """
    interface_mapping = {
        'data_manager': DataManagerInterface,
        'optimizers': OptimizerManagerInterface,
        'losses': LossManagerInterface,
        'models': ModelBuilderInterface,
        'reward_functions': RewardFunctionInterface,
        'training_pipeline': TrainingPipelineInterface,
        'evaluation': EvaluationInterface,
        'export': ExportInterface,
        'deployment': DeploymentInterface,
    }

    if module_name not in interface_mapping:
        raise ValueError(f"未知的模块名称: {module_name}")

    return interface_mapping[module_name]
