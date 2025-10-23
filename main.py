# -*- coding: utf-8 -*-
"""
main.py
深度学习训练框架 - 主入口文件

设计理念：
1. 配置驱动：所有逻辑通过 YAML 配置文件定义
2. 模块化调度：main.py 仅负责调度，不包含业务逻辑
3. 标准化流程：初始化 -> 训练 -> 评估 -> 导出 -> 部署
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# 导入基础模块
from common.common import LoggerManager, load_yaml, call_target
from common.train_context import TrainContext

# 导入业务模块
from modules import (
    data_manager,
    models,
    optimizers,
    losses,
    training_pipeline,
    evaluation,
    export,
    deployment
)

# 初始化日志
logger = LoggerManager.get_logger(__file__)


# ============================================================================
# 核心函数
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数

    返回:
        argparse.Namespace: 命令行参数对象
    """
    parser = argparse.ArgumentParser(
        description="LearnAI 深度学习训练框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基础训练（仅训练和评估）
  python main.py config/config_example.yaml

  # 训练并导出模型
  python main.py config/config_example.yaml --export

  # 训练、导出并部署
  python main.py config/config_example.yaml --export --deploy

  # 仅部署已有模型（跳过训练）
  python main.py config/config_example.yaml --deploy-only

  # 显示详细日志
  python main.py config/config_example.yaml --verbose

  # 指定检查点目录
  python main.py config/config_example.yaml --checkpoint-dir ./my_checkpoints
        """
    )

    # 必需参数
    parser.add_argument(
        "config",
        type=str,
        help="配置文件路径 (YAML 格式)"
    )

    # 可选参数
    parser.add_argument(
        "-e", "--export",
        action="store_true",
        help="训练完成后导出模型（默认：关闭）"
    )

    parser.add_argument(
        "-d", "--deploy",
        action="store_true",
        help="导出后自动部署模型（默认：关闭，需配合 --export 使用）"
    )

    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="仅部署已有模型，跳过训练和导出"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="检查点保存目录（覆盖配置文件中的设置）"
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="跳过评估阶段"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行：仅验证配置文件，不执行训练"
    )

    return parser.parse_args()


def initialize_environment(args: argparse.Namespace) -> None:
    """
    初始化运行环境

    参数:
        args: 命令行参数
    """
    # 设置日志级别
    if args.verbose:
        import logging
        LoggerManager.set_level(logging.DEBUG)
        logger.debug("详细日志模式已启用")

    # 创建必要的目录
    directories = ['data', 'logs', 'checkpoints', 'outputs', 'outputs/onnx', 'outputs/saved_model']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info(f"运行环境初始化完成")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的完整性和正确性

    参数:
        config: 配置字典

    返回:
        bool: 验证是否通过
    """
    logger.info("开始验证配置文件...")

    # 必需的顶级配置节点
    required_sections = ['data_manager', 'models', 'training_pipeline']

    for section in required_sections:
        if section not in config:
            logger.error(f"配置文件缺少必需节点: {section}")
            return False

    # 验证训练模式
    training_mode = config.get('training_pipeline', {}).get('mode')
    valid_modes = ['supervised', 'reinforcement', 'self_supervised', 'custom']

    if training_mode not in valid_modes:
        logger.error(f"不支持的训练模式: {training_mode}. 支持的模式: {valid_modes}")
        return False

    logger.info(f"✓ 配置文件验证通过 (训练模式: {training_mode})")
    return True


def build_train_context(config: Dict[str, Any], args: argparse.Namespace) -> TrainContext:
    """
    构建训练上下文

    参数:
        config: 配置字典
        args: 命令行参数

    返回:
        TrainContext: 训练上下文对象
    """
    from common.train_context import PipelineConfig, StepConfig

    logger.info("构建训练上下文...")

    # 创建 TrainContext（dataclass 不接受 config 参数）
    context = TrainContext()

    # 设置全局配置
    context.global_config = config.get('global', {})

    # 设置训练模式
    training_mode_config = config.get('training_pipeline', {})
    context.training_mode = training_mode_config.get('mode', 'supervised')

    # 解析模型配置到 TrainContext.models（简化版，直接存储配置字典）
    # ModelBuilder 会从这里读取
    models_config = config.get('models', {})
    for model_name, model_cfg in models_config.items():
        # 存储为字典，ModelBuilder 会处理
        context.models[model_name] = model_cfg

    # 解析数据源配置到 TrainContext.data_sources
    data_manager_config = config.get('data_manager', {})
    for source_name, source_cfg in data_manager_config.items():
        if source_name != 'type':  # 跳过 type 字段
            context.data_sources[source_name] = source_cfg

    # 解析优化器配置到 TrainContext.optimizers
    optimizers_config = config.get('optimizers', {})
    for opt_name, opt_cfg in optimizers_config.items():
        context.optimizers[opt_name] = opt_cfg

    # 解析损失函数配置到 TrainContext.losses
    losses_config = config.get('losses', {})
    for loss_name, loss_cfg in losses_config.items():
        context.losses[loss_name] = loss_cfg

    # 解析训练流程配置到 TrainContext.training_pipelines
    # 需要转换为 PipelineConfig 对象
    if 'supervised' in training_mode_config:
        supervised_config = training_mode_config['supervised']

        # 解析步骤序列
        steps = []
        if 'step_sequence' in supervised_config:
            for step_dict in supervised_config.get('step_sequence', []):
                if isinstance(step_dict, dict):
                    step = StepConfig(
                        name=step_dict.get('name', ''),
                        reflection=step_dict.get('reflection', ''),
                        args=step_dict.get('args', {}),
                        bridge=step_dict.get('bridge', None)
                    )
                    steps.append(step)

        # 创建 PipelineConfig 对象
        pipeline = PipelineConfig(
            loop_type=supervised_config.get('loop_type', 'epoch_batch'),
            steps=steps,
            parameters=supervised_config.get('parameters', {}),
            loop_condition=supervised_config.get('loop_condition', None)
        )

        context.training_pipelines['supervised'] = pipeline

    # 存储原始配置到 variables 中作为备份
    context.variables['raw_config'] = config
    context.variables['data_manager_config'] = data_manager_config
    context.variables['models_config'] = models_config
    context.variables['optimizers_config'] = optimizers_config
    context.variables['losses_config'] = losses_config
    context.variables['training_pipeline_config'] = training_mode_config
    context.variables['evaluation_config'] = config.get('evaluation', {})
    context.variables['export_config'] = config.get('export', {})
    context.variables['deployment_config'] = config.get('deployment', {})

    # 应用命令行参数覆盖
    if args.checkpoint_dir:
        context.variables['checkpoint_dir'] = args.checkpoint_dir
        logger.info(f"检查点目录已覆盖: {args.checkpoint_dir}")

    context.variables['export_enabled'] = args.export
    context.variables['deploy_enabled'] = args.deploy
    context.variables['skip_evaluation'] = args.skip_evaluation
    context.variables['deploy_only'] = args.deploy_only

    logger.info(f"✓ 训练上下文构建完成 (模型: {len(context.models)}, 数据源: {len(context.data_sources)}, 训练流程: {len(context.training_pipelines)})")
    return context


def run_training_stage(context: TrainContext) -> bool:
    """
    执行训练阶段

    参数:
        context: 训练上下文

    返回:
        bool: 是否成功
    """
    logger.info("=" * 80)
    logger.info("阶段 1: 数据准备")
    logger.info("=" * 80)

    try:
        # 初始化数据管理器（实例化 DataManager 类）
        from modules.data_manager import DataManager
        dm = DataManager(context)
        dm.initialize()
        data_loaders = dm.execute()
        context.variables['data_loaders'] = data_loaders
        logger.info("✓ 数据管理器初始化成功")
    except Exception as e:
        logger.error(f"✗ 数据管理器初始化失败: {e}")
        logger.debug(traceback.format_exc())
        return False

    logger.info("=" * 80)
    logger.info("阶段 2: 模型构建")
    logger.info("=" * 80)

    try:
        # 构建模型（实例化 ModelBuilder 类）
        from modules.models import ModelBuilder
        mb = ModelBuilder(context)
        mb.initialize()
        models_dict = mb.execute()
        context.variables['models'] = models_dict
        logger.info(f"✓ 成功构建 {len(models_dict)} 个模型")
    except Exception as e:
        logger.error(f"✗ 模型构建失败: {e}")
        logger.debug(traceback.format_exc())
        return False

    logger.info("=" * 80)
    logger.info("阶段 3: 优化器构建")
    logger.info("=" * 80)

    try:
        # 构建优化器（实例化 OptimizerManager 类）
        from modules.optimizers import OptimizerManager
        om = OptimizerManager(context)
        om.initialize()
        optimizers_dict = om.execute()
        context.variables['optimizers'] = optimizers_dict
        logger.info(f"✓ 成功构建 {len(optimizers_dict)} 个优化器")
    except Exception as e:
        logger.error(f"✗ 优化器构建失败: {e}")
        logger.debug(traceback.format_exc())
        return False

    logger.info("=" * 80)
    logger.info("阶段 4: 损失函数构建")
    logger.info("=" * 80)

    try:
        # 构建损失函数（实例化 LossManager 类）
        from modules.losses import LossManager
        lm = LossManager(context)
        lm.initialize()
        losses_dict = lm.execute()
        context.variables['losses'] = losses_dict
        logger.info(f"✓ 成功构建 {len(losses_dict)} 个损失函数")
    except Exception as e:
        logger.error(f"✗ 损失函数构建失败: {e}")
        logger.debug(traceback.format_exc())
        return False

    logger.info("=" * 80)
    logger.info("阶段 5: 开始训练")
    logger.info("=" * 80)

    try:
        # 执行训练（实例化 TrainingPipeline 类）
        from modules.training_pipeline import TrainingPipeline
        tp = TrainingPipeline(context)
        tp.initialize()
        training_results = tp.execute()
        context.variables['training_results'] = training_results
        logger.info("✓ 训练完成")
        return True
    except Exception as e:
        logger.error(f"✗ 训练失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def run_evaluation_stage(context: TrainContext) -> bool:
    """
    执行评估阶段

    参数:
        context: 训练上下文

    返回:
        bool: 是否成功
    """
    if context.variables.get('skip_evaluation'):
        logger.info("跳过评估阶段（--skip-evaluation 已启用）")
        return True

    logger.info("=" * 80)
    logger.info("阶段 6: 模型评估")
    logger.info("=" * 80)

    try:
        # 执行评估（实例化 EvaluationManager 类）
        from modules.evaluation import EvaluationManager
        evaluator = EvaluationManager(context)
        evaluator.initialize()
        eval_results = evaluator.execute()
        context.execution_results['evaluation'] = eval_results
        logger.info("✓ 评估完成")

        # 打印评估结果
        if eval_results:
            logger.info("\n评估结果:")
            for metric_name, metric_value in eval_results.items():
                logger.info(f"  {metric_name}: {metric_value}")

        return True
    except Exception as e:
        logger.error(f"✗ 评估失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def run_export_stage(context: TrainContext) -> bool:
    """
    执行导出阶段

    参数:
        context: 训练上下文

    返回:
        bool: 是否成功
    """
    if not context.variables.get('export_enabled'):
        logger.info("模型导出已跳过（使用 --export 启用）")
        return True

    logger.info("=" * 80)
    logger.info("阶段 7: 模型导出")
    logger.info("=" * 80)

    try:
        # 执行导出（实例化 ModelExporter 类）
        from modules.export import ModelExporter
        exporter = ModelExporter(context)
        exporter.initialize()
        export_results = exporter.execute()
        context.execution_results['export'] = export_results
        logger.info("✓ 模型导出完成")

        # 打印导出信息
        if export_results:
            logger.info("\n导出信息:")
            for key, value in export_results.items():
                logger.info(f"  {key}: {value}")

        return True
    except Exception as e:
        logger.error(f"✗ 模型导出失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def run_deployment_stage(context: TrainContext) -> bool:
    """
    执行部署阶段

    参数:
        context: 训练上下文

    返回:
        bool: 是否成功
    """
    if not context.variables.get('deploy_enabled') and not context.variables.get('deploy_only'):
        logger.info("模型部署已跳过（使用 --deploy 启用）")
        return True

    if context.variables.get('deploy_enabled') and not context.variables.get('export_enabled'):
        logger.warning("部署需要先导出模型，请添加 --export 参数")
        return False

    logger.info("=" * 80)
    logger.info("阶段 8: 模型部署")
    logger.info("=" * 80)

    try:
        # 执行部署（实例化 ModelDeploymentManager 类）
        from modules.deployment import ModelDeploymentManager
        deployer = ModelDeploymentManager(context)
        deployer.initialize()
        deployment_results = deployer.execute()
        context.execution_results['deployment'] = deployment_results
        logger.info("✓ 模型部署完成")

        # 打印部署信息
        if deployment_results:
            logger.info("\n部署信息:")
            for key, value in deployment_results.items():
                logger.info(f"  {key}: {value}")

        return True
    except Exception as e:
        logger.error(f"✗ 模型部署失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def print_summary(context: TrainContext, success: bool) -> None:
    """
    打印执行摘要

    参数:
        context: 训练上下文
        success: 是否成功
    """
    logger.info("=" * 80)
    logger.info("执行摘要")
    logger.info("=" * 80)

    if success:
        logger.info("✓ 所有阶段执行成功！")
    else:
        logger.error("✗ 部分阶段执行失败，请检查日志")

    # 打印关键信息
    training_mode = context.training_mode
    logger.info(f"训练模式: {training_mode}")

    if 'evaluation' in context.execution_results:
        logger.info("评估: ✓ 已完成")

    if 'export' in context.execution_results:
        logger.info("导出: ✓ 已完成")

    if 'deployment' in context.execution_results:
        logger.info("部署: ✓ 已完成")

    logger.info("=" * 80)


# ============================================================================
# 主函数
# ============================================================================

def main() -> int:
    """
    主函数：协调整个训练流程

    流程：
    1. 解析命令行参数
    2. 加载和验证配置文件
    3. 初始化环境
    4. 构建训练上下文
    5. 执行训练流程
    6. 执行评估
    7. 导出模型（可选）
    8. 部署模型（可选）

    返回:
        int: 退出码（0=成功，1=失败）
    """
    try:
        # 打印欢迎信息
        logger.info("=" * 80)
        logger.info("LearnAI 深度学习训练框架")
        logger.info("版本: 2.0")
        logger.info("=" * 80)

        # 1. 解析命令行参数
        args = parse_arguments()
        logger.info(f"配置文件: {args.config}")

        # 2. 加载配置文件
        logger.info("加载配置文件...")
        config = load_yaml(args.config)
        if not config:
            logger.error("配置文件加载失败")
            return 1
        logger.info("✓ 配置文件加载成功")

        # 3. 验证配置文件
        if not validate_config(config):
            logger.error("配置文件验证失败")
            return 1

        # 试运行模式：仅验证配置
        if args.dry_run:
            logger.info("=" * 80)
            logger.info("试运行完成：配置文件验证通过")
            logger.info("=" * 80)
            return 0

        # 4. 初始化环境
        initialize_environment(args)

        # 5. 构建训练上下文
        context = build_train_context(config, args)

        # 6. 执行主流程
        success = True

        # 仅部署模式
        if args.deploy_only:
            logger.info("仅部署模式：跳过训练和评估")
            success = run_deployment_stage(context)
        else:
            # 完整流程：训练 -> 评估 -> 导出 -> 部署
            if not run_training_stage(context):
                success = False
            elif not run_evaluation_stage(context):
                success = False
            elif not run_export_stage(context):
                success = False
            elif not run_deployment_stage(context):
                success = False

        # 7. 打印摘要
        print_summary(context, success)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("\n用户中断执行")
        return 1
    except Exception as e:
        logger.error(f"发生未预期的错误: {e}")
        logger.debug(traceback.format_exc())
        return 1


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
