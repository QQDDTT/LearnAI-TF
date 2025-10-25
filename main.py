# -*- coding: utf-8 -*-
"""
main.py
深度学习训练框架 - 简化版主入口

设计理念：
1. 配置驱动：所有行为由 YAML 配置定义
2. 模块化：直接调用 common 和 modules 资源
3. 简化流程：加载配置 -> 初始化 -> 训练 -> 评估 -> 导出 -> 部署
"""

import sys
import argparse
import traceback
from pathlib import Path

# 导入基础模块
from common.common import LoggerManager, load_yaml
from common.train_context import TrainContext

# 导入业务模块
from modules.data_manager import DataManager
from modules.models import ModelBuilder
from modules.optimizers import OptimizerManager
from modules.losses import LossManager
from modules.training_pipeline import TrainingPipeline
from modules.evaluation import EvaluationManager
from modules.export import ModelExportManager
from modules.deployment import ModelDeploymentManager

# 初始化日志
logger = LoggerManager.get_logger(__file__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LearnAI 深度学习训练框架')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    parser.add_argument('--export', action='store_true', help='训练后导出模型')
    parser.add_argument('--deploy', action='store_true', help='导出后部署模型')
    parser.add_argument('--deploy-only', action='store_true', help='仅部署，跳过训练')
    parser.add_argument('--skip-eval', action='store_true', help='跳过评估')
    parser.add_argument('--checkpoint-dir', type=str, help='检查点目录')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    parser.add_argument('--dry-run', action='store_true', help='仅验证配置')
    return parser.parse_args()


def build_context(config: dict, args) -> TrainContext:
    """构建训练上下文"""
    from common.train_context import PipelineConfig, StepConfig

    logger.info("构建训练上下文...")

    context = TrainContext()

    # 设置全局配置
    context.global_config = config.get('global', {})

    # 设置训练模式
    training_mode_config = config.get('training_mode', {})
    context.training_mode = training_mode_config.get('type', 'supervised')
    logger.info(f"训练模式: {context.training_mode}")

    # 提取模型配置到 TrainContext.models
    models_config = config.get('models', {})
    for model_name, model_cfg in models_config.items():
        if isinstance(model_cfg, dict):
            context.models[model_name] = model_cfg
            logger.debug(f"添加模型配置: {model_name}")

    # 提取数据源配置到 TrainContext.data_sources
    data_sources_config = config.get('data_sources', {})
    for source_name, source_cfg in data_sources_config.items():
        if isinstance(source_cfg, dict):
            context.data_sources[source_name] = source_cfg
            logger.debug(f"添加数据源配置: {source_name}")

    # 提取优化器配置到 TrainContext.optimizers
    optimizers_config = config.get('optimizers', {})
    for opt_name, opt_cfg in optimizers_config.items():
        if isinstance(opt_cfg, dict):
            context.optimizers[opt_name] = opt_cfg
            logger.debug(f"添加优化器配置: {opt_name}")

    # 提取损失函数配置到 TrainContext.losses
    losses_config = config.get('losses', {})
    for loss_name, loss_cfg in losses_config.items():
        if isinstance(loss_cfg, dict):
            context.losses[loss_name] = loss_cfg
            logger.debug(f"添加损失函数配置: {loss_name}")

    # 提取训练流程配置到 TrainContext.training_pipelines
    # 需要创建 PipelineConfig 对象
    training_pipeline_config = config.get('training_pipeline', {})
    if training_pipeline_config:
        # 获取当前训练模式的配置
        mode_config = training_pipeline_config.get(context.training_mode, {})

        if mode_config:
            # 解析步骤序列（如果有）
            steps = []
            if 'step_sequence' in mode_config:
                for step_dict in mode_config.get('step_sequence', []):
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
                loop_type=mode_config.get('loop_type', 'epoch_batch'),
                steps=steps,
                parameters=mode_config.get('parameters', {}),
                loop_condition=mode_config.get('loop_condition', None)
            )

            context.training_pipelines[context.training_mode] = pipeline
            logger.debug(f"添加训练流程配置: {context.training_mode}")

    # 存储原始配置到 variables 中
    context.variables['raw_config'] = config
    context.variables['evaluation_config'] = config.get('evaluation', {})
    context.variables['export_config'] = config.get('export', {})
    context.variables['deployment_config'] = config.get('deployment', {})

    # 应用命令行参数覆盖
    if args.checkpoint_dir:
        context.variables['checkpoint_dir'] = args.checkpoint_dir
        logger.info(f"检查点目录: {args.checkpoint_dir}")

    context.variables['export_enabled'] = args.export
    context.variables['deploy_enabled'] = args.deploy
    context.variables['skip_evaluation'] = args.skip_eval
    context.variables['deploy_only'] = args.deploy_only

    logger.info(f"✓ 训练上下文构建完成 (模型: {len(context.models)}, 数据源: {len(context.data_sources)}, 优化器: {len(context.optimizers)}, 损失: {len(context.losses)})")
    return context


def run_pipeline(context: TrainContext, args) -> bool:
    """执行完整训练流程"""

    try:
        # ========== 阶段 1: 数据准备 ==========
        if not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 1: 数据准备")
            logger.info("=" * 60)

            dm = DataManager(context)
            dm.initialize()
            data_loaders = dm.execute()
            context.variables['data_loaders'] = data_loaders
            logger.info(f"✓ 数据准备完成，加载了 {len(data_loaders)} 个数据源")

        # ========== 阶段 2: 模型构建 ==========
        if not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 2: 模型构建")
            logger.info("=" * 60)

            mb = ModelBuilder(context)
            mb.initialize()
            models = mb.execute()
            context.variables['models'] = models
            logger.info(f"✓ 成功构建 {len(models)} 个模型")

        # ========== 阶段 3: 优化器构建 ==========
        if not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 3: 优化器构建")
            logger.info("=" * 60)

            om = OptimizerManager(context)
            om.initialize()
            optimizers = om.execute()
            context.variables['optimizers'] = optimizers
            logger.info(f"✓ 成功构建 {len(optimizers)} 个优化器")

        # ========== 阶段 4: 损失函数构建 ==========
        if not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 4: 损失函数构建")
            logger.info("=" * 60)

            lm = LossManager(context)
            lm.initialize()
            losses = lm.execute()
            context.variables['losses'] = losses
            logger.info(f"✓ 成功构建 {len(losses)} 个损失函数")

        # ========== 阶段 5: 训练 ==========
        if not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 5: 开始训练")
            logger.info("=" * 60)

            tp = TrainingPipeline(context)
            tp.initialize()
            training_results = tp.execute()
            context.variables['training_results'] = training_results
            logger.info("✓ 训练完成")

        # ========== 阶段 6: 评估 ==========
        if not args.skip_eval and not args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 6: 模型评估")
            logger.info("=" * 60)

            evaluator = EvaluationManager(context)
            evaluator.initialize()
            eval_results = evaluator.execute()
            context.variables['eval_results'] = eval_results
            logger.info("✓ 评估完成")

        # ========== 阶段 7: 导出 ==========
        if args.export or args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 7: 模型导出")
            logger.info("=" * 60)

            exporter = ModelExportManager(context)
            exporter.initialize()
            export_paths = exporter.execute()
            context.variables['export_paths'] = export_paths
            logger.info("✓ 导出完成")

        # ========== 阶段 8: 部署 ==========
        if args.deploy or args.deploy_only:
            logger.info("=" * 60)
            logger.info("阶段 8: 模型部署")
            logger.info("=" * 60)

            deployer = ModelDeploymentManager(context)
            deployer.initialize()
            deploy_info = deployer.execute()
            context.variables['deploy_info'] = deploy_info
            logger.info("✓ 部署完成")

        return True

    except Exception as e:
        logger.error(f"执行失败: {e}")
        logger.debug(traceback.format_exc())
        return False


def print_summary(context: TrainContext, success: bool):
    """打印执行摘要"""
    logger.info("=" * 60)
    logger.info("执行摘要")
    logger.info("=" * 60)
    logger.info(f"训练模式: {context.training_mode}")
    logger.info(f"状态: {'成功 ✓' if success else '失败 ✗'}")

    # 打印关键结果
    if 'training_results' in context.variables:
        logger.info(f"训练结果: {context.variables['training_results']}")

    if 'eval_results' in context.variables:
        logger.info(f"评估结果: {context.variables['eval_results']}")

    if 'export_paths' in context.variables:
        logger.info(f"导出路径: {context.variables['export_paths']}")

    if 'deploy_info' in context.variables:
        logger.info(f"部署信息: {context.variables['deploy_info']}")

    logger.info("=" * 60)


def main():
    """主函数"""
    try:
        # 打印欢迎信息
        logger.info("=" * 60)
        logger.info("LearnAI 深度学习训练框架 v2.0")
        logger.info("=" * 60)

        # 1. 解析参数
        args = parse_args()
        logger.info(f"配置文件: {args.config}")

        # 2. 加载配置
        logger.info("加载配置文件...")
        config = load_yaml(args.config)
        if not config:
            logger.error("配置文件加载失败")
            return 1
        logger.info("✓ 配置加载成功")

        # 3. 试运行模式
        if args.dry_run:
            logger.info("试运行模式：配置验证通过")
            return 0

        # 4. 构建上下文
        context = build_context(config, args)

        # 5. 执行流程
        success = run_pipeline(context, args)

        # 6. 打印摘要
        print_summary(context, success)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("\n用户中断执行")
        return 1
    except Exception as e:
        logger.error(f"发生错误: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
