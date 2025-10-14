# -*- coding: utf-8 -*-
"""
main.py
主入口：配置驱动的训练框架
- 读取配置文件
- 初始化各模块
- 调度训练流程
"""

import sys
import os
from common.common import LoggerManager, load_yaml
from modules import data_manager, models, optimizers, losses, training_pipeline, evaluation, export, deployment

# 获取日志器
logger = LoggerManager.get_logger(__file__)


def main(config_path: str):
    """
    主函数：配置驱动的训练流程

    参数:
        config_path: 配置文件路径
    """
    logger.info("=" * 60)
    logger.info("启动深度学习训练框架")
    logger.info("=" * 60)

    # 1. 加载配置文件
    logger.info(f"加载配置文件: {config_path}")
    config = load_yaml(config_path)

    project_name = config.get("global", {}).get("name", "UnknownProject")
    training_mode = config.get("training_mode", {}).get("type", "supervised")
    logger.info(f"项目: {project_name}")
    logger.info(f"训练模式: {training_mode}")

    # 2. 初始化数据管理器
    logger.info("-" * 60)
    logger.info("初始化数据管理器")
    data_config = config.get("data_manager", {})
    dataloaders = data_manager.initialize(data_config, training_mode)
    logger.info(f"数据加载器初始化完成: {list(dataloaders.keys())}")

    # 3. 构建模型
    logger.info("-" * 60)
    logger.info("构建模型")
    models_config = config.get("models", {})
    model_dict = models.build_all_models(models_config)
    logger.info(f"模型构建完成: {list(model_dict.keys())}")

    # 4. 构建优化器
    logger.info("-" * 60)
    logger.info("构建优化器")
    optimizers_config = config.get("optimizers", {})
    optimizer_dict = optimizers.build_all_optimizers(optimizers_config)
    logger.info(f"优化器构建完成: {list(optimizer_dict.keys())}")

    # 5. 构建损失函数
    logger.info("-" * 60)
    logger.info("构建损失函数")
    losses_config = config.get("losses", {})
    loss_dict = losses.build_all_losses(losses_config)
    logger.info(f"损失函数构建完成: {list(loss_dict.keys())}")

    # 6. 准备训练上下文
    context = {
        "config": config,
        "dataloaders": dataloaders,
        "models": model_dict,
        "optimizers": optimizer_dict,
        "losses": loss_dict,
        "training_mode": training_mode
    }

    # 7. 执行训练流程
    logger.info("-" * 60)
    logger.info("开始训练")
    pipeline_config = config.get("training_pipeline", {}).get(training_mode, {})
    training_pipeline.run_training(pipeline_config, context)

    # 8. 评估
    logger.info("-" * 60)
    logger.info("模型评估")
    eval_config = config.get("evaluation", {})
    eval_results = evaluation.run_evaluation(eval_config, context)
    logger.info(f"评估结果: {eval_results}")

    # 9. 导出模型
    if config.get("export", {}).get("enabled", False):
        logger.info("-" * 60)
        logger.info("导出模型")
        export_config = config.get("export", {})
        export.export_model(export_config, context)

    # 10. 部署（可选）
    if config.get("deployment", {}).get("enabled", False):
        logger.info("-" * 60)
        logger.info("部署模型")
        deployment_config = config.get("deployment", {})
        deployment.deploy_model(deployment_config, context)

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <config_file_path>")
        print("示例: python main.py config/supervised_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在 - {config_file}")
        sys.exit(1)

    try:
        main(config_file)
    except Exception as e:
        logger = LoggerManager.get_logger(__file__)
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
        sys.exit(1)
