# -*- coding: utf-8 -*-
"""
main.py
主入口文件：
- 读取配置文件
- 初始化日志和模块
- 调度各个模块完成完整训练-评估-导出-部署流程
"""

import os
import sys
import argparse
from common.common import LoggerManager, load_config_from_yaml
from modules import data_manager, models, optimizers, losses, training_pipeline
from modules import evaluation, export, deployment

# 获取 logger
logger = LoggerManager.get_logger(__file__)


def main():
    """
    主函数：
    1. 解析命令行参数（配置文件路径）
    2. 加载配置文件
    3. 初始化数据管理器
    4. 构建模型、优化器、损失函数
    5. 执行训练
    6. 执行评估
    7. 导出模型
    8. 部署模型（可选）
    """
    parser = argparse.ArgumentParser(description="LearnAI Training Framework")
    parser.add_argument(
        "config",
        type=str,
        help="配置文件路径 (例如: config/config_example.yaml)"
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("LearnAI Training Framework 启动")
    logger.info("=" * 80)

    # ========== 1. 加载配置 ==========
    logger.info(f"加载配置文件: {config_path}")
    config = load_config_from_yaml(config_path)
    if not config:
        logger.error("配置文件加载失败")
        sys.exit(1)
    logger.info("配置文件加载成功")

    # ========== 2. 初始化数据管理器 ==========
    logger.info("-" * 80)
    logger.info("步骤 1: 初始化数据管理器")
    logger.info("-" * 80)
    data_context = data_manager.initialize(config)
    logger.info("数据管理器初始化完成")

    # ========== 3. 构建模型 ==========
    logger.info("-" * 80)
    logger.info("步骤 2: 构建模型")
    logger.info("-" * 80)
    models_dict = models.build_all_models(config)
    logger.info(f"成功构建 {len(models_dict)} 个模型")

    # ========== 4. 构建优化器 ==========
    logger.info("-" * 80)
    logger.info("步骤 3: 构建优化器")
    logger.info("-" * 80)
    optimizers_dict = optimizers.build_all_optimizers(config)
    logger.info(f"成功构建 {len(optimizers_dict)} 个优化器")

    # ========== 5. 构建损失函数 ==========
    logger.info("-" * 80)
    logger.info("步骤 4: 构建损失函数")
    logger.info("-" * 80)
    losses_dict = losses.build_all_losses(config)
    logger.info(f"成功构建 {len(losses_dict)} 个损失函数")

    # ========== 6. 构建全局上下文 ==========
    context = {
        "config": config,
        "data": data_context,
        "models": models_dict,
        "optimizers": optimizers_dict,
        "losses": losses_dict
    }

    # ========== 7. 执行训练 ==========
    logger.info("-" * 80)
    logger.info("步骤 5: 开始训练")
    logger.info("-" * 80)
    training_results = training_pipeline.run_training(context)
    logger.info("训练完成")

    # ========== 8. 执行评估 ==========
    logger.info("-" * 80)
    logger.info("步骤 6: 开始评估")
    logger.info("-" * 80)
    evaluation_results = evaluation.run_evaluation(context)
    logger.info("评估完成")

    # ========== 9. 导出模型 ==========
    if "export" in config:
        logger.info("-" * 80)
        logger.info("步骤 7: 导出模型")
        logger.info("-" * 80)
        export_results = export.export_model(context)
        logger.info("模型导出完成")
    else:
        logger.info("跳过模型导出（配置文件中未定义）")

    # ========== 10. 部署模型 ==========
    if "deployment" in config:
        logger.info("-" * 80)
        logger.info("步骤 8: 部署模型")
        logger.info("-" * 80)
        deployment_results = deployment.deploy_model(context)
        logger.info("模型部署完成")
    else:
        logger.info("跳过模型部署（配置文件中未定义）")

    # ========== 11. 完成 ==========
    logger.info("=" * 80)
    logger.info("LearnAI Training Framework 执行完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
