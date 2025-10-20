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
from common.common import LoggerManager, load_yaml
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
    8. 部署模型（可选，通过 -d/--deployment 开关控制）
    """
    parser = argparse.ArgumentParser(
        description="LearnAI Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 仅训练和评估（不部署）
  python main.py game2048_training.yaml

  # 训练、评估并部署模型
  python main.py game2048_training.yaml -d
  python main.py game2048_training.yaml --deployment

  # 显示帮助信息
  python main.py --help
        """
    )
    parser.add_argument(
        "config",
        type=str,
        help="配置文件路径 (例如: config_example.yaml)"
    )
    parser.add_argument(
        "-d", "--deployment",
        action="store_true",
        help="启用模型部署（默认：关闭）。即使配置文件中定义了部署配置，也需要此参数才会执行部署"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("LearnAI Training Framework 启动")
    logger.info("=" * 80)

    # ========== 1. 加载配置 ==========
    logger.info(f"加载配置文件: {args.config}")
    config = load_yaml(args.config)
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
    training_pipeline.run_training(context)
    logger.info("训练完成")

    # ========== 8. 执行评估 ==========
    logger.info("-" * 80)
    logger.info("步骤 6: 开始评估")
    logger.info("-" * 80)
    evaluation.run_evaluation(context)
    logger.info("评估完成")

    # ========== 9. 导出模型 ==========
    if "export" in config:
        logger.info("-" * 80)
        logger.info("步骤 7: 导出模型")
        logger.info("-" * 80)
        export.export_model(context)
        logger.info("模型导出完成")
    else:
        logger.info("跳过模型导出（配置文件中未定义）")

    # ========== 10. 部署模型 ==========
    # 仅当使用 -d 或 --deployment 参数时才执行部署
    if args.deployment:
        if "deployment" in config:
            logger.info("-" * 80)
            logger.info("步骤 8: 部署模型")
            logger.info("-" * 80)
            deployment.deploy_model(context)
            logger.info("模型部署完成")
        else:
            logger.warning("无法部署：配置文件中未定义 deployment 配置")
            logger.warning("请在配置文件中添加 deployment 部分")
    else:
        logger.info("跳过模型部署（未使用 -d/--deployment 参数）")

    # ========== 11. 完成 ==========
    logger.info("=" * 80)
    logger.info("LearnAI Training Framework 执行完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
