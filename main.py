# -*- coding: utf-8 -*-
"""
主程序入口
- 解析参数（动作类型、配置路径）
- 加载配置和日志对象
- 初始化模型
- 根据动作类型调用不同的流程（训练/评估/导出/部署）
"""

import argparse
from modules.utils import load_yaml, LoggerManager
from modules.model import build_model
from modules.dataloader import build_dataloader
from modules.train import train_model
from modules.evaluation import evaluate_model
from modules.export import export_onnx
from modules.deployment import deploy_onnx_server


def main():
    # ---------------- 参数解析 ----------------
    parser = argparse.ArgumentParser(description="LearnAI 主程序")
    parser.add_argument("--action", type=str, required=True,
                        choices=["training", "evaluation", "export", "deploy"],
                        help="执行的动作类型")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径")
    args = parser.parse_args()

    # ---------------- 加载配置和日志 ----------------
    config = load_yaml(args.config)
    logger = LoggerManager.get_logger(__file__)

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Action: {args.action}")

    # ---------------- 初始化模型 ----------------
    model_type = config["model"]["target"]
    model = build_model(config)

    # ---------------- 按动作执行 ----------------
    if args.action == "training":
        mode = config["training"].get("mode", "supervised")
        if mode == "supervised":
            train_supervised_step(config, model)
        elif mode == "unsupervised":
            train_unsupervised_step(config, model)
        elif mode == "rl":
            train_rl_step(config, model)
        else:
            logger.error(f"未知训练模式: {mode}")

    elif args.action == "evaluation":
        mode = config["evaluation"].get("mode", "supervised")
        if mode == "supervised":
            evaluate_supervised_step(config, model)
        elif mode == "unsupervised":
            evaluate_unsupervised_step(config, model)
        else:
            logger.error(f"未知评估模式: {mode}")

    elif args.action == "export":
        export_onnx(config, model=model)

    elif args.action == "deploy":
        deploy_onnx_server(config)

    else:
        logger.error(f"未知动作: {args.action}")


if __name__ == "__main__":
    main()
