# -*- coding: utf-8 -*-
"""
测试端到端流程的脚本
依次执行：训练 -> 评估 -> 导出 -> 部署
"""

from modules.utils import load_yaml, Logger
from modules.model import build_model
from modules.dataloader import build_dataloader
from modules.train import train_model
from modules.evaluation import evaluate_model
from modules.export import export_onnx
from modules.deployment import deploy_onnx_server

logger = Logger(__file__)

def run_pipeline(config_path: str):
    # ---------------- 配置 & 日志 ----------------
    config = load_yaml(config_path)
    logger.info(f"Loaded config from {config_path}")

    # ---------------- 模型初始化 ----------------
    model_config = config["stages"][0]["model"]
    model_dict = build_model(model_config)
    logger.info(f"Model build completed")


    # ----------------  加载数据 ----------------
    data_config = config["stages"][0]["data"]
    dataloaders = build_dataloader(data_config)
    logger.info(f"Dataloader build completed")

    # ---------------- 训练 ----------------
    train_config = config["stages"][0]["training"]
    train_model(train_config, model_dict, dataloaders)
    logger.info(f"Training completed")

    # ---------------- 评估 ----------------
    eval_config = config["stages"][0]["evaluation"]
    evaluate_model(eval_config, model_dict)
    logger.info(f"Evaluation completed")

    # ---------------- 导出 ----------------
    exp_config = config["export"]
    export_onnx(exp_config, model_dict)
    logger.info("Exporting model completed")

    # ---------------- 部署 ----------------
    dep_config = config["deployment"]
    deploy_onnx_server(dep_config)
    logger.info("Deploying model server completed")



if __name__ == "__main__":
    # 默认用一个配置文件跑完整流程
    run_pipeline("config_test.yaml")
