# -*- coding: utf-8 -*-
"""
deploy/deployment.py
模型导出与部署模块

该模块提供导出模型为 ONNX 格式以及部署 ONNX 服务的功能。
所有函数都遵循统一的参数签名：
    (config, logger, **kwargs)

- config: 配置文件 dict（包含 export / deployment 部分）
- logger: 日志对象
- kwargs: 预留扩展参数，例如模型对象 model
"""

from modules.utils import call_target, Logger

logger = Logger(__file__)

def deploy_onnx_server(config):
    """
    部署 ONNX Runtime 推理服务
    参数:
        config (dict): 配置文件内容，需包含 "deployment" 部分
        logger (Logger): 日志对象
        **kwargs: 额外参数（例如是否传递模型路径）
    返回:
        None
    """
    host = config["host"]
    port = config["port"]
    logger.info(f"Deploying ONNX server at {host}:{port}")

    logger.warning("Building...")
    pass
