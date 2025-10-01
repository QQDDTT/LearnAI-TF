# -*- coding: utf-8 -*-

from modules.utils import Logger, call_target

logger = Logger(__file__)


def export_onnx(ex_cfg: dict, model_dict: dict):
    """
    导出模型为 ONNX 格式
    参数:
        config (dict): 配置文件内容，需包含 "export" 部分
    返回:
        None
    """
    path = ex_cfg["path"]
    logger.info(f"Exporting model to ONNX at {path}")

    logger.warning("Building...")

    # TODO: 使用 tf2onnx.convert.from_keras(model) 导出真实 ONNX
    return None
