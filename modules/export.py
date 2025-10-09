# -*- coding: utf-8 -*-
"""
modules/export.py
通用模型导出（完全配置驱动）
- 配置文件直接指定函数路径和参数
- 输出路径相对于项目根目录自动解析
- 支持 PyTorch/ONNX/TensorFlow 等导出函数
- 异常处理完整
"""

import os
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)

def export_model(export_cfg: dict, model_dict: dict, project_root: str = None):
    """
    根据配置导出模型
    :param export_cfg: YAML 配置段
        reflection: 调用函数全路径，例如 torch.save 或 torch.onnx.export
        arguments: 参数字典，支持引用 model_dict 中的模型或 last_result
    :param model_dict: 模型字典，例如 {"generator": model}
    :param project_root: 项目根目录，用于解析相对路径
    :return: 导出结果
    """
    try:
        reflection = export_cfg["reflection"]
        arguments = export_cfg.get("args", {})

        # 解析参数：引用 model_dict 或 last_result
        resolved_args = {}
        for k, v in arguments.items():
            if v == "last_result":
                resolved_args[k] = model_dict.get("last_result")
            elif isinstance(v, str) and v in model_dict:
                resolved_args[k] = model_dict[v]
            else:
                resolved_args[k] = v

        # 解析输出路径（f 或 save_path）
        for key in ["f", "save_path"]:
            path = resolved_args.get(key)
            if path:
                if not os.path.isabs(path) and project_root:
                    path = os.path.join(project_root, path)
                    path = os.path.abspath(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                resolved_args[key] = path

        logger.info(f"Exporting model using {reflection} with args={resolved_args}")
        result = call_target(reflection, resolved_args)
        model_dict["last_result"] = result
        logger.info(f"Model export completed: {result}")

        return result

    except Exception as e:
        logger.exception(f"Model export failed: {e}")
