# -*- coding: utf-8 -*-
"""
models/model.py
模型构筑模块
"""

from modules.utils import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)

# ---------------- 构建子模块 ----------------
def build_module(module_cfg: dict,):
    """
    根据模块配置构建 Sequential 或其他子模块
    """
    # 实例化模块
    module = call_target(module_cfg["reflection"], {})
    # 遍历层配置
    for layer_cfg in module_cfg.get("layers", []):
        layer = call_target(layer_cfg["reflection"], layer_cfg.get("args", {}))
        module.add(layer)

    return module

def build_model(model_cfg: dict):
    """
    根据 model_cfg 构建模型、优化器和损失函数
    参数:
        model_cfg: YAML 中 model 部分的字典
        logger: 可选的日志对象
    返回:
        dict: 包含 generator, discriminator, gen_optimizer, dis_optimizer, gen_loss_fn, dis_loss_fn
    """
    try:
        logger.info("Building generator module...")
        generator = build_module(model_cfg["generator"])
    except Exception as e:
        logger.error(f"Failed to build generator: {e}")

    try:
        logger and logger.info("Building discriminator module...")
        discriminator = build_module(model_cfg["discriminator"])
    except Exception as e:
        logger and logger.error(f"Failed to build discriminator: {e}")

    try:
        logger.info("Building optimizers...")
        gen_optimizer = call_target(model_cfg["optimizers"]["generator"]["reflection"],
                                    model_cfg["optimizers"]["generator"].get("args", {}))
        dis_optimizer = call_target(model_cfg["optimizers"]["discriminator"]["reflection"],
                                    model_cfg["optimizers"]["discriminator"].get("args", {}))
        logger.info(f"Generator optimizer: {gen_optimizer}")
        logger.info(f"Discriminator optimizer: {dis_optimizer}")
    except Exception as e:
        logger.error(f"Failed to build optimizers: {e}")

    try:
        logger.info("Building loss functions...")
        gen_loss_fn = call_target(model_cfg["losses"]["generator"])
        dis_loss_fn = call_target(model_cfg["losses"]["discriminator"])

        logger.info(f"Generator loss: {gen_loss_fn}")
        logger.info(f"Discriminator loss: {dis_loss_fn}")
    except Exception as e:
        logger.error(f"Failed to build loss functions: {e}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "gen_optimizer": gen_optimizer,
        "dis_optimizer": dis_optimizer,
        "gen_loss_fn": gen_loss_fn,
        "dis_loss_fn": dis_loss_fn
    }
