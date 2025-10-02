# -*- coding: utf-8 -*-
"""
modules/train.py
通用训练模块
- 使用配置文件定义训练流程
- 上下文拆分为 model_dict 和 dataloaders
"""

from typing import Dict, List
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)

# ======================================================
# 单次训练步骤执行
# ======================================================
def train_one_step(step_config: List[dict], model_dict: Dict, batch: Dict) -> Dict:
    """
    执行单次训练过程
    :param step_config: YAML配置中的training.steps（步骤列表）
    :param model_dict: 模型、优化器、损失函数等字典
    :param batch: 当前批次数据字典
    :return: 更新后的 model_dict（last_result 也会写入）
    """
    last_result = None

    for idx, step in enumerate(step_config):
        reflection = step["reflection"]
        arguments = step.get("args", {})

        # -------------------------------
        # 参数解析：支持 last_result / model_dict / batch 引用
        # -------------------------------
        resolved_args = {}
        for k, v in arguments.items():
            if v == "last_result":
                resolved_args[k] = last_result
            elif isinstance(v, str) and v in model_dict:
                resolved_args[k] = model_dict[v]
            elif isinstance(v, str) and v in batch:
                resolved_args[k] = batch[v]
            else:
                resolved_args[k] = v

        # -------------------------------
        # 反射执行
        # -------------------------------
        logger.debug(f"Step {idx+1}: calling {reflection} with {resolved_args}")
        result = call_target(reflection, resolved_args)

        # 保存结果
        last_result = result
        model_dict["last_result"] = result

    return model_dict


# ======================================================
# 训练主循环
# ======================================================
def train_model(training_config: dict, model_dict: Dict, dataloaders: Dict):
    try:
        epochs = training_config.get("epochs", 1)
        step_config = training_config["steps"]

        logger.info(f"Start training for {epochs} epochs")

        train_loader = dataloaders.get("train")
        if train_loader is None:
            raise ValueError("No 'train' dataloader found in dataloaders")

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs} start")

            for batch_idx, batch in enumerate(train_loader):
                try:
                    model_dict = train_one_step(step_config, model_dict, batch)
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Step {batch_idx+1}: last_result={model_dict['last_result']}")
                except Exception as e:
                    logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1} failed: {e}")
                    continue

        logger.info("Training completed")
        return model_dict
    except Exception as e:
        logger.exception(f"Training aborted: {e}")

