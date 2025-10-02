# -*- coding: utf-8 -*-
"""
modules/evaluate.py
通用评估模块
- 使用配置文件定义评估流程
- 支持多模型、多指标
"""

from typing import Dict, List
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)

# ======================================================
# 单次评估步骤执行
# ======================================================
def evaluate_one_step(step_config: List[dict], model_dict: Dict, batch: Dict) -> Dict:
    """
    执行单次评估步骤
    :param step_config: YAML配置中的evaluation.steps
    :param model_dict: 模型、loss、metrics等
    :param batch: 当前批次数据
    :return: 更新后的 model_dict（会写入 last_result）
    """
    last_result = None

    for idx, step in enumerate(step_config):
        reflection = step["reflection"]
        arguments = step.get("arguments", {})

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

        logger.debug(f"[Eval Step {idx+1}] calling {reflection} with {resolved_args}")
        result = call_target(reflection, resolved_args)

        last_result = result
        model_dict["last_result"] = result

    return model_dict


# ======================================================
# 评估主循环
# ======================================================
def evaluate_model(eval_cfg: dict, model_dict: Dict, dataloaders: Dict):
    """
    通用评估循环
    :param eval_cfg: YAML 中 evaluation 配置段
    :param model_dict: 模型、loss、metrics 等对象
    :param dataloaders: 数据迭代器字典（val/test）
    :return: 每个 batch 的评估结果列表
    """
    try:
        step_config = eval_cfg["steps"]
        dataset_key = eval_cfg.get("dataset", "val")
        tolerance = eval_cfg.get("tolerance", "strict")  # strict / tolerant

        if dataset_key not in dataloaders:
            raise ValueError(f"No dataset '{dataset_key}' found in dataloaders")

        eval_loader = dataloaders[dataset_key]
        results = []

        logger.info(f"Start evaluation on dataset '{dataset_key}' with tolerance='{tolerance}'")

        for batch_idx, batch in enumerate(eval_loader):
            try:
                model_dict = evaluate_one_step(step_config, model_dict, batch)
                results.append(model_dict["last_result"])

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Eval Step {batch_idx+1}: last_result={model_dict['last_result']}")

            except Exception as e:
                if tolerance == "tolerant":
                    logger.error(f"Eval batch {batch_idx+1} failed: {e}, skipped")
                    continue
                else:
                    logger.exception(f"Evaluation stopped at batch {batch_idx+1} due to error")
                    raise

        logger.info("Evaluation completed")
        return results

    except Exception as e:
        logger.exception(f"Evaluation aborted: {e}")
