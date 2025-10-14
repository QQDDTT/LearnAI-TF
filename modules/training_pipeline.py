# -*- coding: utf-8 -*-
"""
modules/training_pipeline.py
训练流程模块：
- 执行不同训练模式的训练循环
- 调度step_sequence中的步骤
- 管理训练状态
"""

import tensorflow as tf
from typing import Dict, Any, List
from common.common import LoggerManager, call_target
from common import utils

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主训练入口
# ======================================================
def run_training(pipeline_config: Dict, context: Dict) -> None:
    """
    根据配置执行训练流程

    参数:
        pipeline_config: 训练流程配置
        context: 训练上下文（包含models、optimizers、losses等）
    """
    training_mode = context["training_mode"]
    logger.info(f"执行 {training_mode} 训练流程")

    loop_type = pipeline_config.get("loop_type", "epoch_batch")
    parameters = pipeline_config.get("parameters", {})
    step_sequence = pipeline_config.get("step_sequence", [])

    # 根据循环类型选择训练函数
    if loop_type == "epoch_batch":
        run_epoch_batch_training(parameters, step_sequence, context)
    elif loop_type == "episode_step":
        run_episode_step_training(parameters, step_sequence, context)
    elif loop_type == "iteration":
        run_iteration_training(parameters, step_sequence, context)
    else:
        raise ValueError(f"未知的循环类型: {loop_type}")


# ======================================================
# Epoch-Batch训练（监督学习、自监督学习）
# ======================================================
def run_epoch_batch_training(
    parameters: Dict,
    step_sequence: List[Dict],
    context: Dict
) -> None:
    """
    Epoch-Batch训练循环

    流程:
        for epoch in epochs:
            for batch in dataloader:
                execute_step_sequence()
    """
    epochs = parameters.get("epochs", 100)
    train_loader = context["dataloaders"].get("train")

    logger.info(f"开始训练，总epoch数: {epochs}")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # 更新上下文
            context["current_batch"] = batch
            context["current_epoch"] = epoch

            # 执行step序列
            step_results = execute_step_sequence(step_sequence, context)

            # 累积损失
            if "compute_loss" in step_results:
                epoch_loss += float(step_results["compute_loss"])
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"  平均损失: {avg_loss:.4f}")

        # 周期性评估
        if (epoch + 1) % parameters.get("eval_frequency", 10) == 0:
            logger.info("  执行评估...")
            # 评估逻辑在evaluation模块中


# ======================================================
# Episode-Step训练（强化学习）
# ======================================================
def run_episode_step_training(
    parameters: Dict,
    step_sequence: List[Dict],
    context: Dict
) -> None:
    """
    Episode-Step训练循环

    流程:
        for episode in episodes:
            state = env.reset()
            for step in steps:
                execute_step_sequence()
    """
    episodes = parameters.get("episodes", 1000)
    steps_per_episode = parameters.get("steps_per_episode", 500)

    # 获取强化学习客户端
    rl_client = context["dataloaders"].get("client")

    # 初始化epsilon（探索率）
    epsilon = parameters.get("epsilon_start", 1.0)
    epsilon_end = parameters.get("epsilon_end", 0.01)
    epsilon_decay = parameters.get("epsilon_decay", 0.995)

    logger.info(f"开始强化学习训练，总episode数: {episodes}")

    for episode in range(episodes):
        logger.info(f"Episode {episode + 1}/{episodes}")

        # 重置环境
        reset_response = rl_client.request("reset")
        state = reset_response.get("state")

        episode_reward = 0.0
        episode_rewards = []
        episode_values = []

        for step in range(steps_per_episode):
            # 更新上下文
            context["current_episode"] = episode
            context["current_step"] = step
            context["current_state"] = state
            context["epsilon"] = epsilon
            context["episode_rewards"] = episode_rewards
            context["episode_values"] = episode_values

            # 执行step序列
            step_results = execute_step_sequence(step_sequence, context)

            # 更新状态和奖励
            if "execute_action" in step_results:
                next_state = step_results["execute_action"].get("next_state")
                reward = step_results.get("compute_reward", 0.0)
                done = step_results["execute_action"].get("done", False)

                episode_reward += reward
                episode_rewards.append(reward)

                state = next_state

                if done:
                    logger.info(f"  Episode结束于step {step + 1}")
                    break

        logger.info(f"  总奖励: {episode_reward:.2f}")

        # 衰减epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)


# ======================================================
# Iteration训练（无监督聚类等）
# ======================================================
def run_iteration_training(
    parameters: Dict,
    step_sequence: List[Dict],
    context: Dict
) -> None:
    """
    Iteration训练循环（直到收敛）

    流程:
        while not converged:
            execute_step_sequence()
            check_convergence()
    """
    max_iterations = parameters.get("max_iterations", 100)

    logger.info(f"开始迭代训练，最大迭代次数: {max_iterations}")

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")

        # 更新上下文
        context["current_iteration"] = iteration

        # 执行step序列
        step_results = execute_step_sequence(step_sequence, context)

        # 检查收敛
        if "check_convergence" in step_results:
            converged = step_results["check_convergence"].get("converged", False)
            if converged:
                logger.info(f"  在第 {iteration + 1} 次迭代收敛")
                break


# ======================================================
# Step序列执行
# ======================================================
def execute_step_sequence(
    step_sequence: List[Dict],
    context: Dict
) -> Dict[str, Any]:
    """
    执行step序列中的每个步骤

    参数:
        step_sequence: 步骤配置列表
        context: 上下文（包含models、optimizers等）

    返回:
        dict: 每个步骤的结果 {step_name: result}
    """
    results = {}
    last_result = None

    for step_config in step_sequence:
        step_name = step_config.get("name", "unnamed_step")
        reflection = step_config.get("reflection")
        args = step_config.get("args", {})

        # 解析参数引用
        resolved_args = resolve_arguments(args, context, results, last_result)

        try:
            # 执行步骤
            result = call_target(reflection, resolved_args)
            results[step_name] = result
            last_result = result

        except Exception as e:
            logger.error(f"步骤 {step_name} 执行失败: {str(e)}")
            raise

    return results


def resolve_arguments(
    arg_cfg: Dict,
    context: Dict,
    step_results: Dict,
    last_result: Any
) -> Dict:
    """
    解析参数配置中的特殊引用，包括 $last_result, ${context_var}, step_name.field。

    参数:
        arg_cfg: 待解析的参数配置字典。
        context: 训练上下文。
        step_results: 之前步骤的结果字典。
        last_result: 上一步的输出结果。

    返回:
        Dict: 解析后的参数字典。
    """
    resolved = {}

    for key, value in arg_cfg.items():
        # --- 优先级最高的特殊字符串解析 ---
        if isinstance(value, str):
            # 1. 引用上一步结果 ($last_result)
            if value == "$last_result":
                resolved[key] = last_result
                continue # 解析完成，进入下一个键值对

            # 2. 引用上下文变量 (${global_lr})
            elif value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved[key] = context.get(var_name)
                continue # 解析完成，进入下一个键值对

            # 3. 引用指定步骤结果 (fetch_data.x)
            # 检查是否有 '.' 且第一部分是已知的步骤名
            elif "." in value:
                step_name, field = value.split(".", 1)
                if step_name in step_results:
                    step_result = step_results[step_name]
                    if isinstance(step_result, dict):
                        resolved[key] = step_result.get(field)
                    else:
                        # 兼容非字典的步骤结果（例如直接返回Tensor）
                        resolved[key] = step_result
                    continue # 解析完成，进入下一个键值对

            # 如果是普通字符串，或者上述特殊引用未能成功解析（如步骤名不存在）
            resolved[key] = value

        # --- 递归解析 ---
        # 字典递归解析
        elif isinstance(value, dict):
            resolved[key] = resolve_arguments(value, context, step_results, last_result)

        # 列表递归解析
        elif isinstance(value, list):
            resolved[key] = [
                resolve_arguments({"item": item}, context, step_results, last_result)["item"]
                if isinstance(item, (dict, str)) else item
                for item in value
            ]

        # 其他类型直接使用
        else:
            resolved[key] = value

    return resolved


def get_nested_value(d: Dict, path: str) -> Any:
    """
    获取嵌套字典的值

    示例: get_nested_value(config, "training_pipeline.supervised.parameters.epochs")
    """
    keys = path.split(".")
    value = d
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value
