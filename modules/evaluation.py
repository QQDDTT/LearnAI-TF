# -*- coding: utf-8 -*-
"""
modules/evaluation.py
评估模块：
- 模型评估
- 指标计算
- 结果记录和可视化
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional
from common.common import LoggerManager
from common import utils

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主评估函数
# ======================================================
def run_evaluation(eval_config: Dict, context: Dict) -> Dict[str, Any]:
    """
    根据训练模式执行评估

    参数:
        eval_config: 评估配置
        context: 训练上下文

    返回:
        dict: 评估结果

    示例:
        >>> results = run_evaluation(eval_config, context)
        >>> print(results)
    """
    training_mode = context["training_mode"]
    logger.info(f"执行 {training_mode} 模式评估")

    if training_mode == "supervised":
        return evaluate_supervised(eval_config.get("supervised_eval", {}), context)
    elif training_mode == "reinforcement":
        return evaluate_reinforcement(eval_config.get("rl_eval", {}), context)
    elif training_mode in ["self_supervised", "unsupervised"]:
        return evaluate_unsupervised(eval_config.get("unsupervised_eval", {}), context)
    else:
        logger.warning(f"未定义的评估方式: {training_mode}")
        return {}


# ======================================================
# 监督学习评估
# ======================================================
def evaluate_supervised(eval_config: Dict, context: Dict) -> Dict[str, float]:
    """
    监督学习评估

    参数:
        eval_config: 评估配置
            - split: 数据集划分 (val/test)
            - metrics: 指标列表
            - batch_size: 批大小
        context: 上下文

    返回:
        dict: 评估指标 {metric_name: value}

    示例:
        >>> results = evaluate_supervised(eval_config, context)
        >>> print(f"准确率: {results['accuracy']:.4f}")
    """
    split = eval_config.get("split", "val")
    metrics_list = eval_config.get("metrics", ["accuracy"])

    # 获取数据加载器和模型
    dataloader = context["dataloaders"].get(split)
    model_name = eval_config.get("model", "classifier")
    model = context["models"].get(model_name)

    if dataloader is None:
        logger.error(f"无法获取 {split} 数据集")
        return {}

    if model is None:
        logger.error(f"无法获取模型: {model_name}")
        return {}

    logger.info(f"在 {split} 集上评估模型 {model_name}")

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    # 获取损失函数（如果配置了）
    loss_fn = None
    if "loss_fn" in eval_config:
        loss_fn = context["losses"].get(eval_config["loss_fn"])

    # 遍历数据集
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x = batch["x"]
            y = batch["y"]

        # 前向传播
        predictions = model(x, training=False)

        # 计算损失
        if loss_fn is not None:
            loss = loss_fn(y, predictions)
            total_loss += float(loss.numpy())

        all_predictions.append(predictions.numpy())
        all_targets.append(y.numpy())
        num_batches += 1

    # 拼接所有批次
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    results = utils.compute_metrics(all_predictions, all_targets, metrics_list)

    # 添加平均损失
    if loss_fn is not None:
        results["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

    # 打印结果
    logger.info("=" * 60)
    logger.info("评估结果:")
    for metric_name, metric_value in results.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("=" * 60)

    return results


# ======================================================
# 强化学习评估
# ======================================================
def evaluate_reinforcement(eval_config: Dict, context: Dict) -> Dict[str, float]:
    """
    强化学习评估

    参数:
        eval_config: 评估配置
            - eval_episodes: 评估的episode数量
            - metrics: 指标列表
            - max_steps: 每个episode的最大步数
        context: 上下文

    返回:
        dict: 评估指标

    示例:
        >>> results = evaluate_reinforcement(eval_config, context)
        >>> print(f"平均奖励: {results['mean_reward']:.2f}")
    """
    eval_episodes = eval_config.get("eval_episodes", 10)
    metrics_list = eval_config.get("metrics", ["mean_reward"])
    max_steps = eval_config.get("max_steps", 1000)

    # 获取模型和客户端
    policy_name = eval_config.get("policy", "actor")
    policy = context["models"].get(policy_name)
    rl_client = context["dataloaders"].get("client")

    if policy is None:
        logger.error(f"无法获取策略模型: {policy_name}")
        return {}

    if rl_client is None:
        logger.error("无法获取RL客户端")
        return {}

    logger.info(f"运行 {eval_episodes} 个评估episode")

    episode_results = []

    for episode in range(eval_episodes):
        # 运行一个完整的episode
        result = run_eval_episode(policy, rl_client, max_steps)
        episode_results.append(result)

        logger.debug(f"  Episode {episode + 1}/{eval_episodes}: "
                    f"奖励={result['total_reward']:.2f}, "
                    f"步数={result['steps']}, "
                    f"成功={'是' if result['success'] else '否'}")

    # 聚合指标
    results = aggregate_episode_metrics(episode_results, metrics_list)

    # 打印结果
    logger.info("=" * 60)
    logger.info("评估结果:")
    for metric_name, metric_value in results.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("=" * 60)

    return results


def run_eval_episode(
    policy: tf.keras.Model,
    rl_client,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """
    运行一个评估episode（确定性策略，不探索）

    参数:
        policy: 策略模型
        rl_client: RL客户端
        max_steps: 最大步数

    返回:
        dict: episode结果
            - total_reward: 总奖励
            - steps: 步数
            - success: 是否成功
    """
    # 重置环境
    reset_response = rl_client.request("reset")
    state = np.array(reset_response["state"], dtype=np.float32)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # 确定性策略（选择概率最大的动作）
        action_probs = policy(np.expand_dims(state, 0), training=False)
        action = int(tf.argmax(action_probs[0]))

        # 执行动作
        try:
            step_response = rl_client.request("step", {"action": action})
            state = np.array(step_response["state"], dtype=np.float32)
            reward = float(step_response["reward"])
            done = bool(step_response["done"])

            total_reward += reward
            steps += 1
        except Exception as e:
            logger.error(f"执行动作时出错: {str(e)}")
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": done
    }


def aggregate_episode_metrics(
    episode_results: List[Dict],
    metrics_list: List[str]
) -> Dict[str, float]:
    """
    聚合多个episode的指标

    参数:
        episode_results: episode结果列表
        metrics_list: 要聚合的指标

    返回:
        dict: 聚合后的指标
    """
    results = {}

    if "mean_reward" in metrics_list:
        rewards = [ep["total_reward"] for ep in episode_results]
        results["mean_reward"] = float(np.mean(rewards))
        results["std_reward"] = float(np.std(rewards))
        results["min_reward"] = float(np.min(rewards))
        results["max_reward"] = float(np.max(rewards))

    if "mean_episode_length" in metrics_list:
        lengths = [ep["steps"] for ep in episode_results]
        results["mean_episode_length"] = float(np.mean(lengths))
        results["std_episode_length"] = float(np.std(lengths))

    if "success_rate" in metrics_list:
        successes = [ep["success"] for ep in episode_results]
        results["success_rate"] = float(np.mean(successes))

    return results


# ======================================================
# 无监督/自监督学习评估
# ======================================================
def evaluate_unsupervised(eval_config: Dict, context: Dict) -> Dict[str, float]:
    """
    无监督/自监督学习评估

    参数:
        eval_config: 评估配置
        context: 上下文

    返回:
        dict: 评估指标

    示例:
        >>> results = evaluate_unsupervised(eval_config, context)
    """
    logger.info("无监督学习评估")

    eval_type = eval_config.get("type", "reconstruction")

    if eval_type == "reconstruction":
        return evaluate_reconstruction(eval_config, context)
    elif eval_type == "clustering":
        return evaluate_clustering(eval_config, context)
    else:
        logger.warning(f"未知的无监督评估类型: {eval_type}")
        return {}


def evaluate_reconstruction(eval_config: Dict, context: Dict) -> Dict[str, float]:
    """
    评估重构质量（自编码器）

    参数:
        eval_config: 评估配置
        context: 上下文

    返回:
        dict: 重构指标
    """
    dataloader = context["dataloaders"].get("val") or context["dataloaders"].get("train")
    encoder = context["models"].get("encoder")
    decoder = context["models"].get("decoder")

    if dataloader is None or encoder is None or decoder is None:
        logger.error("无法获取数据集或模型")
        return {}

    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch["x"]

        # 编码-解码
        z = encoder(x, training=False)
        x_reconstructed = decoder(z, training=False)

        # 计算重构误差
        mse = tf.reduce_mean(tf.square(x - x_reconstructed))
        mae = tf.reduce_mean(tf.abs(x - x_reconstructed))

        total_mse += float(mse.numpy())
        total_mae += float(mae.numpy())
        num_batches += 1

    results = {
        "reconstruction_mse": total_mse / num_batches if num_batches > 0 else 0.0,
        "reconstruction_mae": total_mae / num_batches if num_batches > 0 else 0.0
    }

    logger.info("重构评估结果:")
    for metric_name, metric_value in results.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    return results


def evaluate_clustering(eval_config: Dict, context: Dict) -> Dict[str, float]:
    """
    评估聚类质量

    参数:
        eval_config: 评估配置
        context: 上下文

    返回:
        dict: 聚类指标
    """
    # 获取聚类结果
    centroids = context.get("centroids")
    assignments = context.get("assignments")
    data = context.get("data")

    if centroids is None or assignments is None or data is None:
        logger.error("无法获取聚类结果")
        return {}

    # 计算轮廓系数（需要 scikit-learn）
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        results = {
            "silhouette_score": float(silhouette_score(data, assignments)),
            "davies_bouldin_score": float(davies_bouldin_score(data, assignments)),
            "calinski_harabasz_score": float(calinski_harabasz_score(data, assignments))
        }

        logger.info("聚类评估结果:")
        for metric_name, metric_value in results.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        return results

    except ImportError:
        logger.warning("未安装scikit-learn，无法计算聚类指标")
        logger.info("如需聚类评估功能，请运行: pip install scikit-learn")

        # 返回基本信息
        return {
            "num_clusters": len(np.unique(assignments)),
            "total_samples": len(assignments),
            "note": "需要安装scikit-learn才能计算完整的聚类指标"
        }


# ======================================================
# 评估工具函数
# ======================================================
def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    计算混淆矩阵

    参数:
        predictions: 预测结果 [n_samples, n_classes] 或 [n_samples]
        targets: 真实标签 [n_samples, n_classes] 或 [n_samples]
        num_classes: 类别数（可选，自动推断）

    返回:
        混淆矩阵 [num_classes, num_classes]

    示例:
        >>> cm = compute_confusion_matrix(predictions, targets)
        >>> print(cm)
    """
    # 转换为标签索引
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        pred_labels = np.argmax(predictions, axis=-1)
    else:
        pred_labels = predictions.astype(int)

    if len(targets.shape) > 1 and targets.shape[-1] > 1:
        true_labels = np.argmax(targets, axis=-1)
    else:
        true_labels = targets.astype(int)

    # 推断类别数
    if num_classes is None:
        num_classes = max(pred_labels.max(), true_labels.max()) + 1

    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for true_label, pred_label in zip(true_labels, pred_labels):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


def print_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    打印混淆矩阵

    参数:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称列表

    示例:
        >>> print_confusion_matrix(cm, ["cat", "dog"])
    """
    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class{i}" for i in range(num_classes)]

    logger.info("混淆矩阵:")

    # 打印表头
    header = "真实\\预测 |" + " | ".join([f"{name:^10}" for name in class_names])
    logger.info(header)
    logger.info("-" * len(header))

    # 打印每一行
    for i, row in enumerate(confusion_matrix):
        row_str = f"{class_names[i]:^10} |" + " | ".join([f"{val:^10}" for val in row])
        logger.info(row_str)


def compute_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    生成分类报告

    参数:
        predictions: 预测结果
        targets: 真实标签
        class_names: 类别名称

    返回:
        dict: 每个类别的精确率、召回率、F1分数

    示例:
        >>> report = compute_classification_report(predictions, targets)
        >>> print(report)
    """
    # 转换为标签索引
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        pred_labels = np.argmax(predictions, axis=-1)
        num_classes = predictions.shape[-1]
    else:
        pred_labels = predictions.astype(int)
        num_classes = pred_labels.max() + 1

    if len(targets.shape) > 1 and targets.shape[-1] > 1:
        true_labels = np.argmax(targets, axis=-1)
    else:
        true_labels = targets.astype(int)

    if class_names is None:
        class_names = [f"Class{i}" for i in range(num_classes)]

    report = {}

    for i, class_name in enumerate(class_names):
        # 计算该类别的指标
        tp = np.sum((pred_labels == i) & (true_labels == i))
        fp = np.sum((pred_labels == i) & (true_labels != i))
        fn = np.sum((pred_labels != i) & (true_labels == i))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        support = np.sum(true_labels == i)

        report[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": int(support)
        }

    return report


def print_classification_report(report: Dict[str, Dict[str, float]]) -> None:
    """
    打印分类报告

    参数:
        report: 分类报告字典

    示例:
        >>> print_classification_report(report)
    """
    logger.info("分类报告:")
    logger.info(f"{'类别':^15} | {'精确率':^10} | {'召回率':^10} | {'F1分数':^10} | {'样本数':^10}")
    logger.info("-" * 70)

    for class_name, metrics in report.items():
        logger.info(f"{class_name:^15} | "
                   f"{metrics['precision']:^10.4f} | "
                   f"{metrics['recall']:^10.4f} | "
                   f"{metrics['f1-score']:^10.4f} | "
                   f"{metrics['support']:^10}")


def save_evaluation_results(
    results: Dict[str, float],
    output_path: str,
    format: str = "json"
) -> None:
    """
    保存评估结果到文件

    参数:
        results: 评估结果字典
        output_path: 输出路径
        format: 保存格式 ("json" 或 "csv")

    示例:
        >>> save_evaluation_results(results, "eval_results.json")
    """
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"评估结果已保存到: {output_path}")

    elif format == "csv":
        import pandas as pd
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
        logger.info(f"评估结果已保存到: {output_path}")

    else:
        logger.error(f"未知的保存格式: {format}")
