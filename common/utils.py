# -*- coding: utf-8 -*-
"""
common/utils.py
工具函数模块（可通过反射调用）：
- 网络客户端类（用于环境交互）
- 数据加载函数
- 数据处理函数
- 强化学习工具函数
- 推理函数
- 优化器函数
- 指标计算函数
"""

import requests
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, Optional, List, Tuple, Union
import tensorflow as tf


# ======================================================
# 网络客户端（强化学习环境接口）
# ======================================================
class NetworkClient:
    """
    网络客户端：用于和外部环境交互（强化学习等）

    参数:
        host: 服务器主机地址
        port: 服务器端口
        endpoints: 端点映射字典 {endpoint_name: path}
        timeout: 请求超时时间（秒）

    示例:
        >>> client = NetworkClient("localhost", 8000, {"step": "/api/step"})
        >>> response = client.request("step", {"action": 0})
    """
    def __init__(self, host: str, port: int, endpoints: Dict[str, str], timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.endpoints = endpoints
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, endpoint_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        向环境发送请求，获取响应

        参数:
            endpoint_name: 端点名称
            payload: 请求载荷
        返回:
            dict: 响应数据
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(f"未知端点: {endpoint_name}")

        url = self.base_url + self.endpoints[endpoint_name]
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"请求失败 {url}: {str(e)}")

    def close(self):
        """关闭会话"""
        self.session.close()


# ======================================================
# 数据集构建函数
# ======================================================
def build_web_dataset(
    client: NetworkClient,
    endpoint_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    delay: float = 0.0,
    prefetch: int = 2
) -> tf.data.Dataset:
    """
    动态从网络环境获取数据，并生成 tf.data.Dataset

    参数:
        client: NetworkClient 实例
        endpoint_name: 要请求的端点名称
        batch_size: 批大小
        shuffle: 是否打乱
        buffer_size: shuffle 缓冲区大小
        delay: 每次请求后的等待时间（秒）
        prefetch: 预取批次数
    返回:
        tf.data.Dataset
    """
    def generator():
        while True:
            try:
                data = client.request(endpoint_name)
                x = np.array(data["state"], dtype=np.float32)
                y = np.array(data.get("reward", 0.0), dtype=np.float32)
                yield x, y
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                print(f"数据获取错误: {e}")
                continue

    sample_x, sample_y = next(generator())

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=sample_x.shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)
    if prefetch > 0:
        ds = ds.prefetch(prefetch)

    return ds


def build_csv_dataset(
    file_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_split: float = 0.0,
    target_column: str = None,
    feature_columns: List[str] = None,
    normalize: bool = False
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    从 CSV 文件加载数据并返回 tf.data.Dataset

    参数:
        file_path: CSV 文件路径
        batch_size: 批大小
        shuffle: 是否打乱
        validation_split: 验证集比例 (0.0-1.0)
        target_column: 目标列名，默认使用最后一列
        feature_columns: 特征列名列表，默认使用除目标列外的所有列
        normalize: 是否对特征进行标准化
    返回:
        (train_ds, val_ds) 或 train_ds
    """
    df = pd.read_csv(file_path)

    if target_column is None:
        target_column = df.columns[-1]

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32)

    if normalize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-8)

    if validation_split > 0:
        n_val = int(len(X) * validation_split)
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        X_train, X_val = X[n_val:], X[:n_val]
        y_train, y_val = y[n_val:], y[:n_val]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(X_train))

        train_ds = train_ds.batch(batch_size).prefetch(2)
        val_ds = val_ds.batch(batch_size).prefetch(2)

        return train_ds, val_ds
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(2)
        return ds


def build_numpy_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    validation_split: float = 0.0
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    从 NumPy 数组构建 tf.data.Dataset

    参数:
        X: 特征数组 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        batch_size: 批大小
        shuffle: 是否打乱
        validation_split: 验证集比例
    返回:
        (train_ds, val_ds) 或 train_ds
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if validation_split > 0:
        n_val = int(len(X) * validation_split)
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        X_train, X_val = X[n_val:], X[:n_val]
        y_train, y_val = y[n_val:], y[:n_val]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(X_train))

        train_ds = train_ds.batch(batch_size).prefetch(2)
        val_ds = val_ds.batch(batch_size).prefetch(2)

        return train_ds, val_ds
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(2)
        return ds


# ======================================================
# 数据处理函数
# ======================================================
def get_batch(loader: tf.data.Dataset) -> Dict[str, tf.Tensor]:
    """
    从数据加载器获取一个批次

    参数:
        loader: tf.data.Dataset 实例
    返回:
        dict: {"x": features, "y": labels}
    """
    iterator = iter(loader)
    x, y = next(iterator)
    return {"x": x, "y": y}


def get_all_data(loader: tf.data.Dataset) -> Dict[str, np.ndarray]:
    """
    从数据加载器获取所有数据

    参数:
        loader: tf.data.Dataset 实例
    返回:
        dict: {"x": all_features, "y": all_labels}
    """
    x_list, y_list = [], []
    for x, y in loader:
        x_list.append(x.numpy())
        y_list.append(y.numpy())

    return {
        "x": np.concatenate(x_list, axis=0),
        "y": np.concatenate(y_list, axis=0)
    }


# ======================================================
# 推理函数
# ======================================================
def forward(model: tf.keras.Model, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
    """
    模型前向传播

    参数:
        model: Keras模型
        inputs: 输入张量
        training: 是否训练模式
    返回:
        输出张量
    """
    return model(inputs, training=training)


# ======================================================
# 优化器函数
# ======================================================
def compute_gradients(
    loss: tf.Tensor,
    model: Union[tf.keras.Model, List[tf.keras.Model]]
) -> List[Tuple[tf.Tensor, tf.Variable]]:
    """
    计算梯度

    参数:
        loss: 损失张量
        model: 单个模型或模型列表
    返回:
        梯度列表
    """
    if not isinstance(model, list):
        model = [model]

    trainable_vars = []
    for m in model:
        trainable_vars.extend(m.trainable_variables)

    tape = tf.GradientTape()
    with tape:
        gradients = tape.gradient(loss, trainable_vars)

    return list(zip(gradients, trainable_vars))


def apply_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: List[Tuple[tf.Tensor, tf.Variable]],
    model: Union[tf.keras.Model, List[tf.keras.Model]] = None
) -> None:
    """
    应用梯度更新参数

    参数:
        optimizer: Keras优化器
        gradients: 梯度列表
        model: 模型（可选，用于验证）
    """
    optimizer.apply_gradients(gradients)


# ======================================================
# 损失计算函数
# ======================================================
def compute_loss(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    loss_fn: Union[tf.keras.losses.Loss, str]
) -> tf.Tensor:
    """
    计算损失

    参数:
        predictions: 预测值
        targets: 目标值
        loss_fn: 损失函数实例或名称
    返回:
        损失值
    """
    if isinstance(loss_fn, str):
        loss_fn = tf.keras.losses.get(loss_fn)

    return loss_fn(targets, predictions)


# ======================================================
# 指标计算函数
# ======================================================
def compute_metrics(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    metrics: List[str]
) -> Dict[str, float]:
    """
    计算评估指标

    参数:
        predictions: 预测值
        targets: 目标值
        metrics: 指标名称列表 ["accuracy", "precision", "recall", "f1"]
    返回:
        dict: {metric_name: value}
    """
    results = {}

    # 转换为numpy
    pred_np = predictions.numpy() if isinstance(predictions, tf.Tensor) else predictions
    targ_np = targets.numpy() if isinstance(targets, tf.Tensor) else targets

    # 二分类或多分类
    if len(pred_np.shape) > 1 and pred_np.shape[-1] > 1:
        pred_labels = np.argmax(pred_np, axis=-1)
    else:
        pred_labels = (pred_np > 0.5).astype(int)

    if len(targ_np.shape) > 1 and targ_np.shape[-1] > 1:
        true_labels = np.argmax(targ_np, axis=-1)
    else:
        true_labels = targ_np.astype(int)

    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = float(np.mean(pred_labels == true_labels))
        elif metric == "precision":
            tp = np.sum((pred_labels == 1) & (true_labels == 1))
            fp = np.sum((pred_labels == 1) & (true_labels == 0))
            results["precision"] = float(tp / (tp + fp + 1e-8))
        elif metric == "recall":
            tp = np.sum((pred_labels == 1) & (true_labels == 1))
            fn = np.sum((pred_labels == 0) & (true_labels == 1))
            results["recall"] = float(tp / (tp + fn + 1e-8))
        elif metric == "f1":
            if "precision" not in results:
                tp = np.sum((pred_labels == 1) & (true_labels == 1))
                fp = np.sum((pred_labels == 1) & (true_labels == 0))
                precision = tp / (tp + fp + 1e-8)
            else:
                precision = results["precision"]

            if "recall" not in results:
                tp = np.sum((pred_labels == 1) & (true_labels == 1))
                fn = np.sum((pred_labels == 0) & (true_labels == 1))
                recall = tp / (tp + fn + 1e-8)
            else:
                recall = results["recall"]

            results["f1"] = float(2 * precision * recall / (precision + recall + 1e-8))

    return results


# ======================================================
# 强化学习工具函数
# ======================================================
def compute_reward(
    response: dict,
    score_delta_weight: float = 1.0,
    end_bonus: float = 10.0,
    step_penalty: float = 0.01,
    max_value_bonus: float = 5.0
) -> float:
    """
    根据环境响应计算奖励

    参数:
        response: 环境返回的响应
        score_delta_weight: 分数变化权重
        end_bonus: 完成任务奖励
        step_penalty: 每步惩罚
        max_value_bonus: 最大额外奖励
    返回:
        float: 奖励值
    """
    reward = response.get("reward", 0) * score_delta_weight

    if response.get("done", False):
        reward += end_bonus

    if "steps" in response:
        reward -= step_penalty * response["steps"]

    bonus = min(response.get("bonus", 0), max_value_bonus)
    reward += bonus

    return float(reward)


def select_action(
    model_output: int,
    mapping: Optional[Dict[int, str]] = None,
    epsilon: float = 0.0
) -> str:
    """
    根据模型输出选择动作

    参数:
        model_output: 模型输出的动作索引
        mapping: 动作映射表
        epsilon: 随机探索概率
    返回:
        str: 动作名称
    """
    if mapping is None:
        mapping = {
            0: "up", 1: "down", 2: "left", 3: "right",
            4: "special1", 5: "special2", 6: "special3", 7: "special4"
        }

    if epsilon > 0 and np.random.random() < epsilon:
        model_output = np.random.choice(list(mapping.keys()))

    return mapping.get(model_output, "noop")


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义优势估计 (GAE)

    参数:
        rewards: 奖励序列
        values: 状态价值序列
        next_value: 最后状态的下一个价值
        gamma: 折扣因子
        lambda_: GAE 参数
    返回:
        (advantages, returns)
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)

    return advantages, returns


def normalize_rewards(rewards: List[float], eps: float = 1e-8) -> np.ndarray:
    """
    标准化奖励序列

    参数:
        rewards: 奖励列表
        eps: 防止除零的小常数
    返回:
        np.ndarray: 标准化后的奖励
    """
    rewards = np.array(rewards, dtype=np.float32)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def discount_rewards(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    """
    计算折扣累积奖励

    参数:
        rewards: 奖励序列
        gamma: 折扣因子
    返回:
        np.ndarray: 折扣累积奖励
    """
    discounted = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        discounted.insert(0, cumulative)
    return np.array(discounted, dtype=np.float32)


# ======================================================
# 强化学习环境交互函数
# ======================================================
def observe_state(client: NetworkClient, endpoint: str) -> Dict[str, Any]:
    """
    观察环境状态

    参数:
        client: NetworkClient实例
        endpoint: 端点名称
    返回:
        dict: 包含状态信息
    """
    response = client.request(endpoint)
    return {"state": np.array(response.get("state", []), dtype=np.float32)}


def execute_action(
    client: NetworkClient,
    endpoint: str,
    action: Union[str, int]
) -> Dict[str, Any]:
    """
    执行动作

    参数:
        client: NetworkClient实例
        endpoint: 端点名称
        action: 动作
    返回:
        dict: 包含next_state, reward, done等
    """
    response = client.request(endpoint, {"action": action})
    return {
        "next_state": np.array(response.get("next_state", []), dtype=np.float32),
        "reward": float(response.get("reward", 0.0)),
        "done": bool(response.get("done", False)),
        "info": response.get("info", {})
    }


def store_transition(
    buffer: List,
    state: np.ndarray,
    action: Union[str, int],
    reward: float,
    next_state: np.ndarray,
    done: bool
) -> None:
    """
    存储经验到回放缓冲区

    参数:
        buffer: 经验回放缓冲区（列表）
        state: 当前状态
        action: 动作
        reward: 奖励
        next_state: 下一状态
        done: 是否结束
    """
    buffer.append({
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done
    })


def sample_batch(buffer: List, batch_size: int) -> Dict[str, np.ndarray]:
    """
    从回放缓冲区采样批次

    参数:
        buffer: 经验回放缓冲区
        batch_size: 批次大小
    返回:
        dict: 批次数据
    """
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    batch = [buffer[i] for i in indices]

    return {
        "states": np.array([t["state"] for t in batch]),
        "actions": np.array([t["action"] for t in batch]),
        "rewards": np.array([t["reward"] for t in batch]),
        "next_states": np.array([t["next_state"] for t in batch]),
        "dones": np.array([t["done"] for t in batch])
    }


def eval_episode(
    policy: tf.keras.Model,
    client: NetworkClient,
    reset_endpoint: str = "reset",
    step_endpoint: str = "step",
    max_steps: int = 1000,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    评估一个episode

    参数:
        policy: 策略模型
        client: NetworkClient实例
        reset_endpoint: 重置端点
        step_endpoint: 步进端点
        max_steps: 最大步数
        deterministic: 是否确定性策略
    返回:
        dict: episode统计信息
    """
    # 重置环境
    response = client.request(reset_endpoint)
    state = np.array(response["state"], dtype=np.float32)

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # 选择动作
        action_probs = policy(np.expand_dims(state, 0), training=False)
        if deterministic:
            action = int(tf.argmax(action_probs[0]))
        else:
            action = int(tf.random.categorical(tf.math.log(action_probs), 1)[0, 0])

        # 执行动作
        response = client.request(step_endpoint, {"action": action})
        state = np.array(response["state"], dtype=np.float32)
        reward = float(response["reward"])
        done = bool(response["done"])

        total_reward += reward
        steps += 1

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": done
    }


def aggregate_episode_metrics(
    episodes: List[Dict],
    metrics: List[str]
) -> Dict[str, float]:
    """
    聚合多个episode的指标

    参数:
        episodes: episode结果列表
        metrics: 要聚合的指标
    返回:
        dict: 聚合后的指标
    """
    results = {}

    for metric in metrics:
        if metric == "mean_reward":
            results["mean_reward"] = np.mean([ep["total_reward"] for ep in episodes])
        elif metric == "mean_episode_length":
            results["mean_episode_length"] = np.mean([ep["steps"] for ep in episodes])
        elif metric == "success_rate":
            results["success_rate"] = np.mean([ep["success"] for ep in episodes])

    return results
