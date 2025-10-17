# -*- coding: utf-8 -*-
"""
lib/reinforcement_learning.py
强化学习工具：网络通信、经验回放、策略选择、奖励计算、优势估计等
"""

import numpy as np
import requests
import tensorflow as tf
from typing import Dict, Any, Optional, List
from collections import deque


# ======================================================
# 网络通信
# ======================================================

class NetworkClient:
    """网络客户端：用于与强化学习环境通信"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: int = 30,
        endpoints: Optional[Dict[str, str]] = None
    ):
        """
        初始化网络客户端

        参数:
            host: 服务器地址
            port: 服务器端口
            timeout: 请求超时时间（秒）
            endpoints: API端点配置
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.endpoints = endpoints or {
            "reset": "/api/reset",
            "step": "/api/step",
            "state": "/api/state"
        }
        self.session = requests.Session()

    def request(self, endpoint_name: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        发送HTTP请求

        参数:
            endpoint_name: 端点名称
            data: 请求数据

        返回:
            响应数据
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(f"未知的端点: {endpoint_name}")

        url = self.base_url + self.endpoints[endpoint_name]

        try:
            if data:
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                response = self.session.get(url, timeout=self.timeout)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时: {url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"连接失败: {url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP错误: {url}, 状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {url}, 错误: {e}")

    def close(self):
        """关闭会话"""
        self.session.close()


def build_web_dataset(
    client: NetworkClient,
    endpoint_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    delay: float = 0.01,
    **kwargs
):
    """
    构建基于网络请求的数据集

    参数:
        client: NetworkClient实例
        endpoint_name: 端点名称
        batch_size: 批次大小
        shuffle: 是否打乱
        buffer_size: 缓冲区大小
        delay: 请求延迟（秒）

    返回:
        数据生成器
    """
    if not isinstance(client, NetworkClient):
        raise TypeError(f"client 必须是 NetworkClient 实例")

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0: {batch_size}")

    def data_generator():
        buffer = deque(maxlen=buffer_size)

        while True:
            try:
                response = client.request(endpoint_name)
                data = response.get('data')

                if data:
                    buffer.append(data)

                if len(buffer) >= batch_size:
                    batch = list(buffer)[:batch_size]
                    if shuffle:
                        np.random.shuffle(batch)
                    yield batch

                import time
                time.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"数据生成器错误: {e}")

    return data_generator()


# ======================================================
# 环境交互
# ======================================================

def observe_state(
    client: NetworkClient,
    endpoint: str = "state",
    **kwargs
) -> Dict[str, Any]:
    """
    观察环境状态

    参数:
        client: NetworkClient实例
        endpoint: 端点名称

    返回:
        包含state的字典
    """
    response = client.request(endpoint)

    if "state" not in response:
        raise ValueError("响应中缺少 'state' 字段")

    return {"state": response["state"]}


def execute_action(
    client: NetworkClient,
    endpoint: str,
    action: str,
    **kwargs
) -> Dict[str, Any]:
    """
    执行动作

    参数:
        client: NetworkClient实例
        endpoint: 端点名称
        action: 动作名称

    返回:
        包含next_state, reward, done的字典
    """
    response = client.request(endpoint, data={"action": action})

    return {
        "next_state": response.get("next_state"),
        "reward": response.get("reward", 0.0),
        "done": response.get("done", False),
        "info": response.get("info", {})
    }


# ======================================================
# 策略选择
# ======================================================

def select_action(
    model_output: Any,
    mapping: Dict[int, str],
    epsilon: float = 0.0,
    **kwargs
) -> str:
    """
    选择动作（支持ε-greedy策略）

    参数:
        model_output: 模型输出（动作概率分布或Q值）
        mapping: 动作映射表 {索引: 动作名称}
        epsilon: 探索率

    返回:
        动作名称
    """
    if not 0 <= epsilon <= 1:
        raise ValueError(f"epsilon 必须在 [0, 1] 范围内: {epsilon}")

    if not mapping:
        raise ValueError("动作映射表不能为空")

    # ε-greedy策略
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(mapping))
    else:
        if hasattr(model_output, 'numpy'):
            probs = model_output.numpy()
        else:
            probs = model_output

        action_idx = np.argmax(probs)

    if action_idx not in mapping:
        raise KeyError(f"动作索引 {action_idx} 不在映射表中")

    return mapping[action_idx]


# ======================================================
# 奖励计算
# ======================================================

def compute_reward(
    response: Dict[str, Any],
    score_delta_weight: float = 1.0,
    end_bonus: float = 100.0,
    step_penalty: float = 0.01,
    max_value_bonus: float = 10.0,
    **kwargs
) -> float:
    """
    计算奖励

    参数:
        response: 执行动作的响应
        score_delta_weight: 分数变化的权重
        end_bonus: 完成任务的大额奖励
        step_penalty: 每步小惩罚
        max_value_bonus: 额外奖励的上限

    返回:
        奖励值
    """
    reward = response.get("reward", 0.0)

    # 基础奖励
    total_reward = reward * score_delta_weight

    # 完成奖励
    if response.get("done", False) and response.get("success", False):
        total_reward += end_bonus

    # 步骤惩罚
    total_reward -= step_penalty

    # 限制奖励范围
    total_reward = np.clip(total_reward, -max_value_bonus, max_value_bonus)

    return float(total_reward)


def normalize_rewards(rewards: np.ndarray, **kwargs) -> np.ndarray:
    """
    奖励标准化

    参数:
        rewards: 奖励数组

    返回:
        标准化后的奖励
    """
    if len(rewards) == 0:
        raise ValueError("奖励数组为空")

    mean = np.mean(rewards)
    std = np.std(rewards)

    return (rewards - mean) / (std + 1e-8)


def discount_rewards(
    rewards: List[float],
    gamma: float = 0.99,
    **kwargs
) -> np.ndarray:
    """
    折扣累积奖励

    参数:
        rewards: 奖励列表
        gamma: 折扣因子

    返回:
        折扣后的累积奖励
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma 必须在 [0, 1] 范围内: {gamma}")

    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0

    for t in reversed(range(len(rewards))):
        cumulative = rewards[t] + gamma * cumulative
        discounted[t] = cumulative

    return discounted


# ======================================================
# 经验回放
# ======================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        """
        初始化

        参数:
            capacity: 缓冲区容量
        """
        if capacity <= 0:
            raise ValueError(f"容量必须大于 0: {capacity}")

        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """
        采样批次

        参数:
            batch_size: 批次大小

        返回:
            经验列表
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"批次大小 {batch_size} 超过缓冲区大小 {len(self.buffer)}")

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)


def store_transition(
    buffer: ReplayBuffer,
    state: np.ndarray,
    action: Any,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    **kwargs
):
    """
    存储经验

    参数:
        buffer: ReplayBuffer实例
        state: 当前状态
        action: 动作
        reward: 奖励
        next_state: 下一个状态
        done: 是否结束
    """
    if not isinstance(buffer, ReplayBuffer):
        raise TypeError("buffer 必须是 ReplayBuffer 实例")

    buffer.store(state, action, reward, next_state, done)


def sample_batch(
    buffer: ReplayBuffer,
    batch_size: int = 32,
    **kwargs
) -> List:
    """
    从缓冲区采样批次

    参数:
        buffer: ReplayBuffer实例
        batch_size: 批次大小

    返回:
        经验批次
    """
    return buffer.sample(batch_size)


# ======================================================
# GAE（广义优势估计）
# ======================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    计算广义优势估计（GAE）

    参数:
        rewards: 奖励列表
        values: 状态价值列表
        next_value: 下一个状态的价值
        gamma: 折扣因子
        lambda_: GAE参数

    返回:
        包含advantages和returns的字典
    """
    if len(rewards) != len(values):
        raise ValueError(f"rewards 和 values 长度不匹配: {len(rewards)} vs {len(values)}")

    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma 必须在 [0, 1] 范围内: {gamma}")

    if not 0 <= lambda_ <= 1:
        raise ValueError(f"lambda_ 必须在 [0, 1] 范围内: {lambda_}")

    rewards = np.array(rewards)
    values = np.array(values)

    # 计算TD误差
    values_next = np.append(values[1:], next_value)
    deltas = rewards + gamma * values_next - values

    # 计算GAE
    advantages = np.zeros_like(rewards)
    advantage = 0
    for t in reversed(range(len(rewards))):
        advantage = deltas[t] + gamma * lambda_ * advantage
        advantages[t] = advantage

    # 计算returns
    returns = advantages + values

    return {
        "advantages": advantages,
        "returns": returns
    }


# ======================================================
# Episode评估
# ======================================================

def eval_episode(
    policy: Any,
    client: NetworkClient,
    reset_endpoint: str = "reset",
    step_endpoint: str = "step",
    max_steps: int = 1000,
    deterministic: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    评估一个episode

    参数:
        policy: 策略模型
        client: NetworkClient实例
        reset_endpoint: 重置端点
        step_endpoint: 步骤端点
        max_steps: 最大步数
        deterministic: 是否使用确定性策略

    返回:
        episode结果
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps 必须大于 0: {max_steps}")

    # 重置环境
    response = client.request(reset_endpoint)
    state = response.get("state")

    if state is None:
        raise ValueError("重置响应中缺少 'state' 字段")

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # 选择动作
        action_probs = policy(np.expand_dims(state, 0), training=False)

        if deterministic:
            action_idx = np.argmax(action_probs)
        else:
            action_idx = np.random.choice(len(action_probs[0]), p=action_probs[0])

        # 执行动作
        response = client.request(step_endpoint, data={"action": int(action_idx)})

        reward = response.get("reward", 0.0)
        next_state = response.get("next_state")
        done = response.get("done", False)

        total_reward += reward
        state = next_state
        steps += 1

    return {
        "total_reward": total_reward,
        "episode_length": steps,
        "success": response.get("success", False)
    }


def aggregate_episode_metrics(
    episodes: List[Dict],
    metrics: List[str],
    **kwargs
) -> Dict[str, float]:
    """
    聚合episode指标

    参数:
        episodes: episode结果列表
        metrics: 指标名称列表

    返回:
        聚合后的指标
    """
    if not episodes:
        raise ValueError("episodes 列表为空")

    if not metrics:
        raise ValueError("metrics 列表为空")

    results = {}

    for metric in metrics:
        if metric == "mean_reward":
            results["mean_reward"] = np.mean([ep["total_reward"] for ep in episodes])

        elif metric == "mean_episode_length":
            results["mean_episode_length"] = np.mean([ep["episode_length"] for ep in episodes])

        elif metric == "success_rate":
            results["success_rate"] = np.mean([ep.get("success", False) for ep in episodes])

        else:
            raise ValueError(f"不支持的指标: {metric}")

    return results


# ======================================================
# Actor-Critic更新
# ======================================================

def update_actor_critic(
    actor: Any,
    critic: Any,
    buffer: ReplayBuffer,
    actor_opt: Any,
    critic_opt: Any,
    batch_size: int = 32,
    gamma: float = 0.99,
    **kwargs
):
    """
    更新Actor-Critic网络

    参数:
        actor: Actor网络
        critic: Critic网络
        buffer: 经验回放缓冲区
        actor_opt: Actor优化器
        critic_opt: Critic优化器
        batch_size: 批次大小
        gamma: 折扣因子
    """
    if len(buffer) < batch_size:
        raise ValueError(f"缓冲区大小 {len(buffer)} 小于批次大小 {batch_size}")

    # 采样批次
    batch = buffer.sample(batch_size)

    states = np.array([exp[0] for exp in batch])
    actions = np.array([exp[1] for exp in batch])
    rewards = np.array([exp[2] for exp in batch])
    next_states = np.array([exp[3] for exp in batch])
    dones = np.array([exp[4] for exp in batch])

    # 更新Critic
    with tf.GradientTape() as tape:
        # 当前状态价值
        values = critic(states, training=True)

        # 下一个状态价值
        next_values = critic(next_states, training=False)

        # TD目标
        td_targets = rewards + gamma * next_values * (1 - dones)

        # Critic损失
        critic_loss = tf.reduce_mean(tf.square(td_targets - values))

    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # 更新Actor
    with tf.GradientTape() as tape:
        action_probs = actor(states, training=True)
        values = critic(states, training=False)

        # 优势
        next_values = critic(next_states, training=False)
        td_targets = rewards + gamma * next_values * (1 - dones)
        advantages = td_targets - values

        # Actor损失（策略梯度）
        actor_loss = -tf.reduce_mean(
            tf.math.log(action_probs + 1e-8) * tf.stop_gradient(advantages)
        )

    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return {
        "actor_loss": float(actor_loss.numpy()),
        "critic_loss": float(critic_loss.numpy())
    }
