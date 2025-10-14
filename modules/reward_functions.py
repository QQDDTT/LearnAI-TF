# -*- coding: utf-8 -*-
"""
modules/reward_functions.py
强化学习奖励函数模块：
- 游戏相关奖励函数
- 任务相关奖励函数
- 奖励塑形（Reward Shaping）
"""

from typing import Dict, Any
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 通用奖励函数
# ======================================================
def compute_basic_reward(
    response: Dict[str, Any],
    score_weight: float = 1.0,
    completion_bonus: float = 0.0,
    step_penalty: float = 0.0
) -> float:
    """
    基础奖励计算

    参数:
        response: 环境响应 {"reward": float, "done": bool, "score": float, ...}
        score_weight: 分数权重
        completion_bonus: 完成任务奖励
        step_penalty: 每步惩罚

    返回:
        float: 奖励值
    """
    # 基础奖励
    reward = response.get("reward", 0.0) * score_weight

    # 完成奖励
    if response.get("done", False):
        reward += completion_bonus

    # 步数惩罚（鼓励快速完成）
    if step_penalty > 0:
        reward -= step_penalty

    return float(reward)


def compute_game_reward(
    response: Dict[str, Any],
    win_bonus: float = 100.0,
    lose_penalty: float = -100.0,
    score_delta_weight: float = 1.0
) -> float:
    """
    游戏相关奖励（如Atari游戏）

    参数:
        response: 环境响应
        win_bonus: 胜利奖励
        lose_penalty: 失败惩罚
        score_delta_weight: 分数变化权重

    返回:
        float: 奖励值
    """
    reward = 0.0

    # 分数变化
    score_delta = response.get("score_delta", 0.0)
    reward += score_delta * score_delta_weight

    # 胜负判定
    if response.get("done", False):
        if response.get("win", False):
            reward += win_bonus
        elif response.get("lose", False):
            reward += lose_penalty

    return float(reward)


# ======================================================
# 任务导向奖励函数
# ======================================================
def compute_navigation_reward(
    response: Dict[str, Any],
    goal_reached_bonus: float = 100.0,
    distance_reward_weight: float = 1.0,
    collision_penalty: float = -10.0
) -> float:
    """
    导航任务奖励（如机器人导航）

    参数:
        response: 环境响应
        goal_reached_bonus: 到达目标奖励
        distance_reward_weight: 距离变化权重
        collision_penalty: 碰撞惩罚

    返回:
        float: 奖励值
    """
    reward = 0.0

    # 到达目标
    if response.get("goal_reached", False):
        reward += goal_reached_bonus

    # 距离变化（离目标更近给奖励）
    distance_delta = response.get("distance_delta", 0.0)
    reward += distance_delta * distance_reward_weight

    # 碰撞惩罚
    if response.get("collision", False):
        reward += collision_penalty

    return float(reward)


def compute_manipulation_reward(
    response: Dict[str, Any],
    task_success_bonus: float = 100.0,
    progress_weight: float = 10.0,
    action_penalty: float = 0.01
) -> float:
    """
    操作任务奖励（如机械臂抓取）

    参数:
        response: 环境响应
        task_success_bonus: 任务成功奖励
        progress_weight: 进度奖励权重
        action_penalty: 动作惩罚（鼓励少动作）

    返回:
        float: 奖励值
    """
    reward = 0.0

    # 任务成功
    if response.get("task_success", False):
        reward += task_success_bonus

    # 任务进度
    progress = response.get("progress", 0.0)
    reward += progress * progress_weight

    # 动作惩罚
    reward -= action_penalty

    return float(reward)


# ======================================================
# 奖励塑形（Reward Shaping）
# ======================================================
def apply_potential_based_shaping(
    current_state: Dict,
    next_state: Dict,
    gamma: float = 0.99,
    potential_fn: callable = None
) -> float:
    """
    基于势能的奖励塑形
    保证策略不变性的奖励塑形方法

    公式: F(s, s') = γ * Φ(s') - Φ(s)

    参数:
        current_state: 当前状态
        next_state: 下一状态
        gamma: 折扣因子
        potential_fn: 势能函数

    返回:
        float: 塑形奖励
    """
    if potential_fn is None:
        # 默认势能函数：离目标的距离
        def default_potential(state):
            return -state.get("distance_to_goal", 0.0)
        potential_fn = default_potential

    current_potential = potential_fn(current_state)
    next_potential = potential_fn(next_state)

    shaping_reward = gamma * next_potential - current_potential

    return float(shaping_reward)


def apply_curiosity_bonus(
    state: Dict,
    action: int,
    novelty_score: float
) -> float:
    """
    好奇心驱动的内在奖励
    鼓励探索新状态

    参数:
        state: 状态
        action: 动作
        novelty_score: 新颖度分数（由预测模型计算）

    返回:
        float: 好奇心奖励
    """
    # 新颖度越高，奖励越大
    curiosity_reward = novelty_score

    return float(curiosity_reward)


# ======================================================
# 多目标奖励函数
# ======================================================
def compute_multi_objective_reward(
    response: Dict[str, Any],
    objectives: Dict[str, float]
) -> Dict[str, float]:
    """
    多目标奖励计算

    参数:
        response: 环境响应
        objectives: 目标权重 {"objective1": weight1, "objective2": weight2, ...}

    返回:
        dict: 各目标的奖励和总奖励
    """
    rewards = {}
    total_reward = 0.0

    for objective_name, weight in objectives.items():
        objective_reward = response.get(objective_name, 0.0)
        weighted_reward = objective_reward * weight

        rewards[objective_name] = objective_reward
        total_reward += weighted_reward

    rewards["total"] = total_reward

    return rewards


# ======================================================
# 稀疏奖励处理
# ======================================================
def apply_reward_normalization(
    reward: float,
    running_mean: float,
    running_std: float,
    alpha: float = 0.01
) -> tuple:
    """
    奖励标准化
    处理稀疏奖励问题

    参数:
        reward: 原始奖励
        running_mean: 运行均值
        running_std: 运行标准差
        alpha: 更新率

    返回:
        (normalized_reward, new_mean, new_std)
    """
    # 更新统计量
    new_mean = (1 - alpha) * running_mean + alpha * reward
    new_std = (1 - alpha) * running_std + alpha * abs(reward - running_mean)

    # 标准化
    normalized_reward = (reward - new_mean) / (new_std + 1e-8)

    return float(normalized_reward), float(new_mean), float(new_std)


def apply_reward_clipping(reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    奖励裁剪
    限制奖励的范围

    参数:
        reward: 原始奖励
        min_val: 最小值
        max_val: 最大值

    返回:
        float: 裁剪后的奖励
    """
    return float(max(min_val, min(max_val, reward)))
