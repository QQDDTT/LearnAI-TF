# -*- coding: utf-8 -*-
"""
modules/reward_functions.py
奖励函数构建器：从配置上下文构建强化学习奖励函数

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from common.train_context import TrainContext
from common.common import call_target


class RewardFunctionBuilder:
    """
    奖励函数构建器

    职责：
    1. 读取 TrainContext 中的奖励函数配置
    2. 使用 call_target 实例化奖励函数
    3. 支持简单奖励和复合奖励
    4. 支持奖励塑造和归一化
    5. 支持内在奖励（好奇心驱动）

    注意：所有奖励函数通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化奖励函数构建器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.reward_functions: Dict[str, Any] = {}
        self.reward_stats: Dict[str, Dict[str, float]] = {}

    def build_all_reward_functions(self) -> Dict[str, Callable]:
        """
        构建所有奖励函数

        返回:
            奖励函数字典 {reward_name: reward_function}
        """
        # 检查是否在强化学习模式
        if self.context.training_mode != "reinforcement":
            raise ValueError(
                "奖励函数仅在强化学习模式下使用，"
                f"当前模式: {self.context.training_mode}"
            )

        # 从配置中查找奖励函数定义
        # 通常在 training_pipeline 或自定义配置中
        reward_configs = self._extract_reward_configs()

        if not reward_configs:
            raise ValueError("未找到奖励函数配置")

        rewards = {}

        for reward_name, reward_config in reward_configs.items():
            try:
                reward_fn = self.build_reward_function(reward_name, reward_config)
                rewards[reward_name] = reward_fn
            except Exception as e:
                raise RuntimeError(f"构建奖励函数 '{reward_name}' 失败: {e}")

        self.reward_functions = rewards
        return rewards

    def _extract_reward_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        从上下文中提取奖励函数配置

        返回:
            奖励函数配置字典
        """
        # 尝试从多个位置查找奖励函数配置
        configs = {}

        # 1. 从 training_pipeline 的参数中查找
        if "reinforcement" in self.context.training_pipelines:
            pipeline = self.context.training_pipelines["reinforcement"]
            if hasattr(pipeline, 'parameters') and 'reward_functions' in pipeline.parameters:
                configs = pipeline.parameters['reward_functions']

        # 2. 从自定义字段查找（如果有）
        if hasattr(self.context, 'reward_functions'):
            configs.update(self.context.reward_functions)

        # 3. 如果没有找到，使用默认配置
        if not configs:
            configs = {
                "default_reward": {
                    "reflection": "lib.reinforcement_learning:compute_reward",
                    "args": {}
                }
            }

        return configs

    def build_reward_function(
        self,
        reward_name: str,
        reward_config: Dict[str, Any]
    ) -> Callable:
        """
        构建单个奖励函数

        参数:
            reward_name: 奖励函数名称
            reward_config: 奖励函数配置

        返回:
            奖励函数
        """
        if 'reflection' not in reward_config:
            raise ValueError(f"奖励函数 '{reward_name}' 缺少 reflection 字段")

        # 准备参数
        args = reward_config.get('args', {})

        # 使用 call_target 创建奖励函数
        reward_fn = call_target(
            reflection=reward_config['reflection'],
            args=args
        )

        # 如果返回的不是函数，包装成函数
        if not callable(reward_fn):
            base_fn = reward_fn
            reward_fn = lambda *args, **kwargs: base_fn

        # 初始化统计信息
        self.reward_stats[reward_name] = {
            "total": 0.0,
            "count": 0,
            "mean": 0.0,
            "min": float('inf'),
            "max": float('-inf')
        }

        return reward_fn

    def get_reward_function(self, reward_name: str) -> Callable:
        """
        获取指定的奖励函数

        参数:
            reward_name: 奖励函数名称

        返回:
            奖励函数
        """
        if reward_name not in self.reward_functions:
            raise ValueError(f"奖励函数 '{reward_name}' 不存在")

        return self.reward_functions[reward_name]

    def compute_reward(
        self,
        reward_name: str,
        state: Any,
        action: Any,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算奖励

        参数:
            reward_name: 奖励函数名称
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息

        返回:
            奖励值
        """
        reward_fn = self.get_reward_function(reward_name)

        # 调用奖励函数
        try:
            reward = reward_fn(
                state=state,
                action=action,
                next_state=next_state,
                done=done,
                info=info or {}
            )
        except TypeError:
            # 如果签名不匹配，尝试简化调用
            try:
                reward = reward_fn(next_state, done, info or {})
            except:
                reward = reward_fn(info or {})

        # 更新统计信息
        self._update_stats(reward_name, float(reward))

        return float(reward)

    def _update_stats(self, reward_name: str, reward: float):
        """更新奖励统计信息"""
        if reward_name not in self.reward_stats:
            self.reward_stats[reward_name] = {
                "total": 0.0,
                "count": 0,
                "mean": 0.0,
                "min": float('inf'),
                "max": float('-inf')
            }

        stats = self.reward_stats[reward_name]
        stats["total"] += reward
        stats["count"] += 1
        stats["mean"] = stats["total"] / stats["count"]
        stats["min"] = min(stats["min"], reward)
        stats["max"] = max(stats["max"], reward)

    def create_composite_reward(
        self,
        reward_components: Dict[str, str],
        weights: Optional[Dict[str, float]] = None
    ) -> Callable:
        """
        创建复合奖励函数

        参数:
            reward_components: 奖励组件映射 {component_name: reward_name}
            weights: 权重映射（可选）

        返回:
            复合奖励函数
        """
        if not reward_components:
            raise ValueError("reward_components 不能为空")

        # 获取所有奖励函数
        reward_fns = {}
        for comp_name, reward_name in reward_components.items():
            if reward_name not in self.reward_functions:
                raise ValueError(f"奖励函数 '{reward_name}' 不存在")
            reward_fns[comp_name] = self.reward_functions[reward_name]

        # 设置权重
        if weights is None:
            weights = {name: 1.0 for name in reward_components.keys()}

        def composite_reward(state, action, next_state, done, info):
            total_reward = 0.0

            for comp_name, reward_fn in reward_fns.items():
                try:
                    comp_reward = reward_fn(state, action, next_state, done, info)
                except TypeError:
                    try:
                        comp_reward = reward_fn(next_state, done, info)
                    except:
                        comp_reward = reward_fn(info)

                weight = weights.get(comp_name, 1.0)
                total_reward += weight * comp_reward

            return total_reward

        return composite_reward

    def create_shaped_reward(
        self,
        base_reward_name: str,
        shaping_function: Optional[Callable] = None,
        potential_function: Optional[Callable] = None,
        gamma: float = 0.99
    ) -> Callable:
        """
        创建奖励塑造函数

        参数:
            base_reward_name: 基础奖励函数名称
            shaping_function: 塑形函数（可选）
            potential_function: 势函数（用于基于势的奖励塑造）
            gamma: 折扣因子

        返回:
            塑造后的奖励函数
        """
        if base_reward_name not in self.reward_functions:
            raise ValueError(f"奖励函数 '{base_reward_name}' 不存在")

        base_reward_fn = self.reward_functions[base_reward_name]

        if potential_function is not None:
            # 基于势的奖励塑造：F(s, s') = γΦ(s') - Φ(s)
            def shaped_reward(state, action, next_state, done, info):
                # 基础奖励
                try:
                    base_reward = base_reward_fn(state, action, next_state, done, info)
                except:
                    base_reward = base_reward_fn(info)

                # 势函数差
                if not done:
                    phi_s = potential_function(state)
                    phi_next = potential_function(next_state)
                    shaping = gamma * phi_next - phi_s
                else:
                    phi_s = potential_function(state)
                    shaping = -phi_s

                return base_reward + shaping

        elif shaping_function is not None:
            # 自定义塑形函数
            def shaped_reward(state, action, next_state, done, info):
                try:
                    base_reward = base_reward_fn(state, action, next_state, done, info)
                except:
                    base_reward = base_reward_fn(info)

                shaping = shaping_function(state, action, next_state, done, info)

                return base_reward + shaping

        else:
            # 无塑形，返回原始奖励
            shaped_reward = base_reward_fn

        return shaped_reward

    def create_normalized_reward(
        self,
        base_reward_name: str,
        normalization_type: str = "running_mean_std",
        clip_range: Optional[Tuple[float, float]] = None
    ) -> Callable:
        """
        创建归一化奖励函数

        参数:
            base_reward_name: 基础奖励函数名称
            normalization_type: 归一化类型 ('running_mean_std', 'clip', 'tanh')
            clip_range: 裁剪范围（可选）

        返回:
            归一化奖励函数
        """
        if base_reward_name not in self.reward_functions:
            raise ValueError(f"奖励函数 '{base_reward_name}' 不存在")

        base_reward_fn = self.reward_functions[base_reward_name]

        if normalization_type == "running_mean_std":
            # 运行均值和标准差归一化
            running_mean = call_target(
                reflection="tensorflow:Variable",
                args={
                    "initial_value": 0.0,
                    "trainable": False,
                    "dtype": call_target(
                        reflection="tensorflow:float32",
                        args={}
                    )
                }
            )
            running_var = call_target(
                reflection="tensorflow:Variable",
                args={
                    "initial_value": 1.0,
                    "trainable": False,
                    "dtype": call_target(
                        reflection="tensorflow:float32",
                        args={}
                    )
                }
            )
            count = call_target(
                reflection="tensorflow:Variable",
                args={
                    "initial_value": 0,
                    "trainable": False,
                    "dtype": call_target(
                        reflection="tensorflow:int32",
                        args={}
                    )
                }
            )

            def normalized_reward(state, action, next_state, done, info):
                try:
                    reward = base_reward_fn(state, action, next_state, done, info)
                except:
                    reward = base_reward_fn(info)

                # 更新运行统计
                count.assign_add(1)
                delta = reward - running_mean
                running_mean.assign_add(delta / call_target(
                    reflection="tensorflow:cast",
                    args={"x": count, "dtype": call_target(
                        reflection="tensorflow:float32", args={}
                    )}
                ))
                delta2 = reward - running_mean
                running_var.assign_add(delta * delta2)

                # 归一化
                std = call_target(
                    reflection="tensorflow:sqrt",
                    args={"x": running_var / call_target(
                        reflection="tensorflow:cast",
                        args={"x": call_target(
                            reflection="tensorflow:maximum",
                            args={"x": count, "y": 1}
                        ), "dtype": call_target(
                            reflection="tensorflow:float32", args={}
                        )}
                    )}
                )

                normalized = (reward - running_mean) / (std + 1e-8)

                if clip_range is not None:
                    normalized = call_target(
                        reflection="tensorflow:clip_by_value",
                        args={
                            "t": normalized,
                            "clip_value_min": clip_range[0],
                            "clip_value_max": clip_range[1]
                        }
                    )

                return float(normalized.numpy())

        elif normalization_type == "clip":
            # 简单裁剪
            if clip_range is None:
                raise ValueError("clip 归一化需要指定 clip_range")

            def normalized_reward(state, action, next_state, done, info):
                try:
                    reward = base_reward_fn(state, action, next_state, done, info)
                except:
                    reward = base_reward_fn(info)

                return max(clip_range[0], min(reward, clip_range[1]))

        elif normalization_type == "tanh":
            # tanh 归一化到 [-1, 1]
            def normalized_reward(state, action, next_state, done, info):
                try:
                    reward = base_reward_fn(state, action, next_state, done, info)
                except:
                    reward = base_reward_fn(info)

                normalized = call_target(
                    reflection="tensorflow.math:tanh",
                    args={"x": reward}
                )

                return float(normalized.numpy())

        else:
            raise ValueError(f"不支持的归一化类型: {normalization_type}")

        return normalized_reward

    def create_intrinsic_reward(
        self,
        intrinsic_type: str = "curiosity",
        base_reward_name: Optional[str] = None,
        intrinsic_weight: float = 0.5
    ) -> Callable:
        """
        创建内在奖励函数（好奇心驱动、预测误差等）

        参数:
            intrinsic_type: 内在奖励类型 ('curiosity', 'prediction_error', 'count_based')
            base_reward_name: 基础外在奖励名称（可选）
            intrinsic_weight: 内在奖励权重

        返回:
            内在奖励函数
        """
        base_reward_fn = None
        if base_reward_name is not None:
            if base_reward_name not in self.reward_functions:
                raise ValueError(f"奖励函数 '{base_reward_name}' 不存在")
            base_reward_fn = self.reward_functions[base_reward_name]

        if intrinsic_type == "curiosity":
            # 基于好奇心的内在奖励（需要预测模型）
            # 这里提供一个框架，实际实现需要具体的模型

            def intrinsic_reward(state, action, next_state, done, info):
                # 外在奖励
                extrinsic = 0.0
                if base_reward_fn is not None:
                    try:
                        extrinsic = base_reward_fn(state, action, next_state, done, info)
                    except:
                        extrinsic = base_reward_fn(info)

                # 内在奖励：状态预测误差
                # 注意：这需要训练一个前向模型
                # 这里使用简化版本
                if 'prediction_error' in info:
                    intrinsic = info['prediction_error']
                else:
                    # 默认基于状态差异
                    intrinsic = call_target(
                        reflection="tensorflow:reduce_mean",
                        args={"input_tensor": call_target(
                            reflection="tensorflow:square",
                            args={"x": call_target(
                                reflection="tensorflow:subtract",
                                args={"x": next_state, "y": state}
                            )}
                        )}
                    )
                    intrinsic = float(intrinsic.numpy())

                return extrinsic + intrinsic_weight * intrinsic

        elif intrinsic_type == "prediction_error":
            # 基于预测误差的内在奖励

            def intrinsic_reward(state, action, next_state, done, info):
                extrinsic = 0.0
                if base_reward_fn is not None:
                    try:
                        extrinsic = base_reward_fn(state, action, next_state, done, info)
                    except:
                        extrinsic = base_reward_fn(info)

                # 预测误差作为内在奖励
                if 'prediction_error' in info:
                    intrinsic = info['prediction_error']
                else:
                    intrinsic = 0.0

                return extrinsic + intrinsic_weight * intrinsic

        elif intrinsic_type == "count_based":
            # 基于访问计数的探索奖励
            visit_counts = {}

            def intrinsic_reward(state, action, next_state, done, info):
                extrinsic = 0.0
                if base_reward_fn is not None:
                    try:
                        extrinsic = base_reward_fn(state, action, next_state, done, info)
                    except:
                        extrinsic = base_reward_fn(info)

                # 状态的哈希表示
                state_hash = hash(tuple(state.flatten().tolist()))

                # 更新访问计数
                visit_counts[state_hash] = visit_counts.get(state_hash, 0) + 1

                # 内在奖励：1/sqrt(count)
                intrinsic = 1.0 / (visit_counts[state_hash] ** 0.5)

                return extrinsic + intrinsic_weight * intrinsic

        else:
            raise ValueError(f"不支持的内在奖励类型: {intrinsic_type}")

        return intrinsic_reward

    def create_sparse_to_dense_reward(
        self,
        base_reward_name: str,
        potential_function: Callable,
        gamma: float = 0.99
    ) -> Callable:
        """
        将稀疏奖励转换为稠密奖励

        参数:
            base_reward_name: 基础稀疏奖励名称
            potential_function: 势函数（用于提供中间指导）
            gamma: 折扣因子

        返回:
            稠密奖励函数
        """
        return self.create_shaped_reward(
            base_reward_name=base_reward_name,
            potential_function=potential_function,
            gamma=gamma
        )

    def get_reward_stats(self, reward_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取奖励统计信息

        参数:
            reward_name: 奖励函数名称（可选，None 表示所有）

        返回:
            统计信息字典
        """
        if reward_name is not None:
            if reward_name not in self.reward_stats:
                raise ValueError(f"奖励函数 '{reward_name}' 不存在")
            return {reward_name: self.reward_stats[reward_name]}

        return self.reward_stats.copy()

    def reset_reward_stats(self, reward_name: Optional[str] = None):
        """
        重置奖励统计信息

        参数:
            reward_name: 奖励函数名称（可选，None 表示所有）
        """
        if reward_name is not None:
            if reward_name in self.reward_stats:
                self.reward_stats[reward_name] = {
                    "total": 0.0,
                    "count": 0,
                    "mean": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
        else:
            for name in self.reward_stats:
                self.reward_stats[name] = {
                    "total": 0.0,
                    "count": 0,
                    "mean": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }


def build_reward_functions_from_context(context: TrainContext) -> Dict[str, Callable]:
    """
    从训练上下文构建所有奖励函数的便捷函数

    参数:
        context: 训练上下文

    返回:
        奖励函数字典
    """
    builder = RewardFunctionBuilder(context)
    return builder.build_all_reward_functions()


def create_reward_builder(context: TrainContext) -> RewardFunctionBuilder:
    """
    创建并初始化奖励函数构建器的便捷函数

    参数:
        context: 训练上下文

    返回:
        RewardFunctionBuilder 实例
    """
    builder = RewardFunctionBuilder(context)
    builder.build_all_reward_functions()
    return builder
