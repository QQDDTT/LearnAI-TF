# -*- coding: utf-8 -*-
"""
modules/reward_functions.py
奖励函数管理器：从配置上下文构建强化学习奖励函数

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
from common.interfaces import RewardFunctionInterface
from common.train_context import TrainContext
from common.common import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)


class RewardFunctionManager(RewardFunctionInterface):
    """
    奖励函数管理器

    职责：
    1. 读取 TrainContext 中的奖励函数配置
    2. 使用 call_target 实例化奖励函数
    3. 支持简单奖励和复合奖励
    4. 支持奖励塑造（Reward Shaping）
    5. 支持奖励归一化和缩放
    6. 支持内在奖励（好奇心驱动）

    标准生命周期：
    initialize() -> validate_rewards() -> build_reward_function() ->
    setup_shaping() -> compute_reward() -> finalize()

    注意：所有奖励函数通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化奖励函数管理器

        参数:
            context: 训练上下文
        """
        super().__init__(context)
        self.reward_functions: Dict[str, Callable] = {}
        self.reward_stats: Dict[str, Dict[str, float]] = {}
        self.shaping_functions: Dict[str, Callable] = {}
        self.reward_configs: Dict[str, Any] = {}

    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有奖励函数配置
        2. 加载奖励函数配置数据
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        logger.info("初始化奖励函数管理器")

        # 验证训练模式
        if self.context.training_mode != "reinforcement":
            logger.warning(
                f"奖励函数通常用于强化学习，当前模式: {self.context.training_mode}"
            )

        # 从配置中提取奖励函数定义
        self.reward_configs = self._extract_reward_configs()

        if not self.reward_configs:
            logger.warning("未找到奖励函数配置，将在需要时使用默认奖励")
            self.reward_configs = {
                "default_reward": {
                    "reflection": "common.utils:compute_reward",
                    "args": {}
                }
            }

        logger.info(f"找到 {len(self.reward_configs)} 个奖励函数配置")

        # 初始化统计信息
        self.reward_stats = {}
        self.shaping_functions = {}

        self._initialized = True

    def validate_rewards(self) -> bool:
        """
        验证奖励函数配置

        返回:
            配置是否有效
        """
        logger.info("验证奖励函数配置")

        if not self.reward_configs:
            logger.error("奖励函数配置为空")
            return False

        # 验证每个奖励函数配置
        for reward_name, reward_config in self.reward_configs.items():
            if not isinstance(reward_config, dict):
                logger.error(f"奖励函数 '{reward_name}' 配置必须是字典")
                return False

            if 'reflection' not in reward_config:
                logger.error(f"奖励函数 '{reward_name}' 缺少 'reflection' 字段")
                return False

            # 验证反射字符串格式
            reflection = reward_config['reflection']
            if ':' not in reflection:
                logger.error(
                    f"奖励函数 '{reward_name}' 的 reflection 格式错误: {reflection}"
                )
                return False

        logger.info("奖励函数配置验证通过")
        return True

    def build_reward_function(
        self,
        reward_name: str,
        reward_config: Any
    ) -> Callable:
        """
        构建单个奖励函数

        参数:
            reward_name: 奖励函数名称
            reward_config: 奖励函数配置

        返回:
            奖励函数实例
        """
        logger.info(f"构建奖励函数: {reward_name}")

        if not isinstance(reward_config, dict):
            raise TypeError(f"奖励函数配置必须是字典: {type(reward_config)}")

        if 'reflection' not in reward_config:
            raise ValueError(f"奖励函数 '{reward_name}' 缺少 reflection 字段")

        # 准备参数
        args = reward_config.get('args', {})

        # 使用 call_target 创建奖励函数
        try:
            reward_fn = call_target(
                reflection=reward_config['reflection'],
                args=args
            )

            # 如果返回的不是函数，包装成函数
            if not callable(reward_fn):
                base_value = reward_fn
                reward_fn = lambda *args, **kwargs: base_value

            # 初始化统计信息
            self.reward_stats[reward_name] = {
                "total": 0.0,
                "count": 0,
                "mean": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "std": 0.0
            }

            logger.info(f"奖励函数 '{reward_name}' 构建成功")
            return reward_fn

        except Exception as e:
            logger.error(f"构建奖励函数 '{reward_name}' 失败: {str(e)}", exc_info=True)
            raise

    def setup_shaping(self) -> None:
        """
        设置奖励塑形（可选）

        职责：
        1. 配置势函数（Potential Function）
        2. 设置奖励缩放和归一化
        3. 配置内在奖励
        """
        logger.info("设置奖励塑形")

        # 检查是否有塑形配置
        for reward_name, reward_config in self.reward_configs.items():
            shaping_config = reward_config.get('shaping', None)

            if shaping_config:
                logger.info(f"为奖励 '{reward_name}' 设置塑形函数")

                shaping_type = shaping_config.get('type', 'potential')

                if shaping_type == 'potential':
                    # 基于势的奖励塑造
                    potential_fn_ref = shaping_config.get('potential_function')
                    if potential_fn_ref:
                        potential_fn = call_target(
                            reflection=potential_fn_ref,
                            args=shaping_config.get('args', {})
                        )
                        self.shaping_functions[reward_name] = {
                            'type': 'potential',
                            'function': potential_fn,
                            'gamma': shaping_config.get('gamma', 0.99)
                        }

                elif shaping_type == 'custom':
                    # 自定义塑形函数
                    shaping_fn_ref = shaping_config.get('shaping_function')
                    if shaping_fn_ref:
                        shaping_fn = call_target(
                            reflection=shaping_fn_ref,
                            args=shaping_config.get('args', {})
                        )
                        self.shaping_functions[reward_name] = {
                            'type': 'custom',
                            'function': shaping_fn
                        }

                elif shaping_type == 'normalize':
                    # 奖励归一化
                    self.shaping_functions[reward_name] = {
                        'type': 'normalize',
                        'method': shaping_config.get('method', 'running_mean_std')
                    }

        self._setup_complete = True

    def compute_reward(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励

        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息

        返回:
            奖励值
        """
        # 使用第一个奖励函数（或默认奖励）
        reward_name = list(self.reward_functions.keys())[0] if self.reward_functions else "default_reward"

        return self.compute_reward_by_name(
            reward_name=reward_name,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            info=info
        )

    def execute(self) -> Dict[str, Any]:
        """
        执行核心逻辑：构建所有奖励函数

        返回:
            奖励函数字典
        """
        logger.info("开始构建所有奖励函数")

        if not self.validate_rewards():
            raise ValueError("奖励函数配置验证失败")

        rewards = {}

        for reward_name, reward_config in self.reward_configs.items():
            try:
                reward_fn = self.build_reward_function(reward_name, reward_config)
                rewards[reward_name] = reward_fn
            except Exception as e:
                logger.error(f"构建奖励函数 '{reward_name}' 失败: {str(e)}")
                raise

        self.reward_functions = rewards

        # 设置奖励塑形
        if not self._setup_complete:
            self.setup_shaping()

        logger.info(f"成功构建 {len(rewards)} 个奖励函数")

        return rewards

    def finalize(self) -> None:
        """
        清理和保存结果

        职责：
        1. 将奖励函数缓存到 TrainContext
        2. 保存奖励统计信息
        3. 记录日志
        """
        logger.info("完成奖励函数管理，保存结果")

        # 保存奖励函数到上下文容器
        if self.reward_functions:
            if not hasattr(self.context.container, 'reward_functions'):
                self.context.container.reward_functions = {}
            self.context.container.reward_functions = self.reward_functions

        # 保存统计信息到执行结果
        if self.reward_stats:
            self.context.execution_results['reward_stats'] = self.reward_stats

        logger.info(f"奖励函数已保存到上下文，包含 {len(self.reward_functions)} 个函数")

    # ========================================================================
    # 扩展方法：奖励计算和管理
    # ========================================================================

    def compute_reward_by_name(
        self,
        reward_name: str,
        state: Any,
        action: Any,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        使用指定名称的奖励函数计算奖励

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
        if reward_name not in self.reward_functions:
            raise ValueError(f"奖励函数 '{reward_name}' 不存在")

        reward_fn = self.reward_functions[reward_name]
        info = info or {}

        # 调用奖励函数
        try:
            # 尝试完整签名
            reward = reward_fn(
                state=state,
                action=action,
                next_state=next_state,
                done=done,
                info=info
            )
        except TypeError:
            # 尝试简化签名
            try:
                reward = reward_fn(next_state, done, info)
            except TypeError:
                try:
                    reward = reward_fn(info)
                except TypeError:
                    # 最后尝试无参数调用
                    reward = reward_fn()

        # 转换为浮点数
        reward = float(reward)

        # 应用奖励塑形
        if reward_name in self.shaping_functions:
            reward = self._apply_shaping(
                reward_name=reward_name,
                base_reward=reward,
                state=state,
                next_state=next_state,
                done=done
            )

        # 更新统计信息
        self._update_reward_stats(reward_name, reward)

        return reward

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

    def create_composite_reward(
        self,
        reward_components: Dict[str, str],
        weights: Optional[Dict[str, float]] = None
    ) -> Callable:
        """
        创建复合奖励函数

        参数:
            reward_components: 奖励组件映射 {component_name: reward_name}
            weights: 权重映射（可选，默认均等权重）

        返回:
            复合奖励函数
        """
        logger.info(f"创建复合奖励函数，包含 {len(reward_components)} 个组件")

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
                    except TypeError:
                        try:
                            comp_reward = reward_fn(info)
                        except TypeError:
                            comp_reward = reward_fn()

                weight = weights.get(comp_name, 1.0)
                total_reward += weight * float(comp_reward)

            return total_reward

        return composite_reward

    def create_intrinsic_reward(
        self,
        intrinsic_type: str = "curiosity",
        **kwargs
    ) -> Callable:
        """
        创建内在奖励函数（好奇心驱动）

        参数:
            intrinsic_type: 内在奖励类型 ('curiosity', 'empowerment', 'novelty')
            **kwargs: 额外参数

        返回:
            内在奖励函数
        """
        logger.info(f"创建内在奖励函数: {intrinsic_type}")

        if intrinsic_type == "curiosity":
            # 基于预测误差的好奇心
            def curiosity_reward(state, action, next_state, done, info):
                # 这里需要一个前向模型来预测下一个状态
                # 奖励 = 预测误差
                # 简化实现：使用状态变化的幅度
                if hasattr(state, '__iter__') and hasattr(next_state, '__iter__'):
                    state_array = np.array(state)
                    next_state_array = np.array(next_state)
                    prediction_error = np.linalg.norm(next_state_array - state_array)
                    return float(prediction_error) * kwargs.get('scale', 0.1)
                return 0.0

            return curiosity_reward

        elif intrinsic_type == "novelty":
            # 基于状态访问计数的新颖性
            visit_counts = {}

            def novelty_reward(state, action, next_state, done, info):
                # 将状态转换为可哈希的形式
                state_key = str(state)
                visit_counts[state_key] = visit_counts.get(state_key, 0) + 1
                # 奖励与访问次数成反比
                return 1.0 / np.sqrt(visit_counts[state_key])

            return novelty_reward

        elif intrinsic_type == "empowerment":
            # 基于赋能的内在奖励
            def empowerment_reward(state, action, next_state, done, info):
                # 简化实现：鼓励探索能导致多样化结果的状态
                # 实际实现需要互信息计算
                return kwargs.get('base_empowerment', 0.01)

            return empowerment_reward

        else:
            raise ValueError(f"不支持的内在奖励类型: {intrinsic_type}")

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
                logger.warning(f"奖励函数 '{reward_name}' 没有统计信息")
                return {}
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
                    "max": float('-inf'),
                    "std": 0.0
                }
                logger.info(f"已重置奖励 '{reward_name}' 的统计信息")
        else:
            for name in self.reward_stats:
                self.reward_stats[name] = {
                    "total": 0.0,
                    "count": 0,
                    "mean": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "std": 0.0
                }
            logger.info("已重置所有奖励的统计信息")

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _extract_reward_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        从上下文中提取奖励函数配置

        返回:
            奖励函数配置字典
        """
        configs = {}

        # 1. 从 training_pipeline 的参数中查找
        if "reinforcement" in self.context.training_pipelines:
            pipeline = self.context.training_pipelines["reinforcement"]
            if hasattr(pipeline, 'parameters') and 'reward_functions' in pipeline.parameters:
                configs = pipeline.parameters['reward_functions']
                logger.debug(f"从 training_pipeline 中找到 {len(configs)} 个奖励配置")

        # 2. 从自定义字段查找（如果有）
        if hasattr(self.context, 'reward_functions'):
            configs.update(self.context.reward_functions)
            logger.debug("合并自定义奖励配置")

        # 3. 从全局配置中查找
        if hasattr(self.context, 'global_config'):
            global_config = self.context.global_config
            if isinstance(global_config, dict) and 'reward_functions' in global_config:
                configs.update(global_config['reward_functions'])

        return configs

    def _apply_shaping(
        self,
        reward_name: str,
        base_reward: float,
        state: Any,
        next_state: Any,
        done: bool
    ) -> float:
        """
        应用奖励塑形

        参数:
            reward_name: 奖励函数名称
            base_reward: 基础奖励
            state: 当前状态
            next_state: 下一个状态
            done: 是否结束

        返回:
            塑形后的奖励
        """
        if reward_name not in self.shaping_functions:
            return base_reward

        shaping_config = self.shaping_functions[reward_name]
        shaping_type = shaping_config['type']

        if shaping_type == 'potential':
            # 基于势的奖励塑造：F(s, s') = γΦ(s') - Φ(s)
            potential_fn = shaping_config['function']
            gamma = shaping_config['gamma']

            if not done:
                phi_s = potential_fn(state)
                phi_next = potential_fn(next_state)
                shaping = gamma * phi_next - phi_s
            else:
                phi_s = potential_fn(state)
                shaping = -phi_s

            return base_reward + shaping

        elif shaping_type == 'custom':
            # 自定义塑形函数
            shaping_fn = shaping_config['function']
            shaping = shaping_fn(state, next_state, done)
            return base_reward + shaping

        elif shaping_type == 'normalize':
            # 奖励归一化
            method = shaping_config['method']

            if method == 'running_mean_std':
                stats = self.reward_stats[reward_name]
                if stats['count'] > 1:
                    mean = stats['mean']
                    std = stats['std']
                    if std > 0:
                        return (base_reward - mean) / (std + 1e-8)

            return base_reward

        else:
            return base_reward

    def _update_reward_stats(self, reward_name: str, reward: float):
        """
        更新奖励统计信息

        参数:
            reward_name: 奖励函数名称
            reward: 奖励值
        """
        if reward_name not in self.reward_stats:
            self.reward_stats[reward_name] = {
                "total": 0.0,
                "count": 0,
                "mean": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "std": 0.0
            }

        stats = self.reward_stats[reward_name]
        old_mean = stats["mean"]
        stats["total"] += reward
        stats["count"] += 1
        stats["mean"] = stats["total"] / stats["count"]
        stats["min"] = min(stats["min"], reward)
        stats["max"] = max(stats["max"], reward)

        # 计算标准差（在线算法）
        if stats["count"] > 1:
            # 使用 Welford's online algorithm
            delta = reward - old_mean
            stats["std"] = np.sqrt(
                ((stats["count"] - 2) * stats["std"]**2 + delta**2) / (stats["count"] - 1)
            ) if stats["count"] > 2 else abs(reward - stats["mean"])


# ============================================================================
# 便捷函数
# ============================================================================

def build_reward_functions_from_context(context: TrainContext) -> Dict[str, Callable]:
    """
    从训练上下文构建所有奖励函数的便捷函数

    参数:
        context: 训练上下文

    返回:
        奖励函数字典

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> # ... 配置 context
        >>> rewards = build_reward_functions_from_context(context)
    """
    manager = RewardFunctionManager(context)
    return manager.run()


def create_reward_manager(context: TrainContext) -> RewardFunctionManager:
    """
    创建并初始化奖励函数管理器的便捷函数

    参数:
        context: 训练上下文

    返回:
        RewardFunctionManager 实例

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> manager = create_reward_manager(context)
        >>> reward = manager.compute_reward(state, action, next_state, done, info)
    """
    manager = RewardFunctionManager(context)
    manager.run()
    return manager


def compute_reward_with_manager(
    context: TrainContext,
    reward_name: str,
    state: Any,
    action: Any,
    next_state: Any,
    done: bool,
    info: Dict[str, Any]
) -> float:
    """
    使用管理器计算奖励的便捷函数

    参数:
        context: 训练上下文
        reward_name: 奖励函数名称
        state: 当前状态
        action: 执行的动作
        next_state: 下一个状态
        done: 是否结束
        info: 额外信息

    返回:
        奖励值

    示例:
        >>> reward = compute_reward_with_manager(
        ...     context, "game_reward", state, action, next_state, done, info
        ... )
    """
    manager = RewardFunctionManager(context)
    manager.run()
    return manager.compute_reward_by_name(
        reward_name, state, action, next_state, done, info
    )
