# -*- coding: utf-8 -*-
"""
modules/reward_functions.py
奖励函数管理器：构建和管理强化学习的奖励函数

设计原则：
1. 支持自定义奖励函数（通过 reflection）
2. 支持奖励塑形（reward shaping）
3. 支持内在奖励（intrinsic rewards）
4. 管理奖励统计信息

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 动态调用奖励函数
"""

from typing import Dict, Any, Callable, Optional
from common.train_context import TrainContext
from common.common import call_target, LoggerManager
from common.interfaces import RewardManagerInterface

# 初始化日志
logger = LoggerManager.get_logger(__file__)


class RewardFunctionManager(RewardManagerInterface):
    """
    奖励函数管理器

    职责：
    1. 从 TrainContext 读取奖励函数配置
    2. 构建奖励函数
    3. 管理奖励塑形
    4. 计算和统计奖励值
    """

    def __init__(self, context: TrainContext):
        """初始化奖励函数管理器"""
        super().__init__(context)
        self.reward_functions: Dict[str, Callable] = {}
        self.reward_configs: Dict[str, Dict[str, Any]] = {}
        self.shaping_functions: Dict[str, Callable] = {}
        self.reward_stats: Dict[str, Dict[str, float]] = {}
        logger.info("RewardFunctionManager 初始化完成")

    def initialize(self) -> None:
        """从 TrainContext 初始化奖励函数管理器"""
        logger.info("开始初始化奖励函数管理器")

        # 提取奖励函数配置
        self.reward_configs = self._extract_reward_configs()

        if not self.reward_configs:
            logger.warning("未找到奖励函数配置")
            self.reward_configs = {}
        else:
            logger.info(f"找到 {len(self.reward_configs)} 个奖励函数配置")

        self._initialized = True
        logger.info("奖励函数管理器初始化完成")

    def validate_rewards(self) -> bool:
        """验证奖励函数配置"""
        logger.info("开始验证奖励函数配置")

        if not self.reward_configs:
            logger.warning("奖励函数配置为空，跳过验证")
            return True

        for reward_name, reward_config in self.reward_configs.items():
            logger.debug(f"验证奖励函数: {reward_name}")

            if not isinstance(reward_config, dict):
                logger.error(f"奖励函数 '{reward_name}' 配置格式错误")
                raise ValueError(f"奖励函数 '{reward_name}' 配置格式错误")

            # 检查必需字段
            if 'reflection' not in reward_config and 'type' not in reward_config:
                logger.error(f"奖励函数 '{reward_name}' 缺少 reflection 或 type 字段")
                raise ValueError(f"奖励函数 '{reward_name}' 缺少 reflection 或 type 字段")

        logger.info("奖励函数配置验证通过")
        return True

    def build_reward_function(
        self,
        reward_name: str,
        reward_config: Dict[str, Any]
    ) -> Callable:
        """构建单个奖励函数"""
        logger.info(f"开始构建奖励函数: {reward_name}")

        # 如果提供了 reflection，直接创建
        if 'reflection' in reward_config:
            reflection = reward_config['reflection']
            args = reward_config.get('args', {})

            logger.debug(f"使用反射创建奖励函数: {reflection}")
            reward_fn = call_target(reflection=reflection, args=args)

        # 否则根据类型创建内置奖励函数
        elif 'type' in reward_config:
            reward_type = reward_config['type'].lower()
            logger.debug(f"创建内置奖励函数，类型: {reward_type}")

            if reward_type == 'sparse':
                reward_fn = self._create_sparse_reward(reward_config)
            elif reward_type == 'dense':
                reward_fn = self._create_dense_reward(reward_config)
            elif reward_type == 'shaped':
                reward_fn = self._create_shaped_reward(reward_config)
            else:
                logger.error(f"不支持的奖励类型: {reward_type}")
                raise ValueError(f"不支持的奖励类型: {reward_type}")
        else:
            logger.error(f"奖励函数 '{reward_name}' 配置无效")
            raise ValueError(f"奖励函数 '{reward_name}' 配置无效")

        # 初始化统计信息
        self.reward_stats[reward_name] = {
            "total": 0.0,
            "count": 0,
            "mean": 0.0,
            "min": float('inf'),
            "max": float('-inf')
        }

        logger.info(f"奖励函数 '{reward_name}' 构建完成")
        return reward_fn

    def compute_reward_by_name(
        self,
        reward_name: str,
        state: Any,
        action: Any,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        """根据名称计算奖励"""
        logger.debug(f"计算奖励: {reward_name}")

        if reward_name not in self.reward_functions:
            logger.error(f"奖励函数 '{reward_name}' 不存在")
            raise ValueError(f"奖励函数 '{reward_name}' 不存在")

        reward_fn = self.reward_functions[reward_name]

        # 尝试不同的函数签名
        try:
            reward = reward_fn(state, action, next_state, done, info)
        except TypeError:
            try:
                reward = reward_fn(next_state, done, info)
            except TypeError:
                reward = reward_fn(info)

        # 应用奖励塑形（如果配置）
        if reward_name in self.shaping_functions:
            logger.debug(f"应用奖励塑形: {reward_name}")
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

    def setup_shaping(self) -> None:
        """设置奖励塑形"""
        logger.info("设置奖励塑形")

        for reward_name, reward_config in self.reward_configs.items():
            shaping_config = reward_config.get('shaping')

            if shaping_config:
                logger.info(f"为奖励 '{reward_name}' 配置塑形")

                shaping_type = shaping_config.get('type', 'potential').lower()
                logger.debug(f"塑形类型: {shaping_type}")

                if shaping_type == 'potential':
                    shaping_fn = self._create_potential_shaping(shaping_config)
                elif shaping_type == 'curiosity':
                    shaping_fn = self._create_curiosity_shaping(shaping_config)
                else:
                    logger.warning(f"未知的塑形类型: {shaping_type}")
                    continue

                self.shaping_functions[reward_name] = shaping_fn
                logger.info(f"奖励塑形配置完成: {reward_name}")

        logger.info(f"共配置 {len(self.shaping_functions)} 个奖励塑形")

    def build_all_rewards(self) -> Dict[str, Callable]:
        """构建所有奖励函数"""
        logger.info("开始构建所有奖励函数")

        for reward_name, reward_config in self.reward_configs.items():
            try:
                reward_fn = self.build_reward_function(reward_name, reward_config)
                self.reward_functions[reward_name] = reward_fn
                logger.info(f"✓ 奖励函数 '{reward_name}' 构建成功")
            except Exception as e:
                logger.error(f"✗ 构建奖励函数 '{reward_name}' 失败: {e}", exc_info=True)
                raise RuntimeError(f"构建奖励函数 '{reward_name}' 失败: {e}") from e

        logger.info(f"所有奖励函数构建完成，共 {len(self.reward_functions)} 个")
        return self.reward_functions

    def execute(self) -> Dict[str, Any]:
        """执行核心逻辑：构建所有奖励函数"""
        logger.info("开始执行奖励函数构建流程")

        if not self.validate_rewards():
            raise ValueError("奖励函数配置验证失败")

        rewards = self.build_all_rewards()

        # 设置奖励塑形
        if not self._setup_complete:
            self.setup_shaping()
            self._setup_complete = True

        logger.info("奖励函数构建流程执行完成")
        return rewards

    def finalize(self) -> None:
        """清理和保存结果"""
        logger.info("保存奖励函数到上下文")

        # 保存奖励函数到上下文
        if self.reward_functions:
            if not hasattr(self.context.container, 'reward_functions'):
                self.context.container.reward_functions = {}
            self.context.container.reward_functions.update(self.reward_functions)
            logger.debug(f"已保存 {len(self.reward_functions)} 个奖励函数")

        # 保存塑形函数
        if self.shaping_functions:
            if not hasattr(self.context.container, 'shaping_functions'):
                self.context.container.shaping_functions = {}
            self.context.container.shaping_functions.update(self.shaping_functions)
            logger.debug(f"已保存 {len(self.shaping_functions)} 个塑形函数")

        logger.info("奖励函数保存完成")

    def get_reward_stats(self, reward_name: Optional[str] = None) -> Dict[str, Any]:
        """获取奖励统计信息"""
        if reward_name is not None:
            if reward_name not in self.reward_stats:
                logger.warning(f"奖励函数 '{reward_name}' 没有统计信息")
                return {}
            return {reward_name: self.reward_stats[reward_name]}

        return self.reward_stats.copy()

    def reset_reward_stats(self, reward_name: Optional[str] = None):
        """重置奖励统计信息"""
        if reward_name is not None:
            if reward_name in self.reward_stats:
                self.reward_stats[reward_name] = {
                    "total": 0.0,
                    "count": 0,
                    "mean": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
                logger.info(f"已重置奖励 '{reward_name}' 的统计信息")
        else:
            for name in self.reward_stats:
                self.reward_stats[name] = {
                    "total": 0.0,
                    "count": 0,
                    "mean": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
            logger.info("已重置所有奖励的统计信息")

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _extract_reward_configs(self) -> Dict[str, Dict[str, Any]]:
        """从上下文中提取奖励函数配置"""
        configs = {}

        # 从 training_pipeline 的参数中查找
        if "reinforcement" in self.context.training_pipelines:
            pipeline = self.context.training_pipelines["reinforcement"]
            if hasattr(pipeline, 'parameters') and 'reward_functions' in pipeline.parameters:
                configs = pipeline.parameters['reward_functions']
                logger.debug(f"从 training_pipeline 中找到 {len(configs)} 个奖励配置")

        # 从自定义字段查找
        if hasattr(self.context, 'reward_functions'):
            configs.update(self.context.reward_functions)
            logger.debug("合并自定义奖励配置")

        return configs

    def _update_reward_stats(self, reward_name: str, reward: float):
        """更新奖励统计信息"""
        if reward_name not in self.reward_stats:
            return

        stats = self.reward_stats[reward_name]
        stats["total"] += reward
        stats["count"] += 1
        stats["mean"] = stats["total"] / stats["count"]
        stats["min"] = min(stats["min"], reward)
        stats["max"] = max(stats["max"], reward)

    def _apply_shaping(
        self,
        reward_name: str,
        base_reward: float,
        state: Any,
        next_state: Any,
        done: bool
    ) -> float:
        """应用奖励塑形"""
        if reward_name not in self.shaping_functions:
            return base_reward

        shaping_fn = self.shaping_functions[reward_name]

        try:
            shaped_reward = shaping_fn(base_reward, state, next_state, done)
            logger.debug(f"奖励塑形: {base_reward:.4f} -> {shaped_reward:.4f}")
            return shaped_reward
        except Exception as e:
            logger.warning(f"奖励塑形失败: {e}，使用原始奖励")
            return base_reward

    def _create_sparse_reward(self, config: Dict[str, Any]) -> Callable:
        """创建稀疏奖励函数"""
        logger.debug("创建稀疏奖励函数")

        success_reward = config.get('success_reward', 1.0)
        failure_reward = config.get('failure_reward', 0.0)

        def sparse_reward(state, action, next_state, done, info):
            if done:
                if info.get('success', False):
                    return success_reward
                else:
                    return failure_reward
            return 0.0

        return sparse_reward

    def _create_dense_reward(self, config: Dict[str, Any]) -> Callable:
        """创建密集奖励函数"""
        logger.debug("创建密集奖励函数")

        reward_fn_reflection = config.get('reward_fn')
        if reward_fn_reflection:
            return call_target(reflection=reward_fn_reflection, args=config.get('args', {}))

        # 默认使用距离奖励
        def dense_reward(state, action, next_state, done, info):
            return info.get('reward', 0.0)

        return dense_reward

    def _create_shaped_reward(self, config: Dict[str, Any]) -> Callable:
        """创建塑形奖励函数"""
        logger.debug("创建塑形奖励函数")

        base_reward_fn = self._create_dense_reward(config)
        potential_fn_reflection = config.get('potential_fn')

        if potential_fn_reflection:
            potential_fn = call_target(
                reflection=potential_fn_reflection,
                args=config.get('potential_args', {})
            )
        else:
            # 默认势函数
            def potential_fn(state):
                return 0.0

        gamma = config.get('gamma', 0.99)

        def shaped_reward(state, action, next_state, done, info):
            base_r = base_reward_fn(state, action, next_state, done, info)
            phi_next = 0.0 if done else potential_fn(next_state)
            phi_curr = potential_fn(state)
            return base_r + gamma * phi_next - phi_curr

        return shaped_reward

    def _create_potential_shaping(self, config: Dict[str, Any]) -> Callable:
        """创建势函数塑形"""
        logger.debug("创建势函数塑形")

        potential_fn_reflection = config.get('potential_fn')
        if not potential_fn_reflection:
            logger.warning("势函数塑形缺少 potential_fn，使用默认")
            return lambda r, s, ns, d: r

        potential_fn = call_target(
            reflection=potential_fn_reflection,
            args=config.get('args', {})
        )
        gamma = config.get('gamma', 0.99)

        def shaping(base_reward, state, next_state, done):
            phi_next = 0.0 if done else potential_fn(next_state)
            phi_curr = potential_fn(state)
            return base_reward + gamma * phi_next - phi_curr

        return shaping

    def _create_curiosity_shaping(self, config: Dict[str, Any]) -> Callable:
        """创建好奇心塑形"""
        logger.debug("创建好奇心塑形")

        curiosity_weight = config.get('weight', 0.1)

        def shaping(base_reward, state, next_state, done):
            # 简化的好奇心奖励（实际需要预测误差）
            curiosity = curiosity_weight
            return base_reward + curiosity

        return shaping


# 便捷函数
def build_reward_functions_from_context(context: TrainContext) -> Dict[str, Callable]:
    """从训练上下文构建所有奖励函数的便捷函数"""
    logger.info("使用便捷函数构建奖励函数")
    manager = RewardFunctionManager(context)
    return manager.run()


def create_reward_manager(context: TrainContext) -> RewardFunctionManager:
    """创建并初始化奖励函数管理器的便捷函数"""
    logger.info("创建奖励函数管理器")
    manager = RewardFunctionManager(context)
    manager.run()
    return manager
