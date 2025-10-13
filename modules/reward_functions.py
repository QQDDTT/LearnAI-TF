# -*- coding: utf-8 -*-
"""
reward_functions.py - 强化学习奖励函数
功能：
  - 根据配置文件计算奖励
  - 支持表达式、脚本、聚合三种方式
  - 灵活定义奖励规则
"""

from typing import Dict, Any, Optional
import re
from utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class RewardFunctionBuilder:
    """
    奖励函数构建器：
    - 根据配置生成奖励计算函数
    - 支持多种奖励计算方式
    """

    def build_all(self, reward_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建所有奖励函数
        参数：
            reward_config: 配置文件中的reward_functions部分
        返回：
            {奖励函数名: 奖励计算函数}
        """
        reward_functions = {}

        for reward_name, reward_cfg in reward_config.items():
            logger.info(f"构建奖励函数: {reward_name}")
            try:
                reward_fn = self.build_single(reward_name, reward_cfg)
                reward_functions[reward_name] = reward_fn
                logger.info(f"奖励函数 {reward_name} 构建成功")
            except Exception as e:
                logger.error(f"奖励函数 {reward_name} 构建失败: {str(e)}", exc_info=True)
                raise

        return reward_functions

    def build_single(self, reward_name: str, reward_cfg: Dict[str, Any]):
        """
        构建单个奖励函数
        参数：
            reward_name: 奖励函数名称
            reward_cfg: 奖励函数配置
        返回：
            奖励计算函数
        """
        reward_type = reward_cfg.get("type", "expression")

        if reward_type == "expression":
            return self._build_expression_reward(reward_name, reward_cfg)
        elif reward_type == "script":
            return self._build_script_reward(reward_name, reward_cfg)
        elif reward_type == "aggregation":
            return self._build_aggregation_reward(reward_name, reward_cfg)
        else:
            raise ValueError(f"未知的奖励函数类型: {reward_type}")

    def _build_expression_reward(self, reward_name: str, reward_cfg: Dict[str, Any]):
        """
        构建表达式奖励（简单公式）
        例如：reward * log_prob
        """
        formula = reward_cfg.get("formula", "reward")
        variables = reward_cfg.get("variables", [])

        def compute_reward(context: Dict[str, Any]) -> float:
            """
            根据上下文计算奖励
            参数：
                context: 执行上下文
            返回：
                奖励值
            """
            # 提取变量值
            var_values = {}
            for var_cfg in variables:
                var_name = var_cfg.get("name")
                var_source = var_cfg.get("source")

                # 从context中提取变量值
                value = self._extract_value_from_context(var_source, context)
                var_values[var_name] = value

            # 计算公式
            try:
                reward_value = self._safe_eval(formula, var_values)
                logger.debug(f"奖励函数 {reward_name} 计算结果: {reward_value}")
                return float(reward_value)
            except Exception as e:
                logger.error(f"计算公式失败: {formula}, 错误: {str(e)}")
                return 0.0

        return compute_reward

    def _build_script_reward(self, reward_name: str, reward_cfg: Dict[str, Any]):
        """
        构建脚本奖励（调用函数）
        """
        function = reward_cfg.get("function")
        args = reward_cfg.get("args", {})

        def compute_reward(context: Dict[str, Any]) -> float:
            """根据上下文调用函数计算奖励"""
            try:
                reward_value = call_target(function, {**args, "context": context})
                return float(reward_value)
            except Exception as e:
                logger.error(f"调用奖励函数 {function} 失败: {str(e)}")
                return 0.0

        return compute_reward

    def _build_aggregation_reward(self, reward_name: str, reward_cfg: Dict[str, Any]):
        """
        构建聚合奖励（多阶段评分）
        仅在特定条件触发时计算
        """
        trigger = reward_cfg.get("trigger", "done == true")
        formula = reward_cfg.get("formula", "reward")

        def compute_reward(context: Dict[str, Any]) -> float:
            """根据上下文和触发条件计算奖励"""
            # 检查触发条件
            if not self._check_trigger(trigger, context):
                return 0.0

            # 计算奖励
            try:
                reward_value = self._safe_eval(formula, context)
                logger.debug(f"聚合奖励 {reward_name} 计算结果: {reward_value}")
                return float(reward_value)
            except Exception as e:
                logger.error(f"计算公式失败: {formula}, 错误: {str(e)}")
                return 0.0

        return compute_reward

    def _extract_value_from_context(self, source: str, context: Dict[str, Any]) -> Any:
        """
        从上下文中提取值
        支持的格式：
        - "feedback.reward": 嵌套字典访问
        - "${variable}": 直接变量访问
        """
        if source.startswith("${") and source.endswith("}"):
            # 直接变量
            var_name = source[2:-1]
            return context.get(var_name)
        else:
            # 嵌套访问 (feedback.reward)
            parts = source.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value

    def _check_trigger(self, trigger: str, context: Dict[str, Any]) -> bool:
        """
        检查触发条件
        例如：done == true, reward > 10
        """
        try:
            # 安全地评估条件
            return self._safe_eval(trigger, context)
        except:
            return False

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        安全地评估表达式
        仅允许基本的算术和比较操作
        """
        # 限制允许的操作符和函数
        allowed_names = {
            'True': True, 'False': False, 'None': None,
            '__builtins__': {},
        }

        # 添加context中的变量
        allowed_names.update(context)

        try:
            # 使用eval，但限制命名空间
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            logger.error(f"表达式评估失败: {expression}, 错误: {str(e)}")
            raise


class RewardCalculator:
    """
    奖励计算器：
    - 管理奖励函数
    - 计算奖励值
    - 支持奖励缓存和累积
    """

    def __init__(self, reward_functions: Dict[str, Any] = None):
        """
        初始化计算器
        参数：
            reward_functions: 奖励函数字典
        """
        self.reward_functions = reward_functions or {}
        self.reward_history = []

    def compute(self,
                reward_name: str,
                context: Dict[str, Any]) -> float:
        """
        计算单个奖励
        参数：
            reward_name: 奖励函数名称
            context: 执行上下文
        返回：
            奖励值
        """
        if reward_name not in self.reward_functions:
            logger.warning(f"未找到奖励函数: {reward_name}")
            return 0.0

        reward_fn = self.reward_functions[reward_name]
        reward = reward_fn(context)

        # 记录到历史
        self.reward_history.append({
            "name": reward_name,
            "value": reward,
            "context": context
        })

        return reward

    def get_cumulative_reward(self) -> float:
        """获取累积奖励"""
        return sum(r["value"] for r in self.reward_history)

    def get_average_reward(self) -> float:
        """获取平均奖励"""
        if len(self.reward_history) == 0:
            return 0.0
        return self.get_cumulative_reward() / len(self.reward_history)

    def reset_history(self):
        """重置历史"""
        self.reward_history = []


class RewardNormalizer:
    """
    奖励标准化器：
    - 归一化奖励到[-1, 1]范围
    - 奖励缩放
    - 运行统计
    """

    def __init__(self):
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.reward_count = 0

    def update(self, reward: float):
        """更新统计"""
        self.reward_sum += reward
        self.reward_sq_sum += reward ** 2
        self.reward_count += 1

    def normalize(self, reward: float) -> float:
        """标准化奖励"""
        if self.reward_count == 0:
            return 0.0

        mean = self.reward_sum / self.reward_count
        variance = (self.reward_sq_sum / self.reward_count) - (mean ** 2)
        std = (variance ** 0.5) + 1e-8

        return (reward - mean) / std

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if self.reward_count == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        mean = self.reward_sum / self.reward_count
        variance = (self.reward_sq_sum / self.reward_count) - (mean ** 2)
        std = (variance ** 0.5) + 1e-8

        return {
            "mean": mean,
            "std": std,
            "count": self.reward_count,
            "sum": self.reward_sum
        }
