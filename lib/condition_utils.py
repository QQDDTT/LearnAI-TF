# -*- coding: utf-8 -*-
"""
lib/condition_utils.py
条件检查工具函数
"""

from typing import Dict, Any
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


def check_condition(
    condition: str,
    context: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, bool]:
    """
    检查条件是否满足

    参数:
        condition: 条件表达式（字符串形式）
        context: 上下文变量字典

    返回:
        包含result的字典
    """
    if not condition:
        raise ValueError("条件表达式不能为空")

    try:
        # 创建安全的执行环境
        safe_context = context.copy() if context else {}

        # 评估条件
        result = eval(condition, {"__builtins__": {}}, safe_context)

        logger.debug(f"条件检查: {condition} = {result}")

        return {"result": bool(result)}

    except NameError as e:
        logger.error(f"条件表达式中存在未定义的变量: {e}")
        raise
    except SyntaxError as e:
        logger.error(f"条件表达式语法错误: {e}")
        raise
    except Exception as e:
        logger.error(f"条件检查失败: {e}")
        raise


def check_threshold(
    value: float,
    threshold: float,
    mode: str = "greater",
    **kwargs
) -> Dict[str, bool]:
    """
    检查值是否满足阈值条件

    参数:
        value: 要检查的值
        threshold: 阈值
        mode: 比较模式 (greater, less, equal, greater_equal, less_equal)

    返回:
        包含result的字典
    """
    mode = mode.lower()

    try:
        if mode == "greater":
            result = value > threshold
        elif mode == "less":
            result = value < threshold
        elif mode == "equal":
            result = abs(value - threshold) < 1e-8
        elif mode == "greater_equal":
            result = value >= threshold
        elif mode == "less_equal":
            result = value <= threshold
        else:
            raise ValueError(f"不支持的比较模式: {mode}")

        logger.debug(f"阈值检查: {value} {mode} {threshold} = {result}")

        return {"result": result}

    except Exception as e:
        logger.error(f"阈值检查失败: {e}")
        raise


def check_convergence_simple(
    current_value: float,
    previous_value: float,
    tolerance: float = 1e-4,
    **kwargs
) -> Dict[str, bool]:
    """
    简单的收敛性检查

    参数:
        current_value: 当前值
        previous_value: 前一个值
        tolerance: 容差

    返回:
        包含converged和change的字典
    """
    try:
        change = abs(current_value - previous_value)
        converged = change < tolerance

        logger.debug(f"收敛检查: 变化量={change:.6f}, 容差={tolerance}, 收敛={converged}")

        return {
            "converged": converged,
            "change": float(change)
        }

    except Exception as e:
        logger.error(f"收敛检查失败: {e}")
        raise
