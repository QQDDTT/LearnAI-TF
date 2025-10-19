# -*- coding: utf-8 -*-
# common/validators/patterns.py
"""
配置验证器 - 正则表达式模式
"""
import re
from common.train_context import BridgeAction, ConnectionType

class ReferencePattern:
    """
    引用模式常量
    """
    # 变量引用：${variable} 或 ${step.field}
    VARIABLE = r'\$\{([^}]+)\}'
    VARIABLE_COMPILED = re.compile(VARIABLE)

    # 步骤输出引用：${step_name.field}
    STEP_OUTPUT = r'\$\{([^.}]+)\.([^}]+)\}'

    # 模块引用：${module_name}
    MODULE = r'\$\{([^.}]+)\}'

    # ============ Bridge 表达式模式 ============
    BRIDGE = r'@(\w+)(?::([^?()]+))?(?:\?([^()]+))?(?:\(([^)]+)\))?'
    BRIDGE_COMPILED = re.compile(BRIDGE)

    # 条件表达式的组成部分
    CONDITION_VARIABLE = r'\$\{([^}]+)\}'
    CONDITION_VARIABLE_COMPILED = re.compile(CONDITION_VARIABLE)

    CONDITION_OPERATOR = r'(==|!=|>=|<=|>|<)'
    CONDITION_OPERATOR_COMPILED = re.compile(CONDITION_OPERATOR)

    CONDITION_LOGICAL = r'\b(AND|OR|NOT)\b'
    CONDITION_LOGICAL_COMPILED = re.compile(CONDITION_LOGICAL)

    # ============ Connection 表达式模式 ============
    CONNECTION = r'@(\w+)(?::([^()]+))?(?:\(([^)]+)\))?'
    CONNECTION_COMPILED = re.compile(CONNECTION)
