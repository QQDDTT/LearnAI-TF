# -*- coding: utf-8 -*-
# common/validators/patterns.py
"""
配置验证器 - 正则表达式模式
"""
import re
from common.train_context import BridgeAction, ConnectionType

class ReferencePattern:
    """引用模式和 Bridge 控制模式"""

    # 变量引用：${variable} 或 ${step.field}
    VARIABLE = r'\$\{([^}]+)\}'
    VARIABLE_COMPILED = re.compile(VARIABLE)

    # Bridge 表达式：@ACTION:target?condition(params)
    # 示例：
    # - @SKIP:next_epoch
    # - @SKIP:validation?${epoch}%5!=0
    # - @LOOP:reset_env?${done}==false(max=500)
    # - @BRANCH:path_a,path_b?${loss}<0.5
    BRIDGE = r'@(\w+)(?::([^?()]+))?(?:\?([^()]+))?(?:\(([^)]+)\))?'
    BRIDGE_COMPILED = re.compile(BRIDGE)

    # Connection 表达式：@RESIDUAL:target(params)
    CONNECTION = r'@(\w+)(?::([^()]+))?(?:\(([^)]+)\))?'
    CONNECTION_COMPILED = re.compile(CONNECTION)

    # 条件表达式的组成部分
    CONDITION_VARIABLE = r'\$\{([^}]+)\}'
    CONDITION_OPERATOR = r'(==|!=|>=|<=|>|<|%)'  # 新增 % 取模运算符
    CONDITION_LOGICAL = r'\b(AND|OR|NOT)\b'

    # Bridge 参数解析：(key1=value1,key2=value2)
    BRIDGE_PARAMS = r'(\w+)=([^,\)]+)'
    BRIDGE_PARAMS_COMPILED = re.compile(BRIDGE_PARAMS)
