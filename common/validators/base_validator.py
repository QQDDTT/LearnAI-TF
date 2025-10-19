# -*- coding: utf-8 -*-
# common/validators/base_validator.py
"""
验证器基类
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 避免循环导入
    from common.config_validator import ConfigValidator

class BaseValidator:
    """
    所有验证器模块的基类
    """
    def __init__(self, context: 'ConfigValidator'):
        self.context = context

    def add_error(self, message: str):
        """向主协调器添加错误"""
        self.context.add_error(message)

    def add_warning(self, message: str):
        """向主协调器添加警告"""
        self.context.add_warning(message)

    def validate(self, config: dict):
        """
        子类应重写此方法以执行特定验证
        """
        raise NotImplementedError
