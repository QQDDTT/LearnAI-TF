# -*- coding: utf-8 -*-
# common/validators/structure_validator.py
"""
验证器：基本结构和常量
"""
from .base_validator import BaseValidator
from .constants import ConfigStructure

class StructureValidator(BaseValidator):

    def validate(self, config: dict):
        """验证第一层（模块）和第二层（常量）"""
        self._validate_module_layer(config)
        self._validate_constant_layer(config)

    def _validate_module_layer(self, config: dict):
        """验证顶层必须的模块"""
        for module in ConfigStructure.REQUIRED_MODULES:
            if module not in config:
                self.add_error(f"缺少必须的模块: {module}")

    def _validate_constant_layer(self, config: dict):
        """验证每个模块必须的字段"""
        for module, required_fields in ConfigStructure.MODULE_REQUIRED_FIELDS.items():
            if module not in config:
                continue

            module_config = config[module]
            if not isinstance(module_config, dict):
                self.add_error(f"模块 '{module}' 必须是字典类型")
                continue

            for field in required_fields:
                if field not in module_config:
                    self.add_error(f"模块 '{module}' 缺少必须字段: {field}")
