# -*- coding: utf-8 -*-
# common/config_validator.py
"""
common/config_validator.py
配置文件验证协调器
"""

from typing import Dict, Any, List, Tuple, Set

# 导入所有子验证器
from validators.structure_validator import StructureValidator
from validators.collection_validator import CollectionValidator
from validators.execution_validator import ExecutionValidator
from validators.bridge_validator import BridgeValidator, validate_bridge_expression
from validators.connection_validator import ConnectionValidator, validate_connection_expression
from validators.cross_ref_validator import CrossReferenceValidator

class ConfigValidator:
    """
    配置验证协调器

    此类负责协调验证流程，管理错误/警告状态，
    并维护跨验证阶段所需的共享状态（如层/步骤名称）。

    具体的验证逻辑委托给 /validators/ 目录下的子验证器。
    """

    def __init__(self):
        # 验证结果
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # 跨阶段共享的状态
        self.layer_names: Dict[str, Set[str]] = {}  # 每个模型的层名称集合
        self.step_names: Dict[str, Set[str]] = {}   # 每个流程的步骤名称集合

        # 实例化子验证器
        self.structure_validator = StructureValidator(self)
        self.collection_validator = CollectionValidator(self)
        self.execution_validator = ExecutionValidator(self)
        self.bridge_validator = BridgeValidator(self)
        self.connection_validator = ConnectionValidator(self)
        self.cross_ref_validator = CrossReferenceValidator(self)

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        执行完整的验证流程

        流程分为多个阶段，确保依赖关系正确：
        1. 结构验证 (StructureValidator)
        2. 名称收集 (CollectionValidator) - 必须在其他验证之前，以收集名称
        3. 执行单元验证 (ExecutionValidator)
        4. 深度验证 (Bridge, Connection) - 依赖收集到的名称
        5. 交叉引用验证 (CrossReferenceValidator) - 依赖收集到的名称
        """
        self.errors = []
        self.warnings = []
        self.layer_names = {}
        self.step_names = {}

        if not isinstance(config, dict):
            self.errors.append("配置文件必须是字典类型")
            return False, self.errors, self.warnings

        # --- 验证流程 ---

        # 阶段 1: 基本结构
        self.structure_validator.validate(config)

        # 阶段 2: 收集名称 (此阶段也可能产生验证错误)
        self.collection_validator.validate_and_collect(config)

        # 阶段 3: 验证执行单元 (reflection/args)
        self.execution_validator.validate(config)

        # 阶段 4: 深度验证 Bridge 和 Connection (依赖阶段 2 收集的名称)
        self.bridge_validator.validate(config)
        self.connection_validator.validate(config)

        # 阶段 5: 交叉引用验证 (依赖阶段 2 收集的名称)
        self.cross_ref_validator.validate(config)

        # --- 流程结束 ---

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    # --- 供子验证器调用的公共方法 ---

    def add_error(self, message: str):
        """添加错误信息"""
        self.errors.append(message)

    def add_warning(self, message: str):
        """添加警告信息"""
        self.warnings.append(message)

    def set_layer_names(self, model_name: str, names: Set[str]):
        """注册模型的所有层名称"""
        self.layer_names[model_name] = names

    def set_step_names(self, pipeline_name: str, names: Set[str]):
        """注册流程的所有步骤名称"""
        self.step_names[pipeline_name] = names

    def get_layer_names(self, model_name: str) -> Set[str]:
        """获取特定模型的层名称"""
        return self.layer_names.get(model_name, set())

    def get_step_names(self, pipeline_name: str) -> Set[str]:
        """获取特定流程的步骤名称"""
        return self.step_names.get(pipeline_name, set())

    # --- 报告 ---

    def print_report(self):
        """打印验证报告"""
        if self.errors:
            print("\n" + "=" * 70)
            print("❌ 错误列表:")
            print("=" * 70)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

        if self.warnings:
            print("\n" + "=" * 70)
            print("⚠️  警告列表:")
            print("=" * 70)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n" + "=" * 70)
            print("✅ 配置验证通过，没有错误或警告")
            print("=" * 70)


# ======================================================
# 便捷函数
# ======================================================

def validate_config_file(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    验证配置文件的便捷函数
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)
    validator.print_report()
    return is_valid, errors, warnings

# 导出来自子模块的独立验证函数，保持 API 向后兼容
validate_bridge_expression = validate_bridge_expression
validate_connection_expression = validate_connection_expression
