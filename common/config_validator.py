# -*- coding: utf-8 -*-
"""
common/config_validator.py
配置文件验证器 - 改进版
重点改进：Bridge 和 Connection 表达式的完整验证
"""

import re
from typing import Dict, Any, List, Tuple, Set
from common.train_context import (
    ConfigStructure,
    TrainingModeConstants,
    ReferencePattern,
    BridgeAction,
    ConnectionType,
    parse_bridge,
    parse_connection,
    validate_bridge_condition,
    validate_connection_targets,
    validate_execution_unit
)


class ConfigValidator:
    """
    配置验证器 - 改进版

    新增功能：
    1. 完整的 Bridge 表达式验证
    2. 完整的 Connection 表达式验证
    3. 更详细的错误和警告信息
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.layer_names: Dict[str, Set[str]] = {}  # 每个模型的层名称集合
        self.step_names: Dict[str, Set[str]] = {}   # 每个流程的步骤名称集合

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        验证配置文件

        参数:
            config: 配置字典

        返回:
            (是否通过, 错误列表, 警告列表)
        """
        self.errors = []
        self.warnings = []
        self.layer_names = {}
        self.step_names = {}

        # 第一层验证：模块层
        self._validate_module_layer(config)

        # 第二层验证：常量层
        self._validate_constant_layer(config)

        # 第三层验证：layers层（收集名称信息）
        self._validate_layers_layer(config)

        # 第四层验证：执行层（包含 Bridge 和 Connection）
        self._validate_execution_layer(config)

        # Bridge 和 Connection 的深度验证
        self._validate_bridges(config)
        self._validate_connections(config)

        # 交叉验证
        self._validate_cross_references(config)

        is_valid = len(self.errors) == 0

        return is_valid, self.errors, self.warnings

    def _validate_module_layer(self, config: Dict[str, Any]):
        """验证第一层：模块层"""
        missing_modules = []
        for module in ConfigStructure.REQUIRED_MODULES:
            if module not in config:
                missing_modules.append(module)
                self.errors.append(f"缺少必须的模块: {module}")

    def _validate_constant_layer(self, config: Dict[str, Any]):
        """验证第二层：常量层"""
        for module, required_fields in ConfigStructure.MODULE_REQUIRED_FIELDS.items():
            if module not in config:
                continue

            module_config = config[module]

            for field in required_fields:
                if field not in module_config:
                    self.errors.append(f"模块 '{module}' 缺少必须字段: {field}")

    def _validate_layers_layer(self, config: Dict[str, Any]):
        """
        验证第三层：layers层

        同时收集层名称和步骤名称信息，供后续验证使用
        """
        # 收集模型中的层名称
        if "models" in config:
            for model_name, model_config in config["models"].items():
                if "layers" not in model_config:
                    self.errors.append(f"模型 '{model_name}' 缺少 'layers' 字段")
                    continue

                if not isinstance(model_config["layers"], list):
                    self.errors.append(f"模型 '{model_name}' 的 'layers' 必须是列表")
                    continue

                # 收集该模型的所有层名称
                layer_names = set()
                for i, layer in enumerate(model_config["layers"]):
                    if isinstance(layer, dict):
                        layer_name = layer.get("name", f"layer_{i}")
                        if layer_name in layer_names:
                            self.warnings.append(
                                f"模型 '{model_name}' 中层名称重复: {layer_name}"
                            )
                        layer_names.add(layer_name)

                self.layer_names[model_name] = layer_names

        # 收集训练流程中的步骤名称
        if "training_pipeline" in config:
            for pipeline_name, pipeline_config in config["training_pipeline"].items():
                if "step_sequence" not in pipeline_config:
                    self.errors.append(f"流程 '{pipeline_name}' 缺少 'step_sequence' 字段")
                    continue

                if not isinstance(pipeline_config["step_sequence"], list):
                    self.errors.append(
                        f"流程 '{pipeline_name}' 的 'step_sequence' 必须是列表"
                    )
                    continue

                # 收集该流程的所有步骤名称
                step_names = set()
                for i, step in enumerate(pipeline_config["step_sequence"]):
                    if isinstance(step, dict):
                        step_name = step.get("name", f"step_{i}")
                        if step_name in step_names:
                            self.warnings.append(
                                f"流程 '{pipeline_name}' 中步骤名称重复: {step_name}"
                            )
                        step_names.add(step_name)

                self.step_names[pipeline_name] = step_names

    def _validate_execution_layer(self, config: Dict[str, Any]):
        """验证第四层：执行层"""
        # 验证models中每个层的执行层结构
        if "models" in config:
            for model_name, model_config in config["models"].items():
                # 验证模型本身
                model_errors = validate_execution_unit(model_config, "model")
                for error in model_errors:
                    self.errors.append(f"模型 '{model_name}': {error}")

                # 验证每个层
                if "layers" in model_config and isinstance(model_config["layers"], list):
                    for i, layer in enumerate(model_config["layers"]):
                        layer_errors = validate_execution_unit(layer, "layer")
                        layer_name = layer.get("name", f"layer_{i}")
                        for error in layer_errors:
                            self.errors.append(
                                f"模型 '{model_name}' 的层 '{layer_name}': {error}"
                            )

        # 验证training_pipeline中每个步骤的执行层结构
        if "training_pipeline" in config:
            for pipeline_name, pipeline_config in config["training_pipeline"].items():
                if "step_sequence" in pipeline_config:
                    if isinstance(pipeline_config["step_sequence"], list):
                        for step in pipeline_config["step_sequence"]:
                            step_errors = validate_execution_unit(step, "step")
                            step_name = step.get("name", "unnamed_step")
                            for error in step_errors:
                                self.errors.append(
                                    f"流程 '{pipeline_name}' 的步骤 '{step_name}': {error}"
                                )

    def _validate_bridges(self, config: Dict[str, Any]):
        """
        验证所有 Bridge 表达式（深度验证）

        检查：
        1. Bridge 语法正确性
        2. Action 类型合法性
        3. 目标步骤存在性
        4. 条件表达式有效性
        5. 参数完整性
        """
        if "training_pipeline" not in config:
            return

        for pipeline_name, pipeline_config in config["training_pipeline"].items():
            if "step_sequence" not in pipeline_config:
                continue

            step_sequence = pipeline_config["step_sequence"]
            if not isinstance(step_sequence, list):
                continue

            # 获取该流程的所有步骤名称
            available_steps = self.step_names.get(pipeline_name, set())

            # 收集所有可用的变量（步骤名称 + 全局变量）
            available_vars = available_steps.copy()
            available_vars.update(['epoch', 'step', 'loss', 'accuracy', 'metrics'])

            for i, step in enumerate(step_sequence):
                if not isinstance(step, dict):
                    continue

                step_name = step.get("name", f"step_{i}")

                if "bridge" not in step or not step["bridge"]:
                    continue

                bridge_str = step["bridge"]
                bridge = parse_bridge(bridge_str)

                if bridge is None:
                    self.errors.append(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"Bridge 格式无效: {bridge_str}"
                    )
                    continue

                # 验证 action 类型
                if not bridge.is_valid_action():
                    self.errors.append(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"不支持的 Bridge action: {bridge.action}"
                    )

                # 验证目标步骤
                if bridge.requires_target():
                    if not bridge.targets:
                        self.errors.append(
                            f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                            f"Bridge action '{bridge.action}' 需要指定目标步骤"
                        )
                    else:
                        for target in bridge.targets:
                            if target not in available_steps:
                                self.errors.append(
                                    f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                                    f"Bridge 目标步骤 '{target}' 不存在"
                                )

                # 验证条件表达式
                if bridge.requires_condition():
                    if not bridge.condition:
                        self.errors.append(
                            f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                            f"Bridge action '{bridge.action}' 需要指定条件"
                        )
                    else:
                        condition_errors = validate_bridge_condition(
                            bridge.condition,
                            available_vars
                        )
                        for error in condition_errors:
                            self.errors.append(
                                f"流程 '{pipeline_name}' 步骤 '{step_name}': {error}"
                            )

                # 特定 action 的参数验证
                self._validate_bridge_params(
                    bridge, pipeline_name, step_name
                )

    def _validate_bridge_params(
        self,
        bridge,
        pipeline_name: str,
        step_name: str
    ):
        """验证特定 Bridge action 的参数"""
        # LOOP action 需要的参数
        if bridge.action == BridgeAction.LOOP.value:
            if "max_iter" not in bridge.params:
                self.warnings.append(
                    f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                    f"LOOP action 建议指定 'max_iter' 参数以防止无限循环"
                )

        # PARALLEL action 需要的参数
        if bridge.action == BridgeAction.PARALLEL.value:
            if "max_workers" in bridge.params:
                try:
                    max_workers = int(bridge.params["max_workers"])
                    if max_workers <= 0:
                        self.errors.append(
                            f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                            f"'max_workers' 必须大于 0"
                        )
                except ValueError:
                    self.errors.append(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"'max_workers' 必须是整数"
                    )

        # CHECKPOINT action 的参数验证
        if bridge.action == BridgeAction.CHECKPOINT.value:
            valid_params = ['save_best', 'monitor', 'mode', 'save_freq']
            for param in bridge.params:
                if param not in valid_params:
                    self.warnings.append(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"CHECKPOINT action 的参数 '{param}' 可能无效"
                    )

    def _validate_connections(self, config: Dict[str, Any]):
        """
        验证所有 Connection 表达式（深度验证）

        检查：
        1. Connection 语法正确性
        2. 连接类型合法性
        3. 目标层存在性
        4. 目标层数量要求
        5. 参数完整性
        """
        if "models" not in config:
            return

        for model_name, model_config in config["models"].items():
            if "layers" not in model_config:
                continue

            layers = model_config["layers"]
            if not isinstance(layers, list):
                continue

            # 获取该模型的所有层名称
            available_layers = self.layer_names.get(model_name, set())

            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    continue

                layer_name = layer.get("name", f"layer_{i}")

                if "connection" not in layer or not layer["connection"]:
                    continue

                connection_str = layer["connection"]
                connection = parse_connection(connection_str)

                if connection is None:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"Connection 格式无效: {connection_str}"
                    )
                    continue

                # 验证连接类型
                if not connection.is_valid_type():
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"不支持的 Connection 类型: {connection.type}"
                    )
                    continue

                # 验证目标层
                target_errors = validate_connection_targets(
                    connection.type,
                    connection.targets,
                    available_layers
                )

                for error in target_errors:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': {error}"
                    )

                # 验证连接参数
                self._validate_connection_params(
                    connection, model_name, layer_name
                )

    def _validate_connection_params(
        self,
        connection,
        model_name: str,
        layer_name: str
    ):
        """验证特定 Connection 类型的参数"""
        # RESIDUAL 连接的参数验证
        if connection.type == ConnectionType.RESIDUAL.value:
            if "scale" in connection.params:
                try:
                    scale = float(connection.params["scale"])
                    if scale <= 0:
                        self.errors.append(
                            f"模型 '{model_name}' 层 '{layer_name}': "
                            f"'scale' 必须大于 0"
                        )
                except ValueError:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"'scale' 必须是数字"
                    )

        # ATTENTION 连接的参数验证
        if connection.type == ConnectionType.ATTENTION.value:
            required_params = ['heads']
            for param in required_params:
                if param not in connection.params:
                    self.warnings.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"ATTENTION 连接建议指定 '{param}' 参数"
                    )

            if "heads" in connection.params:
                try:
                    heads = int(connection.params["heads"])
                    if heads <= 0:
                        self.errors.append(
                            f"模型 '{model_name}' 层 '{layer_name}': "
                            f"'heads' 必须大于 0"
                        )
                except ValueError:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"'heads' 必须是整数"
                    )

            if "dropout" in connection.params:
                try:
                    dropout = float(connection.params["dropout"])
                    if not 0 <= dropout < 1:
                        self.errors.append(
                            f"模型 '{model_name}' 层 '{layer_name}': "
                            f"'dropout' 必须在 [0, 1) 范围内"
                        )
                except ValueError:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"'dropout' 必须是数字"
                    )

        # CONCAT 连接的参数验证
        if connection.type == ConnectionType.CONCAT.value:
            if "axis" in connection.params:
                try:
                    axis = int(connection.params["axis"])
                    if axis < -1:
                        self.warnings.append(
                            f"模型 '{model_name}' 层 '{layer_name}': "
                            f"'axis' 值可能无效: {axis}"
                        )
                except ValueError:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"'axis' 必须是整数"
                    )

        # ADD/MULTIPLY 连接的参数验证
        if connection.type in [ConnectionType.ADD.value, ConnectionType.MULTIPLY.value]:
            if "scale" in connection.params:
                try:
                    float(connection.params["scale"])
                except ValueError:
                    self.errors.append(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"'scale' 必须是数字"
                    )

    def _validate_cross_references(self, config: Dict[str, Any]):
        """交叉验证引用完整性和逻辑一致性"""
        # 验证训练模式
        if "training_mode" in config and "type" in config["training_mode"]:
            mode = config["training_mode"]["type"]
            if mode not in TrainingModeConstants.SUPPORTED_MODES:
                self.errors.append(f"不支持的训练模式: {mode}")

        # 验证训练模式与流程的匹配
        if "training_mode" in config and "training_pipeline" in config:
            mode = config["training_mode"].get("type")
            if mode and mode not in config["training_pipeline"]:
                self.errors.append(
                    f"训练模式 '{mode}' 没有对应的流程定义"
                )

        # 验证步骤中的变量引用
        if "training_pipeline" in config:
            for pipeline_name, pipeline_config in config["training_pipeline"].items():
                if "step_sequence" in pipeline_config:
                    self._validate_step_variable_references(
                        pipeline_config["step_sequence"],
                        pipeline_name,
                        config
                    )

    def _validate_step_variable_references(
        self,
        step_sequence: List[Dict],
        pipeline_name: str,
        config: Dict[str, Any]
    ):
        """验证步骤中的变量引用"""
        # 收集所有可用的名称
        available_names = set()

        # 添加模型名称
        if "models" in config:
            available_names.update(config["models"].keys())

        # 添加优化器名称
        if "optimizers" in config:
            available_names.update(config["optimizers"].keys())

        # 添加损失函数名称
        if "losses" in config:
            available_names.update(config["losses"].keys())

        # 添加步骤名称
        available_names.update(self.step_names.get(pipeline_name, set()))

        # 检查每个步骤的 args 中的引用
        for i, step in enumerate(step_sequence):
            if not isinstance(step, dict):
                continue

            step_name = step.get("name", f"step_{i}")

            if "args" in step and isinstance(step["args"], dict):
                self._check_references_in_dict(
                    step["args"],
                    available_names,
                    f"流程 '{pipeline_name}' 步骤 '{step_name}'"
                )

    def _check_references_in_dict(
        self,
        data: Dict[str, Any],
        valid_names: Set[str],
        context: str
    ):
        """递归检查字典中的变量引用"""
        reference_pattern = re.compile(ReferencePattern.VARIABLE)

        for key, value in data.items():
            if isinstance(value, str):
                # 检查引用
                matches = reference_pattern.findall(value)
                for ref in matches:
                    base_name = ref.split('.')[0]
                    if base_name not in valid_names:
                        self.warnings.append(
                            f"{context}.args.{key}: 引用了未定义的名称 '${{{ref}}}'"
                        )
            elif isinstance(value, dict):
                self._check_references_in_dict(
                    value,
                    valid_names,
                    f"{context}.{key}"
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._check_references_in_dict(
                            item,
                            valid_names,
                            f"{context}.{key}[{i}]"
                        )

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


def validate_config_file(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    验证配置文件的便捷函数

    参数:
        config: 配置字典

    返回:
        (是否通过, 错误列表, 警告列表)
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)

    # 打印详细报告
    validator.print_report()

    return is_valid, errors, warnings


# ======================================================
# 额外的验证工具函数
# ======================================================

def validate_bridge_expression(bridge_str: str) -> Tuple[bool, List[str]]:
    """
    独立验证单个 Bridge 表达式

    参数:
        bridge_str: Bridge 字符串

    返回:
        (是否有效, 错误列表)
    """
    errors = []

    bridge = parse_bridge(bridge_str)
    if bridge is None:
        errors.append(f"Bridge 格式无效: {bridge_str}")
        return False, errors

    if not bridge.is_valid_action():
        errors.append(f"不支持的 Bridge action: {bridge.action}")

    if bridge.requires_target() and not bridge.targets:
        errors.append(f"Bridge action '{bridge.action}' 需要指定目标")

    if bridge.requires_condition() and not bridge.condition:
        errors.append(f"Bridge action '{bridge.action}' 需要指定条件")

    return len(errors) == 0, errors


def validate_connection_expression(connection_str: str) -> Tuple[bool, List[str]]:
    """
    独立验证单个 Connection 表达式

    参数:
        connection_str: Connection 字符串

    返回:
        (是否有效, 错误列表)
    """
    errors = []

    connection = parse_connection(connection_str)
    if connection is None:
        errors.append(f"Connection 格式无效: {connection_str}")
        return False, errors

    if not connection.is_valid_type():
        errors.append(f"不支持的 Connection 类型: {connection.type}")

    if connection.requires_targets() and not connection.targets:
        errors.append(f"Connection 类型 '{connection.type}' 需要指定目标层")

    min_count = connection.min_targets_count()
    if len(connection.targets) < min_count:
        errors.append(
            f"Connection 类型 '{connection.type}' 至少需要 {min_count} 个目标层"
        )

    return len(errors) == 0, errors
