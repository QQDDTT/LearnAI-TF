# -*- coding: utf-8 -*-
# common/validators/bridge_validator.py
"""
验证器：Bridge 表达式深度验证
"""
import re
from typing import Optional, List, Dict, Any, Tuple, Set
from .base_validator import BaseValidator
from .patterns import ReferencePattern
from common.train_context import ParsedBridge, BridgeAction

# (从 train_context.py 移入)
def parse_bridge(bridge_str: str) -> Optional[ParsedBridge]:
    if not bridge_str or not bridge_str.startswith('@'):
        return None
    match = ReferencePattern.BRIDGE_COMPILED.match(bridge_str)
    if not match:
        return None
    action, target_str, condition, params_str = match.groups()
    targets = [t.strip() for t in target_str.split(',')] if target_str else []
    params = {}
    if params_str:
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
    return ParsedBridge(action, targets, condition, params, bridge_str)

# (从 train_context.py 移入)
def validate_bridge_condition(condition: str, available_vars: set) -> List[str]:
    errors = []
    if not condition:
        return errors
    variables = ReferencePattern.CONDITION_VARIABLE_COMPILED.findall(condition)
    for var in variables:
        base_name = var.split('.')[0]
        if base_name not in available_vars:
            errors.append(f"条件中引用了未定义的变量: ${{{var}}}")
    operators = ReferencePattern.CONDITION_OPERATOR_COMPILED.findall(condition)
    logicals = ReferencePattern.CONDITION_LOGICAL_COMPILED.findall(condition)
    if not operators and not logicals and variables:
        errors.append(f"条件表达式缺少运算符: {condition}")
    return errors


class BridgeValidator(BaseValidator):

    def validate(self, config: dict):
        """验证所有 Bridge 表达式（深度验证）"""
        if "training_pipeline" not in config or not isinstance(config["training_pipeline"], dict):
            return

        for pipeline_name, pipeline_config in config["training_pipeline"].items():
            if "step_sequence" not in pipeline_config:
                continue
            step_sequence = pipeline_config["step_sequence"]
            if not isinstance(step_sequence, list):
                continue

            available_steps = self.context.get_step_names(pipeline_name)
            available_vars = available_steps.copy()
            available_vars.update(['epoch', 'step', 'loss', 'accuracy', 'metrics'])

            for i, step in enumerate(step_sequence):
                if not isinstance(step, dict): continue

                step_name = step.get("name", f"step_{i}")
                if "bridge" not in step or not step["bridge"]:
                    continue

                bridge_str = step["bridge"]
                bridge = parse_bridge(bridge_str)

                if bridge is None:
                    self.add_error(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"Bridge 格式无效: {bridge_str}"
                    )
                    continue

                self._validate_bridge(bridge, pipeline_name, step_name, available_steps, available_vars)

    def _validate_bridge(self, bridge: ParsedBridge, pipeline_name: str, step_name: str, available_steps: Set[str], available_vars: Set[str]):
        """验证单个解析后的 Bridge"""

        if not bridge.is_valid_action():
            self.add_error(
                f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                f"不支持的 Bridge action: {bridge.action}"
            )

        if bridge.requires_target():
            if not bridge.targets:
                self.add_error(
                    f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                    f"Bridge action '{bridge.action}' 需要指定目标步骤"
                )
            else:
                for target in bridge.targets:
                    if target not in available_steps:
                        self.add_error(
                            f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                            f"Bridge 目标步骤 '{target}' 不存在"
                        )

        if bridge.requires_condition():
            if not bridge.condition:
                self.add_error(
                    f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                    f"Bridge action '{bridge.action}' 需要指定条件"
                )
            else:
                condition_errors = validate_bridge_condition(
                    bridge.condition,
                    available_vars
                )
                for error in condition_errors:
                    self.add_error(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': {error}"
                    )

        self._validate_bridge_params(bridge, pipeline_name, step_name)

    def _validate_bridge_params(self, bridge, pipeline_name: str, step_name: str):
        """验证特定 Bridge action 的参数"""
        if bridge.action == BridgeAction.LOOP.value:
            if "max_iter" not in bridge.params:
                self.add_warning(
                    f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                    f"LOOP action 建议指定 'max_iter' 参数"
                )

        if bridge.action == BridgeAction.PARALLEL.value:
            if "max_workers" in bridge.params:
                try:
                    if int(bridge.params["max_workers"]) <= 0:
                        self.add_error(
                            f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                            f"'max_workers' 必须大于 0"
                        )
                except ValueError:
                    self.add_error(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"'max_workers' 必须是整数"
                    )

        if bridge.action == BridgeAction.CHECKPOINT.value:
            valid_params = ['save_best', 'monitor', 'mode', 'save_freq']
            for param in bridge.params:
                if param not in valid_params:
                    self.add_warning(
                        f"流程 '{pipeline_name}' 步骤 '{step_name}': "
                        f"CHECKPOINT action 的参数 '{param}' 可能无效"
                    )

# 独立验证函数
def validate_bridge_expression(bridge_str: str) -> Tuple[bool, List[str]]:
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
