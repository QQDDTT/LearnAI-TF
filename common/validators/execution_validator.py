# -*- coding: utf-8 -*-
# common/validators/execution_validator.py
"""
验证器：执行单元 (reflection/args/bridge/connection)
"""
from typing import Dict, Any, List
from .base_validator import BaseValidator
from common.train_context import BridgeAction, ConnectionType

# (从 train_context.py 移入)
from common.validators.bridge_validator import parse_bridge
from common.validators.connection_validator import parse_connection


class ExecutionValidator(BaseValidator):

    def validate(self, config: dict):
        """验证第四层：执行层"""

        # 验证models中每个层的执行层结构
        if "models" in config and isinstance(config["models"], dict):
            for model_name, model_config in config["models"].items():
                if not isinstance(model_config, dict): continue

                # 验证模型本身
                model_errors = self.validate_execution_unit(model_config, "model")
                for error in model_errors:
                    self.add_error(f"模型 '{model_name}': {error}")

                # 验证每个层
                if "layers" in model_config and isinstance(model_config["layers"], list):
                    for i, layer in enumerate(model_config["layers"]):
                        layer_errors = self.validate_execution_unit(layer, "layer")
                        layer_name = layer.get("name", f"layer_{i}")
                        for error in layer_errors:
                            self.add_error(
                                f"模型 '{model_name}' 的层 '{layer_name}': {error}"
                            )

        # 验证training_pipeline中每个步骤的执行层结构
        if "training_pipeline" in config and isinstance(config["training_pipeline"], dict):
            for pipeline_name, pipeline_config in config["training_pipeline"].items():
                if not isinstance(pipeline_config, dict): continue

                if "step_sequence" in pipeline_config:
                    if isinstance(pipeline_config["step_sequence"], list):
                        for step in pipeline_config["step_sequence"]:
                            step_errors = self.validate_execution_unit(step, "step")
                            step_name = step.get("name", "unnamed_step")
                            for error in step_errors:
                                self.add_error(
                                    f"流程 '{pipeline_name}' 的步骤 '{step_name}': {error}"
                                )

    def validate_execution_unit(self, unit: Dict[str, Any], unit_type: str = "generic") -> List[str]:
        """
        (从 train_context.py 移入)
        验证执行层单元的结构
        """
        errors = []
        if not isinstance(unit, dict):
            return ["执行单元必须是字典类型"]

        if "reflection" not in unit:
            errors.append(f"执行层缺少 'reflection' 字段")

        if "args" not in unit:
            errors.append(f"执行层缺少 'args' 字段")
        elif "args" in unit and not isinstance(unit["args"], dict):
            errors.append(f"'args' 必须是字典类型")

        if unit_type == "step" and "name" not in unit:
            errors.append(f"步骤执行层缺少 'name' 字段")

        if "reflection" in unit:
            reflection = unit["reflection"]
            if not isinstance(reflection, str) or not reflection:
                errors.append(f"'reflection' 必须是非空字符串")
            elif ":" not in reflection and "." not in reflection:
                errors.append(f"'reflection' 格式无效: {reflection}")

        # 验证 Bridge（如果存在）
        if "bridge" in unit and unit["bridge"]:
            bridge = parse_bridge(unit["bridge"])
            if bridge is None:
                errors.append(f"Bridge 格式无效: {unit['bridge']}")
            elif not bridge.is_valid_action():
                errors.append(
                    f"Bridge action '{bridge.action}' 不在支持列表中"
                )

        # 验证 Connection（如果存在，仅层使用）
        if unit_type == "layer" and "connection" in unit and unit["connection"]:
            connection = parse_connection(unit["connection"])
            if connection is None:
                errors.append(f"Connection 格式无效: {unit['connection']}")
            elif not connection.is_valid_type():
                errors.append(
                    f"Connection type '{connection.type}' 不在支持列表中"
                )

        return errors

    def _validate_step_execution_units(self, config: dict):
        """验证训练步骤的执行单元"""
        if "training_pipeline" not in config:
            return

        for mode_name, mode_config in config["training_pipeline"].items():
            if "step_sequence" not in mode_config:  # 改为从 step_sequence 读取
                continue

            step_sequence = mode_config["step_sequence"]
            for step in step_sequence:
                # 验证 reflection 字段
                if "reflection" not in step:
                    self.add_error(f"步骤缺少 reflection 字段")
                # 验证 args 字段（可选）
                if "args" in step and not isinstance(step["args"], dict):
                    self.add_error(f"步骤的 args 必须是字典类型")
