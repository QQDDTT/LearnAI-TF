# -*- coding: utf-8 -*-
# common/validators/cross_reference_validator.py
"""
验证器：交叉引用
"""
import re
from typing import Dict, Any, List, Set
from .base_validator import BaseValidator
from .constants import TrainingModeConstants
from .patterns import ReferencePattern

class CrossReferenceValidator(BaseValidator):

    def validate(self, config: dict):
        """交叉验证引用完整性和逻辑一致性"""

        self._validate_training_mode(config)
        self._validate_all_step_references(config)

    def _validate_training_mode(self, config: dict):
        """验证训练模式及其与流程的匹配"""
        mode = None
        if "training_mode" in config and isinstance(config["training_mode"], dict):
            if "type" in config["training_mode"]:
                mode = config["training_mode"]["type"]
                if mode not in TrainingModeConstants.SUPPORTED_MODES:
                    self.add_error(f"不支持的训练模式: {mode}")
            else:
                # 'type' 字段已由 structure_validator 检查
                pass

        if mode and "training_pipeline" in config:
            if mode not in config["training_pipeline"]:
                self.add_error(
                    f"训练模式 '{mode}' 没有对应的流程定义"
                )

    def _validate_all_step_references(self, config: dict):
        """验证所有流程步骤中的变量引用"""
        if "training_pipeline" not in config or not isinstance(config["training_pipeline"], dict):
            return

        available_names = self._collect_all_available_names(config)

        for pipeline_name, pipeline_config in config["training_pipeline"].items():
            if "step_sequence" in pipeline_config and isinstance(pipeline_config["step_sequence"], list):

                # 添加当前流程的步骤名称
                pipeline_specific_names = available_names.copy()
                pipeline_specific_names.update(self.context.get_step_names(pipeline_name))

                self._validate_step_list_references(
                    pipeline_config["step_sequence"],
                    pipeline_name,
                    pipeline_specific_names
                )

    def _collect_all_available_names(self, config: dict) -> Set[str]:
        """收集所有顶层可引用的名称"""
        available_names = set()
        if "models" in config and isinstance(config["models"], dict):
            available_names.update(config["models"].keys())
        if "optimizers" in config and isinstance(config["optimizers"], dict):
            available_names.update(config["optimizers"].keys())
        if "losses" in config and isinstance(config["losses"], dict):
            available_names.update(config["losses"].keys())
        if "data_sources" in config and isinstance(config["data_sources"], dict):
            available_names.update(config["data_sources"].keys())
        return available_names

    def _validate_step_list_references(
        self,
        step_sequence: List[Dict],
        pipeline_name: str,
        available_names: Set[str]
    ):
        """验证步骤列表中的变量引用"""
        for i, step in enumerate(step_sequence):
            if not isinstance(step, dict): continue
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
        context_str: str
    ):
        """递归检查字典中的 ${...} 引用"""
        for key, value in data.items():
            if isinstance(value, str):
                matches = ReferencePattern.VARIABLE_COMPILED.findall(value)
                for ref in matches:
                    base_name = ref.split('.')[0]
                    if base_name not in valid_names:
                        self.add_warning(
                            f"{context_str}.args.{key}: 引用了未定义的名称 '${{{ref}}}'"
                        )
            elif isinstance(value, dict):
                self._check_references_in_dict(
                    value,
                    valid_names,
                    f"{context_str}.{key}"
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._check_references_in_dict(
                            item,
                            valid_names,
                            f"{context_str}.{key}[{i}]"
                        )
