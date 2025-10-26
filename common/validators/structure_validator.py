# -*- coding: utf-8 -*-
# common/validators/structure_validator.py
"""
验证器：基本结构和常量
"""
from .base_validator import BaseValidator
from .constants import (
    ConfigStructure,
    TrainingPipelineConstants,
    TrainingModeConstants
)

class StructureValidator(BaseValidator):
    def validate(self, config: dict):
        """验证第一层（模块）和第二层（常量）"""
        self._validate_module_layer(config)
        self._validate_constant_layer(config)
        self._validate_training_pipeline_structure(config)  # 新增

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

    def _validate_training_pipeline_structure(self, config: dict):
        """验证 training_pipeline 的结构"""
        if "training_pipeline" not in config:
            return

        pipeline_config = config["training_pipeline"]
        if not isinstance(pipeline_config, dict):
            self.add_error("training_pipeline 必须是字典类型")
            return

        # 验证每个训练模式的流程配置
        for mode_name, mode_config in pipeline_config.items():
            if not isinstance(mode_config, dict):
                self.add_error(
                    f"training_pipeline.{mode_name} 必须是字典类型"
                )
                continue

            # 验证必需字段：loop_config 和 step_sequence
            for field in TrainingPipelineConstants.PIPELINE_REQUIRED_FIELDS:
                if field not in mode_config:
                    self.add_error(
                        f"training_pipeline.{mode_name} 缺少必须字段: {field}"
                    )

            # 验证 loop_config 结构
            if "loop_config" in mode_config:
                self._validate_loop_config(mode_name, mode_config["loop_config"])

            # 验证 step_sequence 结构
            if "step_sequence" in mode_config:
                self._validate_step_sequence(mode_name, mode_config["step_sequence"])

    def _validate_loop_config(self, mode_name: str, loop_config: dict):
        """验证 loop_config 的结构"""
        if not isinstance(loop_config, dict):
            self.add_error(
                f"training_pipeline.{mode_name}.loop_config 必须是字典类型"
            )
            return

        # 验证必需字段
        for field in TrainingPipelineConstants.LOOP_CONFIG_REQUIRED_FIELDS:
            if field not in loop_config:
                self.add_error(
                    f"training_pipeline.{mode_name}.loop_config "
                    f"缺少必须字段: {field}"
                )

        # 验证循环类型
        if "type" in loop_config:
            loop_type = loop_config["type"]
            if loop_type not in TrainingPipelineConstants.SUPPORTED_LOOP_TYPES:
                self.add_error(
                    f"training_pipeline.{mode_name}.loop_config.type "
                    f"不支持的循环类型: {loop_type}. "
                    f"支持的类型: {TrainingPipelineConstants.SUPPORTED_LOOP_TYPES}"
                )

            # 验证循环类型对应的必需参数
            if "parameters" in loop_config:
                self._validate_loop_parameters(
                    mode_name,
                    loop_type,
                    loop_config["parameters"]
                )

        # 验证终止条件
        if "termination" in loop_config:
            self._validate_termination(mode_name, loop_config["termination"])

    def _validate_loop_parameters(self, mode_name: str, loop_type: str, parameters: dict):
        """验证循环参数"""
        if not isinstance(parameters, dict):
            self.add_error(
                f"training_pipeline.{mode_name}.loop_config.parameters "
                f"必须是字典类型"
            )
            return

        # 检查必需参数
        required_params = TrainingPipelineConstants.LOOP_TYPE_REQUIRED_PARAMS.get(
            loop_type, set()
        )
        for param in required_params:
            if param not in parameters:
                self.add_warning(
                    f"training_pipeline.{mode_name}.loop_config.parameters "
                    f"建议包含参数: {param} (循环类型 '{loop_type}' 通常需要)"
                )

    def _validate_termination(self, mode_name: str, termination: dict):
        """验证终止条件"""
        if not isinstance(termination, dict):
            self.add_error(
                f"training_pipeline.{mode_name}.loop_config.termination "
                f"必须是字典类型"
            )
            return

        # 验证检查类型
        if "check_type" in termination:
            check_type = termination["check_type"]
            if check_type not in TrainingPipelineConstants.SUPPORTED_TERMINATION_TYPES:
                self.add_error(
                    f"training_pipeline.{mode_name}.loop_config.termination.check_type "
                    f"不支持的类型: {check_type}. "
                    f"支持的类型: {TrainingPipelineConstants.SUPPORTED_TERMINATION_TYPES}"
                )

    def _validate_step_sequence(self, mode_name: str, step_sequence):
        """验证 step_sequence 的结构"""
        if not isinstance(step_sequence, list):
            self.add_error(
                f"training_pipeline.{mode_name}.step_sequence 必须是列表类型"
            )
            return

        if len(step_sequence) == 0:
            self.add_warning(
                f"training_pipeline.{mode_name}.step_sequence 为空"
            )
