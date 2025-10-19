# -*- coding: utf-8 -*-
# common/validators/collection_validator.py
"""
验证器：收集层和步骤名称
"""
from .base_validator import BaseValidator

class CollectionValidator(BaseValidator):

    def validate_and_collect(self, config: dict):
        """
        验证第三层：layers层
        同时收集层名称和步骤名称信息，供后续验证使用
        """
        self._collect_layer_names(config)
        self._collect_step_names(config)

    def _collect_layer_names(self, config: dict):
        if "models" not in config or not isinstance(config["models"], dict):
            return

        for model_name, model_config in config["models"].items():
            if "layers" not in model_config:
                self.add_error(f"模型 '{model_name}' 缺少 'layers' 字段")
                continue

            if not isinstance(model_config["layers"], list):
                self.add_error(f"模型 '{model_name}' 的 'layers' 必须是列表")
                continue

            layer_names = set()
            for i, layer in enumerate(model_config["layers"]):
                if isinstance(layer, dict):
                    layer_name = layer.get("name", f"layer_{i}")
                    if layer_name in layer_names:
                        self.add_warning(
                            f"模型 '{model_name}' 中层名称重复: {layer_name}"
                        )
                    layer_names.add(layer_name)

            self.context.set_layer_names(model_name, layer_names)

    def _collect_step_names(self, config: dict):
        if "training_pipeline" not in config or not isinstance(config["training_pipeline"], dict):
            return

        for pipeline_name, pipeline_config in config["training_pipeline"].items():
            if "step_sequence" not in pipeline_config:
                self.add_error(f"流程 '{pipeline_name}' 缺少 'step_sequence' 字段")
                continue

            if not isinstance(pipeline_config["step_sequence"], list):
                self.add_error(
                    f"流程 '{pipeline_name}' 的 'step_sequence' 必须是列表"
                )
                continue

            step_names = set()
            for i, step in enumerate(pipeline_config["step_sequence"]):
                if isinstance(step, dict):
                    step_name = step.get("name", f"step_{i}")
                    if step_name in step_names:
                        self.add_warning(
                            f"流程 '{pipeline_name}' 中步骤名称重复: {step_name}"
                        )
                    step_names.add(step_name)

            self.context.set_step_names(pipeline_name, step_names)
