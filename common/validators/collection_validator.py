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
        """从 models 配置中收集层名称"""
        if "models" not in config:
            return

        models_config = config["models"]
        if not isinstance(models_config, dict):
            return

        for model_name, model_config in models_config.items():
            if not isinstance(model_config, dict):
                continue

            if "layers" not in model_config:
                continue

            layers = model_config["layers"]
            if not isinstance(layers, list):
                continue

            layer_names = set()
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    continue

                # 获取层名称（如果没有提供，使用索引）
                layer_name = layer.get("name", f"layer_{i}")

                # 检测重复
                if layer_name in layer_names:
                    self.add_warning(
                        f"模型 '{model_name}' 中存在重复的层名称: {layer_name}"
                    )

                layer_names.add(layer_name)

            # 注册到协调器
            self.context.set_layer_names(model_name, layer_names)

    def _collect_step_names(self, config: dict):
        """从 training_pipeline 配置中收集步骤名称"""
        if "training_pipeline" not in config:
            return

        pipeline_config = config["training_pipeline"]
        if not isinstance(pipeline_config, dict):
            return

        # 遍历每个训练模式
        for mode_name, mode_config in pipeline_config.items():
            if not isinstance(mode_config, dict):
                continue

            # ⚠️ 关键修改：从 step_sequence 读取（不是根级）
            if "step_sequence" not in mode_config:
                continue

            step_sequence = mode_config["step_sequence"]
            if not isinstance(step_sequence, list):
                continue

            step_names = set()
            for i, step in enumerate(step_sequence):
                if not isinstance(step, dict):
                    continue

                # 获取步骤名称（必须提供）
                if "name" not in step:
                    self.add_error(
                        f"training_pipeline.{mode_name}.step_sequence[{i}] "
                        f"缺少必须字段: name"
                    )
                    continue

                step_name = step["name"]

                # 检测重复
                if step_name in step_names:
                    self.add_warning(
                        f"训练流程 '{mode_name}' 中存在重复的步骤名称: {step_name}"
                    )

                step_names.add(step_name)

            # 注册到协调器
            self.context.set_step_names(mode_name, step_names)
