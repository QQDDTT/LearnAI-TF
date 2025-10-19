# -*- coding: utf-8 -*-
# common/validators/connection_validator.py
"""
验证器：Connection 表达式深度验证
"""
import re
from typing import Optional, List, Dict, Any, Tuple, Set
from .base_validator import BaseValidator
from .patterns import ReferencePattern
from common.train_context import ParsedConnection, ConnectionType

# (从 train_context.py 移入)
def parse_connection(connection_str: str) -> Optional[ParsedConnection]:
    if not connection_str or not connection_str.startswith('@'):
        return None
    match = ReferencePattern.CONNECTION_COMPILED.match(connection_str)
    if not match:
        return None
    conn_type, target_str, params_str = match.groups()
    targets = [t.strip() for t in target_str.split(',')] if target_str else []
    params = {}
    if params_str:
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
    return ParsedConnection(conn_type, targets, params, connection_str)

# (从 train_context.py 移入)
def validate_connection_targets(
    conn_type: str,
    targets: List[str],
    available_layers: set
) -> List[str]:
    errors = []
    parsed = ParsedConnection(type=conn_type, targets=targets, params={}, raw="")
    if not parsed.is_valid_type():
        errors.append(f"不支持的连接类型: {conn_type}")
        return errors
    if parsed.requires_targets():
        if not targets:
            errors.append(f"连接类型 '{conn_type}' 需要指定目标层")
            return errors
        min_count = parsed.min_targets_count()
        if len(targets) < min_count:
            errors.append(
                f"连接类型 '{conn_type}' 至少需要 {min_count} 个目标层，"
                f"但只提供了 {len(targets)} 个"
            )
    for target in targets:
        if target not in available_layers:
            errors.append(f"连接目标层 '{target}' 不存在")
    return errors


class ConnectionValidator(BaseValidator):

    def validate(self, config: dict):
        """验证所有 Connection 表达式（深度验证）"""
        if "models" not in config or not isinstance(config["models"], dict):
            return

        for model_name, model_config in config["models"].items():
            if "layers" not in model_config:
                continue
            layers = model_config["layers"]
            if not isinstance(layers, list):
                continue

            available_layers = self.context.get_layer_names(model_name)

            for i, layer in enumerate(layers):
                if not isinstance(layer, dict): continue

                layer_name = layer.get("name", f"layer_{i}")
                if "connection" not in layer or not layer["connection"]:
                    continue

                connection_str = layer["connection"]
                connection = parse_connection(connection_str)

                if connection is None:
                    self.add_error(
                        f"模型 '{model_name}' 层 '{layer_name}': "
                        f"Connection 格式无效: {connection_str}"
                    )
                    continue

                self._validate_connection(connection, model_name, layer_name, available_layers)

    def _validate_connection(self, connection: ParsedConnection, model_name: str, layer_name: str, available_layers: Set[str]):
        """验证单个解析后的 Connection"""

        if not connection.is_valid_type():
            self.add_error(
                f"模型 '{model_name}' 层 '{layer_name}': "
                f"不支持的 Connection 类型: {connection.type}"
            )
            return # 后续验证无意义

        target_errors = validate_connection_targets(
            connection.type,
            connection.targets,
            available_layers
        )
        for error in target_errors:
            self.add_error(
                f"模型 '{model_name}' 层 '{layer_name}': {error}"
            )

        self._validate_connection_params(connection, model_name, layer_name)

    def _validate_connection_params(self, connection, model_name: str, layer_name: str):
        """验证特定 Connection 类型的参数"""

        if connection.type == ConnectionType.RESIDUAL.value:
            if "scale" in connection.params:
                try:
                    if float(connection.params["scale"]) <= 0:
                        self.add_error(f"模型 '{model_name}' 层 '{layer_name}': 'scale' 必须大于 0")
                except ValueError:
                    self.add_error(f"模型 '{model_name}' 层 '{layer_name}': 'scale' 必须是数字")

        if connection.type == ConnectionType.ATTENTION.value:
            if "heads" not in connection.params:
                self.add_warning(f"模型 '{model_name}' 层 '{layer_name}': ATTENTION 连接建议指定 'heads' 参数")
            elif "heads" in connection.params:
                try:
                    if int(connection.params["heads"]) <= 0:
                        self.add_error(f"模型 '{model_name}' 层 '{layer_name}': 'heads' 必须大于 0")
                except ValueError:
                    self.add_error(f"模型 '{model_name}' 层 '{layer_name}': 'heads' 必须是整数")

        # ... (添加原文件中对 CONCAT, ADD, MULTIPLY 的其他参数验证) ...


# 独立验证函数
def validate_connection_expression(connection_str: str) -> Tuple[bool, List[str]]:
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
