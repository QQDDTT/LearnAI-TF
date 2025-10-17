# -*- coding: utf-8 -*-
"""
modules/models.py
模型构建器：从配置上下文构建 TensorFlow/Keras 模型

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Tuple
from common.train_context import (
    TrainContext,
    ModelConfig,
    LayerConfig,
    ModelType,
    ConnectionType,
    parse_connection
)
from common.common import call_target


class ModelBuilder:
    """
    模型构建器

    职责：
    1. 读取 TrainContext 中的模型配置
    2. 使用 call_target 实例化层和模型
    3. 处理层之间的连接关系（Connection）
    4. 支持三种模型类型：Sequential、Functional、Subclass

    注意：所有 TensorFlow 层和模型通过 call_target 动态创建
    """

    def __init__(self, context: TrainContext):
        """
        初始化模型构建器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.layer_cache: Dict[str, Any] = {}
        self.layer_outputs: Dict[str, Any] = {}

    def build_all_models(self) -> Dict[str, Any]:
        """
        构建所有模型

        返回:
            模型字典 {model_name: model_instance}
        """
        if not self.context.models:
            raise ValueError("TrainContext 中没有定义任何模型")

        models = {}

        for model_name, model_config in self.context.models.items():
            try:
                model = self.build_model(model_name, model_config)
                models[model_name] = model
            except Exception as e:
                raise RuntimeError(f"构建模型 '{model_name}' 失败: {e}")

        # 缓存到上下文
        self.context.instantiated_models = models

        return models

    def build_model(
        self,
        model_name: str,
        model_config: ModelConfig
    ) -> Any:
        """
        构建单个模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Keras 模型实例
        """
        model_type = model_config.type

        # 清空缓存
        self.layer_cache = {}
        self.layer_outputs = {}

        if model_type == ModelType.SEQUENTIAL.value:
            return self._build_sequential_model(model_name, model_config)
        elif model_type == ModelType.FUNCTIONAL.value:
            return self._build_functional_model(model_name, model_config)
        elif model_type == ModelType.SUBCLASS.value:
            return self._build_subclass_model(model_name, model_config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _build_sequential_model(
        self,
        model_name: str,
        model_config: ModelConfig
    ) -> Any:
        """
        构建顺序模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Sequential 模型
        """
        if not model_config.layers:
            raise ValueError(f"模型 '{model_name}' 没有定义层")

        # 检查是否有非顺序连接
        for layer_config in model_config.layers:
            if layer_config.connection and layer_config.connection != "@sequential":
                raise ValueError(
                    f"Sequential 模型不支持非顺序连接，"
                    f"请使用 Functional 模型"
                )

        # 构建层列表
        layers = []
        for i, layer_config in enumerate(model_config.layers):
            layer = self._build_layer(layer_config, f"{model_name}_layer_{i}")
            layers.append(layer)

        # 通过 call_target 创建 Sequential 模型
        model = call_target(
            reflection="tensorflow.keras:Sequential",
            args={
                "layers": layers,
                "name": model_name
            }
        )

        return model

    def _build_functional_model(
        self,
        model_name: str,
        model_config: ModelConfig
    ) -> Any:
        """
        构建函数式模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Functional 模型
        """
        if not model_config.layers:
            raise ValueError(f"模型 '{model_name}' 没有定义层")

        # 找到输入层
        input_layers = [
            layer for layer in model_config.layers
            if layer.is_input
        ]

        if not input_layers:
            raise ValueError(
                f"Functional 模型 '{model_name}' 必须至少有一个输入层"
            )

        # 找到输出层
        output_layers = [
            layer for layer in model_config.layers
            if layer.is_output
        ]

        if not output_layers:
            raise ValueError(
                f"Functional 模型 '{model_name}' 必须至少有一个输出层"
            )

        # 构建输入
        inputs = []
        for layer_config in input_layers:
            input_tensor = self._build_input_layer(layer_config)
            layer_name = layer_config.name or "input"
            self.layer_outputs[layer_name] = input_tensor
            inputs.append(input_tensor)

        # 构建所有层并连接
        for layer_config in model_config.layers:
            if layer_config.is_input:
                continue

            self._build_and_connect_layer(layer_config, model_name)

        # 收集输出
        outputs = []
        for layer_config in output_layers:
            layer_name = layer_config.name or "output"
            if layer_name not in self.layer_outputs:
                raise ValueError(f"输出层 '{layer_name}' 没有被连接")
            outputs.append(self.layer_outputs[layer_name])

        # 创建模型
        if len(inputs) == 1:
            inputs = inputs[0]
        if len(outputs) == 1:
            outputs = outputs[0]

        # 通过 call_target 创建 Model
        model = call_target(
            reflection="tensorflow.keras:Model",
            args={
                "inputs": inputs,
                "outputs": outputs,
                "name": model_name
            }
        )

        return model

    def _build_subclass_model(
        self,
        model_name: str,
        model_config: ModelConfig
    ) -> Any:
        """
        构建子类模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Model 子类实例
        """
        if not model_config.reflection:
            raise ValueError(
                f"Subclass 模型 '{model_name}' 必须指定 reflection"
            )

        # 准备参数
        args = model_config.args.copy()

        # 构建所有层
        if model_config.layers:
            built_layers = []
            for i, layer_config in enumerate(model_config.layers):
                layer = self._build_layer(
                    layer_config,
                    f"{model_name}_layer_{i}"
                )
                built_layers.append(layer)

            # 将层列表传递给模型构造函数
            args['layers'] = built_layers

        args['name'] = model_name

        # 使用 call_target 实例化模型类
        model = call_target(
            reflection=model_config.reflection,
            args=args
        )

        return model

    def _build_layer(
        self,
        layer_config: LayerConfig,
        default_name: str
    ) -> Any:
        """
        构建单个层

        参数:
            layer_config: 层配置
            default_name: 默认层名称

        返回:
            Keras 层实例
        """
        if not layer_config.reflection:
            raise ValueError("层配置缺少 reflection 字段")

        # 准备参数
        args = layer_config.args.copy()

        # 设置层名称
        if 'name' not in args:
            args['name'] = layer_config.name or default_name

        # 使用 call_target 实例化层
        layer = call_target(
            reflection=layer_config.reflection,
            args=args
        )

        # 缓存层
        layer_name = layer_config.name or default_name
        self.layer_cache[layer_name] = layer

        return layer

    def _build_input_layer(
        self,
        layer_config: LayerConfig
    ) -> Any:
        """
        构建输入层

        参数:
            layer_config: 层配置

        返回:
            输入张量
        """
        args = layer_config.args.copy()
        layer_name = layer_config.name or "input"

        if 'name' not in args:
            args['name'] = layer_name

        # 输入层使用 tf.keras.Input（通过 call_target）
        if 'shape' not in args:
            raise ValueError(f"输入层 '{layer_name}' 必须指定 shape")

        input_tensor = call_target(
            reflection="tensorflow.keras:Input",
            args=args
        )

        return input_tensor

    def _build_and_connect_layer(
        self,
        layer_config: LayerConfig,
        model_name: str
    ):
        """
        构建层并根据 Connection 连接

        参数:
            layer_config: 层配置
            model_name: 模型名称
        """
        layer_name = layer_config.name or "unnamed_layer"

        # 构建层
        layer = self._build_layer(layer_config, layer_name)

        # 解析连接
        connection_str = layer_config.connection or "@sequential"
        connection = parse_connection(connection_str)

        if connection is None:
            raise ValueError(
                f"层 '{layer_name}' 的 connection 格式无效: {connection_str}"
            )

        # 根据连接类型连接层
        if connection.type == ConnectionType.SEQUENTIAL.value:
            output = self._connect_sequential(layer, layer_name)

        elif connection.type == ConnectionType.SKIP.value:
            output = self._connect_skip(layer, connection, layer_name)

        elif connection.type == ConnectionType.RESIDUAL.value:
            output = self._connect_residual(layer, connection, layer_name)

        elif connection.type == ConnectionType.CONCAT.value:
            output = self._connect_concat(layer, connection, layer_name)

        elif connection.type == ConnectionType.ADD.value:
            output = self._connect_add(layer, connection, layer_name)

        elif connection.type == ConnectionType.MULTIPLY.value:
            output = self._connect_multiply(layer, connection, layer_name)

        elif connection.type == ConnectionType.ATTENTION.value:
            output = self._connect_attention(layer, connection, layer_name)

        elif connection.type == ConnectionType.BRANCH.value:
            output = self._connect_branch(layer, connection, layer_name)

        elif connection.type == ConnectionType.MERGE.value:
            output = self._connect_merge(layer, connection, layer_name)

        elif connection.type == ConnectionType.DENSE.value:
            output = self._connect_dense(layer, connection, layer_name)

        else:
            raise ValueError(f"不支持的连接类型: {connection.type}")

        # 缓存输出
        self.layer_outputs[layer_name] = output

    def _connect_sequential(
        self,
        layer: Any,
        layer_name: str
    ) -> Any:
        """顺序连接：连接到最后一个层的输出"""
        if not self.layer_outputs:
            raise ValueError(
                f"层 '{layer_name}' 使用顺序连接，但没有前置层"
            )

        # 获取最后一个输出
        last_output = list(self.layer_outputs.values())[-1]
        return layer(last_output)

    def _connect_skip(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """跳跃连接：跳过中间层，直接连接到指定层"""
        if not connection.targets:
            raise ValueError(f"层 '{layer_name}' 的跳跃连接需要指定目标层")

        target_name = connection.targets[0]
        if target_name not in self.layer_outputs:
            raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")

        target_output = self.layer_outputs[target_name]
        return layer(target_output)

    def _connect_residual(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """残差连接：x + F(x)"""
        if not connection.targets:
            raise ValueError(f"层 '{layer_name}' 的残差连接需要指定输入层")

        input_name = connection.targets[0]
        if input_name not in self.layer_outputs:
            raise ValueError(f"输入层 '{input_name}' 不存在或尚未构建")

        x = self.layer_outputs[input_name]
        fx = layer(x)

        # 支持缩放因子（通过 call_target 创建 Lambda 层）
        scale = float(connection.params.get('scale', 1.0))
        if scale != 1.0:
            fx = call_target(
                reflection="tensorflow.keras.layers:Lambda",
                args={"function": lambda t: t * scale}
            )(fx)

        # 残差相加（通过 call_target 创建 Add 层）
        add_layer = call_target(
            reflection="tensorflow.keras.layers:Add",
            args={}
        )
        output = add_layer([x, fx])

        return output

    def _connect_concat(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """拼接连接：沿指定轴拼接多个输入"""
        if len(connection.targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的拼接连接至少需要 2 个目标层"
            )

        # 收集所有输入
        inputs = []
        for target_name in connection.targets:
            if target_name not in self.layer_outputs:
                raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")
            inputs.append(self.layer_outputs[target_name])

        # 拼接（通过 call_target 创建 Concatenate 层）
        axis = int(connection.params.get('axis', -1))
        concat_layer = call_target(
            reflection="tensorflow.keras.layers:Concatenate",
            args={"axis": axis}
        )
        concatenated = concat_layer(inputs)

        # 应用当前层
        return layer(concatenated)

    def _connect_add(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """相加连接：多个输入逐元素相加"""
        if len(connection.targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的相加连接至少需要 2 个目标层"
            )

        # 收集所有输入
        inputs = []
        for target_name in connection.targets:
            if target_name not in self.layer_outputs:
                raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")
            inputs.append(self.layer_outputs[target_name])

        # 相加（通过 call_target 创建 Add 层）
        add_layer = call_target(
            reflection="tensorflow.keras.layers:Add",
            args={}
        )
        added = add_layer(inputs)

        # 支持缩放
        scale = float(connection.params.get('scale', 1.0))
        if scale != 1.0:
            added = call_target(
                reflection="tensorflow.keras.layers:Lambda",
                args={"function": lambda t: t * scale}
            )(added)

        # 应用当前层
        return layer(added)

    def _connect_multiply(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """相乘连接：多个输入逐元素相乘"""
        if len(connection.targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的相乘连接至少需要 2 个目标层"
            )

        # 收集所有输入
        inputs = []
        for target_name in connection.targets:
            if target_name not in self.layer_outputs:
                raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")
            inputs.append(self.layer_outputs[target_name])

        # 相乘（通过 call_target 创建 Multiply 层）
        multiply_layer = call_target(
            reflection="tensorflow.keras.layers:Multiply",
            args={}
        )
        multiplied = multiply_layer(inputs)

        # 应用当前层
        return layer(multiplied)

    def _connect_attention(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """注意力连接：使用 query, key, value 三个输入"""
        if len(connection.targets) != 3:
            raise ValueError(
                f"层 '{layer_name}' 的注意力连接需要 3 个目标层 "
                f"(query, key, value)"
            )

        query_name, key_name, value_name = connection.targets

        if query_name not in self.layer_outputs:
            raise ValueError(f"Query 层 '{query_name}' 不存在或尚未构建")
        if key_name not in self.layer_outputs:
            raise ValueError(f"Key 层 '{key_name}' 不存在或尚未构建")
        if value_name not in self.layer_outputs:
            raise ValueError(f"Value 层 '{value_name}' 不存在或尚未构建")

        query = self.layer_outputs[query_name]
        key = self.layer_outputs[key_name]
        value = self.layer_outputs[value_name]

        # 应用注意力层（假设 layer 是 MultiHeadAttention）
        output = layer(query, key, value)

        return output

    def _connect_branch(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """
        分支连接：当前层的输出会分发到多个后续层

        注意：这里只返回当前层的输出，分支逻辑由后续层处理
        """
        if not self.layer_outputs:
            raise ValueError(f"层 '{layer_name}' 使用分支连接，但没有前置层")

        last_output = list(self.layer_outputs.values())[-1]
        return layer(last_output)

    def _connect_merge(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """合并连接：合并多个分支（默认使用拼接）"""
        if len(connection.targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的合并连接至少需要 2 个目标层"
            )

        # 收集所有输入
        inputs = []
        for target_name in connection.targets:
            if target_name not in self.layer_outputs:
                raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")
            inputs.append(self.layer_outputs[target_name])

        # 默认使用拼接合并
        merge_type = connection.params.get('type', 'concat')

        if merge_type == 'concat':
            axis = int(connection.params.get('axis', -1))
            merge_layer = call_target(
                reflection="tensorflow.keras.layers:Concatenate",
                args={"axis": axis}
            )
        elif merge_type == 'add':
            merge_layer = call_target(
                reflection="tensorflow.keras.layers:Add",
                args={}
            )
        elif merge_type == 'average':
            merge_layer = call_target(
                reflection="tensorflow.keras.layers:Average",
                args={}
            )
        else:
            raise ValueError(f"不支持的合并类型: {merge_type}")

        merged = merge_layer(inputs)

        # 应用当前层
        return layer(merged)

    def _connect_dense(
        self,
        layer: Any,
        connection,
        layer_name: str
    ) -> Any:
        """
        密集连接（DenseNet 风格）：连接到所有前置层
        """
        if not connection.targets:
            raise ValueError(
                f"层 '{layer_name}' 的密集连接需要指定前置层"
            )

        # 收集所有前置层的输出
        inputs = []
        for target_name in connection.targets:
            if target_name not in self.layer_outputs:
                raise ValueError(f"目标层 '{target_name}' 不存在或尚未构建")
            inputs.append(self.layer_outputs[target_name])

        # 拼接所有前置层
        if len(inputs) > 1:
            axis = int(connection.params.get('axis', -1))
            concat_layer = call_target(
                reflection="tensorflow.keras.layers:Concatenate",
                args={"axis": axis}
            )
            concatenated = concat_layer(inputs)
        else:
            concatenated = inputs[0]

        # 应用当前层
        return layer(concatenated)


def build_models_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有模型的便捷函数

    参数:
        context: 训练上下文

    返回:
        模型字典
    """
    builder = ModelBuilder(context)
    return builder.build_all_models()


def build_single_model(
    context: TrainContext,
    model_name: str
) -> Any:
    """
    从训练上下文构建单个模型的便捷函数

    参数:
        context: 训练上下文
        model_name: 模型名称

    返回:
        Keras 模型
    """
    if model_name not in context.models:
        raise ValueError(f"模型 '{model_name}' 不存在于上下文中")

    builder = ModelBuilder(context)
    model_config = context.models[model_name]
    return builder.build_model(model_name, model_config)
