# -*- coding: utf-8 -*-
"""
modules/models.py
模型构建器：从配置上下文构建 TensorFlow/Keras 模型

设计原则：
1. 支持三种模型类型：Sequential, Functional, Subclass
2. 支持复杂的层连接关系（Connection）
3. 使用 call_target 动态创建层和模型
4. 支持简化配置（可选步骤）

导入限制：
- 仅导入 common 文件夹内的函数
- 使用 call_target 调用 TensorFlow API
"""

from typing import Dict, Any, List, Optional
from common.train_context import TrainContext
from common.common import call_target
from common.interfaces import ModelBuilderInterface


class ModelBuilder(ModelBuilderInterface):
    """
    模型构建器

    职责：
    1. 从 TrainContext.models 读取配置
    2. 使用 call_target 实例化层和模型
    3. 处理层之间的连接关系（Connection）
    4. 支持三种模型类型：Sequential、Functional、Subclass

    支持简化配置：
    - connection 可选（默认顺序连接）
    - 层名称可选（自动生成）
    - 输入/输出标记可选
    """

    def __init__(self, context: TrainContext):
        """初始化模型构建器"""
        super().__init__(context)
        self.models: Dict[str, Any] = {}
        self.layer_cache: Dict[str, Any] = {}
        self.layer_outputs: Dict[str, Any] = {}

    def initialize(self) -> None:
        """从 TrainContext 初始化模型构建器"""
        if not self.context.models:
            raise ValueError("TrainContext 中没有定义任何模型")
        self._initialized = True

    def validate_models(self) -> bool:
        """
        验证模型配置

        返回:
            配置是否有效
        """
        for model_name, model_config in self.context.models.items():
            if not isinstance(model_config, dict):
                raise ValueError(f"模型 '{model_name}' 配置格式错误")

            # 检查必需字段
            if 'type' not in model_config:
                raise ValueError(f"模型 '{model_name}' 缺少 type 字段")

            model_type = model_config['type']

            # Sequential 和 Functional 需要 layers
            if model_type in ['sequential', 'functional']:
                if 'layers' not in model_config or not model_config['layers']:
                    raise ValueError(
                        f"模型 '{model_name}' (type={model_type}) 必须定义 layers"
                    )

            # Subclass 需要 reflection
            if model_type == 'subclass':
                if 'reflection' not in model_config:
                    raise ValueError(
                        f"模型 '{model_name}' (type=subclass) 必须定义 reflection"
                    )

        return True

    def build_model(
        self,
        model_name: str,
        model_config: Any
    ) -> Any:
        """
        构建单个模型

        参数:
            model_name: 模型名称
            model_config: 模型配置

        返回:
            Keras 模型实例
        """
        # 转换为字典格式
        if not isinstance(model_config, dict):
            config = {
                'type': getattr(model_config, 'type', 'sequential'),
                'layers': getattr(model_config, 'layers', []),
                'reflection': getattr(model_config, 'reflection', None),
                'args': getattr(model_config, 'args', {})
            }
        else:
            config = model_config

        model_type = config['type']

        # 清空缓存
        self.layer_cache = {}
        self.layer_outputs = {}

        # 根据类型构建
        if model_type == 'sequential':
            return self._build_sequential_model(model_name, config)
        elif model_type == 'functional':
            return self._build_functional_model(model_name, config)
        elif model_type == 'subclass':
            return self._build_subclass_model(model_name, config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    #=========================================================================
    # Sequential 模型构建
    #=========================================================================

    def _build_sequential_model(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """
        构建顺序模型

        参数:
            model_name: 模型名称
            config: 模型配置

        返回:
            Sequential 模型
        """
        layers_config = config.get('layers', [])

        if not layers_config:
            raise ValueError(f"Sequential 模型 '{model_name}' 没有定义层")

        # 检查是否有非顺序连接
        for i, layer_config in enumerate(layers_config):
            connection = layer_config.get('connection')
            if connection and connection not in [None, '@sequential', '']:
                raise ValueError(
                    f"Sequential 模型不支持非顺序连接: {connection}。"
                    f"请使用 Functional 模型"
                )

        # 构建层列表
        layers = []
        for i, layer_config in enumerate(layers_config):
            layer_name = layer_config.get('name', f"{model_name}_layer_{i}")
            layer = self._build_layer(layer_config, layer_name)
            layers.append(layer)

        # 创建 Sequential 模型
        model = call_target(
            reflection="tensorflow.keras:Sequential",
            args={
                "layers": layers,
                "name": model_name
            }
        )

        return model

    #=========================================================================
    # Functional 模型构建
    #=========================================================================

    def _build_functional_model(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """
        构建函数式模型

        参数:
            model_name: 模型名称
            config: 模型配置

        返回:
            Functional 模型
        """
        layers_config = config.get('layers', [])

        if not layers_config:
            raise ValueError(f"Functional 模型 '{model_name}' 没有定义层")

        # 找到输入层
        input_layers = [
            (i, layer) for i, layer in enumerate(layers_config)
            if layer.get('is_input', False)
        ]

        if not input_layers:
            # 如果没有明确标记，第一层默认为输入
            input_layers = [(0, layers_config[0])]

        # 找到输出层
        output_layers = [
            (i, layer) for i, layer in enumerate(layers_config)
            if layer.get('is_output', False)
        ]

        if not output_layers:
            # 如果没有明确标记，最后一层默认为输出
            output_layers = [(len(layers_config) - 1, layers_config[-1])]

        # 构建输入
        inputs = []
        for i, layer_config in input_layers:
            input_tensor = self._build_input_layer(layer_config, i)
            layer_name = layer_config.get('name', f'input_{i}')
            self.layer_outputs[layer_name] = input_tensor
            inputs.append(input_tensor)

        # 构建所有非输入层并连接
        for i, layer_config in enumerate(layers_config):
            if layer_config.get('is_input', i == 0 and len(input_layers) == 1):
                continue

            layer_name = layer_config.get('name', f'layer_{i}')
            self._build_and_connect_layer(layer_config, layer_name)

        # 收集输出
        outputs = []
        for i, layer_config in output_layers:
            layer_name = layer_config.get('name', f'layer_{i}')
            if layer_name not in self.layer_outputs:
                raise ValueError(f"输出层 '{layer_name}' 没有被连接")
            outputs.append(self.layer_outputs[layer_name])

        # 简化单输入/输出
        if len(inputs) == 1:
            inputs = inputs[0]
        if len(outputs) == 1:
            outputs = outputs[0]

        # 创建模型
        model = call_target(
            reflection="tensorflow.keras:Model",
            args={
                "inputs": inputs,
                "outputs": outputs,
                "name": model_name
            }
        )

        return model

    #=========================================================================
    # Subclass 模型构建
    #=========================================================================

    def _build_subclass_model(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """
        构建子类模型

        参数:
            model_name: 模型名称
            config: 模型配置

        返回:
            Model 子类实例
        """
        reflection = config.get('reflection')

        if not reflection:
            raise ValueError(f"Subclass 模型 '{model_name}' 必须指定 reflection")

        # 准备参数
        args = config.get('args', {}).copy()
        args['name'] = model_name

        # 如果定义了 layers，构建并传递
        layers_config = config.get('layers', [])
        if layers_config:
            built_layers = []
            for i, layer_config in enumerate(layers_config):
                layer_name = layer_config.get('name', f'{model_name}_layer_{i}')
                layer = self._build_layer(layer_config, layer_name)
                built_layers.append(layer)

            args['layers'] = built_layers

        # 实例化模型
        model = call_target(reflection=reflection, args=args)

        return model

    #=========================================================================
    # 层构建
    #=========================================================================

    def _build_layer(
        self,
        layer_config: Dict[str, Any],
        layer_name: str
    ) -> Any:
        """
        构建单个层

        参数:
            layer_config: 层配置
            layer_name: 层名称

        返回:
            Keras 层实例
        """
        reflection = layer_config.get('reflection')

        if not reflection:
            raise ValueError(f"层 '{layer_name}' 缺少 reflection 字段")

        # 准备参数
        args = layer_config.get('args', {}).copy()

        # 设置层名称
        if 'name' not in args:
            args['name'] = layer_name

        # 创建层
        layer = call_target(reflection=reflection, args=args)

        # 缓存层
        self.layer_cache[layer_name] = layer

        return layer

    def _build_input_layer(
        self,
        layer_config: Dict[str, Any],
        index: int
    ) -> Any:
        """
        构建输入层

        参数:
            layer_config: 层配置
            index: 索引

        返回:
            输入张量
        """
        args = layer_config.get('args', {}).copy()
        layer_name = layer_config.get('name', f'input_{index}')

        if 'name' not in args:
            args['name'] = layer_name

        # 输入层必须有 shape
        if 'shape' not in args:
            raise ValueError(f"输入层 '{layer_name}' 必须指定 shape")

        # 创建输入张量
        input_tensor = call_target(
            reflection="tensorflow.keras:Input",
            args=args
        )

        return input_tensor

    #=========================================================================
    # 层连接
    #=========================================================================

    def _build_and_connect_layer(
        self,
        layer_config: Dict[str, Any],
        layer_name: str
    ) -> None:
        """
        构建层并根据 connection 连接

        参数:
            layer_config: 层配置
            layer_name: 层名称
        """
        # 构建层
        layer = self._build_layer(layer_config, layer_name)

        # 解析连接
        connection_str = layer_config.get('connection', '@sequential')

        # 如果为空，使用顺序连接
        if not connection_str or connection_str.strip() == '':
            connection_str = '@sequential'

        # 根据连接类型处理
        output = self._process_connection(layer, connection_str, layer_name)

        # 缓存输出
        self.layer_outputs[layer_name] = output

    def _process_connection(
        self,
        layer: Any,
        connection_str: str,
        layer_name: str
    ) -> Any:
        """
        处理层连接

        参数:
            layer: 层实例
            connection_str: 连接字符串
            layer_name: 层名称

        返回:
            层输出
        """
        # 解析连接类型和目标
        if connection_str.startswith('@sequential'):
            return self._connect_sequential(layer, layer_name)

        elif connection_str.startswith('@skip:'):
            target = connection_str.split(':', 1)[1]
            return self._connect_skip(layer, target, layer_name)

        elif connection_str.startswith('@residual:'):
            target = connection_str.split(':', 1)[1]
            return self._connect_residual(layer, target, layer_name)

        elif connection_str.startswith('@concat:'):
            targets = connection_str.split(':', 1)[1]
            target_list = [t.strip() for t in targets.strip('[]').split(',')]
            return self._connect_concat(layer, target_list, layer_name)

        elif connection_str.startswith('@add:'):
            targets = connection_str.split(':', 1)[1]
            target_list = [t.strip() for t in targets.strip('[]').split(',')]
            return self._connect_add(layer, target_list, layer_name)

        else:
            # 默认顺序连接
            return self._connect_sequential(layer, layer_name)

    def _connect_sequential(self, layer: Any, layer_name: str) -> Any:
        """顺序连接"""
        if not self.layer_outputs:
            raise ValueError(
                f"层 '{layer_name}' 使用顺序连接，但没有前置层"
            )

        # 获取最后一个输出
        last_output = list(self.layer_outputs.values())[-1]
        return layer(last_output)

    def _connect_skip(self, layer: Any, target: str, layer_name: str) -> Any:
        """跳跃连接"""
        if target not in self.layer_outputs:
            raise ValueError(
                f"层 '{layer_name}' 的跳跃连接目标 '{target}' 不存在"
            )

        target_output = self.layer_outputs[target]
        return layer(target_output)

    def _connect_residual(self, layer: Any, target: str, layer_name: str) -> Any:
        """残差连接: x + F(x)"""
        if target not in self.layer_outputs:
            raise ValueError(
                f"层 '{layer_name}' 的残差连接目标 '{target}' 不存在"
            )

        x = self.layer_outputs[target]
        fx = layer(x)

        # 残差相加
        add_layer = call_target(
            reflection="tensorflow.keras.layers:Add",
            args={}
        )
        output = add_layer([x, fx])

        return output

    def _connect_concat(
        self,
        layer: Any,
        targets: List[str],
        layer_name: str
    ) -> Any:
        """拼接连接"""
        if len(targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的拼接连接需要至少2个目标层"
            )

        # 收集输入
        inputs = []
        for target in targets:
            if target not in self.layer_outputs:
                raise ValueError(f"目标层 '{target}' 不存在")
            inputs.append(self.layer_outputs[target])

        # 拼接
        concat_layer = call_target(
            reflection="tensorflow.keras.layers:Concatenate",
            args={"axis": -1}
        )
        concatenated = concat_layer(inputs)

        # 应用当前层
        return layer(concatenated)

    def _connect_add(
        self,
        layer: Any,
        targets: List[str],
        layer_name: str
    ) -> Any:
        """相加连接"""
        if len(targets) < 2:
            raise ValueError(
                f"层 '{layer_name}' 的相加连接需要至少2个目标层"
            )

        # 收集输入
        inputs = []
        for target in targets:
            if target not in self.layer_outputs:
                raise ValueError(f"目标层 '{target}' 不存在")
            inputs.append(self.layer_outputs[target])

        # 相加
        add_layer = call_target(
            reflection="tensorflow.keras.layers:Add",
            args={}
        )
        added = add_layer(inputs)

        # 应用当前层
        return layer(added)

    #=========================================================================
    # 接口实现
    #=========================================================================

    def build_all_models(self) -> Dict[str, Any]:
        """
        构建所有模型

        返回:
            模型字典 {model_name: model}
        """
        for model_name, model_config in self.context.models.items():
            try:
                model = self.build_model(model_name, model_config)
                self.models[model_name] = model
            except Exception as e:
                raise RuntimeError(
                    f"构建模型 '{model_name}' 失败: {e}"
                ) from e

        return self.models

    def compile_models(self) -> None:
        """编译模型（可选）"""
        # 如果有编译配置，执行编译
        # 否则跳过（使用自定义训练循环）
        pass

    def execute(self) -> Dict[str, Any]:
        """执行模型构建"""
        models = self.build_all_models()
        self.compile_models()
        return models

    def finalize(self) -> None:
        """将模型缓存到上下文"""
        self.context.container.models = self.models

    #=========================================================================
    # 工具方法
    #=========================================================================

    def get_model(self, model_name: str) -> Any:
        """
        获取模型实例

        参数:
            model_name: 模型名称

        返回:
            模型实例
        """
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        return self.models[model_name]

    def get_model_summary(self, model_name: str) -> str:
        """
        获取模型摘要

        参数:
            model_name: 模型名称

        返回:
            模型摘要字符串
        """
        model = self.get_model(model_name)

        # 捕获 summary 输出
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model.summary()

        return buffer.getvalue()

    def save_model(
        self,
        model_name: str,
        filepath: str,
        save_format: str = 'tf'
    ) -> None:
        """
        保存模型

        参数:
            model_name: 模型名称
            filepath: 保存路径
            save_format: 保存格式 (tf/h5/keras)
        """
        model = self.get_model(model_name)
        model.save(filepath, save_format=save_format)

    def load_model(
        self,
        model_name: str,
        filepath: str
    ) -> Any:
        """
        加载模型

        参数:
            model_name: 模型名称
            filepath: 模型路径

        返回:
            加载的模型
        """
        model = call_target(
            reflection="tensorflow.keras.models:load_model",
            args={'filepath': filepath}
        )

        self.models[model_name] = model
        return model


#=============================================================================
# 便捷函数
#=============================================================================

def build_models_from_context(context: TrainContext) -> Dict[str, Any]:
    """
    从训练上下文构建所有模型的便捷函数

    参数:
        context: 训练上下文

    返回:
        模型字典
    """
    builder = ModelBuilder(context)
    return builder.run()


def build_single_model(
    context: TrainContext,
    model_name: str
) -> Any:
    """
    从上下文构建单个模型

    参数:
        context: 训练上下文
        model_name: 模型名称

    返回:
        模型实例
    """
    if model_name not in context.models:
        raise ValueError(f"模型 '{model_name}' 不存在于上下文中")

    builder = ModelBuilder(context)
    builder.run()
    return builder.get_model(model_name)


def create_model_builder(context: TrainContext) -> ModelBuilder:
    """
    创建并初始化模型构建器

    参数:
        context: 训练上下文

    返回:
        ModelBuilder 实例
    """
    builder = ModelBuilder(context)
    builder.run()
    return builder
