# -*- coding: utf-8 -*-
"""
modules/models.py
模型构建模块：
- 构建神经网络模型
- 模型加载和保存
- 模型管理
"""

import tensorflow as tf
from typing import Dict, Any, List
from common.common import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主构建函数
# ======================================================
def build_all_models(config: Dict) -> Dict[str, tf.keras.Model]:
    """
    构建所有模型

    参数:
        config: models配置

    返回:
        dict: 模型字典 {model_name: model}
    """
    model_dict = {}

    for model_name, model_config in config.items():
        logger.info(f"构建模型: {model_name}")

        model_type = model_config["type"]

        if model_type == "Sequential":
            model = build_sequential_model(model_config)
        elif model_type == "Functional":
            model = build_functional_model(model_config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        model_dict[model_name] = model
        logger.info(f"  模型构建完成，参数量: {model.count_params():,}")

    return model_dict


# ======================================================
# Sequential 模型构建
# ======================================================
def build_sequential_model(config: Dict) -> tf.keras.Sequential:
    """
    构建Sequential顺序模型

    参数:
        config: 模型配置
            - reflection: 模型类路径
            - layers: 层配置列表

    返回:
        tf.keras.Sequential模型
    """
    layers = []

    for layer_config in config.get("layers", []):
        layer = call_target(
            layer_config["reflection"],
            layer_config.get("args", {})
        )
        layers.append(layer)

    # 通过反射创建Sequential模型
    model_class = call_target(config["reflection"], {})

    # 如果返回的是类，则实例化并添加层
    if isinstance(model_class, type):
        model = model_class(layers)
    else:
        # 如果已经是实例，直接返回
        model = model_class
        for layer in layers:
            model.add(layer)

    return model


# ======================================================
# Functional 模型构建
# ======================================================
def build_functional_model(config: Dict) -> tf.keras.Model:
    """
    构建Functional API模型

    参数:
        config: 模型配置 (必须包含 layers 列表)

    返回:
        tf.keras.Model
    """
    layers_config = config.get("layers", [])
    if not layers_config:
        raise ValueError("Functional模型配置中 'layers' 列表不能为空。")

    # 用于存储已创建的 Keras 张量 (Tensor)，键是层名或自定义名称
    tensor_map = {}

    # 用于存储 Input 层对象，以便最后构建 tf.keras.Model
    input_layers = []

    # 跟踪上一个输出张量，用于默认的顺序连接
    last_output_tensor = None

    logger.info("  开始解析函数式模型层连接...")

    for i, layer_cfg in enumerate(layers_config):
        # 1. 获取反射路径和参数
        reflection_path = layer_cfg["reflection"]
        args = layer_cfg.get("args", {})
        layer_name = layer_cfg.get("name", f"layer_{i}")

        # 2. 创建 Keras Layer/Tensor 对象
        try:
            # 2.1. 处理 Input 层 (特殊的创建方式)
            if "Input" in reflection_path:
                # Input 层直接通过 call_target 创建一个 Tensor 对象
                current_tensor = call_target(reflection_path, args)
                input_layers.append(current_tensor)

                # Input 层没有输入，它是图的起点
                current_tensor_input = None

            # 2.2. 处理普通层
            else:
                # 使用 call_target 反射创建 Layer 实例 (如 Dense, Conv2D)
                layer_instance = call_target(reflection_path, args)

                # 确定当前层的输入张量
                input_spec = layer_cfg.get("input_tensor", None)

                if input_spec:
                    # 显式连接：从 tensor_map 中查找指定的输入张量
                    current_tensor_input = tensor_map.get(input_spec)
                    if current_tensor_input is None:
                        raise ValueError(f"层 '{layer_name}' 指定的输入张量 '{input_spec}' 未找到或未命名。")
                else:
                    # 默认连接：使用上一个层的输出
                    current_tensor_input = last_output_tensor

                if current_tensor_input is None:
                    raise ValueError(f"层 '{layer_name}' 缺少输入张量。请确保第一个非Input层有Input连接。")

                # 调用 Layer 实例，实现张量连接：output = layer(input)
                current_tensor = layer_instance(current_tensor_input)

            # 3. 更新张量映射和上一个输出
            tensor_map[layer_name] = current_tensor
            last_output_tensor = current_tensor
            logger.debug(f"    - Layer '{layer_name}' (Output Shape: {current_tensor.shape})")

        except Exception as e:
            logger.error(f"  构建层 '{layer_name}' 失败: {str(e)}", exc_info=True)
            raise

    # 4. 构建最终的 tf.keras.Model
    if not input_layers:
        raise ValueError("Functional模型必须至少有一个 Input 层。")
    if last_output_tensor is None:
        raise ValueError("Functional模型构建完成，但没有找到最终的输出张量。")

    model = tf.keras.Model(
        inputs=input_layers,
        outputs=last_output_tensor,
        name=config.get("name")
    )

    logger.info(f"  函数式模型构建成功，Input 数量: {len(input_layers)}")
    return model


# ======================================================
# 预训练模型加载
# ======================================================
def load_pretrained_model(
    model_name: str,
    weights: str = "imagenet",
    include_top: bool = False,
    input_shape: tuple = None
) -> tf.keras.Model:
    """
    加载预训练模型

    参数:
        model_name: 模型名称 (VGG16/ResNet50/MobileNetV2等)
        weights: 权重 (imagenet/None)
        include_top: 是否包含顶层分类器
        input_shape: 输入形状

    返回:
        预训练模型
    """
    logger.info(f"加载预训练模型: {model_name}")

    if model_name == "VGG16":
        model = tf.keras.applications.VGG16(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape
        )
    elif model_name == "ResNet50":
        model = tf.keras.applications.ResNet50(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape
        )
    elif model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"未知的预训练模型: {model_name}")

    logger.info(f"  预训练模型加载完成")
    return model


# ======================================================
# 模型保存和加载
# ======================================================
def save_model(model: tf.keras.Model, path: str, save_format: str = "tf") -> None:
    """
    保存模型

    参数:
        model: Keras模型
        path: 保存路径
        save_format: 保存格式 (tf/h5)
    """
    logger.info(f"保存模型到: {path}")
    model.save(path, save_format=save_format)
    logger.info("  模型保存完成")


def load_model_from_file(path: str) -> tf.keras.Model:
    """
    从文件加载模型

    参数:
        path: 模型文件路径

    返回:
        加载的模型
    """
    logger.info(f"从文件加载模型: {path}")
    model = tf.keras.models.load_model(path)
    logger.info("  模型加载完成")
    return model


# ======================================================
# 模型工具函数
# ======================================================
def freeze_layers(model: tf.keras.Model, num_layers: int = None) -> None:
    """
    冻结模型的层（用于迁移学习）

    参数:
        model: Keras模型
        num_layers: 冻结的层数（None则冻结全部）
    """
    if num_layers is None:
        num_layers = len(model.layers)

    for i, layer in enumerate(model.layers):
        if i < num_layers:
            layer.trainable = False

    logger.info(f"冻结了前 {num_layers} 层")


def unfreeze_layers(model: tf.keras.Model, num_layers: int = None) -> None:
    """
    解冻模型的层

    参数:
        model: Keras模型
        num_layers: 解冻的层数（None则解冻全部）
    """
    if num_layers is None:
        num_layers = len(model.layers)

    start_idx = len(model.layers) - num_layers
    for i, layer in enumerate(model.layers):
        if i >= start_idx:
            layer.trainable = True

    logger.info(f"解冻了后 {num_layers} 层")


def print_model_summary(model: tf.keras.Model) -> None:
    """
    打印模型摘要

    参数:
        model: Keras模型
    """
    logger.info("模型结构:")
    model.summary(print_fn=logger.info)
