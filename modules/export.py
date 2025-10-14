# -*- coding: utf-8 -*-
"""
modules/export.py
模型导出模块：
- 导出为ONNX格式
- 导出为SavedModel格式
- 导出为TFLite格式
"""

import tensorflow as tf
import os
from typing import Dict, Any
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主导出函数
# ======================================================
def export_model(export_config: Dict, context: Dict) -> None:
    """
    根据配置导出模型

    参数:
        export_config: 导出配置
        context: 训练上下文
    """
    logger.info("开始导出模型")

    # ONNX导出
    if export_config.get("onnx", {}).get("enabled", False):
        export_to_onnx(export_config["onnx"], context)

    # SavedModel导出
    if export_config.get("saved_model", {}).get("enabled", False):
        export_to_saved_model(export_config["saved_model"], context)

    # TFLite导出
    if export_config.get("tflite", {}).get("enabled", False):
        export_to_tflite(export_config["tflite"], context)

    logger.info("模型导出完成")


# ======================================================
# ONNX导出
# ======================================================
def export_to_onnx(onnx_config: Dict, context: Dict) -> None:
    """
    导出模型为ONNX格式

    参数:
        onnx_config: ONNX导出配置
        context: 上下文
    """
    try:
        import tf2onnx
    except ImportError:
        logger.error("未安装tf2onnx，请运行: pip install tf2onnx")
        return

    model_name = onnx_config.get("model", "classifier")
    output_path = onnx_config.get("output_path", "outputs/model.onnx")
    opset_version = onnx_config.get("opset_version", 13)

    model = context["models"].get(model_name)
    if model is None:
        logger.error(f"未找到模型: {model_name}")
        return

    logger.info(f"导出 {model_name} 为ONNX格式")
    logger.info(f"  输出路径: {output_path}")
    logger.info(f"  Opset版本: {opset_version}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 获取输入签名
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]

    # 转换为ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset_version,
        output_path=output_path
    )

    logger.info("  ONNX导出成功")


# ======================================================
# SavedModel导出
# ======================================================
def export_to_saved_model(saved_model_config: Dict, context: Dict) -> None:
    """
    导出模型为TensorFlow SavedModel格式

    参数:
        saved_model_config: SavedModel导出配置
        context: 上下文
    """
    model_name = saved_model_config.get("model", "classifier")
    output_path = saved_model_config.get("output_path", "outputs/saved_model")

    model = context["models"].get(model_name)
    if model is None:
        logger.error(f"未找到模型: {model_name}")
        return

    logger.info(f"导出 {model_name} 为SavedModel格式")
    logger.info(f"  输出路径: {output_path}")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 保存模型
    tf.saved_model.save(model, output_path)

    logger.info("  SavedModel导出成功")


# ======================================================
# TFLite导出
# ======================================================
def export_to_tflite(tflite_config: Dict, context: Dict) -> None:
    """
    导出模型为TFLite格式（用于移动端部署）

    参数:
        tflite_config: TFLite导出配置
        context: 上下文
    """
    model_name = tflite_config.get("model", "classifier")
    output_path = tflite_config.get("output_path", "outputs/model.tflite")
    quantize = tflite_config.get("quantize", False)

    model = context["models"].get(model_name)
    if model is None:
        logger.error(f"未找到模型: {model_name}")
        return

    logger.info(f"导出 {model_name} 为TFLite格式")
    logger.info(f"  输出路径: {output_path}")
    logger.info(f"  量化: {quantize}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 量化设置
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 转换
    tflite_model = converter.convert()

    # 保存
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    logger.info("  TFLite导出成功")


# ======================================================
# 导出工具函数
# ======================================================
def export_weights_only(model: tf.keras.Model, output_path: str) -> None:
    """
    只导出模型权重

    参数:
        model: Keras模型
        output_path: 输出路径
    """
    logger.info(f"导出权重到: {output_path}")
    model.save_weights(output_path)
    logger.info("  权重导出成功")


def export_model_config(model: tf.keras.Model, output_path: str) -> None:
    """
    导出模型配置（JSON格式）

    参数:
        model: Keras模型
        output_path: 输出路径
    """
    import json

    logger.info(f"导出模型配置到: {output_path}")

    config = model.get_config()

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("  配置导出成功")
