# -*- coding: utf-8 -*-
"""
export.py - 模型导出
功能：
  - 将训练好的模型导出为ONNX格式
  - 支持模型转换和优化
  - 生成导出报告
"""

from typing import Dict, Any
import os
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class ModelExporter:
    """
    模型导出器：
    - 导出模型为ONNX格式
    - 支持格式转换
    - 生成导出配置
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化导出器

        参数：
            config: 完整配置文件
        """
        self.config = config
        self.export_config = config.get("export", {})

        # 创建输出目录
        self.output_dir = "outputs/onnx"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"导出器初始化: {self.output_dir}")

    def export(self, models: Dict[str, Any]):
        """
        导出模型

        参数：
            models: 模型字典
        """
        logger.info("=" * 60)
        logger.info("开始导出模型")
        logger.info("=" * 60)

        if not self.export_config:
            logger.warning("未找到export配置")
            return

        model_key = self.export_config.get("model")
        if not model_key:
            logger.error("export配置缺少model字段")
            raise ValueError("缺少model字段")

        if model_key not in models:
            logger.error(f"未找到模型: {model_key}")
            raise ValueError(f"模型 {model_key} 不存在")

        model = models[model_key]

        try:
            self._export_onnx(model, model_key)
            logger.info("模型导出成功")
        except Exception as e:
            logger.error(f"模型导出失败: {str(e)}", exc_info=True)
            raise

    def _export_onnx(self, model: Any, model_name: str):
        """
        导出为ONNX格式

        参数：
            model: 模型实例
            model_name: 模型名称
        """
        logger.info(f"导出模型 {model_name} 为ONNX格式")

        # 获取导出参数
        export_args = self.export_config.get("args", {})
        reflection = self.export_config.get("reflection")

        if not reflection:
            logger.error("导出配置缺少reflection字段")
            raise ValueError("缺少reflection字段")

        # 准备参数
        args = dict(export_args)  # 复制参数，避免修改原配置
        args["model"] = model

        # 更新输出路径
        if "f" in args:
            args["f"] = os.path.join(self.output_dir, os.path.basename(args["f"]))
        else:
            args["f"] = os.path.join(self.output_dir, f"{model_name}.onnx")

        logger.info(f"导出参数: {args}")

        try:
            # 通过反射调用导出函数
            call_target(reflection, args)
            logger.info(f"模型已保存到: {args['f']}")

            # 验证ONNX模型
            self._verify_onnx_model(args["f"])
        except Exception as e:
            logger.error(f"导出失败: {str(e)}", exc_info=True)
            raise

    def export_tensorflow_saved_model(self, model: Any, model_name: str):
        """
        导出TensorFlow SavedModel格式

        参数：
            model: TensorFlow模型
            model_name: 模型名称
        """
        logger.info(f"导出TensorFlow SavedModel: {model_name}")

        save_path = os.path.join(self.output_dir, f"{model_name}_saved_model")

        try:
            call_target("tensorflow.keras.models.save_model", {
                "model": model,
                "filepath": save_path,
                "save_format": "tf"
            })

            logger.info(f"SavedModel已保存到: {save_path}")
        except Exception as e:
            logger.error(f"SavedModel导出失败: {str(e)}", exc_info=True)
            raise

    def export_h5_model(self, model: Any, model_name: str):
        """
        导出H5格式模型

        参数：
            model: TensorFlow模型
            model_name: 模型名称
        """
        logger.info(f"导出H5格式模型: {model_name}")

        save_path = os.path.join(self.output_dir, f"{model_name}.h5")

        try:
            call_target("tensorflow.keras.models.save_model", {
                "model": model,
                "filepath": save_path,
                "save_format": "h5"
            })

            logger.info(f"H5模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"H5模型导出失败: {str(e)}", exc_info=True)
            raise

    def _verify_onnx_model(self, model_path: str):
        """
        验证ONNX模型

        参数：
            model_path: ONNX模型路径
        """
        logger.info(f"验证ONNX模型: {model_path}")

        try:
            import onnx

            # 加载和验证模型
            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            logger.info("ONNX模型验证成功")

            # 打印模型信息
            self._print_onnx_model_info(model)

        except ImportError:
            logger.warning("onnx未安装，跳过验证。请运行: pip install onnx")
        except Exception as e:
            logger.error(f"ONNX验证失败: {str(e)}", exc_info=True)

    def _print_onnx_model_info(self, onnx_model: Any):
        """
        打印ONNX模型信息

        参数：
            onnx_model: ONNX模型实例
        """
        logger.info("=" * 60)
        logger.info("ONNX模型信息")
        logger.info("=" * 60)

        try:
            # 打印输入信息
            logger.info("输入:")
            for input_tensor in onnx_model.graph.input:
                logger.info(f"  名称: {input_tensor.name}")
                if input_tensor.type.tensor_type.shape.dim:
                    shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
                    logger.info(f"  形状: {shape}")

            # 打印输出信息
            logger.info("输出:")
            for output_tensor in onnx_model.graph.output:
                logger.info(f"  名称: {output_tensor.name}")
                if output_tensor.type.tensor_type.shape.dim:
                    shape = [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
                    logger.info(f"  形状: {shape}")

            # 打印模型统计
            logger.info(f"操作数: {len(onnx_model.graph.node)}")
            logger.info(f"初始化器: {len(onnx_model.graph.initializer)}")

        except Exception as e:
            logger.warning(f"打印模型信息失败: {str(e)}")

    def get_model_size(self, model_path: str) -> float:
        """
        获取模型文件大小

        参数：
            model_path: 模型路径
        返回：
            大小（MB）
        """
        try:
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            logger.info(f"模型大小: {size_mb:.2f} MB")
            return size_mb
        except Exception as e:
            logger.error(f"获取模型大小失败: {str(e)}")
            return 0.0


class ExportConfig:
    """
    导出配置管理器
    """

    @staticmethod
    def generate_export_manifest(model_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成导出清单

        参数：
            model_name: 模型名称
            model_info: 模型信息
        返回：
            清单字典
        """
        manifest = {
            "model_name": model_name,
            "export_time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": model_info,
            "framework": "tensorflow",
            "format": "onnx",
            "version": "1.0"
        }

        return manifest

    @staticmethod
    def save_export_manifest(manifest: Dict[str, Any], filepath: str):
        """
        保存导出清单

        参数：
            manifest: 清单字典
            filepath: 保存路径
        """
        import json
        import os

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"导出清单已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存导出清单失败: {str(e)}")
