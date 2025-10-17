# -*- coding: utf-8 -*-
"""
modules/export.py
模型导出模块：从配置上下文导出训练好的模型

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable
from common.train_context import (
    TrainContext,
    ExportConfig
)
from common.common import call_target


class ModelExporter:
    """
    模型导出器

    职责：
    1. 读取 TrainContext 中的导出配置
    2. 导出训练好的模型为多种格式
    3. 支持 SavedModel、HDF5、TFLite 等格式
    4. 支持模型优化和量化
    5. 生成模型元数据和签名

    注意：所有导出操作通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化模型导出器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.export_configs: Dict[str, ExportConfig] = {}
        self.export_paths: Dict[str, str] = {}

    def setup_exports(self):
        """设置所有导出配置"""
        if not self.context.export:
            return

        for export_name, export_config in self.context.export.items():
            self.export_configs[export_name] = export_config

    def export_all(self) -> Dict[str, str]:
        """
        导出所有配置的模型

        返回:
            导出路径字典 {export_name: export_path}
        """
        if not self.export_configs:
            self.setup_exports()

        results = {}

        for export_name, export_config in self.export_configs.items():
            try:
                export_path = self.export_model(export_name, export_config)
                results[export_name] = export_path
            except Exception as e:
                raise RuntimeError(f"导出模型 '{export_name}' 失败: {e}")

        return results

    def export_model(
        self,
        export_name: str,
        export_config: Optional[ExportConfig] = None
    ) -> str:
        """
        导出单个模型

        参数:
            export_name: 导出配置名称
            export_config: 导出配置（可选）

        返回:
            导出路径
        """
        if export_config is None:
            if export_name not in self.export_configs:
                raise ValueError(f"导出配置 '{export_name}' 不存在")
            export_config = self.export_configs[export_name]

        # 准备参数
        args = self._prepare_export_args(export_config.args)

        # 使用 call_target 执行导出
        try:
            result = call_target(
                reflection=export_config.reflection,
                args=args
            )

            # 如果返回的是路径字符串
            if isinstance(result, str):
                export_path = result
            # 如果返回的是字典
            elif isinstance(result, dict) and 'path' in result:
                export_path = result['path']
            else:
                export_path = args.get('export_dir', f"./exports/{export_name}")

        except Exception as e:
            raise RuntimeError(f"执行导出失败: {e}")

        # 记录导出路径
        self.export_paths[export_name] = export_path

        return export_path

    def _prepare_export_args(self, args_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备导出参数（解析变量引用）

        参数:
            args_config: 参数配置

        返回:
            解析后的参数
        """
        prepared_args = {}

        for key, value in args_config.items():
            # 解析变量引用 ${variable}
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                var_name = value[2:-1]
                resolved_value = self._resolve_variable(var_name)
                prepared_args[key] = resolved_value

            elif isinstance(value, dict):
                prepared_args[key] = self._prepare_export_args(value)

            elif isinstance(value, list):
                prepared_args[key] = [
                    self._prepare_export_args(item) if isinstance(item, dict) else item
                    for item in value
                ]

            else:
                prepared_args[key] = value

        return prepared_args

    def _resolve_variable(self, var_name: str) -> Any:
        """
        解析变量引用

        参数:
            var_name: 变量名

        返回:
            变量值
        """
        # 支持点号访问
        if '.' in var_name:
            parts = var_name.split('.')
            base_name = parts[0]
            field_path = parts[1:]

            if base_name in self.context.instantiated_models:
                value = self.context.instantiated_models[base_name]
            elif base_name in self.context.execution_results:
                value = self.context.execution_results[base_name]
            else:
                raise ValueError(f"无法解析变量: ${{{var_name}}}")

            for field in field_path:
                if isinstance(value, dict):
                    value = value.get(field)
                elif hasattr(value, field):
                    value = getattr(value, field)
                else:
                    raise ValueError(f"无法访问 '{var_name}'")

            return value

        # 直接变量名
        if var_name in self.context.instantiated_models:
            return self.context.instantiated_models[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${{{var_name}}}")

    def export_savedmodel(
        self,
        model_name: str,
        export_dir: str,
        signatures: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        导出为 SavedModel 格式

        参数:
            model_name: 模型名称
            export_dir: 导出目录
            signatures: 签名字典
            options: 导出选项

        返回:
            导出路径
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        # 使用 call_target 调用 tf.saved_model.save
        call_target(
            reflection="tensorflow.saved_model:save",
            args={
                "obj": model,
                "export_dir": export_dir,
                "signatures": signatures,
                "options": options
            }
        )

        return export_dir

    def export_h5(
        self,
        model_name: str,
        filepath: str,
        include_optimizer: bool = False,
        save_format: str = "h5"
    ) -> str:
        """
        导出为 HDF5 格式

        参数:
            model_name: 模型名称
            filepath: 文件路径
            include_optimizer: 是否包含优化器状态
            save_format: 保存格式

        返回:
            文件路径
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        # 调用模型的 save 方法
        model.save(
            filepath=filepath,
            save_format=save_format,
            include_optimizer=include_optimizer
        )

        return filepath

    def export_tflite(
        self,
        model_name: str,
        filepath: str,
        quantization: Optional[str] = None,
        optimization: Optional[List[str]] = None
    ) -> str:
        """
        导出为 TFLite 格式

        参数:
            model_name: 模型名称
            filepath: 文件路径
            quantization: 量化类型 ('int8', 'float16', None)
            optimization: 优化选项

        返回:
            文件路径
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        # 创建 TFLite 转换器
        converter = call_target(
            reflection="tensorflow.lite:TFLiteConverter.from_keras_model",
            args={"model": model}
        )

        # 设置优化
        if optimization:
            optimizations = []
            for opt in optimization:
                if opt == "DEFAULT":
                    opt_obj = call_target(
                        reflection="tensorflow.lite:Optimize.DEFAULT",
                        args={}
                    )
                    optimizations.append(opt_obj)

            if optimizations:
                converter.optimizations = optimizations

        # 设置量化
        if quantization == "int8":
            # INT8 量化
            converter.target_spec.supported_ops = [
                call_target(
                    reflection="tensorflow.lite:OpsSet.TFLITE_BUILTINS_INT8",
                    args={}
                )
            ]
            converter.inference_input_type = call_target(
                reflection="tensorflow:int8",
                args={}
            )
            converter.inference_output_type = call_target(
                reflection="tensorflow:int8",
                args={}
            )

        elif quantization == "float16":
            # Float16 量化
            converter.target_spec.supported_types = [
                call_target(
                    reflection="tensorflow:float16",
                    args={}
                )
            ]

        # 转换模型
        tflite_model = converter.convert()

        # 保存到文件
        with open(filepath, 'wb') as f:
            f.write(tflite_model)

        return filepath

    def export_onnx(
        self,
        model_name: str,
        filepath: str,
        opset_version: int = 13
    ) -> str:
        """
        导出为 ONNX 格式

        参数:
            model_name: 模型名称
            filepath: 文件路径
            opset_version: ONNX opset 版本

        返回:
            文件路径
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        try:
            # 使用 tf2onnx 转换（需要通过 call_target）
            call_target(
                reflection="tf2onnx.convert:from_keras",
                args={
                    "model": model,
                    "output_path": filepath,
                    "opset": opset_version
                }
            )
        except Exception as e:
            raise RuntimeError(
                f"导出 ONNX 失败: {e}. "
                f"请确保已安装 tf2onnx: pip install tf2onnx"
            )

        return filepath

    def export_weights_only(
        self,
        model_name: str,
        filepath: str,
        save_format: str = "tf"
    ) -> str:
        """
        仅导出模型权重

        参数:
            model_name: 模型名称
            filepath: 文件路径
            save_format: 保存格式 ('tf' 或 'h5')

        返回:
            文件路径
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        # 保存权重
        model.save_weights(filepath, save_format=save_format)

        return filepath

    def export_with_metadata(
        self,
        model_name: str,
        export_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        导出模型并附带元数据

        参数:
            model_name: 模型名称
            export_dir: 导出目录
            metadata: 元数据字典

        返回:
            导出路径字典
        """
        import json
        import os

        # 创建导出目录
        os.makedirs(export_dir, exist_ok=True)

        # 导出模型
        model_path = os.path.join(export_dir, "model")
        self.export_savedmodel(model_name, model_path)

        # 生成元数据
        if metadata is None:
            metadata = self._generate_metadata(model_name)

        # 保存元数据
        metadata_path = os.path.join(export_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "model_path": model_path,
            "metadata_path": metadata_path
        }

    def _generate_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        生成模型元数据

        参数:
            model_name: 模型名称

        返回:
            元数据字典
        """
        if model_name not in self.context.instantiated_models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.instantiated_models[model_name]

        # 获取 TensorFlow 版本（通过反射）
        try:
            tf_version = call_target(
                reflection="tensorflow:__version__",
                args={}
            )
        except:
            tf_version = "unknown"

        metadata = {
            "model_name": model_name,
            "training_mode": self.context.training_mode,
            "project_name": self.context.project_name,
            "model_version": self.context.model_version,
            "framework": "TensorFlow",
            "framework_version": tf_version
        }

        # 添加模型信息
        try:
            metadata["input_shape"] = str(model.input_shape)
            metadata["output_shape"] = str(model.output_shape)
        except:
            pass

        # 添加参数数量（通过反射调用 tensorflow.size）
        try:
            trainable_params = sum([
                call_target(
                    reflection="tensorflow:size",
                    args={"input": v}
                ).numpy() for v in model.trainable_variables
            ])
            total_params = sum([
                call_target(
                    reflection="tensorflow:size",
                    args={"input": v}
                ).numpy() for v in model.variables
            ])

            metadata["trainable_parameters"] = int(trainable_params)
            metadata["total_parameters"] = int(total_params)
        except:
            pass

        # 添加训练信息
        if hasattr(self.context, 'current_epoch'):
            metadata["trained_epochs"] = self.context.current_epoch

        # 添加最佳指标（如果有）
        if hasattr(self, 'evaluator') and self.evaluator:
            try:
                best_metrics = self.evaluator.get_best_metrics()
                if best_metrics:
                    metadata["best_metrics"] = best_metrics
            except:
                pass

        return metadata

    def export_multi_format(
        self,
        model_name: str,
        export_dir: str,
        formats: List[str]
    ) -> Dict[str, str]:
        """
        导出为多种格式

        参数:
            model_name: 模型名称
            export_dir: 导出目录
            formats: 格式列表 ['savedmodel', 'h5', 'tflite', 'onnx']

        返回:
            导出路径字典
        """
        import os

        os.makedirs(export_dir, exist_ok=True)

        results = {}

        for fmt in formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "savedmodel":
                path = os.path.join(export_dir, "savedmodel")
                results["savedmodel"] = self.export_savedmodel(model_name, path)

            elif fmt_lower == "h5":
                path = os.path.join(export_dir, f"{model_name}.h5")
                results["h5"] = self.export_h5(model_name, path)

            elif fmt_lower == "tflite":
                path = os.path.join(export_dir, f"{model_name}.tflite")
                results["tflite"] = self.export_tflite(model_name, path)

            elif fmt_lower == "onnx":
                path = os.path.join(export_dir, f"{model_name}.onnx")
                results["onnx"] = self.export_onnx(model_name, path)

            elif fmt_lower == "weights":
                path = os.path.join(export_dir, "weights")
                results["weights"] = self.export_weights_only(model_name, path)

            else:
                raise ValueError(f"不支持的导出格式: {fmt}")

        return results

    def get_export_info(self, export_name: str) -> Dict[str, Any]:
        """
        获取导出信息

        参数:
            export_name: 导出名称

        返回:
            导出信息字典
        """
        import os

        if export_name not in self.export_paths:
            return {}

        export_path = self.export_paths[export_name]

        info = {
            "export_name": export_name,
            "export_path": export_path,
            "exists": os.path.exists(export_path)
        }

        # 获取文件/目录大小
        if info["exists"]:
            if os.path.isfile(export_path):
                info["size_bytes"] = os.path.getsize(export_path)
                info["size_mb"] = info["size_bytes"] / (1024 * 1024)
            elif os.path.isdir(export_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(export_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                info["size_bytes"] = total_size
                info["size_mb"] = total_size / (1024 * 1024)

        return info


def create_exporter(context: TrainContext) -> ModelExporter:
    """
    创建模型导出器的便捷函数

    参数:
        context: 训练上下文

    返回:
        ModelExporter 实例
    """
    exporter = ModelExporter(context)
    exporter.setup_exports()
    return exporter


def export_model(
    context: TrainContext,
    model_name: str,
    export_dir: str,
    export_format: str = "savedmodel",
    **kwargs
) -> str:
    """
    导出模型的便捷函数

    参数:
        context: 训练上下文
        model_name: 模型名称
        export_dir: 导出目录
        export_format: 导出格式
        **kwargs: 额外参数

    返回:
        导出路径
    """
    exporter = ModelExporter(context)

    if export_format == "savedmodel":
        return exporter.export_savedmodel(model_name, export_dir, **kwargs)
    elif export_format == "h5":
        return exporter.export_h5(model_name, export_dir, **kwargs)
    elif export_format == "tflite":
        return exporter.export_tflite(model_name, export_dir, **kwargs)
    elif export_format == "onnx":
        return exporter.export_onnx(model_name, export_dir, **kwargs)
    elif export_format == "weights":
        return exporter.export_weights_only(model_name, export_dir, **kwargs)
    else:
        raise ValueError(f"不支持的导出格式: {export_format}")


def export_all_models(
    context: TrainContext,
    export_dir: str,
    formats: Optional[List[str]] = None
) -> Dict[str, Dict[str, str]]:
    """
    导出所有模型的便捷函数

    参数:
        context: 训练上下文
        export_dir: 导出目录
        formats: 格式列表

    返回:
        导出路径字典 {model_name: {format: path}}
    """
    import os

    if formats is None:
        formats = ["savedmodel"]

    exporter = ModelExporter(context)
    results = {}

    for model_name in context.instantiated_models.keys():
        model_export_dir = os.path.join(export_dir, model_name)
        results[model_name] = exporter.export_multi_format(
            model_name,
            model_export_dir,
            formats
        )

    return results
