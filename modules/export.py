# -*- coding: utf-8 -*-
"""
modules/export.py
模型导出管理器：从配置上下文导出模型为多种格式

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from common.interfaces import ExportInterface
from common.train_context import TrainContext
from common.common import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)


class ModelExportManager(ExportInterface):
    """
    模型导出管理器

    职责：
    1. 读取 TrainContext 中的导出配置
    2. 导出模型为多种格式（SavedModel/ONNX/TFLite/H5）
    3. 支持模型优化和量化
    4. 生成模型元数据
    5. 验证导出结果

    标准生命周期：
    initialize() -> validate_export() -> prepare_model() ->
    export_model() -> verify_export() -> finalize()

    支持的导出格式：
    - SavedModel: TensorFlow 标准格式，用于 TensorFlow Serving
    - ONNX: 跨平台格式，用于多框架部署
    - TFLite: 移动端和边缘设备格式
    - H5: Keras HDF5 格式
    - Weights Only: 仅导出权重

    注意：所有导出操作通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化模型导出管理器

        参数:
            context: 训练上下文
        """
        super().__init__(context)
        self.export_configs: Dict[str, Any] = {}
        self.export_paths: Dict[str, str] = {}
        self.export_info: Dict[str, Dict[str, Any]] = {}

    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有导出配置
        2. 加载导出配置数据
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        logger.info("初始化模型导出管理器")

        # 提取导出配置
        self.export_configs = self._extract_export_configs()

        if not self.export_configs:
            logger.warning("未找到导出配置，导出功能将不可用")
            self.export_configs = {}

        logger.info(f"找到 {len(self.export_configs)} 个导出配置")

        # 初始化状态
        self.export_paths = {}
        self.export_info = {}

        self._initialized = True

    def validate_export(self) -> bool:
        """
        验证导出配置

        返回:
            配置是否有效
        """
        logger.info("验证导出配置")

        if not self.export_configs:
            logger.warning("导出配置为空，跳过验证")
            return True

        # 验证每个导出配置
        for export_name, export_config in self.export_configs.items():
            if not isinstance(export_config, dict):
                logger.error(f"导出配置 '{export_name}' 必须是字典")
                return False

            # 检查必需字段
            if 'format' not in export_config and 'reflection' not in export_config:
                logger.error(
                    f"导出配置 '{export_name}' 缺少 'format' 或 'reflection' 字段"
                )
                return False

            # 验证格式
            if 'format' in export_config:
                fmt = export_config['format'].lower()
                supported_formats = ['savedmodel', 'onnx', 'tflite', 'h5', 'weights']
                if fmt not in supported_formats:
                    logger.error(
                        f"导出配置 '{export_name}' 的格式 '{fmt}' 不支持，"
                        f"支持的格式: {supported_formats}"
                    )
                    return False

        logger.info("导出配置验证通过")
        return True

    def prepare_model(self, model: Any) -> Any:
        """
        准备导出模型

        参数:
            model: 待导出模型

        返回:
            准备好的模型
        """
        logger.info("准备模型用于导出")

        # 模型准备步骤（可选）
        # 1. 移除训练特定的层（如Dropout）
        # 2. 固定批归一化层
        # 3. 转换为推理模式

        # 这里直接返回原模型，实际可以添加更多准备逻辑
        return model

    def export_model(
        self,
        export_name: str,
        export_config: Any
    ) -> str:
        """
        导出单个模型

        参数:
            export_name: 导出配置名称
            export_config: 导出配置

        返回:
            导出路径
        """
        logger.info(f"开始导出模型: {export_name}")

        if not isinstance(export_config, dict):
            raise TypeError(f"导出配置必须是字典: {type(export_config)}")

        # 获取模型名称
        model_name = export_config.get('model', list(self.context.container.models.keys())[0])

        if model_name not in self.context.container.models:
            raise ValueError(f"模型 '{model_name}' 不存在")

        model = self.context.container.models[model_name]

        # 准备模型
        prepared_model = self.prepare_model(model)

        # 获取导出路径
        output_path = export_config.get('output_path', f'outputs/{export_name}')

        # 根据格式或反射调用导出
        if 'reflection' in export_config:
            # 使用自定义反射调用
            export_path = self._export_with_reflection(
                export_config,
                prepared_model,
                output_path
            )
        else:
            # 使用内置格式导出
            export_format = export_config['format'].lower()
            export_path = self._export_with_format(
                export_format,
                model_name,
                prepared_model,
                output_path,
                export_config
            )

        # 保存导出路径
        self.export_paths[export_name] = export_path

        # 生成元数据
        metadata = self._generate_metadata(model_name, prepared_model, export_config)
        self.export_info[export_name] = metadata

        logger.info(f"模型 '{export_name}' 导出成功: {export_path}")

        return export_path

    def export_all(self) -> Dict[str, str]:
        """
        导出所有配置的模型

        返回:
            导出路径字典 {export_name: export_path}
        """
        logger.info("开始导出所有模型")

        if not self.export_configs:
            logger.warning("没有导出配置，跳过导出")
            return {}

        all_paths = {}

        for export_name, export_config in self.export_configs.items():
            try:
                export_path = self.export_model(export_name, export_config)
                all_paths[export_name] = export_path

                # 验证导出
                if export_config.get('verify', False):
                    is_valid = self.verify_export(export_path)
                    if not is_valid:
                        logger.warning(f"导出 '{export_name}' 验证失败")

            except Exception as e:
                logger.error(f"导出 '{export_name}' 失败: {str(e)}", exc_info=True)

        logger.info(f"完成导出，成功 {len(all_paths)}/{len(self.export_configs)} 个")

        return all_paths

    def verify_export(self, export_path: str) -> bool:
        """
        验证导出结果（可选）

        参数:
            export_path: 导出路径

        返回:
            验证是否通过
        """
        logger.info(f"验证导出结果: {export_path}")

        # 检查文件/目录是否存在
        if not os.path.exists(export_path):
            logger.error(f"导出路径不存在: {export_path}")
            return False

        # 检查文件大小
        if os.path.isfile(export_path):
            size = os.path.getsize(export_path)
            if size == 0:
                logger.error(f"导出文件为空: {export_path}")
                return False
            logger.info(f"导出文件大小: {size / (1024*1024):.2f} MB")

        # 可以添加更多验证逻辑
        # 例如：尝试加载模型，进行推理测试等

        logger.info("导出验证通过")
        return True

    def execute(self) -> Dict[str, str]:
        """
        执行核心逻辑：导出所有模型

        返回:
            导出路径字典
        """
        logger.info("开始执行导出流程")

        if not self.validate_export():
            raise ValueError("导出配置验证失败")

        # 导出所有模型
        export_paths = self.export_all()

        logger.info("导出流程执行完成")

        return export_paths

    def finalize(self) -> None:
        """
        清理和保存结果

        职责：
        1. 将导出路径写回 TrainContext
        2. 保存导出元数据
        3. 记录日志
        """
        logger.info("完成导出，保存结果")

        # 保存导出路径到上下文
        if self.export_paths:
            self.context.execution_results['export_paths'] = self.export_paths

        # 保存导出元数据
        if self.export_info:
            self.context.execution_results['export_info'] = self.export_info

        logger.info(f"导出信息已保存，包含 {len(self.export_paths)} 个模型")

    # ========================================================================
    # 私有方法：格式化导出
    # ========================================================================

    def _extract_export_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        从上下文中提取导出配置

        返回:
            导出配置字典
        """
        configs = {}

        # 1. 从 training_pipeline 的参数中查找
        training_mode = self.context.training_mode
        if training_mode in self.context.training_pipelines:
            pipeline = self.context.training_pipelines[training_mode]
            if hasattr(pipeline, 'parameters') and 'export' in pipeline.parameters:
                configs = pipeline.parameters['export']
                logger.debug("从 training_pipeline 中找到导出配置")

        # 2. 从全局配置中查找
        if hasattr(self.context, 'global_config'):
            global_config = self.context.global_config
            if isinstance(global_config, dict) and 'export' in global_config:
                configs.update(global_config['export'])

        # 3. 从自定义字段查找
        if hasattr(self.context, 'export'):
            configs.update(self.context.export)

        return configs

    def _export_with_format(
        self,
        export_format: str,
        model_name: str,
        model: Any,
        output_path: str,
        export_config: Dict[str, Any]
    ) -> str:
        """
        根据格式导出模型

        参数:
            export_format: 导出格式
            model_name: 模型名称
            model: 模型对象
            output_path: 输出路径
            export_config: 导出配置

        返回:
            导出路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if export_format == 'savedmodel':
            return self._export_savedmodel(model, output_path, export_config)

        elif export_format == 'onnx':
            return self._export_onnx(model, output_path, export_config)

        elif export_format == 'tflite':
            return self._export_tflite(model, output_path, export_config)

        elif export_format == 'h5':
            return self._export_h5(model, output_path, export_config)

        elif export_format == 'weights':
            return self._export_weights(model, output_path, export_config)

        else:
            raise ValueError(f"不支持的导出格式: {export_format}")

    def _export_savedmodel(
        self,
        model: Any,
        output_path: str,
        config: Dict[str, Any]
    ) -> str:
        """
        导出为 SavedModel 格式

        参数:
            model: 模型对象
            output_path: 输出路径
            config: 配置

        返回:
            导出路径
        """
        logger.info(f"导出 SavedModel: {output_path}")

        try:
            # 使用 TensorFlow 的 save 方法
            call_target(
                reflection="tensorflow.saved_model:save",
                args={
                    "obj": model,
                    "export_dir": output_path
                }
            )

            logger.info(f"SavedModel 导出成功: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"SavedModel 导出失败: {str(e)}", exc_info=True)
            raise

    def _export_onnx(
        self,
        model: Any,
        output_path: str,
        config: Dict[str, Any]
    ) -> str:
        """
        导出为 ONNX 格式

        参数:
            model: 模型对象
            output_path: 输出路径
            config: 配置

        返回:
            导出路径
        """
        logger.info(f"导出 ONNX: {output_path}")

        opset_version = config.get('opset_version', 13)

        try:
            # 使用 tf2onnx 转换
            import tf2onnx

            # 转换模型
            spec = (tf2onnx.convert.from_keras(
                model,
                opset=opset_version
            ))

            # 保存 ONNX 模型
            onnx_model, _ = spec
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"ONNX 导出成功: {output_path}")
            return output_path

        except ImportError:
            logger.error("tf2onnx 未安装，请运行: pip install tf2onnx")
            raise
        except Exception as e:
            logger.error(f"ONNX 导出失败: {str(e)}", exc_info=True)
            raise

    def _export_tflite(
        self,
        model: Any,
        output_path: str,
        config: Dict[str, Any]
    ) -> str:
        """
        导出为 TFLite 格式

        参数:
            model: 模型对象
            output_path: 输出路径
            config: 配置

        返回:
            导出路径
        """
        logger.info(f"导出 TFLite: {output_path}")

        quantization = config.get('quantization', None)

        try:
            import tensorflow as tf

            # 创建转换器
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 设置优化
            optimizations = config.get('optimizations', [])
            if optimizations:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # 设置量化
            if quantization == 'int8':
                logger.info("应用 INT8 量化")
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

                # 需要代表性数据集
                # representative_dataset = config.get('representative_dataset')
                # if representative_dataset:
                #     converter.representative_dataset = representative_dataset

            elif quantization == 'float16':
                logger.info("应用 Float16 量化")
                converter.target_spec.supported_types = [tf.float16]

            # 转换模型
            tflite_model = converter.convert()

            # 保存到文件
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"TFLite 导出成功: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TFLite 导出失败: {str(e)}", exc_info=True)
            raise

    def _export_h5(
        self,
        model: Any,
        output_path: str,
        config: Dict[str, Any]
    ) -> str:
        """
        导出为 H5 格式

        参数:
            model: 模型对象
            output_path: 输出路径
            config: 配置

        返回:
            导出路径
        """
        logger.info(f"导出 H5: {output_path}")

        include_optimizer = config.get('include_optimizer', False)

        try:
            # 使用 Keras 的 save 方法
            model.save(
                filepath=output_path,
                save_format='h5',
                include_optimizer=include_optimizer
            )

            logger.info(f"H5 导出成功: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"H5 导出失败: {str(e)}", exc_info=True)
            raise

    def _export_weights(
        self,
        model: Any,
        output_path: str,
        config: Dict[str, Any]
    ) -> str:
        """
        仅导出权重

        参数:
            model: 模型对象
            output_path: 输出路径
            config: 配置

        返回:
            导出路径
        """
        logger.info(f"导出权重: {output_path}")

        try:
            # 创建目录
            os.makedirs(output_path, exist_ok=True)

            # 保存权重
            weights_path = os.path.join(output_path, 'weights.h5')
            model.save_weights(weights_path)

            logger.info(f"权重导出成功: {weights_path}")
            return weights_path

        except Exception as e:
            logger.error(f"权重导出失败: {str(e)}", exc_info=True)
            raise

    def _export_with_reflection(
        self,
        export_config: Dict[str, Any],
        model: Any,
        output_path: str
    ) -> str:
        """
        使用反射调用导出

        参数:
            export_config: 导出配置
            model: 模型对象
            output_path: 输出路径

        返回:
            导出路径
        """
        logger.info("使用反射调用导出模型")

        reflection = export_config['reflection']
        args = export_config.get('args', {})

        # 添加默认参数
        if 'model' not in args:
            args['model'] = model
        if 'output_path' not in args:
            args['output_path'] = output_path

        # 解析参数中的变量引用
        resolved_args = self._resolve_arguments(args)

        try:
            # 调用导出函数
            result = call_target(
                reflection=reflection,
                args=resolved_args
            )

            # 如果返回路径，使用返回的路径
            if isinstance(result, str):
                return result

            return output_path

        except Exception as e:
            logger.error(f"反射调用导出失败: {str(e)}", exc_info=True)
            raise

    def _resolve_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析参数中的变量引用

        参数:
            args: 原始参数字典

        返回:
            解析后的参数字典
        """
        resolved_args = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith('$'):
                # 变量引用
                var_name = value[1:]
                resolved_args[key] = self._resolve_variable(var_name)
            elif isinstance(value, dict):
                resolved_args[key] = self._resolve_arguments(value)
            elif isinstance(value, list):
                resolved_args[key] = [
                    self._resolve_variable(v[1:]) if isinstance(v, str) and v.startswith('$') else v
                    for v in value
                ]
            else:
                resolved_args[key] = value

        return resolved_args

    def _resolve_variable(self, var_name: str) -> Any:
        """
        解析变量引用

        参数:
            var_name: 变量名称

        返回:
            变量值
        """
        # 支持点号访问
        if '.' in var_name:
            parts = var_name.split('.')
            obj = self.context

            # 尝试从 container 访问
            if parts[0] == 'models':
                obj = self.context.container.models
                parts = parts[1:]

            # 逐层访问
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part)
                elif hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"无法解析变量: ${var_name}")

            return obj

        # 直接变量名
        if hasattr(self.context.container, 'models') and var_name in self.context.container.models:
            return self.context.container.models[var_name]

        if var_name in self.context.execution_results:
            return self.context.execution_results[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${var_name}")

    def _generate_metadata(
        self,
        model_name: str,
        model: Any,
        export_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成模型元数据

        参数:
            model_name: 模型名称
            model: 模型对象
            export_config: 导出配置

        返回:
            元数据字典
        """
        import datetime

        metadata = {
            "model_name": model_name,
            "export_format": export_config.get('format', 'custom'),
            "export_time": datetime.datetime.now().isoformat(),
            "training_mode": self.context.training_mode,
            "project_name": getattr(self.context, 'project_name', 'unknown'),
        }

        # 添加模型信息
        try:
            if hasattr(model, 'input_shape'):
                metadata["input_shape"] = str(model.input_shape)
            if hasattr(model, 'output_shape'):
                metadata["output_shape"] = str(model.output_shape)
        except:
            pass

        # 添加参数数量
        try:
            import tensorflow as tf
            trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
            total_params = sum([tf.size(v).numpy() for v in model.variables])

            metadata["trainable_parameters"] = int(trainable_params)
            metadata["total_parameters"] = int(total_params)
        except:
            pass

        # 添加训练信息
        if hasattr(self.context, 'current_epoch'):
            metadata["trained_epochs"] = self.context.current_epoch

        return metadata


# ============================================================================
# 便捷函数
# ============================================================================

def export_model_from_context(
    context: TrainContext,
    export_name: Optional[str] = None
) -> Dict[str, str]:
    """
    从训练上下文导出模型的便捷函数

    参数:
        context: 训练上下文
        export_name: 导出配置名称（可选）

    返回:
        导出路径字典

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> # ... 配置 context
        >>> paths = export_model_from_context(context)
    """
    manager = ModelExportManager(context)
    return manager.run()


def create_export_manager(context: TrainContext) -> ModelExportManager:
    """
    创建并初始化导出管理器的便捷函数

    参数:
        context: 训练上下文

    返回:
        ModelExportManager 实例

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> manager = create_export_manager(context)
        >>> paths = manager.export_all()
    """
    manager = ModelExportManager(context)
    manager.initialize()
    return manager


def export_single_model(
    context: TrainContext,
    model_name: str,
    output_path: str,
    export_format: str = 'savedmodel',
    **kwargs
) -> str:
    """
    导出单个模型的便捷函数

    参数:
        context: 训练上下文
        model_name: 模型名称
        output_path: 输出路径
        export_format: 导出格式
        **kwargs: 额外配置

    返回:
        导出路径

    示例:
        >>> path = export_single_model(
        ...     context,
        ...     "classifier",
        ...     "outputs/model",
        ...     "savedmodel"
        ... )
    """
    manager = ModelExportManager(context)
    manager.initialize()

    # 构造导出配置
    export_config = {
        'model': model_name,
        'format': export_format,
        'output_path': output_path,
        **kwargs
    }

    return manager.export_model('temp_export', export_config)
