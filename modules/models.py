# -*- coding: utf-8 -*-
"""
models.py - 模型定义和生成
功能：
  - 通过反射调用从配置文件生成模型
  - 支持任意深度的模型（深度学习或浅层）
  - 通过配置文件中的reflection字段动态构建
"""

from typing import Dict, Any, List, Optional
import os
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class ModelBuilder:
    """
    模型构建器：
    - 根据配置文件动态生成模型
    - 使用反射调用TensorFlow/PyTorch的API
    - 支持任意模型结构
    """

    def build_all(self, models_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建所有配置中的模型

        参数：
            models_config: 配置文件中的models部分
                {
                    "generator": {...},
                    "discriminator": {...},
                    "actor": {...},
                    "critic": {...}
                }
        返回：
            {模型名: 模型实例}
        """
        models = {}

        if not models_config:
            logger.warning("模型配置为空")
            return models

        for model_name, model_cfg in models_config.items():
            logger.info(f"构建模型: {model_name}")
            try:
                model = self.build_single(model_name, model_cfg)
                models[model_name] = model
                logger.info(f"模型 {model_name} 构建成功")
            except Exception as e:
                logger.error(f"模型 {model_name} 构建失败: {str(e)}", exc_info=True)
                raise

        return models

    def build_single(self, model_name: str, model_cfg: Dict[str, Any]) -> Any:
        """
        构建单个模型

        参数：
            model_name: 模型名称
            model_cfg: 模型配置
                {
                    "type": "Functional",
                    "reflection": "tensorflow.keras.Sequential",
                    "layers": [...]
                }
        返回：
            模型实例
        """
        if not model_cfg:
            raise ValueError(f"模型 {model_name} 配置为空")

        model_type = model_cfg.get("type", "Functional")
        reflection = model_cfg.get("reflection")

        if not reflection:
            raise ValueError(f"模型 {model_name} 缺少reflection字段")

        logger.debug(f"模型类型: {model_type}, 反射路径: {reflection}")

        if model_type == "Functional":
            return self._build_functional(model_name, model_cfg)
        elif model_type == "Sequential":
            return self._build_sequential(model_name, model_cfg)
        elif model_type == "Custom":
            return self._build_custom(model_name, model_cfg)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _build_functional(self, model_name: str, model_cfg: Dict[str, Any]) -> Any:
        """
        构建函数式模型（Sequential或Functional API）

        流程：
        1. 调用model_cfg.reflection中的类构造函数
        2. 逐层添加layers中定义的层

        参数：
            model_name: 模型名称
            model_cfg: 模型配置
        返回：
            模型实例
        """
        reflection = model_cfg.get("reflection")
        layers_cfg = model_cfg.get("layers", [])

        logger.debug(f"构建函数式模型: {model_name}")

        # 步骤1：创建模型容器（通常是Sequential）
        logger.debug(f"创建模型容器: {reflection}")
        try:
            model = call_target(reflection)
        except Exception as e:
            logger.error(f"创建模型容器失败: {str(e)}")
            raise

        # 步骤2：添加所有层
        if not layers_cfg:
            logger.warning(f"模型 {model_name} 没有层配置")
            return model

        for idx, layer_cfg in enumerate(layers_cfg):
            layer_reflection = layer_cfg.get("reflection")
            layer_args = layer_cfg.get("args", {})

            if not layer_reflection:
                logger.warning(f"第{idx}层缺少reflection字段，跳过")
                continue

            logger.debug(f"添加第{idx}层: {layer_reflection}, 参数: {layer_args}")

            try:
                # 通过反射调用构造函数，创建层实例
                layer = call_target(layer_reflection, layer_args)

                # 将层添加到模型
                if hasattr(model, 'add'):
                    model.add(layer)
                else:
                    logger.warning(f"模型没有add方法，无法添加第{idx}层")
                    raise ValueError(f"模型类型不支持add方法")

                logger.debug(f"第{idx}层添加成功")
            except Exception as e:
                logger.error(f"添加第{idx}层失败: {str(e)}", exc_info=True)
                raise

        logger.debug(f"模型 {model_name} 构建完成，共{len(layers_cfg)}层")
        return model

    def _build_sequential(self, model_name: str, model_cfg: Dict[str, Any]) -> Any:
        """
        构建Sequential模型
        这是_build_functional的别名，保留用于向后兼容

        参数：
            model_name: 模型名称
            model_cfg: 模型配置
        返回：
            模型实例
        """
        return self._build_functional(model_name, model_cfg)

    def _build_custom(self, model_name: str, model_cfg: Dict[str, Any]) -> Any:
        """
        构建自定义模型

        参数：
            model_name: 模型名称
            model_cfg: 模型配置
        返回：
            模型实例
        """
        reflection = model_cfg.get("reflection")
        args = model_cfg.get("args", {})

        logger.debug(f"构建自定义模型: {model_name}")

        try:
            model = call_target(reflection, args)
            logger.debug(f"自定义模型 {model_name} 构建完成")
            return model
        except Exception as e:
            logger.error(f"构建自定义模型失败: {str(e)}", exc_info=True)
            raise


class ModelManager:
    """
    模型管理器：
    - 保存和加载模型
    - 模型版本管理
    - 获取模型权重
    - 模型验证
    """

    def __init__(self, model_dir: str = "outputs/models"):
        """
        初始化管理器

        参数：
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"模型管理器初始化: {model_dir}")

    def save_model(self, model: Any, model_name: str, version: int = 0, format: str = "keras"):
        """
        保存模型

        参数：
            model: 模型实例
            model_name: 模型名称
            version: 版本号
            format: 保存格式 (keras/h5/savedmodel/checkpoint)
        """
        if format == "keras":
            self._save_keras_model(model, model_name, version)
        elif format == "h5":
            self._save_h5_model(model, model_name, version)
        elif format == "savedmodel":
            self._save_savedmodel(model, model_name, version)
        elif format == "checkpoint":
            self._save_checkpoint(model, model_name, version)
        else:
            raise ValueError(f"不支持的保存格式: {format}")

    def _save_keras_model(self, model: Any, model_name: str, version: int):
        """保存为Keras格式"""
        save_path = os.path.join(self.model_dir, f"{model_name}_v{version}")

        logger.info(f"保存模型为Keras格式: {save_path}")

        try:
            call_target("tensorflow.keras.models.save_model", {
                "model": model,
                "filepath": save_path,
                "save_format": "keras"
            })
            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}", exc_info=True)
            raise

    def _save_h5_model(self, model: Any, model_name: str, version: int):
        """保存为H5格式"""
        save_path = os.path.join(self.model_dir, f"{model_name}_v{version}.h5")

        logger.info(f"保存模型为H5格式: {save_path}")

        try:
            call_target("tensorflow.keras.models.save_model", {
                "model": model,
                "filepath": save_path,
                "save_format": "h5"
            })
            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存H5模型失败: {str(e)}", exc_info=True)
            raise

    def _save_savedmodel(self, model: Any, model_name: str, version: int):
        """保存为SavedModel格式"""
        save_path = os.path.join(self.model_dir, f"{model_name}_v{version}")

        logger.info(f"保存模型为SavedModel格式: {save_path}")

        try:
            call_target("tensorflow.keras.models.save_model", {
                "model": model,
                "filepath": save_path,
                "save_format": "tf"
            })
            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存SavedModel失败: {str(e)}", exc_info=True)
            raise

    def _save_checkpoint(self, model: Any, model_name: str, version: int):
        """保存为检查点格式"""
        save_path = os.path.join(self.model_dir, f"{model_name}_v{version}_ckpt")

        logger.info(f"保存检查点: {save_path}")

        try:
            # 保存权重
            weights = model.get_weights()
            import pickle
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "weights.pkl"), 'wb') as f:
                pickle.dump(weights, f)
            logger.info(f"检查点已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}", exc_info=True)
            raise

    def load_model(self, model_name: str, version: int = 0, custom_objects: Dict = None) -> Any:
        """
        加载模型

        参数：
            model_name: 模型名称
            version: 版本号
            custom_objects: 自定义对象字典
        返回：
            模型实例
        """
        load_path = os.path.join(self.model_dir, f"{model_name}_v{version}")

        logger.info(f"加载模型: {load_path}")

        try:
            args = {
                "filepath": load_path,
                "custom_objects": custom_objects or {}
            }
            model = call_target("tensorflow.keras.models.load_model", args)
            logger.info(f"模型加载成功")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise

    def get_weights(self, model: Any) -> List[Any]:
        """
        获取模型权重

        参数：
            model: 模型实例
        返回：
            权重列表
        """
        try:
            weights = model.get_weights()
            logger.debug(f"获取模型权重成功，共{len(weights)}个权重矩阵")
            return weights
        except Exception as e:
            logger.error(f"获取权重失败: {str(e)}")
            return []

    def set_weights(self, model: Any, weights: List[Any]):
        """
        设置模型权重

        参数：
            model: 模型实例
            weights: 权重列表
        """
        try:
            model.set_weights(weights)
            logger.debug(f"设置模型权重成功")
        except Exception as e:
            logger.error(f"设置权重失败: {str(e)}")
            raise

    def clone_model(self, model: Any, model_name: str, version: int = 0) -> Any:
        """
        克隆模型

        参数：
            model: 要克隆的模型
            model_name: 新模型名称
            version: 版本号
        返回：
            克隆的模型
        """
        try:
            cloned = call_target("tensorflow.keras.models.clone_model", {
                "model": model
            })
            logger.info(f"模型克隆成功: {model_name}_v{version}")
            return cloned
        except Exception as e:
            logger.error(f"克隆模型失败: {str(e)}")
            raise

    def get_model_summary(self, model: Any) -> str:
        """
        获取模型摘要信息

        参数：
            model: 模型实例
        返回：
            摘要字符串
        """
        try:
            from io import StringIO
            import sys

            # 捕获summary的输出
            buffer = StringIO()
            model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            summary = buffer.getvalue()

            return summary
        except Exception as e:
            logger.warning(f"获取模型摘要失败: {str(e)}")
            return ""

    def print_model_info(self, model: Any):
        """
        打印模型信息

        参数：
            model: 模型实例
        """
        logger.info("=" * 60)
        logger.info("模型信息")
        logger.info("=" * 60)

        try:
            summary = self.get_model_summary(model)
            logger.info(summary)
        except Exception as e:
            logger.warning(f"打印模型信息失败: {str(e)}")


class ModelValidator:
    """
    模型验证器：
    - 验证模型结构
    - 检查模型权重
    - 验证模型输入输出
    """

    @staticmethod
    def validate_model(model: Any) -> bool:
        """
        验证模型

        参数：
            model: 模型实例
        返回：
            验证是否通过
        """
        try:
            # 检查模型是否有必要的方法
            if not hasattr(model, 'predict'):
                logger.error("模型缺少predict方法")
                return False

            if not hasattr(model, 'get_weights'):
                logger.error("模型缺少get_weights方法")
                return False

            logger.info("模型验证通过")
            return True
        except Exception as e:
            logger.error(f"模型验证失败: {str(e)}")
            return False

    @staticmethod
    def check_weights_shape(model: Any) -> Dict[str, tuple]:
        """
        检查权重形状

        参数：
            model: 模型实例
        返回：
            {权重名: 形状}
        """
        try:
            weights = model.get_weights()
            weight_shapes = {}

            for i, weight in enumerate(weights):
                weight_shapes[f"weight_{i}"] = weight.shape

            logger.info(f"权重形状检查完成: {len(weight_shapes)}个权重")
            return weight_shapes
        except Exception as e:
            logger.error(f"权重形状检查失败: {str(e)}")
            return {}
