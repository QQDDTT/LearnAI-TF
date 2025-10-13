# -*- coding: utf-8 -*-
"""
optimizers.py - 优化器定义和生成
功能：
  - 通过反射调用从配置文件生成优化器
  - 支持Adam、SGD、RMSprop等各种优化器
  - 为不同的模型配置不同的学习率和参数
"""

from typing import Dict, Any
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class OptimizerBuilder:
    """
    优化器构建器：
    - 根据配置文件动态生成优化器
    - 使用反射调用TensorFlow的Optimizer API
    - 支持任意优化器类型
    """

    def build_all(self, optimizers_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建所有配置中的优化器

        参数：
            optimizers_config: 配置文件中的optimizers部分
                {
                    "generator_opt": {...},
                    "discriminator_opt": {...},
                    "actor_opt": {...}
                }
        返回：
            {优化器名: 优化器实例}
        """
        optimizers = {}

        if not optimizers_config:
            logger.warning("优化器配置为空")
            return optimizers

        for opt_name, opt_cfg in optimizers_config.items():
            logger.info(f"构建优化器: {opt_name}")
            try:
                optimizer = self.build_single(opt_name, opt_cfg)
                optimizers[opt_name] = optimizer
                logger.info(f"优化器 {opt_name} 构建成功")
            except Exception as e:
                logger.error(f"优化器 {opt_name} 构建失败: {str(e)}", exc_info=True)
                raise

        return optimizers

    def build_single(self, opt_name: str, opt_cfg: Dict[str, Any]) -> Any:
        """
        构建单个优化器

        参数：
            opt_name: 优化器名称
            opt_cfg: 优化器配置
                {
                    "reflection": "tensorflow.keras.optimizers.Adam",
                    "args": {
                        "learning_rate": 0.001,
                        "beta_1": 0.9,
                        ...
                    }
                }
        返回：
            优化器实例
        """
        reflection = opt_cfg.get("reflection")
        args = opt_cfg.get("args", {})

        if not reflection:
            raise ValueError(f"优化器 {opt_name} 缺少reflection字段")

        logger.debug(f"创建优化器: {reflection}, 参数: {args}")

        try:
            # 通过反射调用优化器的构造函数
            optimizer = call_target(reflection, args)
            logger.debug(f"优化器 {opt_name} 创建成功")
            return optimizer
        except Exception as e:
            logger.error(f"优化器创建失败: {str(e)}", exc_info=True)
            raise


class OptimizerManager:
    """
    优化器管理器：
    - 管理优化器的学习率调整
    - 实现学习率衰减
    - 保存和加载优化器状态
    """

    def __init__(self):
        """初始化优化器管理器"""
        self.learning_rate_history = []

    def set_learning_rate(self, optimizer: Any, learning_rate: float):
        """
        设置优化器的学习率

        参数：
            optimizer: 优化器实例
            learning_rate: 新的学习率
        """
        logger.info(f"设置学习率: {learning_rate}")

        try:
            # TensorFlow 2.x: 直接赋值learning_rate属性
            if hasattr(optimizer, 'learning_rate'):
                optimizer.learning_rate = learning_rate
                logger.debug(f"学习率已更新")
            else:
                logger.warning(f"优化器没有learning_rate属性")
        except Exception as e:
            logger.error(f"设置学习率失败: {str(e)}")
            raise

    def decay_learning_rate(self, optimizer: Any, decay_factor: float = 0.9, min_lr: float = 1e-6):
        """
        衰减学习率（乘以衰减因子）

        参数：
            optimizer: 优化器实例
            decay_factor: 衰减因子（通常0.9, 0.95等）
            min_lr: 最小学习率
        """
        try:
            current_lr = self.get_learning_rate(optimizer)
            if current_lr is None:
                logger.warning("无法获取当前学习率")
                return

            new_lr = max(current_lr * decay_factor, min_lr)
            self.set_learning_rate(optimizer, new_lr)

            self.learning_rate_history.append({
                "old_lr": current_lr,
                "new_lr": new_lr
            })

            logger.info(f"学习率衰减: {current_lr:.6f} -> {new_lr:.6f}")
        except Exception as e:
            logger.error(f"学习率衰减失败: {str(e)}")

    def get_learning_rate(self, optimizer: Any) -> float:
        """
        获取优化器的学习率

        参数：
            optimizer: 优化器实例
        返回：
            当前学习率
        """
        try:
            if not hasattr(optimizer, 'learning_rate'):
                logger.warning("优化器没有learning_rate属性")
                return None

            lr = optimizer.learning_rate

            # 处理不同的学习率类型
            if hasattr(lr, 'numpy'):
                return float(lr.numpy())
            elif callable(lr):
                # 某些情况下learning_rate可能是可调用的
                return float(lr())
            else:
                return float(lr)
        except Exception as e:
            logger.warning(f"获取学习率失败: {str(e)}")
            return None

    def get_config(self, optimizer: Any) -> Dict[str, Any]:
        """
        获取优化器配置

        参数：
            optimizer: 优化器实例
        返回：
            配置字典
        """
        try:
            if hasattr(optimizer, 'get_config'):
                return optimizer.get_config()
            else:
                logger.warning("优化器没有get_config方法")
                return {}
        except Exception as e:
            logger.error(f"获取优化器配置失败: {str(e)}")
            return {}

    def save_optimizer_state(self, optimizer: Any, filepath: str):
        """
        保存优化器状态（权重和学习率等）

        参数：
            optimizer: 优化器实例
            filepath: 保存路径
        """
        import pickle
        import os

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        try:
            state = {
                'learning_rate': self.get_learning_rate(optimizer),
                'config': self.get_config(optimizer),
                'weights': optimizer.get_weights() if hasattr(optimizer, 'get_weights') else None
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"优化器状态已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存优化器状态失败: {str(e)}")
            raise

    def load_optimizer_state(self, optimizer: Any, filepath: str):
        """
        加载优化器状态

        参数：
            optimizer: 优化器实例
            filepath: 加载路径
        """
        import pickle

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # 恢复学习率
            if state.get('learning_rate'):
                self.set_learning_rate(optimizer, state['learning_rate'])

            # 恢复权重
            if state.get('weights') and hasattr(optimizer, 'set_weights'):
                optimizer.set_weights(state['weights'])

            logger.info(f"优化器状态已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载优化器状态失败: {str(e)}")
            raise

    def print_optimizer_info(self, optimizer: Any):
        """
        打印优化器信息

        参数：
            optimizer: 优化器实例
        """
        logger.info("=" * 60)
        logger.info("优化器信息")
        logger.info("=" * 60)

        try:
            logger.info(f"优化器类型: {type(optimizer).__name__}")

            lr = self.get_learning_rate(optimizer)
            if lr is not None:
                logger.info(f"学习率: {lr}")

            config = self.get_config(optimizer)
            if config:
                logger.info("配置:")
                for key, value in config.items():
                    logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.warning(f"打印优化器信息失败: {str(e)}")
