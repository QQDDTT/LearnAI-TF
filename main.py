# -*- coding: utf-8 -*-
"""
main.py - 主流程入口
功能：
  - 加载配置文件
  - 初始化所有组件（模型、优化器、损失函数、数据加载器）
  - 选择对应的训练流程
  - 执行训练、评估、导出
"""

import sys
import os
from typing import Dict, Any

from modules.utils import LoggerManager, load_yaml, call_target
from modules.models import ModelBuilder
from modules.optimizers import OptimizerBuilder
from modules.losses import LossBuilder
from modules.data_manager import DataManager
from modules.training_pipeline import TrainingPipeline
from modules.evaluation import Evaluator
from modules.export import ModelExporter
from modules.deployment import ModelDeployer

# 初始化日志
logger = LoggerManager.get_logger(__file__)


class UniversalTrainer:
    """
    统一的训练框架：
    - 支持所有训练方式
    - 通过配置文件驱动
    - 自动选择对应的训练流程
    """

    def __init__(self, config_file: str):
        """
        初始化训练器
        参数：
            config_file: 配置文件名称（从config目录读取）
        """
        logger.info(f"加载配置文件: {config_file}")
        self.config = load_yaml(config_file)

        # 记录训练模式
        self.training_mode = self.config.get("training_mode", {})
        logger.info(f"训练模式: {self.training_mode.get('type')} "
                   f"({self.training_mode.get('subtype', 'default')})")

        # 初始化所有组件
        logger.info("初始化模型...")
        self.models = self._build_models()

        logger.info("初始化优化器...")
        self.optimizers = self._build_optimizers()

        logger.info("初始化损失函数...")
        self.losses = self._build_losses()

        logger.info("初始化数据加载器...")
        self.data = self._build_data()

        # 初始化训练流程
        logger.info("初始化训练流程...")
        self.training_pipeline = TrainingPipeline(
            config=self.config,
            models=self.models,
            optimizers=self.optimizers,
            losses=self.losses,
            data=self.data
        )

        # 初始化评估器
        self.evaluator = Evaluator(config=self.config, models=self.models)

        # 初始化导出器
        self.exporter = ModelExporter(config=self.config)

        # 初始化部署器
        self.deployer = ModelDeployer(config=self.config)

    def _build_models(self) -> Dict[str, Any]:
        """
        根据配置文件构建所有模型
        返回：{模型名: 模型实例}
        """
        models_config = self.config.get("models", {})
        builder = ModelBuilder()
        return builder.build_all(models_config)

    def _build_optimizers(self) -> Dict[str, Any]:
        """
        根据配置文件构建所有优化器
        返回：{优化器名: 优化器实例}
        """
        optimizers_config = self.config.get("optimizers", {})
        builder = OptimizerBuilder()
        return builder.build_all(optimizers_config)

    def _build_losses(self) -> Dict[str, Any]:
        """
        根据配置文件构建所有损失函数
        返回：{损失函数名: 损失函数实例}
        """
        losses_config = self.config.get("losses", {})
        builder = LossBuilder()
        return builder.build_all(losses_config)

    def _build_data(self) -> Dict[str, Any]:
        """
        根据配置文件构建数据加载器
        返回：{数据集名: 数据加载器实例}
        """
        data_config = self.config.get("data_manager", {})
        training_mode = self.training_mode.get("type")

        manager = DataManager(config=self.config, training_mode=training_mode)
        return manager.build_all(data_config)

    def train(self):
        """
        执行训练流程
        - 自动选择对应的训练方式
        - 执行训练循环
        - 保存中间检查点
        """
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)

        try:
            self.training_pipeline.execute()
            logger.info("训练完成！")
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}", exc_info=True)
            raise

    def evaluate(self):
        """
        执行评估流程
        - 在验证/测试集上评估模型
        - 输出评估指标
        """
        logger.info("=" * 60)
        logger.info("开始评估")
        logger.info("=" * 60)

        try:
            self.evaluator.evaluate()
            logger.info("评估完成！")
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}", exc_info=True)
            raise

    def export(self):
        """
        导出模型为ONNX格式
        - 转换为跨平台格式
        - 保存到指定路径
        """
        logger.info("=" * 60)
        logger.info("开始导出模型")
        logger.info("=" * 60)

        try:
            self.exporter.export(self.models)
            logger.info("导出完成！")
        except Exception as e:
            logger.error(f"导出过程出错: {str(e)}", exc_info=True)
            raise

    def deploy(self):
        """
        部署模型为服务
        - 创建Docker容器
        - 启动推理服务
        """
        logger.info("=" * 60)
        logger.info("开始部署模型")
        logger.info("=" * 60)

        try:
            self.deployer.deploy(self.models)
            logger.info("部署完成！")
        except Exception as e:
            logger.error(f"部署过程出错: {str(e)}", exc_info=True)
            raise

    def run_all(self):
        """
        执行完整的训练流程：训练 → 评估 → 导出 → 部署
        """
        self.train()
        self.evaluate()
        self.export()
        self.deploy()


def main():
    """
    主函数：解析命令行参数，执行对应的操作

    使用方式：
        python main.py --config config_advanced_example.yaml --mode train
        python main.py --config config_advanced_example.yaml --mode eval
        python main.py --config config_advanced_example.yaml --mode export
        python main.py --config config_advanced_example.yaml --mode deploy
        python main.py --config config_advanced_example.yaml --mode all
    """
    import argparse

    parser = argparse.ArgumentParser(description="通用深度学习训练框架")
    parser.add_argument(
        "--config",
        type=str,
        default="config_advanced_example.yaml",
        help="配置文件名称（从config目录读取）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "export", "deploy", "all"],
        default="train",
        help="执行模式"
    )

    args = parser.parse_args()

    # 创建训练器
    trainer = UniversalTrainer(config_file=args.config)

    # 执行对应的操作
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        trainer.evaluate()
    elif args.mode == "export":
        trainer.export()
    elif args.mode == "deploy":
        trainer.deploy()
    elif args.mode == "all":
        trainer.run_all()

    logger.info("执行完成！")


if __name__ == "__main__":
    main()
