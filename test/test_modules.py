# -*- coding: utf-8 -*-
"""
test_modules.py - 单元测试文件
功能：
  - 测试各个模块的函数是否正常工作
  - 验证配置文件的正确性
  - 测试反射调用机制
"""

import sys
import os
from modules.utils import LoggerManager, load_yaml, call_target

logger = LoggerManager.get_logger(__file__)


class TestModules:
    """
    测试各个模块的类
    """

    def __init__(self, config_file: str = "config_test.yaml"):
        """
        初始化测试

        参数：
            config_file: 配置文件名称
        """
        logger.info("=" * 70)
        logger.info("开始模块测试")
        logger.info("=" * 70)

        try:
            self.config = load_yaml(config_file)
            logger.info(f"配置文件加载成功: {config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}", exc_info=True)
            raise

    def test_config_structure(self):
        """
        测试配置文件结构

        返回：
            测试是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试1: 配置文件结构完整性")
        logger.info("=" * 70)

        try:
            # 检查必要的配置部分
            required_sections = [
                "global",
                "training_mode",
                "models",
                "optimizers",
                "losses",
                "data_manager",
                "training_pipeline",
                "evaluation",
                "export",
                "deployment"
            ]

            missing_sections = []
            for section in required_sections:
                if section not in self.config:
                    missing_sections.append(section)
                    logger.warning(f"  ✗ 缺少 {section}")
                else:
                    logger.info(f"  ✓ {section}")

            if missing_sections:
                logger.warning(f"缺少以下配置部分: {missing_sections}")
                return False

            logger.info("✓ 配置文件结构完整")
            return True
        except Exception as e:
            logger.error(f"配置文件检查失败: {str(e)}", exc_info=True)
            return False

    def test_models(self):
        """
        测试模型构建

        返回：
            模型字典，如果失败返回None
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试2: 模型构建")
        logger.info("=" * 70)

        try:
            from modules.models import ModelBuilder

            models_config = self.config.get("models", {})

            if not models_config:
                logger.warning("  模型配置为空")
                return None

            logger.info(f"  开始构建{len(models_config)}个模型...")
            builder = ModelBuilder()
            models = builder.build_all(models_config)

            logger.info(f"  ✓ 模型构建成功，共{len(models)}个模型:")
            for model_name, model in models.items():
                model_type = type(model).__name__
                logger.info(f"    - {model_name}: {model_type}")

            return models
        except Exception as e:
            logger.error(f"模型测试失败: {str(e)}", exc_info=True)
            return None

    def test_optimizers(self):
        """
        测试优化器构建

        返回：
            优化器字典，如果失败返回None
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试3: 优化器构建")
        logger.info("=" * 70)

        try:
            from modules.optimizers import OptimizerBuilder

            optimizers_config = self.config.get("optimizers", {})

            if not optimizers_config:
                logger.warning("  优化器配置为空")
                return None

            logger.info(f"  开始构建{len(optimizers_config)}个优化器...")
            builder = OptimizerBuilder()
            optimizers = builder.build_all(optimizers_config)

            logger.info(f"  ✓ 优化器构建成功，共{len(optimizers)}个优化器:")
            for opt_name, opt in optimizers.items():
                opt_type = type(opt).__name__
                logger.info(f"    - {opt_name}: {opt_type}")

            return optimizers
        except Exception as e:
            logger.error(f"优化器测试失败: {str(e)}", exc_info=True)
            return None

    def test_losses(self):
        """
        测试损失函数构建

        返回：
            损失函数字典，如果失败返回None
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试4: 损失函数构建")
        logger.info("=" * 70)

        try:
            from modules.losses import LossBuilder

            losses_config = self.config.get("losses", {})

            if not losses_config:
                logger.warning("  损失函数配置为空")
                return None

            logger.info(f"  开始构建{len(losses_config)}个损失函数...")
            builder = LossBuilder()
            losses = builder.build_all(losses_config)

            logger.info(f"  ✓ 损失函数构建成功，共{len(losses)}个损失函数:")
            for loss_name, loss in losses.items():
                loss_type = type(loss).__name__
                logger.info(f"    - {loss_name}: {loss_type}")

            return losses
        except Exception as e:
            logger.error(f"损失函数测试失败: {str(e)}", exc_info=True)
            return None

    def test_data_manager(self):
        """
        测试数据管理器

        返回：
            数据加载器字典，如果失败返回None
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试5: 数据管理器")
        logger.info("=" * 70)

        try:
            from modules.data_manager import DataManager

            training_mode = self.config.get("training_mode", {}).get("type", "supervised")
            data_config = self.config.get("data_manager", {})

            if not data_config:
                logger.warning("  数据管理器配置为空")
                return None

            logger.info(f"  训练模式: {training_mode}")
            logger.info(f"  开始构建数据加载器...")

            manager = DataManager(self.config, training_mode)
            data_loaders = manager.build_all(data_config)

            logger.info(f"  ✓ 数据加载器构建成功，共{len(data_loaders)}个加载器:")
            for loader_name, loader in data_loaders.items():
                loader_type = type(loader).__name__
                logger.info(f"    - {loader_name}: {loader_type}")

            return data_loaders
        except Exception as e:
            logger.error(f"数据管理器测试失败: {str(e)}", exc_info=True)
            return None

    def test_parameter_binding(self):
        """
        测试参数绑定

        返回：
            测试是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("测试6: 参数绑定系统")
        logger.info("=" * 70)

        try:
            from modules.training_pipeline import TrainingPipeline

            logger.info("  创建测试上下文...")

            # 创建虚拟对象进行测试
            context = {
                "epoch": 1,
                "batch_idx": 10,
                "epsilon": 0.5,
                "config": self.config,
                "batch": ([1, 2, 3], [0, 1])
            }

            logger.info("  初始化训练流程对象...")
            # 创建管道对象（用于测试参数绑定）
            pipeline = TrainingPipeline(
                config=self.config,
                models={},
                optimizers={},
                losses={},
                data={}
            )

            logger.info("  测试参数绑定...")
            # 测试参数绑定
            args_cfg = {
                "epoch_val": "${epoch}",
                "batch_val": "${batch_idx}",
                "epsilon_val": "${epsilon}",
                "batch_x": "batch.x",
                "config_val": "config.training_mode.type",
                "normal_val": "test"
            }

            bound_args = pipeline._bind_arguments(args_cfg, context, None)

            logger.info("  ✓ 参数绑定成功:")
            expected_bindings = {
                "epoch_val": 1,
                "batch_val": 10,
                "epsilon_val": 0.5,
                "batch_x": [1, 2, 3],
                "config_val": "supervised",
                "normal_val": "test"
            }

            for key, expected_value in expected_bindings.items():
                actual_value = bound_args.get(key)
                if actual_value == expected_value:
                    logger.info(f"    ✓ {key}: {actual_value}")
                else:
                    logger.warning(f"    ✗ {key}: 期望{expected_value}, 得到{actual_value}")
                    return False

            logger.info("✓ 参数绑定系统正常")
            return True
        except Exception as e:
            logger.error(f"参数绑定测试失败: {str(e)}", exc_info=True)
            return False

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "=" * 70)
        logger.info("运行所有模块测试")
        logger.info("=" * 70)

        results = {}

        results["配置文件结构"] = self.test_config_structure()
        results["模型构建"] = self.test_models() is not None
        results["优化器构建"] = self.test_optimizers() is not None
        results["损失函数构建"] = self.test_losses() is not None
        results["数据管理器"] = self.test_data_manager() is not None
        results["参数绑定系统"] = self.test_parameter_binding()

        logger.info("\n" + "=" * 70)
        logger.info("测试结果汇总")
        logger.info("=" * 70)

        passed = 0
        failed = 0
        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"  {test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1

        logger.info("=" * 70)
        logger.info(f"总计: {passed}个通过, {failed}个失败")
        logger.info("=" * 70)

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="模块测试工具")
    parser.add_argument(
        "--config",
        type=str,
        default="config_test.yaml",
        help="配置文件名称"
    )

    args = parser.parse_args()

    try:
        # 运行测试
        tester = TestModules(config_file=args.config)
        results = tester.run_all_tests()

        # 返回状态码
        failed_count = sum(1 for v in results.values() if not v)
        return failed_count
    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
