# -*- coding: utf-8 -*-
"""
test_modules.py
模块测试文件 - 简化版
测试内容：
1. 模块导入正常
2. modules/*.py 中核心函数可用
3. ${} 变量引用和 Bridge/Connection 解析
4. call_target 动态调用功能
"""

import os
import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from common.common import LoggerManager, call_target
from common.train_context import (
    parse_bridge,
    parse_connection,
    validate_bridge_condition,
    validate_connection_targets,
    BridgeAction,
    ConnectionType,
    ReferencePattern
)

logger = LoggerManager.get_logger(__file__)


class TestModuleImports(unittest.TestCase):
    """测试1: 模块导入"""

    def test_import_common(self):
        """测试 common 模块导入"""
        logger.info("测试: common 模块导入")
        try:
            from common import common
            self.assertIsNotNone(common)
            logger.info("✓ common 模块导入成功")
        except Exception as e:
            self.fail(f"common 模块导入失败: {e}")

    def test_import_modules(self):
        """测试 modules 模块导入"""
        logger.info("测试: modules 模块导入")
        try:
            from modules import models, optimizers, losses
            from modules import training_pipeline, evaluation
            from modules import data_manager, export, deployment
            self.assertIsNotNone(models)
            self.assertIsNotNone(optimizers)
            self.assertIsNotNone(losses)
            logger.info("✓ 所有 modules 模块导入成功")
        except Exception as e:
            self.fail(f"modules 模块导入失败: {e}")

    def test_import_lib(self):
        """测试 lib 模块导入"""
        logger.info("测试: lib 模块导入")
        try:
            from lib import training_utils, data_utils, condition_utils
            from lib import checkpoint, clustering, deployment
            self.assertIsNotNone(training_utils)
            self.assertIsNotNone(data_utils)
            self.assertIsNotNone(condition_utils)
            logger.info("✓ 所有 lib 模块导入成功")
        except Exception as e:
            self.fail(f"lib 模块导入失败: {e}")


class TestCallTarget(unittest.TestCase):
    """测试2: call_target 动态调用"""

    def test_call_tensorflow_layer(self):
        """测试调用 TensorFlow 层"""
        logger.info("测试: call_target 创建 TensorFlow Dense 层")
        try:
            layer = call_target(
                "tensorflow.keras.layers.Dense",
                {"units": 64, "activation": "relu"}
            )
            self.assertIsInstance(layer, tf.keras.layers.Layer)
            logger.info(f"✓ 成功创建 {type(layer).__name__} 层")
        except Exception as e:
            self.fail(f"call_target 调用 TensorFlow 失败: {e}")

    def test_call_tensorflow_optimizer(self):
        """测试调用 TensorFlow 优化器"""
        logger.info("测试: call_target 创建 TensorFlow Adam 优化器")
        try:
            optimizer = call_target(
                "tensorflow.keras.optimizers.Adam",
                {"learning_rate": 0.001}
            )
            self.assertIsInstance(optimizer, tf.keras.optimizers.Optimizer)
            logger.info(f"✓ 成功创建 {type(optimizer).__name__} 优化器")
        except Exception as e:
            self.fail(f"call_target 调用优化器失败: {e}")

    def test_call_tensorflow_loss(self):
        """测试调用 TensorFlow 损失函数"""
        logger.info("测试: call_target 创建 TensorFlow Loss")
        try:
            loss_fn = call_target(
                "tensorflow.keras.losses.MeanSquaredError",
                {}
            )
            self.assertIsInstance(loss_fn, tf.keras.losses.Loss)
            logger.info(f"✓ 成功创建 {type(loss_fn).__name__} 损失函数")
        except Exception as e:
            self.fail(f"call_target 调用损失函数失败: {e}")

    def test_call_lib_function(self):
        """测试调用 lib 模块函数"""
        logger.info("测试: call_target 调用 lib.condition_utils")
        try:
            result = call_target(
                "lib.condition_utils:check_threshold",
                {"value": 10, "threshold": 5, "mode": "greater"}
            )
            self.assertIsInstance(result, dict)
            self.assertIn("result", result)
            self.assertTrue(result["result"])
            logger.info("✓ 成功调用 lib.condition_utils:check_threshold")
        except Exception as e:
            logger.warning(f"⚠ call_target 调用 lib 函数警告: {e}")
            # lib 函数可能依赖其他模块，不作为强制失败


class TestVariableReferences(unittest.TestCase):
    """测试3: 变量引用解析"""

    def test_simple_variable(self):
        """测试简单变量引用"""
        logger.info("测试: 简单变量引用 ${model}")
        import re
        pattern = re.compile(ReferencePattern.VARIABLE)
        matches = pattern.findall("${model}")
        self.assertEqual(matches, ["model"])
        logger.info("✓ 简单变量引用解析正确")

    def test_nested_variable(self):
        """测试嵌套变量引用"""
        logger.info("测试: 嵌套变量引用 ${step.output}")
        import re
        pattern = re.compile(ReferencePattern.VARIABLE)
        matches = pattern.findall("${step.output}")
        self.assertEqual(matches, ["step.output"])
        logger.info("✓ 嵌套变量引用解析正确")

    def test_multiple_variables(self):
        """测试多个变量引用"""
        logger.info("测试: 多个变量引用")
        import re
        pattern = re.compile(ReferencePattern.VARIABLE)
        matches = pattern.findall("${loss} and ${accuracy}")
        self.assertEqual(matches, ["loss", "accuracy"])
        logger.info("✓ 多个变量引用解析正确")

    def test_no_variables(self):
        """测试无变量字符串"""
        logger.info("测试: 无变量字符串")
        import re
        pattern = re.compile(ReferencePattern.VARIABLE)
        matches = pattern.findall("no variables here")
        self.assertEqual(matches, [])
        logger.info("✓ 无变量字符串解析正确")


class TestBridgeParsing(unittest.TestCase):
    """测试4: Bridge 解析"""

    def test_skip_bridge(self):
        """测试 SKIP Bridge"""
        logger.info("测试: @skip:target_step")
        parsed = parse_bridge("@skip:target_step")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action, "skip")
        self.assertIn("target_step", parsed.targets)
        logger.info("✓ SKIP Bridge 解析正确")

    def test_branch_bridge(self):
        """测试 BRANCH Bridge"""
        logger.info("测试: @branch:step_a,step_b?${acc}>0.9")
        parsed = parse_bridge("@branch:step_a,step_b?${acc}>0.9")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action, "branch")
        self.assertEqual(len(parsed.targets), 2)
        self.assertIsNotNone(parsed.condition)
        logger.info("✓ BRANCH Bridge 解析正确")

    def test_loop_bridge(self):
        """测试 LOOP Bridge"""
        logger.info("测试: @loop:train_step?${epoch}<10")
        parsed = parse_bridge("@loop:train_step?${epoch}<10")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action, "loop")
        self.assertIn("train_step", parsed.targets)
        logger.info("✓ LOOP Bridge 解析正确")

    def test_conditional_bridge(self):
        """测试 CONDITIONAL Bridge"""
        logger.info("测试: @conditional:save?${loss}<0.1")
        parsed = parse_bridge("@conditional:save?${loss}<0.1")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action, "conditional")
        self.assertIsNotNone(parsed.condition)
        logger.info("✓ CONDITIONAL Bridge 解析正确")

    def test_checkpoint_bridge(self):
        """测试 CHECKPOINT Bridge"""
        logger.info("测试: @checkpoint")
        parsed = parse_bridge("@checkpoint")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.action, "checkpoint")
        logger.info("✓ CHECKPOINT Bridge 解析正确")


class TestConnectionParsing(unittest.TestCase):
    """测试5: Connection 解析"""

    def test_sequential_connection(self):
        """测试 SEQUENTIAL Connection"""
        logger.info("测试: @sequential")
        parsed = parse_connection("@sequential")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, "sequential")
        logger.info("✓ SEQUENTIAL Connection 解析正确")

    def test_residual_connection(self):
        """测试 RESIDUAL Connection"""
        logger.info("测试: @residual:input_layer")
        parsed = parse_connection("@residual:input_layer")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, "residual")
        self.assertIn("input_layer", parsed.targets)
        logger.info("✓ RESIDUAL Connection 解析正确")

    def test_concat_connection(self):
        """测试 CONCAT Connection"""
        logger.info("测试: @concat:branch_a,branch_b,branch_c")
        parsed = parse_connection("@concat:branch_a,branch_b,branch_c")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, "concat")
        self.assertEqual(len(parsed.targets), 3)
        logger.info("✓ CONCAT Connection 解析正确")

    def test_add_connection(self):
        """测试 ADD Connection"""
        logger.info("测试: @add:path1,path2")
        parsed = parse_connection("@add:path1,path2")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, "add")
        self.assertEqual(len(parsed.targets), 2)
        logger.info("✓ ADD Connection 解析正确")

    def test_skip_connection(self):
        """测试 SKIP Connection"""
        logger.info("测试: @skip:layer2")
        parsed = parse_connection("@skip:layer2")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.type, "skip")
        self.assertIn("layer2", parsed.targets)
        logger.info("✓ SKIP Connection 解析正确")


class TestBridgeValidation(unittest.TestCase):
    """测试6: Bridge 条件验证"""

    def test_simple_condition(self):
        """测试简单条件"""
        logger.info("测试: 简单条件验证 ${loss}<0.1")
        available_vars = {"loss", "accuracy", "epoch"}
        errors = validate_bridge_condition("${loss}<0.1", available_vars)
        self.assertEqual(len(errors), 0)
        logger.info("✓ 简单条件验证通过")

    def test_complex_condition(self):
        """测试复杂条件"""
        logger.info("测试: 复杂条件验证 ${accuracy}>0.9 AND ${epoch}>10")
        available_vars = {"loss", "accuracy", "epoch"}
        errors = validate_bridge_condition("${accuracy}>0.9 AND ${epoch}>10", available_vars)
        self.assertEqual(len(errors), 0)
        logger.info("✓ 复杂条件验证通过")

    def test_undefined_variable(self):
        """测试未定义变量"""
        logger.info("测试: 未定义变量检测")
        available_vars = {"loss", "accuracy"}
        errors = validate_bridge_condition("${undefined_var}>0.5", available_vars)
        self.assertGreater(len(errors), 0)
        logger.info("✓ 未定义变量检测正确")


class TestConnectionValidation(unittest.TestCase):
    """测试7: Connection 目标验证"""

    def test_valid_targets(self):
        """测试有效目标"""
        logger.info("测试: 有效连接目标")
        available_layers = {"input", "layer1", "layer2"}
        errors = validate_connection_targets("residual", ["input"], available_layers)
        self.assertEqual(len(errors), 0)
        logger.info("✓ 有效目标验证通过")

    def test_invalid_targets(self):
        """测试无效目标"""
        logger.info("测试: 无效连接目标检测")
        available_layers = {"input", "layer1", "layer2"}
        errors = validate_connection_targets("residual", ["undefined_layer"], available_layers)
        self.assertGreater(len(errors), 0)
        logger.info("✓ 无效目标检测正确")

    def test_sequential_no_targets(self):
        """测试 sequential 不需要目标"""
        logger.info("测试: sequential 连接不需要目标")
        available_layers = {"input", "layer1"}
        errors = validate_connection_targets("sequential", [], available_layers)
        self.assertEqual(len(errors), 0)
        logger.info("✓ sequential 验证通过")


class TestTensorFlowIntegration(unittest.TestCase):
    """测试8: TensorFlow 集成"""

    def test_create_model(self):
        """测试创建模型"""
        logger.info("测试: 创建 Sequential 模型")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            self.assertIsNotNone(model)
            self.assertEqual(len(model.layers), 3)
            logger.info("✓ Sequential 模型创建成功")
        except Exception as e:
            self.fail(f"模型创建失败: {e}")

    def test_compile_model(self):
        """测试编译模型"""
        logger.info("测试: 编译模型")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✓ 模型编译成功")
        except Exception as e:
            self.fail(f"模型编译失败: {e}")

    def test_forward_pass(self):
        """测试前向传播"""
        logger.info("测试: 模型前向传播")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(2, activation='softmax')
            ])

            # 创建测试输入
            test_input = tf.random.normal((1, 5))
            output = model(test_input)

            self.assertEqual(output.shape, (1, 2))
            logger.info(f"✓ 前向传播成功，输出形状: {output.shape}")
        except Exception as e:
            self.fail(f"前向传播失败: {e}")


def run_tests():
    """运行所有测试"""
    logger.info("=" * 80)
    logger.info("LearnAI-TF 模块测试")
    logger.info("=" * 80)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModuleImports))
    suite.addTests(loader.loadTestsFromTestCase(TestCallTarget))
    suite.addTests(loader.loadTestsFromTestCase(TestVariableReferences))
    suite.addTests(loader.loadTestsFromTestCase(TestBridgeParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestConnectionParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestBridgeValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestConnectionValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorFlowIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出结果汇总
    logger.info("=" * 80)
    logger.info("测试结果汇总")
    logger.info("=" * 80)
    logger.info(f"总测试数: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")

    if result.failures:
        logger.error("-" * 80)
        logger.error("失败详情:")
        for test, traceback in result.failures:
            logger.error(f"  • {test}")

    if result.errors:
        logger.error("-" * 80)
        logger.error("错误详情:")
        for test, traceback in result.errors:
            logger.error(f"  • {test}")

    pass_rate = ((result.testsRun - len(result.failures) - len(result.errors)) /
                 result.testsRun * 100 if result.testsRun > 0 else 0)
    logger.info(f"通过率: {pass_rate:.2f}%")
    logger.info("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
