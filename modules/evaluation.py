# -*- coding: utf-8 -*-
"""
modules/evaluation.py
评估管理器：从配置上下文执行模型评估

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
from common.interfaces import EvaluationInterface
from common.train_context import TrainContext
from common.common import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)


class EvaluationManager(EvaluationInterface):
    """
    评估管理器

    职责：
    1. 读取 TrainContext 中的评估配置
    2. 在验证集/测试集上评估模型
    3. 计算评估指标
    4. 生成评估报告
    5. 跟踪最佳模型指标

    标准生命周期：
    initialize() -> validate_evaluation() -> setup_metrics() ->
    evaluate() -> generate_report() -> finalize()

    注意：所有评估步骤通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化评估管理器

        参数:
            context: 训练上下文
        """
        super().__init__(context)
        self.evaluation_configs: Dict[str, Any] = {}
        self.metrics: Dict[str, Callable] = {}
        self.eval_history: Dict[str, List[Dict[str, Any]]] = {}
        self.best_metrics: Dict[str, Dict[str, float]] = {}
        self.evaluation_results: Dict[str, Any] = {}

    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有评估配置
        2. 加载评估配置数据
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        logger.info("初始化评估管理器")

        # 提取评估配置
        self.evaluation_configs = self._extract_evaluation_configs()

        if not self.evaluation_configs:
            logger.warning("未找到评估配置，评估功能将不可用")
            self.evaluation_configs = {}

        logger.info(f"找到 {len(self.evaluation_configs)} 个评估配置")

        # 初始化状态
        self.eval_history = {}
        self.best_metrics = {}
        self.evaluation_results = {}

        self._initialized = True

    def validate_evaluation(self) -> bool:
        """
        验证评估配置

        返回:
            配置是否有效
        """
        logger.info("验证评估配置")

        if not self.evaluation_configs:
            logger.warning("评估配置为空，跳过验证")
            return True

        # 验证每个评估配置
        for eval_name, eval_config in self.evaluation_configs.items():
            if not isinstance(eval_config, dict):
                logger.error(f"评估配置 '{eval_name}' 必须是字典")
                return False

            # 检查必需字段
            if 'steps' not in eval_config and 'reflection' not in eval_config:
                logger.error(
                    f"评估配置 '{eval_name}' 缺少 'steps' 或 'reflection' 字段"
                )
                return False

        logger.info("评估配置验证通过")
        return True

    def setup_metrics(self) -> Dict[str, Any]:
        """
        设置评估指标

        返回:
            指标字典 {metric_name: metric_fn}
        """
        logger.info("设置评估指标")

        metrics = {}

        # 从评估配置中提取指标
        for eval_name, eval_config in self.evaluation_configs.items():
            metric_names = eval_config.get('metrics', [])

            for metric_name in metric_names:
                if metric_name not in metrics:
                    # 尝试构建指标函数
                    metric_fn = self._build_metric_function(metric_name)
                    if metric_fn:
                        metrics[metric_name] = metric_fn

        self.metrics = metrics
        logger.info(f"设置了 {len(metrics)} 个评估指标")

        return metrics

    def evaluate(
        self,
        model: Any,
        dataset: Any,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        执行评估

        参数:
            model: 待评估模型
            dataset: 评估数据集
            metrics: 评估指标

        返回:
            评估结果 {metric_name: value}
        """
        logger.info("开始执行评估")

        results = {}

        # 收集所有预测和标签
        all_predictions = []
        all_labels = []

        try:
            # 遍历数据集
            for batch in dataset:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch
                    labels = None

                # 模型推理
                predictions = model(inputs, training=False)

                all_predictions.append(predictions)
                if labels is not None:
                    all_labels.append(labels)

            # 合并所有批次
            if all_predictions:
                import tensorflow as tf
                all_predictions = tf.concat(all_predictions, axis=0)
                if all_labels:
                    all_labels = tf.concat(all_labels, axis=0)

                # 计算指标
                for metric_name, metric_fn in metrics.items():
                    try:
                        if all_labels is not None:
                            metric_value = metric_fn(all_labels, all_predictions)
                        else:
                            metric_value = metric_fn(all_predictions)

                        results[metric_name] = float(metric_value)
                        logger.info(f"{metric_name}: {results[metric_name]:.4f}")

                    except Exception as e:
                        logger.error(f"计算指标 '{metric_name}' 失败: {str(e)}")

        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}", exc_info=True)
            raise

        return results

    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        评估所有模型

        返回:
            所有模型的评估结果 {model_name: {metric_name: value}}
        """
        logger.info("开始评估所有模型")

        all_results = {}

        # 获取所有模型
        try:
            from common.train_context import ObjectType
            models = self.context.container.get_all(ObjectType.MODEL)
        except (KeyError, AttributeError):
            models = self.context.variables.get('models', {})

        if not models:
            logger.warning("没有找到可评估的模型")
            return all_results

        if not models:
            logger.warning("没有找到可评估的模型")
            return all_results

        # 获取评估数据集
        try:
            dataloaders = self.context.container.get_all(ObjectType.DATALOADER)
        except (KeyError, AttributeError):
            if hasattr(self.context.container, 'dataloaders'):
                dataloaders = self.context.container.dataloaders
            else:
                dataloaders = self.context.variables.get('data_loaders', {})

        # 为每个评估配置执行评估
        for eval_name, eval_config in self.evaluation_configs.items():
            logger.info(f"执行评估配置: {eval_name}")

            # 获取评估模型名称
            model_name = eval_config.get('model', list(models.keys())[0])
            if model_name not in models:
                logger.warning(f"模型 '{model_name}' 不存在，跳过")
                continue

            model = models[model_name]

            # 获取评估数据集
            dataset_name = eval_config.get('dataset', 'val')
            dataset = self._get_dataset(dataset_name, dataloaders)

            if dataset is None:
                logger.warning(f"数据集 '{dataset_name}' 不存在，跳过")
                continue

            # 执行评估
            try:
                if 'steps' in eval_config:
                    # 使用步骤序列评估
                    eval_result = self._evaluate_with_steps(eval_config)
                elif 'reflection' in eval_config:
                    # 使用反射调用评估
                    eval_result = self._evaluate_with_reflection(eval_config)
                else:
                    # 使用标准评估
                    metrics_to_use = {
                        name: fn for name, fn in self.metrics.items()
                        if name in eval_config.get('metrics', [])
                    }
                    eval_result = self.evaluate(model, dataset, metrics_to_use)

                all_results[eval_name] = eval_result

                # 记录历史
                if eval_name not in self.eval_history:
                    self.eval_history[eval_name] = []

                self.eval_history[eval_name].append({
                    "epoch": self.context.current_epoch,
                    "metrics": eval_result
                })

                # 更新最佳指标
                self._update_best_metrics(eval_name, eval_result)

            except Exception as e:
                logger.error(f"评估 '{eval_name}' 失败: {str(e)}", exc_info=True)

        self.evaluation_results = all_results
        logger.info(f"完成 {len(all_results)} 个评估配置")

        return all_results

    def generate_report(self) -> Dict[str, Any]:
        """
        生成评估报告（可选）

        返回:
            评估报告字典
        """
        logger.info("生成评估报告")

        report = {
            "summary": {},
            "best_metrics": self.best_metrics.copy(),
            "history": self.eval_history.copy(),
            "latest_results": self.evaluation_results.copy()
        }

        # 生成摘要
        for eval_name, results in self.evaluation_results.items():
            report["summary"][eval_name] = {
                "num_metrics": len(results),
                "metrics": results
            }

        return report

    def execute(self) -> Dict[str, Any]:
        """
        执行核心逻辑：评估所有模型

        返回:
            评估结果
        """
        logger.info("开始执行评估流程")

        if not self.validate_evaluation():
            raise ValueError("评估配置验证失败")

        # 设置指标
        if not self._setup_complete:
            self.setup_metrics()

        # 评估所有模型
        results = self.evaluate_all_models()

        # 生成报告
        report = self.generate_report()

        return {
            "results": results,
            "report": report
        }

    def finalize(self) -> None:
        """
        清理和保存结果

        职责：
        1. 将评估结果写回 TrainContext
        2. 保存最佳模型信息
        3. 记录日志
        """
        logger.info("完成评估，保存结果")

        # 保存评估结果到上下文
        if self.evaluation_results:
            self.context.execution_results['evaluation'] = self.evaluation_results

        # 保存最佳指标
        if self.best_metrics:
            self.context.execution_results['best_metrics'] = self.best_metrics

        # 保存评估历史
        if self.eval_history:
            self.context.execution_results['eval_history'] = self.eval_history

        logger.info(f"评估结果已保存，包含 {len(self.evaluation_results)} 个配置")

    # ========================================================================
    # 私有方法：辅助功能
    # ========================================================================

    def _extract_evaluation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        从上下文中提取评估配置

        返回:
            评估配置字典
        """
        configs = {}

        # 1. 从 training_pipeline 的参数中查找
        training_mode = self.context.training_mode
        if training_mode in self.context.training_pipelines:
            pipeline = self.context.training_pipelines[training_mode]
            if hasattr(pipeline, 'parameters') and 'evaluation' in pipeline.parameters:
                configs = pipeline.parameters['evaluation']
                logger.debug(f"从 training_pipeline 中找到评估配置")

        # 2. 从全局配置中查找
        if hasattr(self.context, 'global_config'):
            global_config = self.context.global_config
            if isinstance(global_config, dict) and 'evaluation' in global_config:
                configs.update(global_config['evaluation'])

        # 3. 从自定义字段查找
        if hasattr(self.context, 'evaluation'):
            configs.update(self.context.evaluation)

        return configs

    def _build_metric_function(self, metric_name: str) -> Optional[Callable]:
        """
        构建指标函数

        参数:
            metric_name: 指标名称

        返回:
            指标函数
        """
        try:
            # 尝试从 TensorFlow metrics 构建
            import tensorflow as tf

            metric_mapping = {
                'accuracy': tf.keras.metrics.Accuracy(),
                'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(),
                'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
                'binary_accuracy': tf.keras.metrics.BinaryAccuracy(),
                'precision': tf.keras.metrics.Precision(),
                'recall': tf.keras.metrics.Recall(),
                'auc': tf.keras.metrics.AUC(),
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
            }

            if metric_name.lower() in metric_mapping:
                metric_obj = metric_mapping[metric_name.lower()]

                def metric_fn(y_true, y_pred):
                    metric_obj.reset_states()
                    metric_obj.update_state(y_true, y_pred)
                    return metric_obj.result()

                return metric_fn

            # 尝试通过反射构建
            try:
                metric_fn = call_target(
                    reflection=f"tensorflow.keras.metrics:{metric_name}",
                    args={}
                )
                return metric_fn
            except:
                pass

        except Exception as e:
            logger.warning(f"无法构建指标 '{metric_name}': {str(e)}")

        return None

    def _get_dataset(
        self,
        dataset_name: str,
        dataloaders: Dict[str, Any]
    ) -> Optional[Any]:
        """
        获取评估数据集

        参数:
            dataset_name: 数据集名称
            dataloaders: 数据加载器字典

        返回:
            数据集
        """
        # 尝试多种可能的数据集位置
        if dataset_name in dataloaders:
            return dataloaders[dataset_name]

        # 尝试从嵌套字典中查找
        for source_name, source_splits in dataloaders.items():
            if isinstance(source_splits, dict) and dataset_name in source_splits:
                return source_splits[dataset_name]

        return None

    def _evaluate_with_steps(self, eval_config: Dict[str, Any]) -> Dict[str, float]:
        """
        使用步骤序列执行评估

        参数:
            eval_config: 评估配置

        返回:
            评估结果
        """
        logger.info("使用步骤序列执行评估")

        steps = eval_config.get('steps', [])
        results = {}

        # 执行每个步骤
        for step in steps:
            step_name = step.get('name', 'unnamed_step')
            reflection = step.get('reflection')
            args = step.get('args', {})

            if not reflection:
                logger.warning(f"步骤 '{step_name}' 缺少 reflection，跳过")
                continue

            try:
                # 解析参数中的变量引用
                resolved_args = self._resolve_arguments(args)

                # 调用步骤
                step_result = call_target(
                    reflection=reflection,
                    args=resolved_args
                )

                # 保存结果
                if isinstance(step_result, dict):
                    results.update(step_result)
                else:
                    results[step_name] = step_result

                # 保存到上下文
                output_key = step.get('output_key', step_name)
                self.context.execution_results[output_key] = step_result

            except Exception as e:
                logger.error(f"执行评估步骤 '{step_name}' 失败: {str(e)}")

        # 提取指标
        metrics = eval_config.get('metrics', [])
        metric_results = {}

        for metric_name in metrics:
            if metric_name in results:
                metric_results[metric_name] = float(results[metric_name])

        return metric_results if metric_results else results

    def _evaluate_with_reflection(
        self,
        eval_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        使用反射调用执行评估

        参数:
            eval_config: 评估配置

        返回:
            评估结果
        """
        logger.info("使用反射调用执行评估")

        reflection = eval_config.get('reflection')
        args = eval_config.get('args', {})

        # 解析参数
        resolved_args = self._resolve_arguments(args)

        # 调用评估函数
        try:
            result = call_target(
                reflection=reflection,
                args=resolved_args
            )

            # 如果返回字典，直接使用
            if isinstance(result, dict):
                return result

            # 否则，尝试从结果中提取指标
            metrics = eval_config.get('metrics', [])
            if metrics and hasattr(result, '__iter__'):
                return {
                    metric_name: float(value)
                    for metric_name, value in zip(metrics, result)
                }

            return {"result": float(result)}

        except Exception as e:
            logger.error(f"反射调用评估失败: {str(e)}", exc_info=True)
            return {}

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
            elif parts[0] == 'dataloaders':
                obj = self.context.container.dataloaders
                parts = parts[1:]
            elif parts[0] == 'optimizers':
                obj = self.context.container.optimizers
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

        if hasattr(self.context.container, 'dataloaders') and var_name in self.context.container.dataloaders:
            return self.context.container.dataloaders[var_name]

        if var_name in self.context.execution_results:
            return self.context.execution_results[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${var_name}")

    def _update_best_metrics(
        self,
        eval_name: str,
        current_metrics: Dict[str, float]
    ):
        """
        更新最佳指标

        参数:
            eval_name: 评估名称
            current_metrics: 当前指标
        """
        if eval_name not in self.best_metrics:
            self.best_metrics[eval_name] = current_metrics.copy()
            logger.info(f"初始化 '{eval_name}' 的最佳指标")
            return

        best = self.best_metrics[eval_name]

        for metric_name, metric_value in current_metrics.items():
            # 判断指标是越大越好还是越小越好
            is_higher_better = self._is_higher_better(metric_name)

            if metric_name not in best:
                best[metric_name] = metric_value
                logger.info(f"新指标 '{metric_name}': {metric_value:.4f}")
                continue

            if is_higher_better:
                if metric_value > best[metric_name]:
                    logger.info(
                        f"'{metric_name}' 提升: "
                        f"{best[metric_name]:.4f} -> {metric_value:.4f}"
                    )
                    best[metric_name] = metric_value
            else:
                if metric_value < best[metric_name]:
                    logger.info(
                        f"'{metric_name}' 降低: "
                        f"{best[metric_name]:.4f} -> {metric_value:.4f}"
                    )
                    best[metric_name] = metric_value

    def _is_higher_better(self, metric_name: str) -> bool:
        """
        判断指标是否越大越好

        参数:
            metric_name: 指标名称

        返回:
            是否越大越好
        """
        # 常见的"越小越好"指标
        lower_better = ['loss', 'error', 'mse', 'mae', 'rmse', 'mape']

        metric_lower = metric_name.lower()

        for keyword in lower_better:
            if keyword in metric_lower:
                return False

        # 默认越大越好（如 accuracy, precision, recall, f1）
        return True

    def get_eval_history(self, eval_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取评估历史

        参数:
            eval_name: 评估名称（None 表示所有）

        返回:
            评估历史列表
        """
        if eval_name is None:
            return self.eval_history

        return self.eval_history.get(eval_name, [])

    def get_best_metrics(self, eval_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取最佳指标

        参数:
            eval_name: 评估名称（None 表示所有）

        返回:
            最佳指标字典
        """
        if eval_name is None:
            return self.best_metrics.copy()

        return self.best_metrics.get(eval_name, {})


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_model_from_context(
    context: TrainContext,
    eval_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    从训练上下文评估模型的便捷函数

    参数:
        context: 训练上下文
        eval_name: 评估配置名称（可选）

    返回:
        评估结果

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> # ... 配置 context
        >>> results = evaluate_model_from_context(context)
    """
    manager = EvaluationManager(context)
    return manager.run()


def create_evaluation_manager(context: TrainContext) -> EvaluationManager:
    """
    创建并初始化评估管理器的便捷函数

    参数:
        context: 训练上下文

    返回:
        EvaluationManager 实例

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> manager = create_evaluation_manager(context)
        >>> results = manager.evaluate_all_models()
    """
    manager = EvaluationManager(context)
    manager.initialize()
    manager.setup_metrics()
    return manager


def get_best_model_metrics(context: TrainContext, eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    获取最佳模型指标的便捷函数

    参数:
        context: 训练上下文
        eval_name: 评估配置名称（可选）

    返回:
        最佳指标字典

    示例:
        >>> best_metrics = get_best_model_metrics(context, "supervised_eval")
        >>> print(f"Best accuracy: {best_metrics['accuracy']}")
    """
    if 'best_metrics' in context.execution_results:
        best_metrics = context.execution_results['best_metrics']
        if eval_name:
            return best_metrics.get(eval_name, {})
        return best_metrics

    # 如果没有保存的最佳指标，创建管理器并获取
    manager = EvaluationManager(context)
    manager.initialize()
    return manager.get_best_metrics(eval_name)
