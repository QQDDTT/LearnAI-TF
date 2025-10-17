# -*- coding: utf-8 -*-
"""
modules/evaluation.py
评估模块：从配置上下文执行模型评估

导入限制：
- 仅导入 common 文件夹内的函数
- lib 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from common.train_context import (
    TrainContext,
    EvaluationConfig,
    EvaluationFrequency
)
from common.common import call_target


class Evaluator:
    """
    评估器

    职责：
    1. 读取 TrainContext 中的评估配置
    2. 执行模型评估
    3. 计算评估指标
    4. 支持多种评估频率（每epoch、每step、自定义）
    5. 管理评估结果和历史

    注意：所有评估步骤通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化评估器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.eval_configs: Dict[str, EvaluationConfig] = {}
        self.eval_history: Dict[str, List[Dict[str, Any]]] = {}
        self.best_metrics: Dict[str, Dict[str, float]] = {}

    def setup_evaluations(self):
        """设置所有评估配置"""
        training_mode = self.context.training_mode

        # 获取当前训练模式的评估配置
        if training_mode in self.context.evaluations:
            eval_config = self.context.evaluations[training_mode]
            self.eval_configs[training_mode] = eval_config
            self.eval_history[training_mode] = []
            self.best_metrics[training_mode] = {}
        else:
            # 查找通用评估配置
            if "default" in self.context.evaluations:
                eval_config = self.context.evaluations["default"]
                self.eval_configs["default"] = eval_config
                self.eval_history["default"] = []
                self.best_metrics["default"] = {}

    def should_evaluate(
        self,
        frequency: str,
        current_epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        eval_interval: int = 1
    ) -> bool:
        """
        判断是否应该执行评估

        参数:
            frequency: 评估频率
            current_epoch: 当前 epoch
            current_step: 当前 step
            eval_interval: 评估间隔

        返回:
            是否应该评估
        """
        if frequency == EvaluationFrequency.EPOCH.value:
            if current_epoch is None:
                return False
            return current_epoch % eval_interval == 0

        elif frequency == EvaluationFrequency.STEP.value:
            if current_step is None:
                return False
            return current_step % eval_interval == 0

        elif frequency == EvaluationFrequency.CUSTOM.value:
            # 自定义频率由外部控制
            return True

        else:
            return False

    def evaluate(
        self,
        eval_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行评估

        参数:
            eval_name: 评估配置名称（可选）
            **kwargs: 额外参数

        返回:
            评估结果
        """
        # 确定使用哪个评估配置
        if eval_name is None:
            eval_name = self.context.training_mode
            if eval_name not in self.eval_configs:
                eval_name = "default"

        if eval_name not in self.eval_configs:
            raise ValueError(f"评估配置 '{eval_name}' 不存在")

        eval_config = self.eval_configs[eval_name]

        # 执行评估步骤
        results = self._execute_evaluation_steps(eval_config, **kwargs)

        # 计算指标
        metrics = self._compute_metrics(eval_config, results)

        # 保存结果
        eval_result = {
            "epoch": self.context.current_epoch,
            "step": self.context.current_step,
            "metrics": metrics,
            "raw_results": results
        }

        self.eval_history[eval_name].append(eval_result)

        # 更新最佳指标
        self._update_best_metrics(eval_name, metrics)

        return eval_result

    def _execute_evaluation_steps(
        self,
        eval_config: EvaluationConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行评估步骤序列

        参数:
            eval_config: 评估配置
            **kwargs: 额外参数

        返回:
            评估步骤结果
        """
        results = {}

        if not eval_config.steps:
            # 如果没有定义步骤，执行默认评估
            return self._default_evaluation(eval_config, **kwargs)

        # 执行每个评估步骤
        for step in eval_config.steps:
            step_name = step.name

            # 准备参数
            args = self._prepare_eval_args(step.args, **kwargs)

            # 执行步骤
            try:
                result = call_target(
                    reflection=step.reflection,
                    args=args
                )
                results[step_name] = result
            except Exception as e:
                raise RuntimeError(f"执行评估步骤 '{step_name}' 失败: {e}")

        return results

    def _default_evaluation(
        self,
        eval_config: EvaluationConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        默认评估流程

        参数:
            eval_config: 评估配置
            **kwargs: 额外参数

        返回:
            评估结果
        """
        results = {}

        # 获取评估数据
        split = eval_config.split or "val"

        # 这里需要从数据管理器获取数据
        # 简化处理
        results["split"] = split
        results["metrics"] = {}

        return results

    def _prepare_eval_args(
        self,
        args_config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备评估参数（解析变量引用）

        参数:
            args_config: 参数配置
            **kwargs: 额外参数

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
                prepared_args[key] = self._prepare_eval_args(value, **kwargs)

            elif isinstance(value, list):
                prepared_args[key] = [
                    self._prepare_eval_args(item, **kwargs) if isinstance(item, dict) else item
                    for item in value
                ]

            else:
                prepared_args[key] = value

        # 添加额外参数
        prepared_args.update(kwargs)

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

            # 从不同位置查找基础变量
            if base_name in self.context.instantiated_models:
                value = self.context.instantiated_models[base_name]
            elif base_name in self.context.instantiated_dataloaders:
                value = self.context.instantiated_dataloaders[base_name]
            elif base_name in self.context.execution_results:
                value = self.context.execution_results[base_name]
            else:
                raise ValueError(f"无法解析变量: ${{{var_name}}}")

            # 逐层访问字段
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

        if var_name in self.context.instantiated_optimizers:
            return self.context.instantiated_optimizers[var_name]

        if var_name in self.context.instantiated_losses:
            return self.context.instantiated_losses[var_name]

        if var_name in self.context.instantiated_dataloaders:
            return self.context.instantiated_dataloaders[var_name]

        if var_name in self.context.execution_results:
            return self.context.execution_results[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${{{var_name}}}")

    def _compute_metrics(
        self,
        eval_config: EvaluationConfig,
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算评估指标

        参数:
            eval_config: 评估配置
            results: 评估步骤结果

        返回:
            指标字典
        """
        metrics = {}

        if not eval_config.metrics:
            # 尝试从结果中提取指标
            for step_name, step_result in results.items():
                if isinstance(step_result, dict):
                    for key, value in step_result.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{step_name}_{key}"] = float(value)
            return metrics

        # 根据配置的指标列表计算
        for metric_name in eval_config.metrics:
            metric_value = self._extract_metric(metric_name, results)
            if metric_value is not None:
                metrics[metric_name] = metric_value

        return metrics

    def _extract_metric(
        self,
        metric_name: str,
        results: Dict[str, Any]
    ) -> Optional[float]:
        """
        从结果中提取指标值

        参数:
            metric_name: 指标名称
            results: 结果字典

        返回:
            指标值
        """
        # 支持点号访问：step_name.metric_name
        if '.' in metric_name:
            parts = metric_name.split('.')
            step_name = parts[0]
            field_path = parts[1:]

            if step_name not in results:
                return None

            value = results[step_name]
            for field in field_path:
                if isinstance(value, dict):
                    value = value.get(field)
                elif hasattr(value, field):
                    value = getattr(value, field)
                else:
                    return None

            if isinstance(value, (int, float)):
                return float(value)

            return None

        # 直接查找
        for step_result in results.values():
            if isinstance(step_result, dict) and metric_name in step_result:
                value = step_result[metric_name]
                if isinstance(value, (int, float)):
                    return float(value)

        return None

    def _update_best_metrics(
        self,
        eval_name: str,
        metrics: Dict[str, float]
    ):
        """
        更新最佳指标

        参数:
            eval_name: 评估名称
            metrics: 当前指标
        """
        if eval_name not in self.best_metrics:
            self.best_metrics[eval_name] = {}

        best = self.best_metrics[eval_name]

        for metric_name, metric_value in metrics.items():
            # 判断是否是"越大越好"的指标
            is_higher_better = self._is_higher_better(metric_name)

            if metric_name not in best:
                best[metric_name] = metric_value
            else:
                if is_higher_better:
                    if metric_value > best[metric_name]:
                        best[metric_name] = metric_value
                else:
                    if metric_value < best[metric_name]:
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
        lower_better = ['loss', 'error', 'mse', 'mae', 'rmse']

        metric_lower = metric_name.lower()

        for keyword in lower_better:
            if keyword in metric_lower:
                return False

        # 默认越大越好（如 accuracy, precision, recall, f1）
        return True

    def get_eval_history(
        self,
        eval_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取评估历史

        参数:
            eval_name: 评估名称（None 表示当前模式）

        返回:
            评估历史列表
        """
        if eval_name is None:
            eval_name = self.context.training_mode
            if eval_name not in self.eval_history:
                eval_name = "default"

        if eval_name not in self.eval_history:
            return []

        return self.eval_history[eval_name]

    def get_best_metrics(
        self,
        eval_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        获取最佳指标

        参数:
            eval_name: 评估名称（None 表示当前模式）

        返回:
            最佳指标字典
        """
        if eval_name is None:
            eval_name = self.context.training_mode
            if eval_name not in self.best_metrics:
                eval_name = "default"

        if eval_name not in self.best_metrics:
            return {}

        return self.best_metrics[eval_name].copy()

    def get_latest_metrics(
        self,
        eval_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        获取最新的评估指标

        参数:
            eval_name: 评估名称（None 表示当前模式）

        返回:
            最新指标字典
        """
        history = self.get_eval_history(eval_name)

        if not history:
            return {}

        return history[-1]["metrics"]

    def compare_with_best(
        self,
        current_metrics: Dict[str, float],
        eval_name: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        将当前指标与最佳指标比较

        参数:
            current_metrics: 当前指标
            eval_name: 评估名称

        返回:
            比较结果 {metric_name: is_better}
        """
        best_metrics = self.get_best_metrics(eval_name)

        comparison = {}

        for metric_name, current_value in current_metrics.items():
            if metric_name not in best_metrics:
                comparison[metric_name] = True
                continue

            best_value = best_metrics[metric_name]
            is_higher_better = self._is_higher_better(metric_name)

            if is_higher_better:
                comparison[metric_name] = current_value > best_value
            else:
                comparison[metric_name] = current_value < best_value

        return comparison

    def is_best_epoch(
        self,
        metric_name: str,
        eval_name: Optional[str] = None
    ) -> bool:
        """
        判断当前是否是最佳 epoch

        参数:
            metric_name: 关注的指标名称
            eval_name: 评估名称

        返回:
            是否是最佳 epoch
        """
        latest_metrics = self.get_latest_metrics(eval_name)

        if metric_name not in latest_metrics:
            return False

        best_metrics = self.get_best_metrics(eval_name)

        if metric_name not in best_metrics:
            return True

        return latest_metrics[metric_name] == best_metrics[metric_name]

    def format_metrics(
        self,
        metrics: Dict[str, float],
        precision: int = 4
    ) -> str:
        """
        格式化指标输出

        参数:
            metrics: 指标字典
            precision: 小数精度

        返回:
            格式化字符串
        """
        if not metrics:
            return "No metrics"

        parts = []
        for name, value in metrics.items():
            parts.append(f"{name}: {value:.{precision}f}")

        return ", ".join(parts)

    def get_metrics_summary(
        self,
        eval_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取指标摘要

        参数:
            eval_name: 评估名称

        返回:
            摘要字典
        """
        history = self.get_eval_history(eval_name)
        best_metrics = self.get_best_metrics(eval_name)
        latest_metrics = self.get_latest_metrics(eval_name)

        summary = {
            "total_evaluations": len(history),
            "best_metrics": best_metrics,
            "latest_metrics": latest_metrics,
            "improvement_over_best": {}
        }

        # 计算相对于最佳的改进/退步
        for metric_name, latest_value in latest_metrics.items():
            if metric_name in best_metrics:
                best_value = best_metrics[metric_name]
                if best_value != 0:
                    improvement = (latest_value - best_value) / abs(best_value) * 100
                    summary["improvement_over_best"][metric_name] = improvement

        return summary


class EpisodeEvaluator(Evaluator):
    """
    Episode 评估器（用于强化学习）

    扩展 Evaluator，添加 episode 相关的评估功能
    """

    def evaluate_episodes(
        self,
        num_episodes: int,
        deterministic: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        评估多个 episodes

        参数:
            num_episodes: 评估的 episode 数量
            deterministic: 是否使用确定性策略
            **kwargs: 额外参数

        返回:
            评估结果
        """
        eval_name = kwargs.get('eval_name', self.context.training_mode)

        if eval_name not in self.eval_configs:
            raise ValueError(f"评估配置 '{eval_name}' 不存在")

        eval_config = self.eval_configs[eval_name]

        # 获取 episode 评估参数
        eval_episodes = eval_config.eval_episodes or num_episodes

        episodes_results = []

        # 执行多个 episodes
        for episode_idx in range(eval_episodes):
            episode_result = self._evaluate_single_episode(
                eval_config,
                deterministic=deterministic,
                **kwargs
            )
            episodes_results.append(episode_result)

        # 聚合结果
        aggregated_metrics = self._aggregate_episode_metrics(episodes_results)

        # 保存结果
        eval_result = {
            "epoch": self.context.current_epoch,
            "num_episodes": eval_episodes,
            "metrics": aggregated_metrics,
            "episodes": episodes_results
        }

        self.eval_history[eval_name].append(eval_result)
        self._update_best_metrics(eval_name, aggregated_metrics)

        return eval_result

    def _evaluate_single_episode(
        self,
        eval_config: EvaluationConfig,
        deterministic: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        评估单个 episode

        参数:
            eval_config: 评估配置
            deterministic: 是否确定性
            **kwargs: 额外参数

        返回:
            Episode 结果
        """
        # 执行评估步骤
        results = self._execute_evaluation_steps(
            eval_config,
            deterministic=deterministic,
            **kwargs
        )

        return results

    def _aggregate_episode_metrics(
        self,
        episodes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        聚合多个 episodes 的指标

        参数:
            episodes: Episodes 结果列表

        返回:
            聚合后的指标
        """
        if not episodes:
            return {}

        # 收集所有指标
        all_metrics = {}
        for episode in episodes:
            if 'total_reward' in episode:
                if 'total_reward' not in all_metrics:
                    all_metrics['total_reward'] = []
                all_metrics['total_reward'].append(episode['total_reward'])

            if 'episode_length' in episode:
                if 'episode_length' not in all_metrics:
                    all_metrics['episode_length'] = []
                all_metrics['episode_length'].append(episode['episode_length'])

            if 'success' in episode:
                if 'success_rate' not in all_metrics:
                    all_metrics['success_rate'] = []
                all_metrics['success_rate'].append(float(episode['success']))

        # 计算聚合统计
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if metric_name == 'success_rate':
                aggregated['success_rate'] = sum(values) / len(values)
            else:
                aggregated[f'mean_{metric_name}'] = sum(values) / len(values)
                aggregated[f'std_{metric_name}'] = (
                    sum((x - aggregated[f'mean_{metric_name}'])**2 for x in values) / len(values)
                ) ** 0.5
                aggregated[f'min_{metric_name}'] = min(values)
                aggregated[f'max_{metric_name}'] = max(values)

        return aggregated


def create_evaluator(context: TrainContext) -> Evaluator:
    """
    创建评估器的便捷函数

    参数:
        context: 训练上下文

    返回:
        Evaluator 实例
    """
    # 根据训练模式选择评估器
    if context.training_mode == "reinforcement":
        evaluator = EpisodeEvaluator(context)
    else:
        evaluator = Evaluator(context)

    evaluator.setup_evaluations()
    return evaluator


def evaluate_model(
    context: TrainContext,
    eval_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    评估模型的便捷函数

    参数:
        context: 训练上下文
        eval_name: 评估配置名称
        **kwargs: 额外参数

    返回:
        评估结果
    """
    evaluator = create_evaluator(context)
    return evaluator.evaluate(eval_name, **kwargs)
