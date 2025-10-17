# -*- coding: utf-8 -*-
"""
modules/training_pipeline.py
训练流程执行器：从配置上下文执行训练流程

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from common.train_context import (
    TrainContext,
    PipelineConfig,
    StepConfig,
    LoopType,
    BridgeAction,
    parse_bridge
)
from common.common import call_target


class TrainingPipeline:
    """
    训练流程执行器

    职责：
    1. 读取 TrainContext 中的训练流程配置
    2. 执行训练步骤序列
    3. 处理循环控制（epoch/episode/iteration）
    4. 处理 Bridge 控制流（跳转、分支、循环等）
    5. 管理训练状态和结果

    注意：所有执行步骤通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化训练流程执行器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.current_pipeline: Optional[PipelineConfig] = None
        self.step_results: Dict[str, Any] = {}
        self.loop_counters: Dict[str, int] = {}
        self.should_stop: bool = False

    def run(self) -> Dict[str, Any]:
        """
        运行训练流程

        返回:
            训练结果字典
        """
        # 获取当前训练模式的流程
        training_mode = self.context.training_mode

        if training_mode not in self.context.training_pipelines:
            raise ValueError(
                f"训练模式 '{training_mode}' 没有对应的流程配置"
            )

        pipeline_config = self.context.training_pipelines[training_mode]
        self.current_pipeline = pipeline_config

        # 根据循环类型执行不同的训练循环
        loop_type = pipeline_config.loop_type

        if loop_type == LoopType.EPOCH_BATCH.value:
            return self._run_epoch_batch_loop()

        elif loop_type == LoopType.EPISODE_STEP.value:
            return self._run_episode_step_loop()

        elif loop_type == LoopType.ITERATION.value:
            return self._run_iteration_loop()

        elif loop_type == LoopType.CUSTOM.value:
            return self._run_custom_loop()

        else:
            raise ValueError(f"不支持的循环类型: {loop_type}")

    def _run_epoch_batch_loop(self) -> Dict[str, Any]:
        """
        执行 epoch-batch 循环（监督学习）

        返回:
            训练结果
        """
        pipeline = self.current_pipeline
        parameters = pipeline.parameters
        loop_condition = pipeline.loop_condition

        # 获取参数
        max_epochs = loop_condition.max_epochs if loop_condition else \
                     parameters.get('max_epochs', 100)

        results = {
            "epochs": [],
            "final_metrics": {}
        }

        # Epoch 循环
        for epoch in range(max_epochs):
            if self.should_stop:
                break

            self.context.current_epoch = epoch
            self.loop_counters['epoch'] = epoch

            # 执行一个 epoch
            epoch_result = self._execute_epoch()
            results["epochs"].append(epoch_result)

            # 检查终止条件
            if self._check_loop_termination():
                break

        results["final_metrics"] = self.step_results

        return results

    def _execute_epoch(self) -> Dict[str, Any]:
        """
        执行一个 epoch

        返回:
            Epoch 结果
        """
        epoch_result = {
            "epoch": self.context.current_epoch,
            "steps": []
        }

        # 执行步骤序列
        step_sequence = self.current_pipeline.step_sequence

        step_idx = 0
        while step_idx < len(step_sequence):
            step_config = step_sequence[step_idx]

            # 执行步骤
            step_result = self._execute_step(step_config)
            epoch_result["steps"].append(step_result)

            # 处理 Bridge 控制
            next_idx = self._handle_bridge(step_config, step_idx, step_sequence)

            if next_idx is None:
                # 正常前进
                step_idx += 1
            else:
                # Bridge 跳转
                step_idx = next_idx

        return epoch_result

    def _run_episode_step_loop(self) -> Dict[str, Any]:
        """
        执行 episode-step 循环（强化学习）

        返回:
            训练结果
        """
        pipeline = self.current_pipeline
        parameters = pipeline.parameters
        loop_condition = pipeline.loop_condition

        # 获取参数
        max_episodes = loop_condition.max_episodes if loop_condition else \
                       parameters.get('max_episodes', 1000)
        max_steps_per_episode = parameters.get('max_steps_per_episode', 1000)

        results = {
            "episodes": [],
            "final_metrics": {}
        }

        # Episode 循环
        for episode in range(max_episodes):
            if self.should_stop:
                break

            self.context.current_episode = episode
            self.loop_counters['episode'] = episode

            # 执行一个 episode
            episode_result = self._execute_episode(max_steps_per_episode)
            results["episodes"].append(episode_result)

            # 检查终止条件
            if self._check_loop_termination():
                break

        results["final_metrics"] = self.step_results

        return results

    def _execute_episode(self, max_steps: int) -> Dict[str, Any]:
        """
        执行一个 episode

        参数:
            max_steps: 最大步数

        返回:
            Episode 结果
        """
        episode_result = {
            "episode": self.context.current_episode,
            "steps": [],
            "total_reward": 0.0,
            "done": False
        }

        step_sequence = self.current_pipeline.step_sequence

        # Step 循环
        for step in range(max_steps):
            self.context.current_step = step
            self.loop_counters['step'] = step

            # 执行步骤序列
            step_idx = 0
            while step_idx < len(step_sequence):
                step_config = step_sequence[step_idx]

                # 执行步骤
                step_result = self._execute_step(step_config)
                episode_result["steps"].append(step_result)

                # 累积奖励
                if 'reward' in step_result:
                    episode_result["total_reward"] += step_result['reward']

                # 检查是否结束
                if step_result.get('done', False):
                    episode_result["done"] = True
                    return episode_result

                # 处理 Bridge 控制
                next_idx = self._handle_bridge(step_config, step_idx, step_sequence)

                if next_idx is None:
                    step_idx += 1
                else:
                    step_idx = next_idx

        return episode_result

    def _run_iteration_loop(self) -> Dict[str, Any]:
        """
        执行迭代循环（无监督学习、聚类等）

        返回:
            训练结果
        """
        pipeline = self.current_pipeline
        parameters = pipeline.parameters
        loop_condition = pipeline.loop_condition

        # 获取参数
        max_iterations = loop_condition.max_iterations if loop_condition else \
                         parameters.get('max_iterations', 1000)
        convergence_threshold = parameters.get('convergence_threshold', 1e-4)
        convergence_field = loop_condition.convergence_field if loop_condition else \
                           'loss'

        results = {
            "iterations": [],
            "final_metrics": {},
            "converged": False
        }

        prev_value = None

        # 迭代循环
        for iteration in range(max_iterations):
            if self.should_stop:
                break

            self.loop_counters['iteration'] = iteration

            # 执行步骤序列
            iteration_result = self._execute_iteration()
            results["iterations"].append(iteration_result)

            # 检查收敛
            if convergence_field in self.step_results:
                current_value = self.step_results[convergence_field]

                if prev_value is not None:
                    change = abs(current_value - prev_value)
                    if change < convergence_threshold:
                        results["converged"] = True
                        break

                prev_value = current_value

            # 检查终止条件
            if self._check_loop_termination():
                break

        results["final_metrics"] = self.step_results

        return results

    def _execute_iteration(self) -> Dict[str, Any]:
        """
        执行一次迭代

        返回:
            迭代结果
        """
        iteration_result = {
            "iteration": self.loop_counters.get('iteration', 0),
            "steps": []
        }

        step_sequence = self.current_pipeline.step_sequence

        step_idx = 0
        while step_idx < len(step_sequence):
            step_config = step_sequence[step_idx]

            # 执行步骤
            step_result = self._execute_step(step_config)
            iteration_result["steps"].append(step_result)

            # 处理 Bridge 控制
            next_idx = self._handle_bridge(step_config, step_idx, step_sequence)

            if next_idx is None:
                step_idx += 1
            else:
                step_idx = next_idx

        return iteration_result

    def _run_custom_loop(self) -> Dict[str, Any]:
        """
        执行自定义循环

        返回:
            训练结果
        """
        pipeline = self.current_pipeline
        parameters = pipeline.parameters

        # 获取自定义循环函数
        if 'custom_loop_fn' in parameters:
            custom_loop_fn = call_target(
                reflection=parameters['custom_loop_fn'],
                args={}
            )

            # 执行自定义循环
            return custom_loop_fn(self)

        else:
            raise ValueError("自定义循环需要指定 'custom_loop_fn'")

    def _execute_step(self, step_config: StepConfig) -> Dict[str, Any]:
        """
        执行单个步骤

        参数:
            step_config: 步骤配置

        返回:
            步骤执行结果
        """
        step_name = step_config.name

        # 准备参数
        args = self._prepare_step_args(step_config.args)

        # 使用 call_target 执行步骤
        try:
            result = call_target(
                reflection=step_config.reflection,
                args=args
            )
        except Exception as e:
            raise RuntimeError(
                f"执行步骤 '{step_name}' 失败: {e}"
            )

        # 保存结果
        self.step_results[step_name] = result
        self.context.execution_results[step_name] = result

        return {
            "step_name": step_name,
            "result": result
        }

    def _prepare_step_args(self, args_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备步骤参数（解析变量引用）

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

                # 从不同位置查找变量
                resolved_value = self._resolve_variable(var_name)
                prepared_args[key] = resolved_value

            elif isinstance(value, dict):
                # 递归处理嵌套字典
                prepared_args[key] = self._prepare_step_args(value)

            elif isinstance(value, list):
                # 处理列表
                prepared_args[key] = [
                    self._prepare_step_args(item) if isinstance(item, dict) else item
                    for item in value
                ]

            else:
                prepared_args[key] = value

        return prepared_args

    def _resolve_variable(self, var_name: str) -> Any:
        """
        解析变量引用

        参数:
            var_name: 变量名（可能包含点号，如 step.output）

        返回:
            变量值
        """
        # 支持点号访问：step_name.field
        if '.' in var_name:
            parts = var_name.split('.')
            base_name = parts[0]
            field_path = parts[1:]

            # 从步骤结果中查找
            if base_name in self.step_results:
                value = self.step_results[base_name]

                # 逐层访问字段
                for field in field_path:
                    if isinstance(value, dict):
                        value = value.get(field)
                    elif hasattr(value, field):
                        value = getattr(value, field)
                    else:
                        raise ValueError(
                            f"无法访问 '{base_name}.{'.'.join(field_path)}'"
                        )

                return value

        # 直接变量名
        # 1. 从步骤结果查找
        if var_name in self.step_results:
            return self.step_results[var_name]

        # 2. 从实例化对象查找
        if var_name in self.context.instantiated_models:
            return self.context.instantiated_models[var_name]

        if var_name in self.context.instantiated_optimizers:
            return self.context.instantiated_optimizers[var_name]

        if var_name in self.context.instantiated_losses:
            return self.context.instantiated_losses[var_name]

        if var_name in self.context.instantiated_dataloaders:
            return self.context.instantiated_dataloaders[var_name]

        # 3. 从循环计数器查找
        if var_name in self.loop_counters:
            return self.loop_counters[var_name]

        # 4. 从上下文属性查找
        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${{{var_name}}}")

    def _handle_bridge(
        self,
        step_config: StepConfig,
        current_idx: int,
        step_sequence: List[StepConfig]
    ) -> Optional[int]:
        """
        处理 Bridge 控制流

        参数:
            step_config: 当前步骤配置
            current_idx: 当前步骤索引
            step_sequence: 步骤序列

        返回:
            下一个步骤索引（None 表示正常前进）
        """
        if not step_config.bridge:
            return None

        bridge = parse_bridge(step_config.bridge)

        if bridge is None:
            return None

        action = bridge.action

        # 检查条件（如果有）
        if bridge.condition:
            if not self._evaluate_condition(bridge.condition):
                return None  # 条件不满足，正常前进

        # 执行不同的 Bridge 动作
        if action == BridgeAction.SKIP.value:
            return self._handle_skip(bridge, step_sequence)

        elif action == BridgeAction.BRANCH.value:
            return self._handle_branch(bridge, step_sequence)

        elif action == BridgeAction.LOOP.value:
            return self._handle_loop(bridge, current_idx, step_sequence)

        elif action == BridgeAction.CONDITIONAL.value:
            return self._handle_conditional(bridge, current_idx)

        elif action == BridgeAction.ERROR_HANDLER.value:
            return self._handle_error_handler(bridge, step_sequence)

        elif action == BridgeAction.CHECKPOINT.value:
            self._handle_checkpoint(bridge)
            return None

        else:
            # 其他 action 暂不处理
            return None

    def _evaluate_condition(self, condition: str) -> bool:
        """
        评估条件表达式

        参数:
            condition: 条件字符串

        返回:
            条件是否满足
        """
        # 解析条件中的变量引用
        import re

        # 替换变量引用
        def replace_var(match):
            var_name = match.group(1)
            try:
                value = self._resolve_variable(var_name)
                return str(value)
            except:
                return "None"

        condition_resolved = re.sub(r'\$\{([^}]+)\}', replace_var, condition)

        # 安全地评估表达式
        try:
            # 构建安全的命名空间
            safe_dict = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None,
                "AND": lambda a, b: a and b,
                "OR": lambda a, b: a or b,
                "NOT": lambda a: not a
            }

            # 处理逻辑运算符
            condition_resolved = condition_resolved.replace(' AND ', ' and ')
            condition_resolved = condition_resolved.replace(' OR ', ' or ')
            condition_resolved = condition_resolved.replace('NOT ', 'not ')

            result = eval(condition_resolved, safe_dict)
            return bool(result)

        except Exception as e:
            raise ValueError(f"条件评估失败: {condition}, 错误: {e}")

    def _handle_skip(
        self,
        bridge,
        step_sequence: List[StepConfig]
    ) -> Optional[int]:
        """处理 SKIP 动作"""
        if not bridge.targets:
            return None

        target_name = bridge.targets[0]

        # 查找目标步骤
        for idx, step in enumerate(step_sequence):
            if step.name == target_name:
                return idx

        raise ValueError(f"找不到目标步骤: {target_name}")

    def _handle_branch(
        self,
        bridge,
        step_sequence: List[StepConfig]
    ) -> Optional[int]:
        """处理 BRANCH 动作"""
        if len(bridge.targets) < 2:
            return None

        # 第一个目标：条件为真时跳转
        # 第二个目标：条件为假时跳转
        true_target = bridge.targets[0]
        false_target = bridge.targets[1] if len(bridge.targets) > 1 else None

        # 条件已经在 _handle_bridge 中检查
        # 如果执行到这里，说明条件为真
        target_name = true_target

        # 查找目标步骤
        for idx, step in enumerate(step_sequence):
            if step.name == target_name:
                return idx

        return None

    def _handle_loop(
        self,
        bridge,
        current_idx: int,
        step_sequence: List[StepConfig]
    ) -> Optional[int]:
        """处理 LOOP 动作"""
        if not bridge.targets:
            return None

        target_name = bridge.targets[0]

        # 获取最大迭代次数
        max_iter = int(bridge.params.get('max_iter', 1000))

        # 循环计数
        loop_key = f"loop_{target_name}"
        if loop_key not in self.loop_counters:
            self.loop_counters[loop_key] = 0

        self.loop_counters[loop_key] += 1

        # 检查是否超过最大迭代次数
        if self.loop_counters[loop_key] >= max_iter:
            self.loop_counters[loop_key] = 0
            return None  # 退出循环

        # 查找目标步骤
        for idx, step in enumerate(step_sequence):
            if step.name == target_name:
                return idx

        return None

    def _handle_conditional(
        self,
        bridge,
        current_idx: int
    ) -> Optional[int]:
        """处理 CONDITIONAL 动作"""
        # 条件执行：如果条件满足，继续；否则跳过
        # 条件已在 _handle_bridge 中检查
        return None

    def _handle_error_handler(
        self,
        bridge,
        step_sequence: List[StepConfig]
    ) -> Optional[int]:
        """处理 ERROR_HANDLER 动作"""
        if not bridge.targets:
            return None

        # 如果有错误，跳转到错误处理步骤
        # 这里简化处理，实际需要更复杂的错误检测
        return None

    def _handle_checkpoint(self, bridge):
        """处理 CHECKPOINT 动作"""
        # 保存检查点
        save_best = bridge.params.get('save_best', 'true').lower() == 'true'

        if save_best:
            # 使用 call_target 调用保存函数
            # 这里需要从配置中获取保存函数
            pass

    def _check_loop_termination(self) -> bool:
        """
        检查循环终止条件

        返回:
            是否应该终止
        """
        if self.should_stop:
            return True

        loop_condition = self.current_pipeline.loop_condition

        if not loop_condition:
            return False

        # 检查收敛条件
        if loop_condition.convergence_field:
            field = loop_condition.convergence_field

            # 检查字段是否存在于结果中
            if field in self.step_results:
                # 这里需要更复杂的收敛检测逻辑
                pass

        return False

    def stop(self):
        """停止训练"""
        self.should_stop = True

    def get_results(self) -> Dict[str, Any]:
        """
        获取训练结果

        返回:
            结果字典
        """
        return {
            "step_results": self.step_results,
            "loop_counters": self.loop_counters,
            "context_results": self.context.execution_results
        }


def run_training_pipeline(context: TrainContext) -> Dict[str, Any]:
    """
    运行训练流程的便捷函数

    参数:
        context: 训练上下文

    返回:
        训练结果
    """
    pipeline = TrainingPipeline(context)
    return pipeline.run()


def create_training_pipeline(context: TrainContext) -> TrainingPipeline:
    """
    创建训练流程执行器的便捷函数

    参数:
        context: 训练上下文

    返回:
        TrainingPipeline 实例
    """
    return TrainingPipeline(context)
