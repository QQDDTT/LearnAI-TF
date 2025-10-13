# -*- coding: utf-8 -*-
"""
training_pipeline.py - 训练流程执行
功能：
  - 根据配置文件执行训练流程
  - 支持所有训练方式
  - 自动循环和参数管理
"""

from typing import Dict, Any, Optional
import time
from modules.utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class TrainingPipeline:
    """
    训练流程管理器：
    - 根据配置执行训练
    - 自动选择循环类型（epoch_batch/episode_step/iteration）
    - 管理上下文和参数传递
    """

    def __init__(self,
                 config: Dict[str, Any],
                 models: Dict[str, Any],
                 optimizers: Dict[str, Any],
                 losses: Dict[str, Any],
                 data: Dict[str, Any]):
        """
        初始化训练流程

        参数：
            config: 完整配置文件
            models: 模型字典
            optimizers: 优化器字典
            losses: 损失函数字典
            data: 数据加载器字典
        """
        self.config = config
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.data = data

        self.training_mode = config.get("training_mode", {}).get("type", "supervised")
        logger.info(f"初始化训练流程: {self.training_mode}")

    def execute(self):
        """
        执行训练流程
        - 根据training_mode自动选择对应的流程
        - 执行主循环
        """
        pipeline_config = self.config.get("training_pipeline", {})

        if self.training_mode not in pipeline_config:
            raise ValueError(f"未找到 {self.training_mode} 的训练流程配置")

        pipeline_cfg = pipeline_config[self.training_mode]
        loop_type = pipeline_cfg.get("loop_type", "epoch_batch")

        logger.info(f"开始训练: 循环类型={loop_type}")

        try:
            if loop_type == "epoch_batch":
                self._execute_epoch_batch_loop(pipeline_cfg)
            elif loop_type == "episode_step":
                self._execute_episode_step_loop(pipeline_cfg)
            elif loop_type == "iteration":
                self._execute_iteration_loop(pipeline_cfg)
            else:
                raise ValueError(f"未知的循环类型: {loop_type}")
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}", exc_info=True)
            raise

    def _execute_epoch_batch_loop(self, pipeline_cfg: Dict[str, Any]):
        """
        执行 epoch-batch 循环
        典型用途：监督学习、自监督学习
        """
        parameters = pipeline_cfg.get("parameters", {})
        epochs = parameters.get("epochs", 10)
        steps_cfg = pipeline_cfg.get("step_sequence", [])

        logger.info(f"循环配置: epochs={epochs}")

        for epoch in range(epochs):
            logger.info(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
            epoch_start_time = time.time()

            # 获取数据加载器（通常是train数据）
            train_loader = self.data.get("train")
            if train_loader is None:
                logger.error("未找到train数据加载器")
                raise ValueError("需要train数据加载器")

            batch_idx = 0
            try:
                for batch in train_loader:
                    # 准备上下文
                    context = {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch": batch,
                        "models": self.models,
                        "optimizers": self.optimizers,
                        "losses": self.losses,
                        "config": self.config,
                        "data": self.data
                    }

                    # 执行step序列
                    self._execute_step_sequence(steps_cfg, context)
                    batch_idx += 1

                    if batch_idx % 10 == 0:
                        logger.debug(f"Batch {batch_idx}: 处理中...")
            except Exception as e:
                logger.error(f"Epoch {epoch + 1} 出错: {str(e)}", exc_info=True)
                raise

            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} 完成，处理{batch_idx}个batch，耗时: {epoch_time:.2f}s")

    def _execute_episode_step_loop(self, pipeline_cfg: Dict[str, Any]):
        """
        执行 episode-step 循环
        典型用途：强化学习
        """
        parameters = pipeline_cfg.get("parameters", {})
        episodes = parameters.get("episodes", 100)
        steps_per_episode = parameters.get("steps_per_episode", 500)
        steps_cfg = pipeline_cfg.get("step_sequence", [])

        logger.info(f"循环配置: episodes={episodes}, steps_per_episode={steps_per_episode}")

        for episode in range(episodes):
            logger.info(f"\n========== Episode {episode + 1}/{episodes} ==========")
            episode_start_time = time.time()

            session_id = f"ep_{episode}_{int(time.time() * 1000)}"
            total_reward = 0.0

            try:
                for step in range(steps_per_episode):
                    # 准备上下文
                    context = {
                        "episode": episode,
                        "step": step,
                        "session_id": session_id,
                        "epsilon": max(0.01, 1.0 - episode / episodes),  # ε衰减
                        "models": self.models,
                        "optimizers": self.optimizers,
                        "losses": self.losses,
                        "config": self.config,
                        "data": self.data,
                        "total_reward": total_reward
                    }

                    # 执行step序列
                    last_result = self._execute_step_sequence(steps_cfg, context)

                    # 提取奖励
                    if isinstance(last_result, dict) and "reward" in last_result:
                        total_reward += last_result["reward"]

                    if step % 50 == 0:
                        logger.debug(f"Step {step}: 累积奖励={total_reward:.2f}")
            except Exception as e:
                logger.error(f"Episode {episode + 1} 出错: {str(e)}", exc_info=True)
                raise

            episode_time = time.time() - episode_start_time
            logger.info(f"Episode {episode + 1} 完成, 总奖励={total_reward:.2f}, 耗时={episode_time:.2f}s")

    def _execute_iteration_loop(self, pipeline_cfg: Dict[str, Any]):
        """
        执行迭代循环
        典型用途：无监督学习（聚类）
        """
        parameters = pipeline_cfg.get("parameters", {})
        max_iterations = parameters.get("max_iterations", 100)
        convergence_threshold = parameters.get("convergence_threshold", 1e-4)
        steps_cfg = pipeline_cfg.get("step_sequence", [])
        loop_condition = pipeline_cfg.get("loop_condition", {})

        logger.info(f"循环配置: max_iterations={max_iterations}, threshold={convergence_threshold}")

        converged = False
        iteration = 0

        try:
            while iteration < max_iterations and not converged:
                logger.info(f"\n========== Iteration {iteration + 1}/{max_iterations} ==========")
                iteration_start_time = time.time()

                # 准备上下文
                context = {
                    "iteration": iteration,
                    "models": self.models,
                    "optimizers": self.optimizers,
                    "losses": self.losses,
                    "config": self.config,
                    "data": self.data
                }

                # 执行step序列
                last_result = self._execute_step_sequence(steps_cfg, context)

                # 检查收敛条件
                if isinstance(last_result, dict) and last_result.get("converged", False):
                    converged = True
                    logger.info("模型已收敛")

                iteration_time = time.time() - iteration_start_time
                logger.info(f"Iteration {iteration + 1} 完成，耗时: {iteration_time:.2f}s")

                iteration += 1
        except Exception as e:
            logger.error(f"迭代过程出错: {str(e)}", exc_info=True)
            raise

        if converged:
            logger.info(f"训练收敛，总迭代数: {iteration}")
        else:
            logger.warning(f"达到最大迭代数 {max_iterations}，但未收敛")

    def _execute_step_sequence(self, steps_cfg: list, context: Dict[str, Any]) -> Any:
        """
        执行step序列

        参数：
            steps_cfg: step配置列表
            context: 执行上下文
        返回：
            最后一个step的结果
        """
        last_result = None

        for idx, step_cfg in enumerate(steps_cfg):
            step_name = step_cfg.get("name", f"step_{idx}")
            reflection = step_cfg.get("reflection")
            args_cfg = step_cfg.get("args", {})

            logger.debug(f"执行step: {step_name}")

            try:
                # 参数绑定：替换 ${...} 和 "last_result" 等
                args = self._bind_arguments(args_cfg, context, last_result)

                # 反射调用
                result = call_target(reflection, args)

                # 将结果保存到context，供下一步使用
                context[step_name] = result
                context["last_result"] = result
                last_result = result

                logger.debug(f"Step {step_name} 完成")

            except Exception as e:
                logger.error(f"Step {step_name} 执行失败: {str(e)}", exc_info=True)
                raise

        return last_result

    def _bind_arguments(self,
                       args_cfg: Dict[str, Any],
                       context: Dict[str, Any],
                       last_result: Any) -> Dict[str, Any]:
        """
        参数绑定：将配置中的参数替换为实际值

        支持的格式：
        - "${variable}": 运行时变量
        - "last_result": 上一步的结果
        - "batch.x": batch中的字段
        - "config.path.to.value": 配置文件中的值

        参数：
            args_cfg: 参数配置
            context: 执行上下文
            last_result: 上一步的结果
        返回：
            绑定后的参数字典
        """
        bound_args = {}

        for key, value in args_cfg.items():
            if isinstance(value, str):
                # 处理字符串值
                if value.startswith("${") and value.endswith("}"):
                    # 运行时变量
                    var_name = value[2:-1]
                    bound_args[key] = context.get(var_name)
                    logger.debug(f"绑定变量: {key}=${{{var_name}}} = {bound_args[key]}")
                elif value == "last_result":
                    # 上一步的结果
                    bound_args[key] = last_result
                    logger.debug(f"绑定结果: {key}=last_result")
                elif value.startswith("batch."):
                    # batch字段
                    field = value[6:]
                    batch = context.get("batch", {})
                    if isinstance(batch, (tuple, list)):
                        # 如果batch是元组或列表
                        if field == "x":
                            bound_args[key] = batch[0] if len(batch) > 0 else None
                        elif field == "y":
                            bound_args[key] = batch[1] if len(batch) > 1 else None
                    else:
                        bound_args[key] = batch.get(field) if isinstance(batch, dict) else None
                elif value.startswith("config."):
                    # 配置文件值
                    path = value[7:].split(".")
                    val = self.config
                    for p in path:
                        val = val.get(p) if isinstance(val, dict) else None
                    bound_args[key] = val
                    logger.debug(f"绑定配置: {key}={value} = {val}")
                else:
                    # 普通字符串
                    bound_args[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                bound_args[key] = self._bind_arguments(value, context, last_result)
            elif isinstance(value, list):
                # 处理列表
                bound_args[key] = [
                    self._bind_value(v, context, last_result) for v in value
                ]
            else:
                # 非字符串值直接传递
                bound_args[key] = value

        return bound_args

    def _bind_value(self, value: Any, context: Dict[str, Any], last_result: Any) -> Any:
        """
        绑定单个值

        参数：
            value: 值
            context: 上下文
            last_result: 最后结果
        返回：
            绑定后的值
        """
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                return context.get(var_name)
            elif value == "last_result":
                return last_result
            else:
                return value
        elif isinstance(value, dict):
            return self._bind_arguments(value, context, last_result)
        elif isinstance(value, list):
            return [self._bind_value(v, context, last_result) for v in value]
        else:
            return value
