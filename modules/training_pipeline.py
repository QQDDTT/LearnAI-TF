# -*- coding: utf-8 -*-
"""
modules/training_pipeline.py
训练流程执行器：从配置上下文执行训练流程

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用

优化说明：
- 添加了安全的配置读取方法 _safe_get_config()
- 所有配置访问都通过安全方法进行
- 支持默认值，避免因配置缺失导致程序崩溃
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from common.interfaces import TrainingPipelineInterface
from common.train_context import (
    TrainContext,
    PipelineConfig,
    StepConfig,
    LoopType,
    BridgeAction
)
from common.common import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)


class TrainingPipeline(TrainingPipelineInterface):
    """
    训练流程执行器

    职责：
    1. 读取 TrainContext 中的训练流程配置
    2. 执行训练步骤序列
    3. 处理循环控制（epoch/episode/iteration）
    4. 处理 Bridge 控制流（跳转、分支、循环等）
    5. 管理训练状态和结果

    标准生命周期：
    initialize() -> validate_pipeline() -> setup_loop() -> execute_training() -> finalize()

    注意：所有执行步骤通过 call_target 动态调用

    优化特性：
    - 安全的配置读取，支持默认值
    - 关键配置缺失时会报错，非关键配置使用默认值
    """

    # 定义默认配置值
    DEFAULT_CONFIG = {
        'max_epochs': 100,
        'max_episodes': 1000,
        'max_iterations': 1000,
        'max_steps_per_episode': 1000,
        'convergence_threshold': 1e-4,
        'convergence_field': 'loss',
        'checkpoint_interval': 0,
        'checkpoint_enabled': False,
    }

    def __init__(self, context: TrainContext):
        """
        初始化训练流程执行器

        参数:
            context: 训练上下文
        """
        super().__init__(context)
        self.current_pipeline: Optional[PipelineConfig] = None
        self.step_results: Dict[str, Any] = {}
        self.loop_counters: Dict[str, int] = {}
        self.should_stop: bool = False
        self.training_results: Dict[str, Any] = {}

        # 缓存配置值，避免重复获取
        self._cached_configs: Dict[str, Any] = {}

    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有训练流程配置
        2. 加载当前训练模式的流程配置
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        logger.info("初始化训练流程执行器")

        # 验证训练模式
        if not self.context.training_mode:
            raise ValueError("TrainContext 中未设置 training_mode")

        # 验证训练流程配置
        if not self.context.training_pipelines:
            raise ValueError("TrainContext 中未设置 training_pipelines")

        training_mode = self.context.training_mode

        if training_mode not in self.context.training_pipelines:
            raise ValueError(
                f"训练模式 '{training_mode}' 没有对应的流程配置，"
                f"可用配置: {list(self.context.training_pipelines.keys())}"
            )

        # 加载流程配置
        self.current_pipeline = self.context.training_pipelines[training_mode]
        logger.info(f"加载训练流程: {training_mode}, 循环类型: {self.current_pipeline.loop_type}")

        # 初始化状态
        self.step_results = {}
        self.loop_counters = {}
        self.should_stop = False
        self.training_results = {}
        self._cached_configs = {}  # 清空缓存

        self._initialized = True

    def validate_pipeline(self) -> bool:
        """
        验证训练流程配置

        返回:
            配置是否有效
        """
        if not self.current_pipeline:
            logger.error("未加载训练流程配置")
            return False

        # 验证循环类型
        loop_type = self.current_pipeline.loop_type
        valid_loop_types = [e.value for e in LoopType]

        if loop_type not in valid_loop_types:
            logger.error(f"不支持的循环类型: {loop_type}, 有效类型: {valid_loop_types}")
            return False

        # 验证步骤序列
        if not self.current_pipeline.steps:
            logger.error(
                "⚠️  训练流程步骤列表为空！\n"
                "这意味着不会执行任何实际的训练步骤。\n"
                "请在配置文件中添加训练步骤，例如:\n"
                "  - forward_pass (前向传播)\n"
                "  - compute_loss (计算损失)\n"
                "  - backward_pass (反向传播)\n"
                "  - update_weights (更新权重)\n"
                "如果这是测试环境，可以忽略此警告。"
            )
            # 不返回 False，允许继续但给出明显警告
            return True

        # 验证每个步骤
        for step_config in self.current_pipeline.steps:
            if not step_config.reflection:
                logger.error(f"步骤 '{step_config.name}' 缺少 reflection 配置")
                return False

        logger.info("训练流程配置验证通过")
        return True

    # ========================================================================
    # 安全配置读取方法
    # ========================================================================

    def _safe_get_config(
        self,
        config_key: str,
        default: Any = None,
        source: str = 'auto',
        required: bool = False,
        use_cache: bool = True
    ) -> Any:
        """
        安全获取配置值

        优先级（source='auto'时）:
        1. 缓存的配置值（如果 use_cache=True）
        2. loop_condition 对象属性
        3. loop_condition 字典
        4. parameters 字典
        5. 提供的 default 值
        6. DEFAULT_CONFIG 中的默认值

        参数:
            config_key: 配置键名
            default: 自定义默认值（优先于 DEFAULT_CONFIG）
            source: 配置来源 ('auto', 'loop_condition', 'parameters')
            required: 是否为必需配置（缺失时抛出异常）
            use_cache: 是否使用缓存（默认 True）

        返回:
            配置值

        抛出:
            ValueError: 如果是必需配置且未找到
        """
        # 检查缓存
        if use_cache and config_key in self._cached_configs:
            return self._cached_configs[config_key]

        loop_condition = self.current_pipeline.loop_condition if self.current_pipeline else None
        parameters = self.current_pipeline.parameters if self.current_pipeline else {}

        value = None
        found = False

        # 自动查找模式
        if source == 'auto':
            # 1. 尝试从 loop_condition 获取
            if loop_condition:
                # 尝试对象属性访问
                if hasattr(loop_condition, config_key):
                    value = getattr(loop_condition, config_key, None)
                    if value is not None:
                        found = True
                        logger.debug(f"从 loop_condition 属性获取配置: {config_key} = {value}")

                # 尝试字典访问（如果 loop_condition 是字典）
                elif isinstance(loop_condition, dict) and config_key in loop_condition:
                    value = loop_condition[config_key]
                    found = True
                    logger.debug(f"从 loop_condition 字典获取配置: {config_key} = {value}")

            # 2. 尝试从 parameters 获取
            if not found and config_key in parameters:
                value = parameters[config_key]
                found = True
                logger.debug(f"从 parameters 获取配置: {config_key} = {value}")

        # 指定来源模式
        elif source == 'loop_condition' and loop_condition:
            if hasattr(loop_condition, config_key):
                value = getattr(loop_condition, config_key, None)
                found = value is not None
            elif isinstance(loop_condition, dict) and config_key in loop_condition:
                value = loop_condition[config_key]
                found = True

        elif source == 'parameters' and config_key in parameters:
            value = parameters[config_key]
            found = True

        # 3. 使用自定义默认值
        if not found and default is not None:
            value = default
            found = True
            logger.debug(f"使用自定义默认值: {config_key} = {value}")

        # 4. 使用预定义默认值
        if not found and config_key in self.DEFAULT_CONFIG:
            value = self.DEFAULT_CONFIG[config_key]
            found = True
            logger.debug(f"使用预定义默认值: {config_key} = {value}")

        # 5. 检查是否为必需配置
        if not found and required:
            raise ValueError(
                f"缺少必需的配置项: {config_key}\n"
                f"请在 loop_condition 或 parameters 中提供此配置。"
            )

        # 缓存配置值
        if use_cache and found:
            self._cached_configs[config_key] = value

        return value

    def _safe_get_nested(
        self,
        obj: Any,
        key: str,
        default: Any = None
    ) -> Any:
        """
        安全获取嵌套对象的属性或字典值

        参数:
            obj: 对象或字典
            key: 键名
            default: 默认值

        返回:
            值或默认值
        """
        if obj is None:
            return default

        # 尝试对象属性
        if hasattr(obj, key):
            value = getattr(obj, key, None)
            if value is not None:
                return value

        # 尝试字典访问
        if isinstance(obj, dict):
            return obj.get(key, default)

        return default

    # ========================================================================
    # 循环设置和执行
    # ========================================================================

    def setup_loop(self) -> None:
        """
        设置训练循环

        职责：
        1. 初始化循环计数器
        2. 配置循环条件
        3. 设置早停条件

        优化：
        - 预先获取并缓存所有配置，避免训练循环中重复获取
        - 减少日志输出
        """
        logger.info("设置训练循环")

        loop_type = self.current_pipeline.loop_type

        # 预先获取并缓存所有常用配置
        logger.debug("预加载配置...")
        self._preload_configs()

        # 根据循环类型初始化计数器
        if loop_type == LoopType.EPOCH_BATCH.value:
            self.loop_counters['epoch'] = 0
            self.loop_counters['batch'] = 0
            max_epochs = self._cached_configs.get('max_epochs', 100)
            logger.info(f"设置 Epoch-Batch 循环，最大轮数: {max_epochs}")

        elif loop_type == LoopType.EPISODE_STEP.value:
            self.loop_counters['episode'] = 0
            self.loop_counters['step'] = 0
            max_episodes = self._cached_configs.get('max_episodes', 1000)
            logger.info(f"设置 Episode-Step 循环，最大回合数: {max_episodes}")

        elif loop_type == LoopType.ITERATION.value:
            self.loop_counters['iteration'] = 0
            max_iterations = self._cached_configs.get('max_iterations', 1000)
            logger.info(f"设置迭代循环，最大迭代次数: {max_iterations}")

        elif loop_type == LoopType.CUSTOM.value:
            logger.info("设置自定义循环")

        # 设置早停条件
        convergence_field = self._cached_configs.get('convergence_field')
        convergence_threshold = self._cached_configs.get('convergence_threshold')

        if convergence_field:
            logger.info(
                f"启用收敛检测: 字段={convergence_field}, "
                f"阈值={convergence_threshold}"
            )

        self._setup_complete = True
        logger.debug("训练循环设置完成")

    def _preload_configs(self) -> None:
        """
        预加载所有常用配置到缓存

        避免在训练循环中重复调用 _safe_get_config
        """
        # 清空现有缓存
        self._cached_configs.clear()

        # 预加载所有常用配置
        config_keys = [
            'max_epochs',
            'max_episodes',
            'max_iterations',
            'max_steps_per_episode',
            'convergence_threshold',
            'convergence_field',
            'checkpoint_interval',
            'checkpoint_enabled',
        ]

        for key in config_keys:
            # use_cache=True 会自动缓存值
            self._safe_get_config(key, use_cache=True)

        logger.debug(f"预加载了 {len(self._cached_configs)} 个配置项")

    def execute_step(self, step_name: str, step_config: Any) -> Any:
        """
        执行单个训练步骤

        参数:
            step_name: 步骤名称
            step_config: 步骤配置

        返回:
            步骤执行结果
        """
        logger.debug(f"执行步骤: {step_name}")

        # 解析参数引用
        resolved_args = self._resolve_arguments(step_config.args)

        # 调用目标函数
        try:
            result = call_target(
                reflection=step_config.reflection,
                args=resolved_args
            )

            # 保存结果到上下文
            if step_config.output_key:
                self.context.execution_results[step_config.output_key] = result
                self.step_results[step_config.output_key] = result

            # 返回结果
            step_result = {
                "step_name": step_name,
                "step_id": step_config.step_id
            }

            if isinstance(result, dict):
                step_result.update(result)
            else:
                step_result["result"] = result

            return step_result

        except Exception as e:
            logger.error(f"步骤 {step_name} 执行失败: {str(e)}", exc_info=True)
            raise

    def execute_training(self) -> Dict[str, Any]:
        """
        执行训练循环

        返回:
            训练结果字典
        """
        loop_type = self.current_pipeline.loop_type
        logger.info(f"开始执行训练，循环类型: {loop_type}")

        if loop_type == LoopType.EPOCH_BATCH.value:
            results = self._run_epoch_batch_loop()

        elif loop_type == LoopType.EPISODE_STEP.value:
            results = self._run_episode_step_loop()

        elif loop_type == LoopType.ITERATION.value:
            results = self._run_iteration_loop()

        elif loop_type == LoopType.CUSTOM.value:
            results = self._run_custom_loop()

        else:
            raise ValueError(f"不支持的循环类型: {loop_type}")

        self.training_results = results
        logger.info("训练执行完成")

        return results

    def save_checkpoint(self, epoch: int) -> None:
        """
        保存检查点（可选）

        参数:
            epoch: 当前 epoch

        优化：使用安全的配置读取
        """
        logger.info(f"保存检查点: epoch={epoch}")

        # 安全获取 checkpoint 配置
        parameters = self.current_pipeline.parameters if self.current_pipeline else {}
        checkpoint_config = parameters.get('checkpoint', {})

        # 安全获取 enabled 配置
        enabled = checkpoint_config.get('enabled', False) if isinstance(checkpoint_config, dict) else False

        if not enabled:
            logger.debug("检查点未启用")
            return

        # 使用 call_target 调用保存函数
        save_fn = checkpoint_config.get('save_fn') if isinstance(checkpoint_config, dict) else None

        if save_fn:
            try:
                call_target(
                    reflection=save_fn,
                    args={
                        'context': self.context,
                        'epoch': epoch,
                        'models': self.context.container.models,
                        'optimizers': self.context.container.optimizers,
                        'step_results': self.step_results
                    }
                )
                logger.info(f"检查点已保存: epoch={epoch}")
            except Exception as e:
                logger.error(f"保存检查点失败: {str(e)}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点（可选）

        参数:
            checkpoint_path: 检查点路径
        """
        logger.info(f"加载检查点: {checkpoint_path}")

        parameters = self.current_pipeline.parameters if self.current_pipeline else {}
        checkpoint_config = parameters.get('checkpoint', {})
        load_fn = checkpoint_config.get('load_fn') if isinstance(checkpoint_config, dict) else None

        if load_fn:
            try:
                loaded_data = call_target(
                    reflection=load_fn,
                    args={
                        'checkpoint_path': checkpoint_path,
                        'context': self.context
                    }
                )

                # 恢复状态
                if isinstance(loaded_data, dict):
                    if 'epoch' in loaded_data:
                        self.loop_counters['epoch'] = loaded_data['epoch']
                    if 'step_results' in loaded_data:
                        self.step_results = loaded_data['step_results']

                logger.info("检查点加载成功")
            except Exception as e:
                logger.error(f"加载检查点失败: {str(e)}")
                raise

    def execute(self) -> Dict[str, Any]:
        """
        执行核心逻辑

        返回:
            训练结果
        """
        # 验证配置
        if not self.validate_pipeline():
            raise ValueError("训练流程配置验证失败")

        # 设置循环
        if not self._setup_complete:
            self.setup_loop()

        # 执行训练
        return self.execute_training()

    def finalize(self) -> None:
        """
        清理和保存结果

        职责：
        1. 将执行结果写回 TrainContext
        2. 清理临时资源
        3. 记录日志
        """
        logger.info("完成训练流程，保存结果")

        # 保存训练结果到上下文
        if self.training_results:
            self.context.execution_results['training'] = self.training_results

        # 保存最终步骤结果
        if self.step_results:
            self.context.execution_results['training_step_results'] = self.step_results

        # 保存循环计数器
        if self.loop_counters:
            self.context.execution_results['training_loop_counters'] = self.loop_counters

        logger.info(f"训练结果已保存到上下文，包含 {len(self.step_results)} 个步骤结果")

    # ========================================================================
    # 私有方法：循环实现
    # ========================================================================

    def _run_epoch_batch_loop(self) -> Dict[str, Any]:
        """
        执行 epoch-batch 循环（监督学习）

        返回:
            训练结果

        优化：直接使用缓存的配置，避免重复调用 _safe_get_config
        """
        # 直接从缓存获取配置
        max_epochs = self._cached_configs.get('max_epochs', 100)
        checkpoint_interval = self._cached_configs.get('checkpoint_interval', 0)

        logger.info(f"开始 Epoch-Batch 训练，最大轮数: {max_epochs}")

        results = {
            "epochs": [],
            "final_metrics": {}
        }

        # Epoch 循环
        for epoch in range(max_epochs):
            if self.should_stop:
                logger.warning(f"训练在 epoch {epoch} 被终止")
                break

            self.context.current_epoch = epoch
            self.loop_counters['epoch'] = epoch
            logger.info(f"开始 Epoch {epoch + 1}/{max_epochs}")

            # 执行一个 epoch
            epoch_result = self._execute_epoch()
            results["epochs"].append(epoch_result)

            # 记录 epoch 结果
            if 'loss' in epoch_result:
                logger.info(f"Epoch {epoch + 1} 完成, Loss: {epoch_result.get('loss', 'N/A')}")

            # 保存检查点（直接使用缓存的值）
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch)

            # 检查终止条件
            if self._check_loop_termination():
                logger.info(f"满足终止条件，在 epoch {epoch + 1} 停止训练")
                break

        results["final_metrics"] = self.step_results
        logger.info("训练完成")

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
        steps = self.current_pipeline.steps

        step_idx = 0
        while step_idx < len(steps):
            step_config = steps[step_idx]

            # 执行步骤
            step_result = self.execute_step(step_config.name, step_config)
            epoch_result["steps"].append(step_result)

            # 更新 epoch 结果（如损失、指标等）
            for key, value in step_result.items():
                if key not in ['step_name', 'step_id']:
                    epoch_result[key] = value

            # 处理 Bridge 控制
            next_idx = self._handle_bridge(step_config, step_idx, steps)

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

        优化：直接使用缓存的配置
        """
        # 直接从缓存获取参数
        max_episodes = self._cached_configs.get('max_episodes', 1000)
        max_steps_per_episode = self._cached_configs.get('max_steps_per_episode', 1000)

        logger.info(f"开始 Episode-Step 训练，最大回合数: {max_episodes}, 每回合最大步数: {max_steps_per_episode}")

        results = {
            "episodes": [],
            "final_metrics": {}
        }

        # Episode 循环
        for episode in range(max_episodes):
            if self.should_stop:
                logger.warning(f"训练在 episode {episode} 被终止")
                break

            self.context.current_episode = episode
            self.loop_counters['episode'] = episode
            logger.info(f"开始 Episode {episode + 1}/{max_episodes}")

            # 执行一个 episode
            episode_result = self._execute_episode(max_steps_per_episode)
            results["episodes"].append(episode_result)

            # 记录 episode 结果
            total_reward = episode_result.get('total_reward', 0)
            logger.info(f"Episode {episode + 1} 完成, 总奖励: {total_reward:.2f}, 步数: {len(episode_result['steps'])}")

            # 检查终止条件
            if self._check_loop_termination():
                logger.info(f"满足终止条件，在 episode {episode + 1} 停止训练")
                break

        results["final_metrics"] = self.step_results
        logger.info("强化学习训练完成")

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

        steps = self.current_pipeline.steps

        # Step 循环
        for step in range(max_steps):
            self.context.current_step = step
            self.loop_counters['step'] = step

            # 执行步骤序列
            step_idx = 0
            while step_idx < len(steps):
                step_config = steps[step_idx]

                # 执行步骤
                step_result = self.execute_step(step_config.name, step_config)
                episode_result["steps"].append(step_result)

                # 累积奖励
                if 'reward' in step_result:
                    episode_result["total_reward"] += step_result['reward']

                # 检查是否结束
                if step_result.get('done', False):
                    episode_result["done"] = True
                    logger.debug(f"Episode 在步骤 {step + 1} 结束")
                    return episode_result

                # 处理 Bridge 控制
                next_idx = self._handle_bridge(step_config, step_idx, steps)

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

        优化：直接使用缓存的配置
        """
        # 直接从缓存获取参数
        max_iterations = self._cached_configs.get('max_iterations', 1000)
        convergence_threshold = self._cached_configs.get('convergence_threshold', 1e-4)
        convergence_field = self._cached_configs.get('convergence_field', 'loss')

        logger.info(f"开始迭代训练，最大迭代次数: {max_iterations}, 收敛阈值: {convergence_threshold}")

        results = {
            "iterations": [],
            "final_metrics": {},
            "converged": False
        }

        prev_value = None

        # 迭代循环
        for iteration in range(max_iterations):
            if self.should_stop:
                logger.warning(f"训练在迭代 {iteration} 被终止")
                break

            self.loop_counters['iteration'] = iteration
            logger.debug(f"迭代 {iteration + 1}/{max_iterations}")

            # 执行步骤序列
            iteration_result = self._execute_iteration()
            results["iterations"].append(iteration_result)

            # 检查收敛
            if convergence_field and convergence_field in iteration_result:
                current_value = iteration_result[convergence_field]

                if prev_value is not None:
                    change = abs(current_value - prev_value)
                    if change < convergence_threshold:
                        results["converged"] = True
                        logger.info(f"迭代 {iteration + 1} 达到收敛: {convergence_field} 变化 {change} < {convergence_threshold}")
                        break

                prev_value = current_value

            # 检查终止条件
            if self._check_loop_termination():
                logger.info(f"满足终止条件，在迭代 {iteration + 1} 停止训练")
                break

        results["final_metrics"] = self.step_results
        logger.info(f"迭代训练完成，{'已收敛' if results['converged'] else '未收敛'}")

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

        steps = self.current_pipeline.steps

        # 执行步骤序列
        step_idx = 0
        while step_idx < len(steps):
            step_config = steps[step_idx]

            # 执行步骤
            step_result = self.execute_step(step_config.name, step_config)
            iteration_result["steps"].append(step_result)

            # 更新迭代结果
            for key, value in step_result.items():
                if key not in ['step_name', 'step_id']:
                    iteration_result[key] = value

            # 处理 Bridge 控制
            next_idx = self._handle_bridge(step_config, step_idx, steps)

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
        logger.info("开始自定义循环")

        results = {
            "steps": [],
            "final_metrics": {}
        }

        steps = self.current_pipeline.steps

        # 执行步骤序列（单次）
        step_idx = 0
        while step_idx < len(steps):
            if self.should_stop:
                break

            step_config = steps[step_idx]

            # 执行步骤
            step_result = self.execute_step(step_config.name, step_config)
            results["steps"].append(step_result)

            # 处理 Bridge 控制
            next_idx = self._handle_bridge(step_config, step_idx, steps)

            if next_idx is None:
                step_idx += 1
            else:
                step_idx = next_idx

        results["final_metrics"] = self.step_results
        logger.info("自定义循环完成")

        return results

    # ========================================================================
    # 私有方法：参数解析
    # ========================================================================

    def _resolve_arguments(self, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        解析参数引用

        支持的引用格式：
        - $context.xxx: 从上下文获取
        - $results.step_name.xxx: 从步骤结果获取
        - $loop.xxx: 从循环计数器获取
        - $models.xxx: 从容器获取模型
        - $optimizers.xxx: 从容器获取优化器

        参数:
            args: 参数字典

        返回:
            解析后的参数字典
        """
        if not args:
            return {}

        resolved = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith('$'):
                # 解析引用
                resolved[key] = self._resolve_reference(value)
            elif isinstance(value, dict):
                # 递归解析嵌套字典
                resolved[key] = self._resolve_arguments(value)
            elif isinstance(value, list):
                # 解析列表中的引用
                resolved[key] = [
                    self._resolve_reference(item) if isinstance(item, str) and item.startswith('$') else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _resolve_reference(self, ref: str) -> Any:
        """
        解析单个引用

        参数:
            ref: 引用字符串，如 $context.training_mode

        返回:
            引用的值
        """
        if not ref.startswith('$'):
            return ref

        # 移除 $ 符号
        ref = ref[1:]
        parts = ref.split('.')

        if not parts:
            return None

        source = parts[0]

        try:
            if source == 'context':
                # 从上下文获取
                obj = self.context
                for part in parts[1:]:
                    obj = getattr(obj, part, None) if hasattr(obj, part) else obj.get(part, None) if isinstance(obj, dict) else None
                return obj

            elif source == 'results':
                # 从步骤结果获取
                if len(parts) < 2:
                    return None
                step_name = parts[1]
                if step_name in self.step_results:
                    result = self.step_results[step_name]
                    if len(parts) > 2:
                        for part in parts[2:]:
                            result = result.get(part) if isinstance(result, dict) else getattr(result, part, None)
                    return result
                return None

            elif source == 'loop':
                # 从循环计数器获取
                if len(parts) < 2:
                    return None
                counter_name = parts[1]
                return self.loop_counters.get(counter_name)

            elif source == 'models':
                # 从容器获取模型
                if len(parts) < 2:
                    return None
                model_name = parts[1]
                return self.context.container.models.get(model_name)

            elif source == 'optimizers':
                # 从容器获取优化器
                if len(parts) < 2:
                    return None
                optimizer_name = parts[1]
                return self.context.container.optimizers.get(optimizer_name)

            else:
                logger.warning(f"未知的引用源: {source}")
                return None

        except Exception as e:
            logger.error(f"解析引用失败: {ref}, 错误: {str(e)}")
            return None

    # ========================================================================
    # 私有方法：Bridge 控制流
    # ========================================================================

    def _handle_bridge(
        self,
        step_config: StepConfig,
        current_idx: int,
        steps: List[StepConfig]
    ) -> Optional[int]:
        """
        处理 Bridge 控制流

        参数:
            step_config: 当前步骤配置
            current_idx: 当前步骤索引
            steps: 步骤序列

        返回:
            下一个步骤的索引，None 表示正常继续
        """
        if not step_config.bridge:
            return None

        bridge = step_config.bridge
        action = bridge.action

        logger.debug(f"处理 Bridge: {action}")

        if action == BridgeAction.JUMP.value:
            return self._handle_jump(bridge, steps)

        elif action == BridgeAction.CONDITIONAL.value:
            return self._handle_conditional(bridge, current_idx, steps)

        elif action == BridgeAction.LOOP.value:
            return self._handle_loop_bridge(bridge, current_idx)

        elif action == BridgeAction.BREAK.value:
            self.should_stop = True
            return None

        elif action == BridgeAction.CONTINUE.value:
            # 跳到下一个循环迭代
            return len(steps)

        elif action == BridgeAction.ERROR_HANDLER.value:
            return self._handle_error(bridge)

        elif action == BridgeAction.CHECKPOINT.value:
            self._handle_checkpoint(bridge)
            return None

        else:
            logger.warning(f"未知的 Bridge 动作: {action}")
            return None

    def _handle_jump(self, bridge, steps: List[StepConfig]) -> Optional[int]:
        """处理 JUMP 动作"""
        if not bridge.targets:
            return None

        target_step_id = bridge.targets[0]

        # 查找目标步骤
        for idx, step in enumerate(steps):
            if step.step_id == target_step_id:
                logger.debug(f"跳转到步骤: {target_step_id}")
                return idx

        logger.warning(f"未找到目标步骤: {target_step_id}")
        return None

    def _handle_conditional(
        self,
        bridge,
        current_idx: int,
        steps: List[StepConfig]
    ) -> Optional[int]:
        """处理 CONDITIONAL 动作"""
        condition = bridge.condition

        if not condition:
            return None

        # 评估条件
        condition_result = self._evaluate_condition(condition)

        if condition_result:
            # 条件为真，跳转到第一个目标
            if bridge.targets:
                target_step_id = bridge.targets[0]
                for idx, step in enumerate(steps):
                    if step.step_id == target_step_id:
                        logger.debug(f"条件为真，跳转到: {target_step_id}")
                        return idx
        else:
            # 条件为假，跳转到第二个目标（如果有）
            if len(bridge.targets) > 1:
                target_step_id = bridge.targets[1]
                for idx, step in enumerate(steps):
                    if step.step_id == target_step_id:
                        logger.debug(f"条件为假，跳转到: {target_step_id}")
                        return idx

        return None

    def _evaluate_condition(self, condition: str) -> bool:
        """
        评估条件表达式

        支持的条件格式：
        - $results.step_name.field > 0.9
        - $loop.epoch >= 10
        - $context.training_mode == 'supervised'

        参数:
            condition: 条件表达式字符串

        返回:
            条件评估结果
        """
        try:
            # 简单的条件评估
            # 替换引用
            import re
            refs = re.findall(r'\$[\w.]+', condition)

            eval_condition = condition
            for ref in refs:
                value = self._resolve_reference(ref)
                # 将引用替换为实际值
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(ref, f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(ref, str(value))

            # 评估条件
            result = eval(eval_condition)
            return bool(result)

        except Exception as e:
            logger.error(f"条件评估失败: {condition}, 错误: {str(e)}")
            return False

    def _handle_loop_bridge(self, bridge, current_idx: int) -> Optional[int]:
        """
        处理 LOOP 动作

        优化：使用安全的配置读取方法
        """
        # 安全获取 bridge params
        loop_var = self._safe_get_nested(bridge, 'params', {}).get('loop_var', 'i') if hasattr(bridge, 'params') else \
                   bridge.get('params', {}).get('loop_var', 'i') if isinstance(bridge, dict) else 'i'

        start = int(self._safe_get_nested(bridge, 'params', {}).get('start', 0) if hasattr(bridge, 'params') else \
                    bridge.get('params', {}).get('start', 0) if isinstance(bridge, dict) else 0)

        end = int(self._safe_get_nested(bridge, 'params', {}).get('end', 10) if hasattr(bridge, 'params') else \
                  bridge.get('params', {}).get('end', 10) if isinstance(bridge, dict) else 10)

        step = int(self._safe_get_nested(bridge, 'params', {}).get('step', 1) if hasattr(bridge, 'params') else \
                   bridge.get('params', {}).get('step', 1) if isinstance(bridge, dict) else 1)

        # 初始化循环变量
        if loop_var not in self.loop_counters:
            self.loop_counters[loop_var] = start

        current_value = self.loop_counters[loop_var]

        if current_value < end:
            # 继续循环，跳转到目标步骤
            self.loop_counters[loop_var] += step
            targets = getattr(bridge, 'targets', None) if hasattr(bridge, 'targets') else bridge.get('targets') if isinstance(bridge, dict) else None

            if targets:
                target_step_id = targets[0]
                # 查找目标步骤索引
                for idx, step_cfg in enumerate(self.current_pipeline.steps):
                    if step_cfg.step_id == target_step_id:
                        return idx
        else:
            # 循环结束
            del self.loop_counters[loop_var]
            return None

        return None

    def _handle_error(self, bridge) -> Optional[int]:
        """处理 ERROR_HANDLER 动作"""
        # 简化的错误处理
        targets = getattr(bridge, 'targets', None) if hasattr(bridge, 'targets') else bridge.get('targets') if isinstance(bridge, dict) else None

        if not targets:
            return None

        # 如果有错误，跳转到错误处理步骤
        # 这里简化处理，实际需要更复杂的错误检测
        return None

    def _handle_checkpoint(self, bridge):
        """
        处理 CHECKPOINT 动作

        优化：使用安全的配置读取方法
        """
        # 安全获取 save_best 参数
        params = self._safe_get_nested(bridge, 'params', {})
        save_best_str = params.get('save_best', 'true') if isinstance(params, dict) else 'true'
        save_best = str(save_best_str).lower() == 'true'

        if save_best:
            logger.info("触发检查点保存")
            epoch = self.loop_counters.get('epoch', 0)
            self.save_checkpoint(epoch)

    def _check_loop_termination(self) -> bool:
        """
        检查循环终止条件

        返回:
            是否应该终止

        优化：直接使用缓存的配置
        """
        if self.should_stop:
            return True

        # 直接从缓存获取收敛配置
        convergence_field = self._cached_configs.get('convergence_field')
        convergence_threshold = self._cached_configs.get('convergence_threshold')

        # 检查收敛条件
        if convergence_field:
            # 检查字段是否存在于结果中
            if convergence_field in self.step_results:
                # 这里可以实现更复杂的收敛检测逻辑
                # 例如：检查最近几次迭代的变化
                pass

        return False
