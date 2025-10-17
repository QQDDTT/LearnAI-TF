# -*- coding: utf-8 -*-
"""
common/train_context.py
训练上下文静态结构定义 - 改进版
重点改进：Bridge 和 Connection 表达式定义
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import re


# ======================================================
# 模型类型枚举
# ======================================================

class ModelType(Enum):
    """模型类型枚举"""
    SEQUENTIAL = "sequential"      # 顺序模型
    FUNCTIONAL = "functional"      # 函数式模型
    SUBCLASS = "subclass"          # 子类模型


class LoopType(Enum):
    """训练循环类型枚举"""
    EPOCH_BATCH = "epoch_batch"    # 按 epoch 和 batch 循环（监督学习）
    EPISODE_STEP = "episode_step"  # 按 episode 和 step 循环（强化学习）
    ITERATION = "iteration"        # 简单迭代循环
    CUSTOM = "custom"              # 自定义循环


class DataSourceType(Enum):
    """数据源类型枚举"""
    SUPERVISED = "supervised"      # 监督学习数据
    REINFORCEMENT = "reinforcement"  # 强化学习环境
    CUSTOM = "custom"              # 自定义数据源


class EvaluationFrequency(Enum):
    """评估频率枚举"""
    EPOCH = "epoch"                # 每个epoch评估一次
    BATCH = "batch"                # 每个batch评估一次
    STEP = "step"                  # 每个step评估一次
    EPISODE = "episode"            # 每个episode评估一次
    CUSTOM = "custom"              # 自定义频率


# ======================================================
# Bridge 和 Connection 相关枚举
# ======================================================

class BridgeAction(Enum):
    """训练流程Bridge动作类型枚举"""
    SKIP = "skip"                      # 跳转到指定步骤
    BRANCH = "branch"                  # 条件分支
    LOOP = "loop"                      # 循环控制
    CONDITIONAL = "conditional"        # 条件执行
    ERROR_HANDLER = "error_handler"    # 错误处理
    PARALLEL = "parallel"              # 并行执行
    CALLBACK = "callback"              # 回调
    CHECKPOINT = "checkpoint"          # 检查点


class ConnectionType(Enum):
    """层连接类型枚举"""
    SEQUENTIAL = "sequential"          # 顺序连接（默认）
    SKIP = "skip"                      # 跳跃连接
    RESIDUAL = "residual"              # 残差连接 (x + F(x))
    CONCAT = "concat"                  # 拼接连接（沿特征维度）
    ADD = "add"                        # 相加连接
    MULTIPLY = "multiply"              # 相乘连接
    BRANCH = "branch"                  # 分支（输出到多个层）
    MERGE = "merge"                    # 合并（从多个层输入）
    ATTENTION = "attention"            # 注意力连接
    DENSE = "dense"                    # 密集连接（DenseNet）


class LogicalOperator(Enum):
    """逻辑运算符枚举"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class ComparisonOperator(Enum):
    """比较运算符枚举"""
    EQ = "=="
    NE = "!="
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="


# ======================================================
# Bridge 和 Connection 表达式模式定义（改进版）
# ======================================================

class ReferencePattern:
    """
    引用模式常量 - 改进版

    定义配置中支持的引用格式，增强了 Bridge 和 Connection 的表达能力
    """

    # 变量引用：${variable} 或 ${step.field}
    VARIABLE = r'\$\{([^}]+)\}'

    # 步骤输出引用：${step_name.field}
    STEP_OUTPUT = r'\$\{([^.}]+)\.([^}]+)\}'

    # 模块引用：${module_name}
    MODULE = r'\$\{([^.}]+)\}'

    # ============ Bridge 表达式模式（增强版） ============

    # 完整 Bridge 格式：@action:target?condition(params)
    # 示例：
    # - @skip:validation_step
    # - @branch:step_a,step_b?${accuracy}>0.9
    # - @loop:train_step?${epoch}<10(max_iter=100)
    # - @conditional:save_model?${loss}<${best_loss} AND ${epoch}>5
    BRIDGE = r'@(\w+)(?::([^?()]+))?(?:\?([^()]+))?(?:\(([^)]+)\))?'

    # Bridge action 类型（用于验证）
    BRIDGE_ACTION = r'@(' + '|'.join([e.value for e in BridgeAction]) + r')'

    # Bridge 目标步骤（支持多个目标，逗号分隔）
    BRIDGE_TARGET = r':([a-zA-Z_][\w,]*)'

    # Bridge 条件表达式（支持变量引用和逻辑运算）
    # 示例：${loss}<0.1, ${acc}>0.9 AND ${epoch}>10, NOT ${converged}
    BRIDGE_CONDITION = r'\?([^()]+)'

    # Bridge 参数（键值对，逗号分隔）
    # 示例：(max_iter=100,timeout=60)
    BRIDGE_PARAMS = r'\(([^)]+)\)'

    # 条件表达式的组成部分
    CONDITION_VARIABLE = r'\$\{([^}]+)\}'
    CONDITION_OPERATOR = r'(==|!=|>=|<=|>|<)'
    CONDITION_LOGICAL = r'\b(AND|OR|NOT)\b'

    # ============ Connection 表达式模式（增强版） ============

    # 完整 Connection 格式：@type:layers(params)
    # 示例：
    # - @sequential
    # - @residual:layer_1
    # - @concat:layer_1,layer_2,layer_3
    # - @add:layer_1,layer_2(scale=0.5)
    # - @attention:query_layer,key_layer,value_layer(heads=8)
    CONNECTION = r'@(\w+)(?::([^()]+))?(?:\(([^)]+)\))?'

    # Connection 类型（用于验证）
    CONNECTION_TYPE = r'@(' + '|'.join([e.value for e in ConnectionType]) + r')'

    # Connection 目标层（支持多个层，逗号分隔）
    CONNECTION_TARGET = r':([a-zA-Z_][\w,]*)'

    # Connection 参数（键值对，逗号分隔）
    CONNECTION_PARAMS = r'\(([^)]+)\)'


# ======================================================
# Bridge 和 Connection 解析和验证工具
# ======================================================

@dataclass
class ParsedBridge:
    """解析后的 Bridge 表达式"""
    action: str                        # 动作类型
    targets: List[str]                 # 目标步骤列表
    condition: Optional[str]           # 条件表达式
    params: Dict[str, Any]             # 参数字典
    raw: str                           # 原始表达式

    def is_valid_action(self) -> bool:
        """验证 action 是否合法"""
        return self.action in [e.value for e in BridgeAction]

    def requires_target(self) -> bool:
        """判断该 action 是否需要目标"""
        no_target_actions = [
            BridgeAction.CHECKPOINT.value,
            BridgeAction.CALLBACK.value
        ]
        return self.action not in no_target_actions

    def requires_condition(self) -> bool:
        """判断该 action 是否需要条件"""
        condition_required = [
            BridgeAction.BRANCH.value,
            BridgeAction.CONDITIONAL.value,
            BridgeAction.LOOP.value
        ]
        return self.action in condition_required


@dataclass
class ParsedConnection:
    """解析后的 Connection 表达式"""
    type: str                          # 连接类型
    targets: List[str]                 # 目标层列表
    params: Dict[str, Any]             # 参数字典
    raw: str                           # 原始表达式

    def is_valid_type(self) -> bool:
        """验证 type 是否合法"""
        return self.type in [e.value for e in ConnectionType]

    def requires_targets(self) -> bool:
        """判断该连接类型是否需要目标层"""
        no_target_types = [
            ConnectionType.SEQUENTIAL.value
        ]
        return self.type not in no_target_types

    def min_targets_count(self) -> int:
        """获取最小目标层数量"""
        min_counts = {
            ConnectionType.RESIDUAL.value: 1,
            ConnectionType.SKIP.value: 1,
            ConnectionType.CONCAT.value: 2,
            ConnectionType.ADD.value: 2,
            ConnectionType.MULTIPLY.value: 2,
            ConnectionType.MERGE.value: 2,
            ConnectionType.ATTENTION.value: 3,  # query, key, value
            ConnectionType.BRANCH.value: 2,
        }
        return min_counts.get(self.type, 1)


def parse_bridge(bridge_str: str) -> Optional[ParsedBridge]:
    """
    解析 Bridge 表达式

    参数:
        bridge_str: Bridge 字符串

    返回:
        ParsedBridge 对象或 None
    """
    if not bridge_str or not bridge_str.startswith('@'):
        return None

    pattern = re.compile(ReferencePattern.BRIDGE)
    match = pattern.match(bridge_str)

    if not match:
        return None

    action, target_str, condition, params_str = match.groups()

    # 解析目标列表
    targets = []
    if target_str:
        targets = [t.strip() for t in target_str.split(',')]

    # 解析参数
    params = {}
    if params_str:
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()

    return ParsedBridge(
        action=action,
        targets=targets,
        condition=condition,
        params=params,
        raw=bridge_str
    )


def parse_connection(connection_str: str) -> Optional[ParsedConnection]:
    """
    解析 Connection 表达式

    参数:
        connection_str: Connection 字符串

    返回:
        ParsedConnection 对象或 None
    """
    if not connection_str or not connection_str.startswith('@'):
        return None

    pattern = re.compile(ReferencePattern.CONNECTION)
    match = pattern.match(connection_str)

    if not match:
        return None

    conn_type, target_str, params_str = match.groups()

    # 解析目标列表
    targets = []
    if target_str:
        targets = [t.strip() for t in target_str.split(',')]

    # 解析参数
    params = {}
    if params_str:
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()

    return ParsedConnection(
        type=conn_type,
        targets=targets,
        params=params,
        raw=connection_str
    )


def validate_bridge_condition(condition: str, available_vars: set) -> List[str]:
    """
    验证 Bridge 条件表达式

    参数:
        condition: 条件表达式
        available_vars: 可用变量集合

    返回:
        错误列表
    """
    errors = []

    if not condition:
        return errors

    # 提取所有变量引用
    var_pattern = re.compile(ReferencePattern.CONDITION_VARIABLE)
    variables = var_pattern.findall(condition)

    # 检查变量是否存在
    for var in variables:
        base_name = var.split('.')[0]
        if base_name not in available_vars:
            errors.append(f"条件中引用了未定义的变量: ${{{var}}}")

    # 检查比较运算符
    op_pattern = re.compile(ReferencePattern.CONDITION_OPERATOR)
    operators = op_pattern.findall(condition)

    # 检查逻辑运算符
    logical_pattern = re.compile(ReferencePattern.CONDITION_LOGICAL)
    logicals = logical_pattern.findall(condition)

    # 简单的语法检查：条件表达式应该包含比较或逻辑运算
    if not operators and not logicals and variables:
        errors.append(f"条件表达式缺少运算符: {condition}")

    return errors


def validate_connection_targets(
    conn_type: str,
    targets: List[str],
    available_layers: set
) -> List[str]:
    """
    验证 Connection 目标层

    参数:
        conn_type: 连接类型
        targets: 目标层列表
        available_layers: 可用层集合

    返回:
        错误列表
    """
    errors = []

    parsed = ParsedConnection(type=conn_type, targets=targets, params={}, raw="")

    # 检查连接类型是否合法
    if not parsed.is_valid_type():
        errors.append(f"不支持的连接类型: {conn_type}")
        return errors

    # 检查是否需要目标层
    if parsed.requires_targets() and not targets:
        errors.append(f"连接类型 '{conn_type}' 需要指定目标层")
        return errors

    # 检查目标层数量（仅在需要目标层时检查）
    if parsed.requires_targets():
        min_count = parsed.min_targets_count()
        if len(targets) < min_count:
            errors.append(
                f"连接类型 '{conn_type}' 至少需要 {min_count} 个目标层，"
                f"但只提供了 {len(targets)} 个"
            )

    # 检查目标层是否存在
    for target in targets:
        if target not in available_layers:
            errors.append(f"连接目标层 '{target}' 不存在")

    return errors


# ======================================================
# Bridge 和 Connection 的示例和文档
# ======================================================

class BridgeExamples:
    """Bridge 表达式示例"""

    # 基本跳转
    SKIP_SIMPLE = "@skip:validation_step"

    # 条件分支（单条件）
    BRANCH_SIMPLE = "@branch:step_a,step_b?${accuracy}>0.9"

    # 条件分支（复杂条件）
    BRANCH_COMPLEX = "@branch:continue,stop?${loss}<0.1 AND ${epoch}>10"

    # 循环控制
    LOOP = "@loop:train_step?${epoch}<${max_epochs}(max_iter=1000)"

    # 条件执行（带否定）
    CONDITIONAL = "@conditional:save_model?NOT ${early_stopped}"

    # 错误处理
    ERROR_HANDLER = "@error_handler:fallback_step"

    # 检查点（无目标）
    CHECKPOINT = "@checkpoint(save_best=true)"

    # 并行执行
    PARALLEL = "@parallel:task_a,task_b,task_c(max_workers=4)"


class ConnectionExamples:
    """Connection 表达式示例"""

    # 顺序连接（默认）
    SEQUENTIAL = "@sequential"

    # 残差连接
    RESIDUAL = "@residual:input_layer"

    # 跳跃连接
    SKIP = "@skip:layer_2"

    # 拼接连接
    CONCAT = "@concat:branch_a,branch_b,branch_c"

    # 相加连接（带参数）
    ADD = "@add:path_1,path_2(scale=0.5)"

    # 注意力连接
    ATTENTION = "@attention:query,key,value(heads=8,dropout=0.1)"

    # 密集连接
    DENSE = "@dense:layer_1,layer_2,layer_3,layer_4"

    # 分支连接
    BRANCH = "@branch:branch_a,branch_b"


# ======================================================
# 配置数据类
# ======================================================

@dataclass
class LayerConfig:
    """层配置"""
    reflection: str                    # 层类型反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 层参数
    name: Optional[str] = None        # 层名称
    connection: Optional[str] = None  # 连接表达式
    is_input: bool = False            # 是否为输入层
    is_output: bool = False           # 是否为输出层


@dataclass
class ModelConfig:
    """模型配置"""
    type: str                         # 模型类型: sequential, functional, subclass
    layers: List[LayerConfig] = field(default_factory=list)  # 层列表
    reflection: Optional[str] = None  # 自定义模型类的反射路径（用于subclass类型）
    args: Dict[str, Any] = field(default_factory=dict)  # 模型参数


@dataclass
class DataSourceConfig:
    """数据源配置"""
    reflection: str                   # 数据源反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 数据源参数
    name: Optional[str] = None        # 数据源名称


@dataclass
class OptimizerConfig:
    """优化器配置"""
    reflection: str                   # 优化器反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 优化器参数
    name: Optional[str] = None        # 优化器名称


@dataclass
class LossConfig:
    """损失函数配置"""
    reflection: str                   # 损失函数反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 损失函数参数
    name: Optional[str] = None        # 损失函数名称


@dataclass
class StepConfig:
    """训练步骤配置"""
    name: str                         # 步骤名称
    reflection: str                   # 步骤函数反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 步骤参数
    bridge: Optional[str] = None      # Bridge表达式


@dataclass
class PipelineConfig:
    """训练流程配置"""
    loop_type: str                    # 循环类型: epoch_batch, episode_step, iteration, custom
    steps: List[StepConfig] = field(default_factory=list)  # 步骤列表
    parameters: Dict[str, Any] = field(default_factory=dict)  # 流程参数
    loop_condition: Optional[str] = None  # 循环条件


@dataclass
class EvaluationConfig:
    """评估配置"""
    reflection: str                   # 评估函数反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 评估参数
    frequency: str = "epoch"          # 评估频率
    metrics: List[str] = field(default_factory=list)  # 评估指标
    name: Optional[str] = None        # 评估名称


@dataclass
class DeploymentConfig:
    """部署配置"""
    reflection: str                   # 部署函数反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 部署参数
    target: str = "local"             # 部署目标: local, cloud, edge
    name: Optional[str] = None        # 部署名称


@dataclass
class ExportConfig:
    """导出配置"""
    reflection: str                   # 导出函数反射路径
    args: Dict[str, Any] = field(default_factory=dict)  # 导出参数
    format: str = "savedmodel"        # 导出格式: savedmodel, onnx, tflite, etc.
    name: Optional[str] = None        # 导出名称


@dataclass
class TrainContext:
    """
    训练上下文

    存储所有训练相关的配置和运行时状态
    """
    # 配置数据
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    optimizers: Dict[str, OptimizerConfig] = field(default_factory=dict)
    losses: Dict[str, LossConfig] = field(default_factory=dict)
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    training_pipelines: Dict[str, PipelineConfig] = field(default_factory=dict)

    # 训练模式
    training_mode: str = "supervised"  # 训练模式: supervised, reinforcement, etc.

    # 运行时实例缓存
    instantiated_models: Dict[str, Any] = field(default_factory=dict)
    instantiated_optimizers: Dict[str, Any] = field(default_factory=dict)
    instantiated_losses: Dict[str, Any] = field(default_factory=dict)
    instantiated_data: Dict[str, Any] = field(default_factory=dict)
    instantiated_dataloaders: Dict[str, Any] = field(default_factory=dict)

    # 训练状态
    variables: Dict[str, Any] = field(default_factory=dict)  # 运行时变量
    global_config: Dict[str, Any] = field(default_factory=dict)  # 全局配置
    execution_results: Dict[str, Any] = field(default_factory=dict)  # 执行结果

    # 当前状态（循环计数器）
    current_epoch: int = 0
    current_episode: int = 0
    current_step: int = 0


# ======================================================
# 辅助验证函数（供 config_validator.py 使用）
# ======================================================

def validate_execution_unit(unit: Dict[str, Any], unit_type: str = "generic") -> List[str]:
    """
    验证执行层单元的结构（增强版）

    参数:
        unit: 执行层单元字典
        unit_type: 单元类型 (generic, step, layer, model)

    返回:
        错误列表
    """
    errors = []

    # 检查必须字段
    if "reflection" not in unit:
        errors.append(f"执行层缺少 'reflection' 字段")

    if "args" not in unit:
        errors.append(f"执行层缺少 'args' 字段")

    # 步骤类型必须有name
    if unit_type == "step" and "name" not in unit:
        errors.append(f"步骤执行层缺少 'name' 字段")

    # 检查reflection格式
    if "reflection" in unit:
        reflection = unit["reflection"]
        if not isinstance(reflection, str) or not reflection:
            errors.append(f"'reflection' 必须是非空字符串")
        elif ":" not in reflection and "." not in reflection:
            errors.append(f"'reflection' 格式无效: {reflection}")

    # 检查args类型
    if "args" in unit:
        if not isinstance(unit["args"], dict):
            errors.append(f"'args' 必须是字典类型")

    # 验证 Bridge（如果存在）
    if "bridge" in unit and unit["bridge"]:
        bridge = parse_bridge(unit["bridge"])
        if bridge is None:
            errors.append(f"Bridge 格式无效: {unit['bridge']}")
        else:
            if not bridge.is_valid_action():
                errors.append(
                    f"Bridge action '{bridge.action}' 不在支持列表中: "
                    f"{[e.value for e in BridgeAction]}"
                )
            if bridge.requires_target() and not bridge.targets:
                errors.append(f"Bridge action '{bridge.action}' 需要指定目标")
            if bridge.requires_condition() and not bridge.condition:
                errors.append(f"Bridge action '{bridge.action}' 需要指定条件")

    # 验证 Connection（如果存在，仅层使用）
    if unit_type == "layer" and "connection" in unit and unit["connection"]:
        connection = parse_connection(unit["connection"])
        if connection is None:
            errors.append(f"Connection 格式无效: {unit['connection']}")
        else:
            if not connection.is_valid_type():
                errors.append(
                    f"Connection type '{connection.type}' 不在支持列表中: "
                    f"{[e.value for e in ConnectionType]}"
                )
            # 注意：目标层的存在性验证需要在有上下文时进行

    return errors
