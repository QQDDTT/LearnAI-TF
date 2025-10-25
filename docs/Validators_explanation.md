# common/validators 目录文件说明

## 📁 目录结构

```
common/validators/
├── __init__.py                    # 模块初始化
├── base_validator.py              # 验证器基类
├── constants.py                   # 常量定义
├── patterns.py                    # 正则表达式模式
├── structure_validator.py         # 结构验证器
├── collection_validator.py        # 名称收集验证器
├── execution_validator.py         # 执行单元验证器
├── bridge_validator.py            # Bridge 表达式验证器
├── connection_validator.py        # Connection 表达式验证器
└── cross_ref_validator.py         # 交叉引用验证器
```

## 🎯 设计理念

### 分层验证架构
配置验证采用**分阶段、分职责**的设计模式：

```
ConfigValidator (协调器)
    ↓
├── StructureValidator        (阶段 1: 基础结构)
├── CollectionValidator       (阶段 2: 名称收集)
├── ExecutionValidator        (阶段 3: 执行单元)
├── BridgeValidator          (阶段 4: Bridge 验证)
├── ConnectionValidator      (阶段 4: Connection 验证)
└── CrossReferenceValidator  (阶段 5: 交叉引用)
```

### 核心特点
1. **职责分离**: 每个验证器负责特定层面的验证
2. **依赖管理**: 后续阶段依赖前面阶段收集的信息
3. **状态共享**: 通过协调器共享验证状态（如层名称、步骤名称）
4. **错误收集**: 统一的错误/警告收集机制

---

## 📄 各文件详解

### 1. base_validator.py - 验证器基类

**作用**: 所有验证器的抽象基类

**核心功能**:
```python
class BaseValidator:
    def __init__(self, context: ConfigValidator):
        self.context = context  # 引用主协调器

    def add_error(self, message: str):
        """向协调器添加错误"""
        self.context.add_error(message)

    def add_warning(self, message: str):
        """向协调器添加警告"""
        self.context.add_warning(message)

    def validate(self, config: dict):
        """子类必须实现的验证方法"""
        raise NotImplementedError
```

**职责**:
- 提供统一的错误/警告报告接口
- 定义验证器的基本契约
- 通过 `context` 访问协调器的共享状态

---

### 2. constants.py - 常量定义

**作用**: 定义配置验证所需的所有常量

**核心内容**:

#### ConfigStructure - 配置结构常量
```python
class ConfigStructure:
    # 必需的顶级模块
    REQUIRED_MODULES = {
        "training_mode",
        "models",
        "data_sources",
        "optimizers",
        "losses",
        "training_pipeline"
    }

    # 每个模块必需的字段
    MODULE_REQUIRED_FIELDS = {
        "training_mode": {"type"},
        "data_sources": {"default"},
        "optimizers": {"default"},
        "losses": {"default"},
    }
```

#### TrainingModeConstants - 训练模式常量
```python
class TrainingModeConstants:
    SUPPORTED_MODES = {
        "supervised",
        "reinforcement",
        "unsupervised",
        "custom"
    }
```

**职责**:
- 集中管理所有验证规则相关的常量
- 便于维护和修改验证规则
- 提供清晰的配置要求文档

---

### 3. patterns.py - 正则表达式模式

**作用**: 定义所有用于解析配置表达式的正则模式

**核心内容**:

```python
class ReferencePattern:
    # 变量引用：${variable} 或 ${step.field}
    VARIABLE = r'\$\{([^}]+)\}'
    VARIABLE_COMPILED = re.compile(VARIABLE)

    # Bridge 表达式：@JUMP:target?condition(params)
    BRIDGE = r'@(\w+)(?::([^?()]+))?(?:\?([^()]+))?(?:\(([^)]+)\))?'
    BRIDGE_COMPILED = re.compile(BRIDGE)

    # Connection 表达式：@RESIDUAL:target(params)
    CONNECTION = r'@(\w+)(?::([^()]+))?(?:\(([^)]+)\))?'
    CONNECTION_COMPILED = re.compile(CONNECTION)

    # 条件表达式的组成部分
    CONDITION_VARIABLE = r'\$\{([^}]+)\}'
    CONDITION_OPERATOR = r'(==|!=|>=|<=|>|<)'
    CONDITION_LOGICAL = r'\b(AND|OR|NOT)\b'
```

**职责**:
- 集中管理所有正则表达式
- 提供预编译的正则对象以提高性能
- 支持 Bridge、Connection 等高级表达式的解析

---

### 4. structure_validator.py - 结构验证器

**作用**: 验证配置文件的基本结构（阶段 1）

**验证内容**:

#### 第一层：模块层
- 检查是否包含所有必需的顶级模块
- 验证项：`training_mode`, `models`, `data_sources`, `optimizers`, `losses`, `training_pipeline`

#### 第二层：常量层
- 检查每个模块是否包含必需的字段
- 验证模块配置是否为字典类型

**示例**:
```python
class StructureValidator(BaseValidator):
    def validate(self, config: dict):
        self._validate_module_layer(config)
        self._validate_constant_layer(config)

    def _validate_module_layer(self, config: dict):
        for module in ConfigStructure.REQUIRED_MODULES:
            if module not in config:
                self.add_error(f"缺少必须的模块: {module}")
```

**错误示例**:
```
❌ 缺少必须的模块: models
❌ 模块 'training_mode' 必须是字典类型
❌ 模块 'training_mode' 缺少必须字段: type
```

---

### 5. collection_validator.py - 名称收集验证器

**作用**: 收集层名称和步骤名称（阶段 2）

**核心功能**:

#### 收集层名称
- 从 `models` 配置中收集所有层的名称
- 检测重复的层名称（警告）
- 将层名称注册到协调器

#### 收集步骤名称
- 从 `training_pipeline` 配置中收集所有步骤名称
- 检测重复的步骤名称（警告）
- 将步骤名称注册到协调器

**示例**:
```python
class CollectionValidator(BaseValidator):
    def validate_and_collect(self, config: dict):
        self._collect_layer_names(config)
        self._collect_step_names(config)

    def _collect_layer_names(self, config: dict):
        for model_name, model_config in config["models"].items():
            layer_names = set()
            for layer in model_config["layers"]:
                layer_name = layer.get("name", f"layer_{i}")
                layer_names.add(layer_name)

            # 注册到协调器
            self.context.set_layer_names(model_name, layer_names)
```

**为什么重要**:
- 后续验证器（Bridge、Connection）需要这些名称
- 验证引用的目标是否存在
- 检测命名冲突

---

### 6. execution_validator.py - 执行单元验证器

**作用**: 验证执行单元的 `reflection` 和 `args`（阶段 3）

**验证内容**:

#### 模型层验证
- 验证模型本身的 `reflection` 和 `args`
- 验证每个层的 `reflection` 和 `args`
- 验证层的 `connection` 表达式

#### 步骤验证
- 验证训练流程中每个步骤的 `reflection` 和 `args`
- 验证步骤的 `bridge` 表达式

**示例**:
```python
class ExecutionValidator(BaseValidator):
    def validate_execution_unit(self, unit: dict, unit_type: str) -> List[str]:
        errors = []

        # 验证 reflection 字段
        if "reflection" not in unit:
            errors.append(f"{unit_type} 缺少 'reflection' 字段")
        elif not isinstance(unit["reflection"], str):
            errors.append(f"{unit_type} 的 'reflection' 必须是字符串")

        # 验证 args 字段
        if "args" in unit and not isinstance(unit["args"], dict):
            errors.append(f"{unit_type} 的 'args' 必须是字典")

        return errors
```

**错误示例**:
```
❌ 模型 'classifier': 缺少 'reflection' 字段
❌ 模型 'classifier' 的层 'dense1': 'args' 必须是字典
❌ 流程 'supervised' 步骤 'forward_pass': 缺少 'reflection' 字段
```

---

### 7. bridge_validator.py - Bridge 表达式验证器

**作用**: 深度验证 Bridge 控制流表达式（阶段 4）

**Bridge 表达式格式**:
```
@ACTION:target?condition(params)
```

**验证内容**:

#### 语法验证
- 使用正则表达式解析 Bridge 表达式
- 验证 action 是否有效

#### 目标验证
- 验证目标步骤是否存在
- 验证目标数量是否满足要求

#### 条件验证
- 验证条件表达式中的变量引用
- 验证运算符和逻辑操作符

#### 参数验证
- 验证特定 action 所需的参数

**支持的 Bridge Actions**:
- `JUMP`: 跳转到指定步骤
- `CONDITIONAL`: 条件分支
- `LOOP`: 循环控制
- `BREAK`: 中断循环
- `CONTINUE`: 继续下一次循环
- `ERROR_HANDLER`: 错误处理
- `CHECKPOINT`: 保存检查点

**示例**:
```python
# 解析 Bridge 表达式
def parse_bridge(bridge_str: str) -> Optional[ParsedBridge]:
    match = ReferencePattern.BRIDGE_COMPILED.match(bridge_str)
    if not match:
        return None

    action, target_str, condition, params_str = match.groups()
    # 解析目标和参数...
    return ParsedBridge(action, targets, condition, params, bridge_str)

# 验证 Bridge
class BridgeValidator(BaseValidator):
    def validate(self, config: dict):
        for pipeline_name, pipeline_config in config["training_pipeline"].items():
            for step in pipeline_config["step_sequence"]:
                if "bridge" in step:
                    bridge = parse_bridge(step["bridge"])
                    self._validate_bridge(bridge, ...)
```

**错误示例**:
```
❌ Bridge 格式无效: @JUMP
❌ 不支持的 Bridge action: UNKNOWN
❌ Bridge action 'JUMP' 需要指定目标步骤
❌ Bridge 目标步骤 'nonexistent_step' 不存在
❌ 条件中引用了未定义的变量: ${unknown_var}
```

---

### 8. connection_validator.py - Connection 表达式验证器

**作用**: 深度验证 Connection 层连接表达式（阶段 4）

**Connection 表达式格式**:
```
@TYPE:target(params)
```

**验证内容**:

#### 语法验证
- 使用正则表达式解析 Connection 表达式
- 验证连接类型是否有效

#### 目标验证
- 验证目标层是否存在
- 验证目标数量是否满足连接类型要求

#### 参数验证
- 验证连接类型特定的参数

**支持的 Connection Types**:
- `SEQUENTIAL`: 顺序连接（默认）
- `RESIDUAL`: 残差连接
- `SKIP`: 跳跃连接
- `CONCAT`: 拼接连接
- `ADD`: 相加连接
- `MULTIPLY`: 相乘连接
- `ATTENTION`: 注意力机制
- `BRANCH`: 分支连接

**示例**:
```python
# 解析 Connection 表达式
def parse_connection(connection_str: str) -> Optional[ParsedConnection]:
    match = ReferencePattern.CONNECTION_COMPILED.match(connection_str)
    if not match:
        return None

    conn_type, target_str, params_str = match.groups()
    # 解析目标和参数...
    return ParsedConnection(conn_type, targets, params, connection_str)

# 验证目标层
def validate_connection_targets(
    conn_type: str,
    targets: List[str],
    available_layers: set
) -> List[str]:
    errors = []

    # 验证类型
    if conn_type not in valid_types:
        errors.append(f"不支持的连接类型: {conn_type}")

    # 验证目标数量
    min_count = get_min_targets_count(conn_type)
    if len(targets) < min_count:
        errors.append(f"连接类型 '{conn_type}' 至少需要 {min_count} 个目标层")

    # 验证目标存在性
    for target in targets:
        if target not in available_layers:
            errors.append(f"连接目标层 '{target}' 不存在")

    return errors
```

**错误示例**:
```
❌ Connection 格式无效: @RESIDUAL
❌ 不支持的连接类型: UNKNOWN
❌ 连接类型 'CONCAT' 至少需要 2 个目标层，但只提供了 1 个
❌ 连接目标层 'nonexistent_layer' 不存在
```

---

### 9. cross_ref_validator.py - 交叉引用验证器

**作用**: 验证配置间的引用完整性和逻辑一致性（阶段 5）

**验证内容**:

#### 训练模式验证
- 验证训练模式是否受支持
- 验证训练模式是否有对应的流程定义

#### 步骤引用验证
- 验证步骤中的 `${variable}` 引用
- 检查引用的变量是否定义
- 递归检查嵌套字典中的引用

**示例**:
```python
class CrossReferenceValidator(BaseValidator):
    def validate(self, config: dict):
        self._validate_training_mode(config)
        self._validate_all_step_references(config)

    def _validate_training_mode(self, config: dict):
        mode = config["training_mode"]["type"]

        # 验证模式是否支持
        if mode not in TrainingModeConstants.SUPPORTED_MODES:
            self.add_error(f"不支持的训练模式: {mode}")

        # 验证是否有对应的流程
        if mode not in config["training_pipeline"]:
            self.add_error(f"训练模式 '{mode}' 没有对应的流程定义")

    def _validate_all_step_references(self, config: dict):
        # 收集所有可引用的名称
        available_names = self._collect_all_available_names(config)

        # 验证每个步骤的引用
        for step in pipeline_config["step_sequence"]:
            self._check_references_in_dict(
                step["args"],
                available_names,
                context_str
            )
```

**错误示例**:
```
❌ 不支持的训练模式: custom_mode
❌ 训练模式 'supervised' 没有对应的流程定义
⚠️  步骤 'forward_pass'.args.model: 引用了未定义的名称 '${unknown_model}'
```

---

## 🔄 验证流程

### 完整验证流程图

```
ConfigValidator.validate(config)
    ↓
阶段 1: StructureValidator
    ├─ 验证顶级模块存在性
    ├─ 验证模块类型
    └─ 验证必需字段
    ↓
阶段 2: CollectionValidator
    ├─ 收集所有层名称
    ├─ 收集所有步骤名称
    └─ 检测名称重复
    ↓
阶段 3: ExecutionValidator
    ├─ 验证模型的执行单元
    ├─ 验证层的执行单元
    └─ 验证步骤的执行单元
    ↓
阶段 4: BridgeValidator & ConnectionValidator（并行）
    ├─ 验证 Bridge 表达式
    │   ├─ 语法解析
    │   ├─ 目标验证
    │   ├─ 条件验证
    │   └─ 参数验证
    └─ 验证 Connection 表达式
        ├─ 语法解析
        ├─ 目标验证
        └─ 参数验证
    ↓
阶段 5: CrossReferenceValidator
    ├─ 验证训练模式
    ├─ 验证变量引用
    └─ 验证逻辑一致性
    ↓
返回验证结果
    ├─ is_valid: bool
    ├─ errors: List[str]
    └─ warnings: List[str]
```

### 依赖关系

```
阶段 1 (StructureValidator)
    ↓ (必须先通过基本结构检查)
阶段 2 (CollectionValidator)
    ↓ (收集名称供后续使用)
阶段 3 (ExecutionValidator)
    ↓ (并行)
阶段 4 (BridgeValidator + ConnectionValidator)
    ↓ (依赖阶段 2 收集的名称)
阶段 5 (CrossReferenceValidator)
    ↓ (依赖阶段 2 收集的名称)
完成
```

---

## 📊 验证示例

### 配置示例

```yaml
training_mode:
  type: supervised

models:
  classifier:
    type: sequential
    layers:
      - name: input
        reflection: tensorflow.keras.layers.InputLayer
        args:
          input_shape: [784]

      - name: dense1
        reflection: tensorflow.keras.layers.Dense
        args:
          units: 128
        connection: "@SEQUENTIAL"

      - name: residual
        reflection: tensorflow.keras.layers.Dense
        args:
          units: 128
        connection: "@RESIDUAL:dense1"

training_pipeline:
  supervised:
    step_sequence:
      - name: load_batch
        reflection: common.utils:load_batch
        args:
          data_source: ${mnist}

      - name: forward_pass
        reflection: common.utils:forward
        args:
          model: ${classifier}
        bridge: "@JUMP:backward_pass?${loss}<0.5"
```

### 验证过程

#### 阶段 1: StructureValidator
```
✅ 检查 training_mode 存在
✅ 检查 models 存在
✅ 检查 training_mode.type 存在
```

#### 阶段 2: CollectionValidator
```
✅ 收集模型 'classifier' 的层名称: {input, dense1, residual}
✅ 收集流程 'supervised' 的步骤名称: {load_batch, forward_pass}
```

#### 阶段 3: ExecutionValidator
```
✅ 验证 input 层的 reflection: tensorflow.keras.layers.InputLayer
✅ 验证 dense1 层的 args 是字典
✅ 验证 load_batch 步骤的 reflection
```

#### 阶段 4: BridgeValidator & ConnectionValidator
```
✅ 解析 Connection: @RESIDUAL:dense1
✅ 验证目标层 'dense1' 存在
✅ 解析 Bridge: @JUMP:backward_pass?${loss}<0.5
⚠️  Bridge 目标步骤 'backward_pass' 不存在 (需要添加)
```

#### 阶段 5: CrossReferenceValidator
```
✅ 训练模式 'supervised' 受支持
✅ 流程 'supervised' 已定义
⚠️  引用 ${mnist} 未定义 (需要在 data_sources 中定义)
```

---

## 💡 使用建议

### 1. 扩展验证器

如果需要添加新的验证逻辑：

```python
# 1. 创建新的验证器类
class MyCustomValidator(BaseValidator):
    def validate(self, config: dict):
        # 实现验证逻辑
        if some_condition:
            self.add_error("错误信息")
        if some_warning:
            self.add_warning("警告信息")

# 2. 在 ConfigValidator 中注册
class ConfigValidator:
    def __init__(self):
        # ... 其他验证器
        self.my_custom_validator = MyCustomValidator(self)

    def validate(self, config):
        # ... 其他阶段
        self.my_custom_validator.validate(config)
```

### 2. 添加新的常量

在 `constants.py` 中添加：

```python
class MyConstants:
    SUPPORTED_FORMATS = {"json", "yaml", "xml"}
    MAX_LAYERS = 100
```

### 3. 添加新的正则模式

在 `patterns.py` 中添加：

```python
class ReferencePattern:
    # ... 现有模式
    MY_PATTERN = r'...'
    MY_PATTERN_COMPILED = re.compile(MY_PATTERN)
```

---

## 🎯 总结

### 核心优势

1. **模块化**: 每个验证器职责单一，易于维护
2. **可扩展**: 易于添加新的验证逻辑
3. **分阶段**: 确保验证依赖关系正确
4. **状态共享**: 通过协调器共享验证状态
5. **错误收集**: 统一的错误报告机制

### 文件对应关系

| 文件 | 阶段 | 职责 |
|------|------|------|
| `base_validator.py` | 基础 | 提供基类和通用接口 |
| `constants.py` | 基础 | 定义验证常量 |
| `patterns.py` | 基础 | 定义正则模式 |
| `structure_validator.py` | 1 | 验证基本结构 |
| `collection_validator.py` | 2 | 收集名称 |
| `execution_validator.py` | 3 | 验证执行单元 |
| `bridge_validator.py` | 4 | 验证 Bridge 表达式 |
| `connection_validator.py` | 4 | 验证 Connection 表达式 |
| `cross_ref_validator.py` | 5 | 验证交叉引用 |

### 验证层级

```
第 1 层：模块和字段存在性
第 2 层：名称收集和重复检测
第 3 层：执行单元格式
第 4 层：表达式深度验证
第 5 层：引用完整性和逻辑一致性
```

这种分层设计确保了配置验证的**全面性**、**准确性**和**可维护性**。
