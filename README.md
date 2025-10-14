# 深度学习训练框架 - 项目结构设计

## 📁 项目目录结构

```
LearnAI/
├── main.py                          # 主入口：读取配置，调度各模块
├── config/
│   ├── config_example.yaml         # 示例配置文件
│   ├── supervised_config.yaml      # 监督学习配置
│   ├── reinforcement_config.yaml   # 强化学习配置
│   └── self_supervised_config.yaml # 自监督学习配置
├── common/
│   ├── __init__.py
│   ├── common.py                   # 基础公共模块（日志、反射、配置加载）
│   └── utils.py                    # 通用工具函数（可被反射调用）
├── modules/                        # 核心模块（每个对应配置文件的一级节点）
│   ├── __init__.py
│   ├── data_manager.py            # 数据管理模块
│   ├── models.py                  # 模型构建模块
│   ├── optimizers.py              # 优化器模块
│   ├── losses.py                  # 损失函数模块
│   ├── training_pipeline.py       # 训练流程模块
│   ├── evaluation.py              # 评估模块
│   ├── reward_functions.py        # 奖励函数模块（强化学习）
│   ├── export.py                  # 模型导出模块
│   └── deployment.py              # 部署模块
├── data/                          # 数据目录
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── checkpoints/                   # 检查点目录
├── logs/                         # 日志目录
└── outputs/                      # 输出目录
    ├── onnx/
    └── saved_model/
```

## 🎯 核心设计理念

### 1. 配置驱动
- **main.py** 只负责读取配置文件，调度各模块
- 所有逻辑通过 **reflection** 调用，无需修改代码

### 2. 模块职责清晰
每个模块对应配置文件的一级节点：

| 配置节点 | 模块文件 | 职责 |
|---------|---------|------|
| `data_manager` | `modules/data_manager.py` | 数据加载、增强、预处理 |
| `models` | `modules/models.py` | 模型构建、管理 |
| `optimizers` | `modules/optimizers.py` | 优化器管理 |
| `losses` | `modules/losses.py` | 损失函数管理 |
| `training_pipeline` | `modules/training_pipeline.py` | 训练循环逻辑 |
| `evaluation` | `modules/evaluation.py` | 评估逻辑 |
| `reward_functions` | `modules/reward_functions.py` | 强化学习奖励 |
| `export` | `modules/export.py` | 模型导出 |
| `deployment` | `modules/deployment.py` | 模型部署 |

### 3. 函数调用层次
```
配置文件 (YAML)
    ↓ (reflection 反射调用)
modules/* (核心模块) ← 优先调用
    ↓ (内部调用)
common/utils.py (通用工具) ← 被模块调用
    ↓ (基础支持)
common/common.py (基础功能) ← 被工具调用
```

## 📋 各模块详细设计

### modules/data_manager.py
**职责**：数据加载、增强、预处理
```python
# 核心函数
def load_supervised_data(config: dict) -> Dataset
def load_rl_data(config: dict) -> NetworkClient
def load_unsupervised_data(config: dict) -> Dataset

# 数据增强逻辑（内部使用）
class DataAugmentation:
    - apply_image_augmentation()
    - apply_text_augmentation()
    - create_contrastive_pair()
```

### modules/models.py
**职责**：模型构建和管理
```python
def build_model(config: dict) -> tf.keras.Model
def load_pretrained_model(path: str) -> tf.keras.Model
def save_model(model, path: str) -> None
```

### modules/losses.py
**职责**：损失函数管理和自定义损失
```python
# TensorFlow内置损失（通过反射调用）
# 自定义损失类
class ContrastiveLoss(tf.keras.losses.Loss)
class MultiTaskLoss(tf.keras.losses.Loss)
class FocalLoss(tf.keras.losses.Loss)

# 损失聚合
def aggregate_multi_task_losses(losses, weights) -> tf.Tensor
```

### modules/training_pipeline.py
**职责**：训练循环的执行逻辑
```python
# 主训练循环
def run_supervised_training(config: dict) -> None
def run_reinforcement_training(config: dict) -> None
def run_self_supervised_training(config: dict) -> None

# 单步执行（由主循环调用）
def execute_training_step(step_config: dict, context: dict) -> Any
```

### modules/evaluation.py
**职责**：模型评估
```python
def evaluate_supervised(model, dataset, metrics) -> dict
def evaluate_reinforcement(policy, env, episodes) -> dict
def compute_metrics(predictions, targets, metric_names) -> dict
```

### common/utils.py
**职责**：被模块调用的通用工具函数
```python
# 数据处理
def build_csv_dataset(...)
def build_numpy_dataset(...)
def build_web_dataset(...)

# 强化学习工具
def compute_reward(...)
def select_action(...)
def compute_gae(...)

# 推理和优化
def forward(...)
def compute_gradients(...)
def apply_gradients(...)

# 聚类工具（被data_manager调用）
def kmeans_cluster(...)
```

## 🔄 执行流程

```
1. main.py 启动
   ↓
2. 读取配置文件 (common.load_yaml)
   ↓
3. 根据 training_mode.type 选择训练流程
   ↓
4. 初始化各模块
   ├─ data_manager.load_data(config['data_manager'])
   ├─ models.build_model(config['models'])
   ├─ optimizers.build_optimizer(config['optimizers'])
   └─ losses.build_loss(config['losses'])
   ↓
5. 执行训练循环
   training_pipeline.run_training(config['training_pipeline'])
   ↓
6. 评估
   evaluation.evaluate(config['evaluation'])
   ↓
7. 导出和部署
   ├─ export.export_model(config['export'])
   └─ deployment.deploy(config['deployment'])
```

## 🎨 配置文件映射关系

```yaml
# config_example.yaml

data_manager:              # → modules/data_manager.py
  supervised_source:
    train:
      reflection: modules.data_manager:load_csv_data

models:                    # → modules/models.py
  classifier:
    reflection: modules.models:build_sequential_model

optimizers:                # → modules/optimizers.py
  main_optimizer:
    reflection: tensorflow.keras.optimizers.Adam

losses:                    # → modules/losses.py
  classification_loss:
    reflection: tensorflow.keras.losses.CategoricalCrossentropy
  # 或自定义
  contrastive_loss:
    reflection: modules.losses:ContrastiveLoss

training_pipeline:         # → modules/training_pipeline.py
  supervised:
    reflection: modules.training_pipeline:run_supervised_training

evaluation:                # → modules/evaluation.py
  supervised_eval:
    reflection: modules.evaluation:evaluate_supervised

reward_functions:          # → modules/reward_functions.py (仅强化学习)
  compute_reward:
    reflection: modules.reward_functions:compute_game_reward

export:                    # → modules/export.py
  reflection: modules.export:export_to_onnx

deployment:                # → modules/deployment.py
  reflection: modules.deployment:deploy_rest_api
```

## ✅ 关键原则

1. **配置文件是唯一的控制中心**
   - 所有行为通过配置文件定义
   - main.py 不包含业务逻辑

2. **模块职责单一**
   - 每个模块只负责一类功能
   - 模块之间通过 main.py 协调

3. **反射调用优先级**
   - 优先调用 modules/* 中的函数
   - modules 内部可调用 common.utils
   - 避免循环依赖

4. **扩展性**
   - 新增训练方式：只需添加配置文件和对应的 training_pipeline 函数
   - 新增模块：在 modules/ 下新建文件
   - 新增工具函数：在 common/utils.py 中添加

5. **清晰的边界**
   - **modules/** = 业务逻辑
   - **common/utils.py** = 可复用的工具函数
   - **common/common.py** = 基础设施（日志、反射、配置）

## 📝 下一步工作

1. 重写 `main.py` - 配置驱动的主控制器
2. 重写 `modules/data_manager.py` - 数据管理（包含增强逻辑）
3. 重写 `modules/losses.py` - 损失函数（包含对比损失、多任务）
4. 重写 `modules/training_pipeline.py` - 训练流程控制
5. 重写 `modules/evaluation.py` - 评估逻辑
6. 调整 `common/utils.py` - 只保留纯工具函数
7. 更新 `config_example.yaml` - 反映新的模块结构
