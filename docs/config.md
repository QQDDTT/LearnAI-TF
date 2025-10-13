# 通用配置设计指南 - 适配所有训练方式

## 核心设计原则

### 问题分析
**原配置的限制**：
```
model → data(train/val/test) → training → evaluation
```
这个结构假定了**批处理的离线监督学习**。

**新的要求**：
- 强化学习是**交互式**，每次请求一条数据
- 无监督学习**没有标签**，需要自生成目标
- 自监督学习需要**数据增强**
- 多任务学习需要**并行处理**多个任务头

---

## 新框架的11层结构

```
┌─────────────────────────────────────┐
│ 1. 全局配置（Global）                │
├─────────────────────────────────────┤
│ 2. 训练方式声明（Training Mode）     │ ← 关键：告诉系统用哪种训练方式
├─────────────────────────────────────┤
│ 3. 模型定义（Models）                 │ ← 与训练方式无关
├─────────────────────────────────────┤
│ 4. 优化器定义（Optimizers）           │
├─────────────────────────────────────┤
│ 5. 损失函数定义（Losses）             │
├─────────────────────────────────────┤
│ 6. 数据管理（Data Manager）          │ ← 根据训练方式选择数据源
├─────────────────────────────────────┤
│ 7. 训练流程定义（Training Pipeline） │ ← 核心创新：不同方式用不同流程
├─────────────────────────────────────┤
│ 8. 奖励函数（Reward Functions）      │ ← 仅强化学习使用
├─────────────────────────────────────┤
│ 9. 评估流程（Evaluation）            │
├─────────────────────────────────────┤
│ 10. 导出配置（Export）               │
├─────────────────────────────────────┤
│ 11. 部署配置（Deployment）           │
└─────────────────────────────────────┘
```

---

## 关键创新点

### 1. 训练方式声明（第2层）

```yaml
training_mode:
  type: "supervised"  # 这是关键，告诉Python选择哪种训练流程
  subtype: null       # 细分方式（如 q_learning, policy_gradient 等）
```

**Python中的处理**：
```python
if config.training_mode.type == "supervised":
    use_pipeline("supervised")
elif config.training_mode.type == "reinforcement":
    use_pipeline("reinforcement")
elif config.training_mode.type == "unsupervised_clustering":
    use_pipeline("unsupervised_clustering")
# ... 等等
```

### 2. 数据管理解耦（第6层）

**原设计**：数据硬编码为 train/val/test

**新设计**：根据训练方式选择不同的数据源

```yaml
data_manager:
  supervised_source:    # 监督学习：批加载
    train: {...}
    val: {...}
    test: {...}

  rl_source:           # 强化学习：交互式
    environment: {...}
    request: {...}
    feedback: {...}

  unsupervised_source: # 无监督：无标签
    reflection: ...

  self_supervised_source:  # 自监督：带增强
    reflection: ...
    augmentation: {...}
```

**Python选择逻辑**：
```python
if training_mode == "supervised":
    loader = load_supervised_source()
elif training_mode == "reinforcement":
    env = init_rl_environment()
elif training_mode == "self_supervised":
    loader = load_self_supervised_source()
```

### 3. 训练流程定义（第7层）- 最核心的创新

**原设计**：假设固定的 Forward → Loss → Backward 流程

**新设计**：每种训练方式有自己的 `step_sequence`

#### 监督学习流程
```yaml
supervised:
  loop_type: "epoch_batch"  # 外层epoch，内层batch
  parameters:
    epochs: 100

  step_sequence:
    - fetch_batch
    - forward_pass
    - compute_loss
    - backward_pass
    - update_params

  loop_condition:
    check_type: "epoch_based"
```

#### 强化学习流程
```yaml
reinforcement:
  loop_type: "episode_step"  # 外层episode，内层step
  parameters:
    episodes: 1000
    steps_per_episode: 500

  step_sequence:
    - observe_state        # ← 强化学习特有
    - select_action        # ← 强化学习特有
    - execute_action       # ← 强化学习特有
    - get_feedback         # ← 强化学习特有
    - store_experience     # ← 强化学习特有
    - compute_td_error
    - update_networks
```

#### 无监督学习（聚类）流程
```yaml
unsupervised_clustering:
  loop_type: "iteration"  # 迭代直到收敛
  parameters:
    max_iterations: 100
    convergence_threshold: 1e-4

  step_sequence:
    - load_data
    - assign_clusters      # ← 聚类特有
    - update_centroids     # ← 聚类特有
    - check_convergence    # ← 聚类特有

  loop_condition:
    check_type: "convergence_based"  # ← 不是epoch-based！
```

---

## 使用示例

### 示例1：监督学习（GAN）

```yaml
training_mode:
  type: "supervised"  # 实际上是多任务（两个网络）

models:
  generator: {...}
  discriminator: {...}

training_pipeline:
  supervised:
    parameters:
      epochs: 100
    step_sequence:
      - compute_gen_loss
      - update_generator
      - compute_disc_loss
      - update_discriminator
```

### 示例2：强化学习（游戏AI）

```yaml
training_mode:
  type: "reinforcement"
  subtype: "actor_critic"

models:
  actor: {...}
  critic: {...}

data_manager:
  rl_source:
    environment: GameEnvironment
    request: fetch_state_from_server
    feedback: submit_action_to_server

training_pipeline:
  reinforcement:
    parameters:
      episodes: 1000
      discount_factor: 0.99
    step_sequence:
      - observe_state
      - select_action
      - execute_action
      - get_feedback
      - compute_td_error
      - update_networks
```

### 示例3：自监督学习（对比学习）

```yaml
training_mode:
  type: "self_supervised"

data_manager:
  self_supervised_source:
    augmentation:
      - RandomCrop
      - RandomFlip
      - RandomColorJitter

training_pipeline:
  self_supervised:
    step_sequence:
      - fetch_batch
      - augment_data        # 创建两个视图
      - forward_view1
      - forward_view2
      - compute_contrastive_loss
      - backward_pass
      - update_params
```

---

## Python实现框架

```python
def universal_train_loop(config):
    """统一的训练入口"""

    # 1. 解析训练方式
    training_mode = config.training_mode.type

    # 2. 初始化模型、数据、优化器
    models = build_models(config.models)
    data = load_data_source(config.data_manager, training_mode)
    optimizers = build_optimizers(config.optimizers)

    # 3. 选择对应的训练流程
    pipeline = config.training_pipeline[training_mode]

    # 4. 获取循环参数
    loop_type = pipeline.loop_type  # "epoch_batch" / "episode_step" / "iteration"
    parameters = pipeline.parameters

    # 5. 执行训练流程
    if loop_type == "epoch_batch":
        for epoch in range(parameters.epochs):
            for batch in data:
                execute_step_sequence(pipeline.step_sequence, {
                    "models": models,
                    "batch": batch,
                    "optimizers": optimizers
                })

    elif loop_type == "episode_step":
        for episode in range(parameters.episodes):
            for step in range(parameters.steps_per_episode):
                execute_step_sequence(pipeline.step_sequence, {...})

    elif loop_type == "iteration":
        iteration = 0
        while not converged:
            execute_step_sequence(pipeline.step_sequence, {...})
            iteration += 1

def execute_step_sequence(steps, context):
    """通用的step执行器（反射调用）"""

    last_result = None
    for step_config in steps:
        # 反射调用
        func = import_function(step_config.reflection)

        # 参数绑定（支持 ${变量} 引用）
        args = bind_args(step_config.args, context, last_result)

        # 执行函数
        last_result = func(**args)

        # 保存结果供下一步使用
        context[step_config.name] = last_result
```

---

## 总结：为什么这个设计适配所有方式

| 特性 | 原设计 | 新设计 |
|-----|--------|--------|
| **数据源** | 固定 train/val/test | 根据方式选择（supervised/rl/unsupervised/...） |
| **循环结构** | 固定 epoch-batch | 灵活定义（epoch_batch/episode_step/iteration） |
| **反馈机制** | 标签 | 标签/奖励/自生成（配置定义） |
| **循环条件** | 固定 epoch-based | 灵活定义（epoch/convergence/episode） |
| **模型数量** | 单一 | 多个（shared encoder + task heads） |
| **优化器** | 单一 | 多个（不同模块独立优化） |
| **评估方式** | 固定精度指标 | 灵活评估（监督/强化/无监督各不同） |

### 关键改进对照

**原设计的局限**：
```yaml
stages:
  - model: {...}
    data:           # 假设有标签！
      train: {...}
    training:       # 假设 epoch-batch！
      epochs: 3
    evaluation:     # 假设有val集！
      dataset: val
```

**新设计的灵活性**：
```yaml
training_mode:
  type: "reinforcement"  # 明确告诉系统用什么方式

models:
  actor: {...}
  critic: {...}

data_manager:
  rl_source:           # 没有train/val/test！
    environment: {...}
    request: {...}
    feedback: {...}

training_pipeline:
  reinforcement:       # 特定的训练流程
    loop_type: "episode_step"
    step_sequence:
      - observe_state
      - select_action
      - ...
```

---

## 最佳实践建议

### 1. 为每种训练方式设计 step_sequence

**监督学习**：预测 → 计算损失 → 反向传播 → 更新

**强化学习**：观察 → 选择 → 执行 → 获得奖励 → 存储 → 计算误差 → 更新

**无监督**：聚类或降维需要迭代收敛

**自监督**：需要数据增强生成对比对

**多任务**：共享层 + 任务特定头 + 加权损失

### 2. 使用模板变量实现参数动态绑定

```yaml
# 参考其他字段
learning_rate: "${config.training_pipeline.supervised.parameters.base_lr}"

# 引用运行时变量
epsilon: "${epsilon}"
session_id: "${session_id}"

# 引用上一步结果
inputs: "last_result"
```

### 3. 支持条件表达式

```yaml
loop_condition:
  check_type: "convergence_based"
  # 支持 ==, !=, >, <, >=, <= 和逻辑运算
  convergence_field: "last_result.converged"
```

### 4. 模型复用策略

**共享编码器**：
```yaml
models:
  shared_encoder: {...}
  task1_head: {...}
  task2_head: {...}
```

**独立模型**：
```yaml
models:
  actor: {...}
  critic: {...}
```

### 5. 优化器独立配置

```yaml
optimizers:
  actor_opt: {...}    # Actor有自己的学习率
  critic_opt: {...}   # Critic有自己的学习率
  encoder_opt: {...}  # 编码器有自己的学习率
```

这样可以为不同部分设置不同的学习策略。

---

## 迁移指南：从旧配置到新配置

### 步骤1：识别训练方式

```python
# 旧配置
if "generator" in models and "discriminator" in models:
    training_mode = "adversarial"  # 实际是多任务的特殊情况
elif "actor" in models and "critic" in models:
    training_mode = "reinforcement"
```

### 步骤2：重组数据源

```yaml
# 旧配置
data:
  train:
    file_path: "data/train.csv"
  val:
    file_path: "data/val.csv"

# 新配置
data_manager:
  supervised_source:
    train:
      reflection: modules.utils.build_csv_loader
      args:
        file_path: "data/train.csv"
    val:
      reflection: modules.utils.build_csv_loader
      args:
        file_path: "data/val.csv"
```

### 步骤3：定义训练流程

```yaml
# 旧配置（只能处理简单的epoch-batch）
training:
  epochs: 3
  steps: [...]

# 新配置（支持任意流程）
training_pipeline:
  supervised:
    loop_type: "epoch_batch"
    parameters:
      epochs: 3
    step_sequence:
      - fetch_batch
      - forward_pass
      - compute_loss
      - backward_pass
      - update_params
    loop_condition:
      check_type: "epoch_based"
      max_epochs: 3
```

---

## 配置验证清单

创建配置时检查：

- ✅ `training_mode.type` 已设置（决定使用哪个pipeline）
- ✅ `models` 中包含所有需要的模型
- ✅ `optimizers` 为每个模型配置优化器
- ✅ `losses` 包含所有需要的损失函数
- ✅ `data_manager` 有对应方式的数据源
- ✅ `training_pipeline[type]` 定义了完整的 `step_sequence`
- ✅ `loop_condition` 与 `loop_type` 匹配
- ✅ `step_sequence` 中的 `reflection` 都是有效函数
- ✅ 参数通过 `args` 正确传递或通过 `${...}` 引用
- ✅ `evaluation` 配置适配训练方式

---

## 常见问题

### Q1: 如何处理动态参数（如 epsilon 衰减）？

**A**: 在Python中动态更新，而不是在配置中：

```python
def train_rl():
    epsilon = 1.0
    for episode in range(episodes):
        context = {
            "epsilon": epsilon,
            "episode": episode
        }
        execute_step_sequence(steps, context)
        epsilon *= epsilon_decay
```

### Q2: 如何在两个网络之间共享梯度？

**A**: 在 `step_sequence` 中明确指定：

```yaml
- name: "update_actor"
  reflection: modules.optimizer.apply_gradients
  args:
    optimizer: "actor_opt"
    loss: "actor_loss"
    model: "actor"

- name: "update_critic"
  reflection: modules.optimizer.apply_gradients
  args:
    optimizer: "critic_opt"
    loss: "critic_loss"
    model: "critic"
```

### Q3: 如何实现经验回放（Replay Buffer）？

**A**: 在 `step_sequence` 中添加存储步骤：

```yaml
- name: "store_experience"
  reflection: modules.rl.store_transition
  args:
    buffer: "replay_buffer"
    state: "observe_state.state"
    action: "select_action.action"

- name: "sample_batch"
  reflection: modules.rl.sample_from_buffer
  args:
    buffer: "replay_buffer"
    batch_size: 32
```

### Q4: 如何处理不同的评估指标？

**A**: 在 `evaluation` 中定义方式特定的评估：

```yaml
evaluation:
  supervised_eval:
    metrics: ["accuracy", "f1", "precision"]

  rl_eval:
    metrics: ["total_reward", "win_rate"]

  clustering_eval:
    metrics: ["silhouette_score", "davies_bouldin_index"]
```

---

## 扩展建议

### 新增训练方式的步骤

1. 在 `training_mode.type` 中添加新值
2. 在 `data_manager` 中定义该方式的数据源
3. 在 `training_pipeline` 中创建新条目，定义 `loop_type` 和 `step_sequence`
4. 在 Python 的 `universal_train_loop` 中处理新的循环逻辑

### 新增的模块函数

只需确保函数签名匹配反射调用的参数即可：

```python
def compute_td_error(algorithm, state, action, reward, next_state, actor, critic):
    """强化学习损失计算"""
    if algorithm == "q_learning":
        ...
    elif algorithm == "actor_critic":
        ...
```

---

## 总结

**这个设计实现了**：
- ✅ **配置驱动**：训练方式由配置决定，不需要修改代码
- ✅ **高度复用**：相同的反射机制适用所有方式
- ✅ **灵活扩展**：新增训练方式只需添加配置
- ✅ **模块解耦**：模型、数据、优化、流程完全分离
- ✅ **参数管理**：统一的参数绑定和引用机制

你的项目现在可以支持**任何机器学习训练方式**了！
