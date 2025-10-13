# 完整的机器学习训练方式分类体系

> **重要前提**：深度学习 vs 浅层学习只影响**模型构筑**，不影响**训练流程**。
> 训练方式由**数据来源、标签来源、反馈机制**决定。

---

## 一、训练方式的根本分类维度

```
┌─────────────────────────────────────┐
│   机器学习训练方式                   │
├─────────────────────────────────────┤
│ 按监督程度：有监督 ↔ 无监督 ↔ 半监督  │
│ 按反馈方式：标签 ↔ 奖励 ↔ 自生成       │
│ 按数据流：离线 ↔ 在线 ↔ 流式           │
│ 按学习目标：预测 ↔ 决策 ↔ 表示         │
└─────────────────────────────────────┘
```

---

## 二、主要训练方式详细列表

### 1. 监督学习（Supervised Learning）

#### 特征
- ✅ 数据+标签配对（X, Y）
- ✅ 标签预先已知/人工标注
- ✅ 离线学习
- ✅ 即时反馈

#### 训练流程
```
加载批次(X, Y) → Forward → 计算Loss(Y_pred, Y) → Backward → 更新参数
```

#### 常见算法
- 分类：逻辑回归、SVM、决策树、随机森林
- 回归：线性回归、岭回归
- 神经网络：MLP、CNN、RNN

#### 配置示例
```yaml
training:
  training_type: supervised
  data:
    loader: csv_loader
    has_labels: true
  steps:
    - compute_loss
    - backprop
```

#### 代码
```python
for epoch in epochs:
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)  # ← 标签已知
        loss.backward()
        optimizer.step()
```

---

### 2. 无监督学习（Unsupervised Learning）

#### 特征
- ❌ 无标签数据
- ✅ 自生成目标
- ✅ 离线学习
- ✅ 内部反馈（如重构误差）

#### 子类型

##### 2.1 聚类（Clustering）
```
无标签数据 → 模型 → 自动分组
```

**常见算法**：K-means、GMM、DBSCAN

**流程**：
```
初始化簇心 → 分配样本 → 重新计算簇心 → 收敛
```

**配置**：
```yaml
training:
  training_type: unsupervised_clustering
  algorithm: kmeans
  params:
    n_clusters: 5
  steps:
    - assign_clusters
    - update_centroids
```

##### 2.2 降维（Dimensionality Reduction）
```
高维数据 → 模型 → 低维表示
```

**常见算法**：PCA、t-SNE、自编码器

**流程（以自编码器为例）**：
```
数据 → 编码器 → 低维向量 → 解码器 → 重构数据
      ← 最小化重构误差 ←
```

**配置**：
```yaml
training:
  training_type: unsupervised_autoencoder
  model:
    encoder:
      layers: [input:784 → 128 → 64]
    decoder:
      layers: [64 → 128 → 784]
  steps:
    - forward_encoder_decoder
    - compute_reconstruction_loss
    - backprop
```

**代码**：
```python
for epoch in epochs:
    for X_batch in data_loader:  # ← 无标签
        encoded = encoder(X_batch)
        decoded = decoder(encoded)
        loss = mse(decoded, X_batch)  # 自生成目标：重构
        loss.backward()
        optimizer.step()
```

---

### 3. 强化学习（Reinforcement Learning）

#### 特征
- ❌ 无预标签
- ✅ 奖励反馈
- ✅ 交互式学习
- ✅ 延迟反馈

#### 训练流程
```
Agent观察状态 → 选择动作 → 环境返回(奖励, 新状态)
      ← Agent根据奖励更新策略 ←
```

#### 主要算法

##### 3.1 值方法（Value-Based）
目标：学习最优价值函数 V(s) 或 Q(s,a)

```
Q-learning: Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
```

**配置**：
```yaml
training:
  training_type: reinforcement_learning
  algorithm: q_learning
  params:
    discount_factor: 0.99
    learning_rate: 0.01
  steps:
    - observe_state
    - select_action_epsilon_greedy
    - execute_action
    - get_reward_next_state
    - compute_td_error
    - update_q_table
```

##### 3.2 策略方法（Policy-Based）
目标：直接优化策略 π(a|s)

```
Policy Gradient: ∇J(θ) = ∇log π(a|s) * R(τ)
```

**配置**：
```yaml
training:
  training_type: reinforcement_learning
  algorithm: policy_gradient
  steps:
    - observe_state
    - forward_policy_network
    - sample_action_from_distribution
    - execute_action
    - get_reward_trajectory
    - compute_policy_gradient_loss
    - backprop
```

##### 3.3 行动者-评论家方法（Actor-Critic）
结合值方法和策略方法

```
Actor（策略）：选择动作
Critic（价值）：评估动作好坏
```

**配置**：
```yaml
training:
  training_type: reinforcement_learning
  algorithm: actor_critic
  model:
    actor: policy_network
    critic: value_network
  steps:
    - observe_state
    - actor_predict_action
    - execute_action
    - critic_evaluate_value
    - compute_advantage: reward + γ*V(s') - V(s)
    - update_actor: policy_gradient * advantage
    - update_critic: mse(advantage)
```

**代码**：
```python
for episode in episodes:
    state = env.reset()
    for step in max_steps:
        action = actor(state)  # 策略网络
        next_state, reward, done = env.step(action)

        value = critic(state)  # 价值网络
        next_value = critic(next_state)
        advantage = reward + γ*next_value - value

        actor_loss = -log_prob(action) * advantage
        critic_loss = advantage**2

        backprop(actor_loss + critic_loss)
```

---

### 4. 自监督学习（Self-Supervised Learning）

#### 特征
- ❌ 无人工标签
- ✅ 从数据本身生成标签
- ✅ 离线学习
- ✅ 即时反馈

#### 常见方法

##### 4.1 对比学习（Contrastive Learning）
```
同一样本的两个增强视图 → 模型 → 应该相近
不同样本的增强视图    → 模型 → 应该相远
```

**配置**：
```yaml
training:
  training_type: self_supervised_contrastive
  augmentation:
    - random_crop
    - random_flip
    - random_color_jitter
  steps:
    - augment_data_twice
    - forward_both_views
    - compute_contrastive_loss
    - backprop
```

**代码**：
```python
for batch in data_loader:
    # 同一样本的两个增强版本（自动生成标签概念）
    x_i = augment(batch)
    x_j = augment(batch)

    z_i = encoder(x_i)
    z_j = encoder(x_j)

    # 对比损失：相似度最大化
    loss = contrastive_loss(z_i, z_j)
    loss.backward()
```

##### 4.2 掩码语言建模（MLM，如BERT）
```
文本 "我爱[MASK]学习" → 模型 → 预测[MASK]=深度
```

**配置**：
```yaml
training:
  training_type: self_supervised_mlm
  masking_ratio: 0.15
  steps:
    - mask_tokens
    - forward_model
    - compute_mlm_loss
    - backprop
```

##### 4.3 自编码变分（VAE）
```
数据 → 编码为分布 → 采样 → 解码 → 重构
      ← 最小化重构误差 + KL散度 ←
```

---

### 5. 半监督学习（Semi-Supervised Learning）

#### 特征
- 🟡 部分标注、部分无标签
- ✅ 结合监督+无监督
- ✅ 离线学习

#### 常见方法

##### 5.1 伪标签（Pseudo-Labeling）
```
有标签数据 → 训练 → 模型
无标签数据 → 模型预测 → 伪标签 → 作为标签重新训练
```

**配置**：
```yaml
training:
  training_type: semi_supervised_pseudo_labeling
  labeled_ratio: 0.1
  confidence_threshold: 0.9
  steps:
    - train_on_labeled
    - predict_on_unlabeled
    - filter_high_confidence
    - add_pseudo_labels
    - retrain_on_all
```

**代码**：
```python
labeled_loader, unlabeled_loader = split_data()

for epoch in epochs:
    # 第1步：在有标签数据上训练
    for X, y in labeled_loader:
        y_pred = model(X)
        loss = cross_entropy(y_pred, y)
        loss.backward()

    # 第2步：在无标签数据上生成伪标签
    pseudo_labels = []
    for X_unlabeled in unlabeled_loader:
        y_pred = model(X_unlabeled)
        if max(y_pred) > confidence_threshold:
            pseudo_labels.append(argmax(y_pred))

    # 第3步：混合训练
    for X, y_pseudo in zip(unlabeled_loader, pseudo_labels):
        y_pred = model(X)
        loss = cross_entropy(y_pred, y_pseudo)
        loss.backward()
```

##### 5.2 一致性正则化（Consistency Regularization）
```
有标签数据 → 标准监督损失
无标签数据 → 两个增强版本应预测一致
```

**配置**：
```yaml
training:
  training_type: semi_supervised_consistency
  steps:
    - supervised_loss_labeled
    - consistency_loss_unlabeled
    - combine: λ1*sup_loss + λ2*cons_loss
```

---

### 6. 多任务学习（Multi-Task Learning）

#### 特征
- ✅ 同时学习多个相关任务
- ✅ 共享表示，任务特定头
- ✅ 离线学习

#### 训练流程
```
共享编码器 → Task1头 → Loss1
         → Task2头 → Loss2
         → Task3头 → Loss3
总Loss = α1*Loss1 + α2*Loss2 + α3*Loss3
```

**配置**：
```yaml
training:
  training_type: multi_task
  tasks:
    - name: task_classification
      weight: 0.5
    - name: task_segmentation
      weight: 0.3
    - name: task_depth_estimation
      weight: 0.2
  steps:
    - forward_shared_encoder
    - forward_task1_head
    - forward_task2_head
    - forward_task3_head
    - compute_combined_loss
    - backprop
```

---

### 7. 迁移学习（Transfer Learning）

#### 特征
- ✅ 先在源任务预训练
- ✅ 再在目标任务微调
- ✅ 离线学习

#### 训练流程
```
阶段1：源任务（大数据集）
ImageNet → 训练 → 预训练权重

阶段2：目标任务（小数据集）
医学图像 → 加载预训练权重 → 微调
```

**配置**：
```yaml
training:
  training_type: transfer_learning
  stages:
    - name: pretraining
      dataset: imagenet
      epochs: 100
      freeze_layers: none

    - name: fine_tuning
      dataset: medical_images
      epochs: 20
      freeze_layers: [0, 1, 2]  # 冻结前3层
      learning_rate: 0.0001
```

---

### 8. 元学习（Meta-Learning）

#### 特征
- ✅ 学习如何学习
- ✅ 快速适应新任务
- ✅ 小样本学习

#### 常见算法：MAML、Prototypical Networks

**配置**：
```yaml
training:
  training_type: meta_learning
  algorithm: maml
  params:
    inner_lr: 0.01
    outer_lr: 0.001
    inner_steps: 5
  steps:
    - sample_task
    - inner_loop_optimize
    - outer_loop_optimize
```

---

### 9. 在线学习（Online Learning）

#### 特征
- ✅ 数据流式到达
- ✅ 每次1条或小批量
- ✅ 立即更新
- ✅ 适应非平稳分布

#### 训练流程
```
对于每条数据：
  预测 → 观察标签 → 更新 → 下一条数据
```

**配置**：
```yaml
training:
  training_type: online_learning
  data:
    stream: true
    batch_size: 1
  steps:
    - fetch_single_instance
    - predict
    - get_label
    - update_model
```

**代码**：
```python
for data_point in streaming_data:
    y_pred = model(data_point)
    loss = loss_fn(y_pred, true_label)
    loss.backward()
    optimizer.step()  # ← 每条数据都更新
```

---

### 10. 主动学习（Active Learning）

#### 特征
- 🟡 半监督
- ✅ 模型主动选择标注哪些样本
- ✅ 减少标注成本

#### 训练流程
```
模型 → 评估不确定性 → 选择最难样本 → 人工标注
   ← 重新训练 ←
```

**配置**：
```yaml
training:
  training_type: active_learning
  selection_strategy: uncertainty_sampling
  params:
    initial_labeled: 0.05
    query_size: 100
  steps:
    - train_on_labeled
    - evaluate_uncertainty_unlabeled
    - query_top_uncertain
    - get_labels_human
    - retrain
```

---

### 11. 对抗训练（Adversarial Training）

#### 特征
- ✅ 两个网络对抗
- ✅ 生成器 vs 判别器
- ✅ 离线学习

#### 常见算法：GAN

**配置**：
```yaml
training:
  training_type: adversarial
  algorithm: gan
  model:
    generator: gen_network
    discriminator: disc_network
  steps:
    - sample_noise
    - generator_forward
    - discriminator_real
    - discriminator_fake
    - compute_gen_loss
    - compute_disc_loss
    - update_generator
    - update_discriminator
```

---

### 12. 课程学习（Curriculum Learning）

#### 特征
- ✅ 按难度递进学习
- ✅ 从易到难
- ✅ 提升训练稳定性

**配置**：
```yaml
training:
  training_type: supervised_curriculum
  curriculum:
    - stage: easy
      epochs: 10
      data_filter: difficulty < 0.3
    - stage: medium
      epochs: 15
      data_filter: 0.3 <= difficulty < 0.7
    - stage: hard
      epochs: 20
      data_filter: difficulty >= 0.7
  steps:
    - load_curriculum_stage
    - train_supervised
```

---

## 三、训练方式对比矩阵

| 方式 | 标签需求 | 数据流 | 反馈 | 交互 | 典型应用 |
|-----|--------|--------|------|------|--------|
| **监督学习** | ✅必需 | 离线 | 即时标签 | ❌ | 分类、回归 |
| **无监督学习** | ❌无 | 离线 | 内部损失 | ❌ | 聚类、降维 |
| **强化学习** | ❌无 | 交互 | 延迟奖励 | ✅ | 游戏、机器人 |
| **自监督学习** | ❌无 | 离线 | 自生成目标 | ❌ | 预训练、表示学习 |
| **半监督学习** | 🟡部分 | 离线 | 混合 | ❌ | 标注稀缺场景 |
| **多任务学习** | ✅必需 | 离线 | 多目标 | ❌ | 多个相关任务 |
| **迁移学习** | 🟡部分 | 离线 | 混合 | ❌ | 小样本学习 |
| **元学习** | 🟡部分 | 离线 | 梯度 | ❌ | 快速适应 |
| **在线学习** | ✅必需 | 流式 | 即时标签 | ❌ | 流式数据 |
| **主动学习** | 🟡部分 | 混合 | 即时标签 | ✅ | 成本优化 |
| **对抗训练** | 🟡部分 | 离线 | 对抗 | ❌ | 生成任务 |
| **课程学习** | ✅必需 | 离线 | 难度递进 | ❌ | 难任务学习 |

---

## 四、针对你的项目的建议

根据你的描述："每次请求仅能获取一条数据，查看执行结果还需要一次请求"

**这是典型的在线强化学习**，但也可以组合其他方式：

```yaml
training:
  training_type: "online_reinforcement_learning"
  # 可选结合：对抗训练（GAN）
  # 可选结合：课程学习（从简单游戏到复杂）
  # 可选结合：主动学习（选择最难样本）
```

关键是你的配置架构支持**任意训练方式**，因为：
- ✅ 模型构筑（深度/浅层）独立
- ✅ 数据流、奖励、目标通过配置定义
- ✅ Python代码只需实现不同的主循环逻辑
