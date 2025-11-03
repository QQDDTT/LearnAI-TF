# LearnAI 深度学习训练框架

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**配置驱动 · 模块化设计 · 多模式训练 · 生产就绪**

[快速开始](#快速开始) • [功能特性](#功能特性) • [文档](#文档) • [示例](#示例) • [贡献](#贡献)

</div>

---

## 📖 目录

- [简介](#简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [安装](#安装)
- [使用指南](#使用指南)
- [配置文件](#配置文件)
- [支持的训练模式](#支持的训练模式)
- [模型导出与部署](#模型导出与部署)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 简介

**LearnAI** 是一个基于 TensorFlow 2.x 的企业级深度学习训练框架，采用配置驱动架构，支持监督学习、强化学习、自监督学习等多种训练范式。框架通过 YAML 配置文件定义全部训练流程，无需修改代码即可完成复杂的深度学习任务。

### 为什么选择 LearnAI？

- ✅ **零代码训练**：仅通过 YAML 配置即可完成模型训练
- ✅ **多模式支持**：支持 7 种主流训练范式
- ✅ **生产就绪**：内置模型导出、部署和监控功能
- ✅ **高度灵活**：反射机制支持动态加载任何 Python 组件
- ✅ **企业级质量**：完整的测试覆盖、日志系统和错误处理

---

## 核心特性

### 🎯 配置驱动架构

```yaml
# 一个配置文件定义完整训练流程
global:
  name: "image_classifier"
  version: "v1.0.0"

training_mode:
  type: "supervised"

models:
  classifier:
    # 使用反射机制动态加载模型
    reflection: "tensorflow.keras.Sequential"
    layers:
      - name: "conv1"
        reflection: "tensorflow.keras.layers.Conv2D"
        args: {filters: 32, kernel_size: [3,3]}
```

### 🚀 支持的功能

| 功能类别 | 支持内容 |
|---------|---------|
| **训练模式** | 监督学习、强化学习、无监督学习、自监督学习、半监督学习、多任务学习、自定义训练 |
| **数据源** | CSV、NumPy、图像目录、TFRecord、网络 API、自定义加载器 |
| **模型架构** | Keras Sequential、Functional API、Model Subclassing、预训练模型、自定义模型 |
| **优化器** | Adam、SGD、RMSprop、AdaGrad 等所有 TensorFlow 优化器 + 学习率调度 |
| **损失函数** | TensorFlow 内置损失 + 自定义损失（对比损失、Focal Loss、多任务损失等） |
| **导出格式** | SavedModel、ONNX、TensorFlow Lite、H5、仅权重 |
| **部署方式** | REST API、gRPC、TensorFlow Serving、Docker、自定义部署 |

### 🛠️ 反射机制

通过 `reflection` 字段动态调用任何 Python 类或函数：

```yaml
# 调用 TensorFlow 组件
reflection: "tensorflow.keras.optimizers.Adam"

# 调用自定义函数
reflection: "modules.custom:my_training_function"

# 调用第三方库
reflection: "sklearn.preprocessing.StandardScaler"
```

### 📊 高级训练流程控制

支持 Bridge 表达式进行条件控制：

```yaml
step_sequence:
  - name: "validation"
    reflection: "modules.evaluation:validate"
    bridge: "@skip:validation?${epoch}%10!=0"  # 每10轮验证一次

  - name: "early_stop"
    reflection: "common.utils:check_convergence"
    bridge: "@jump:save_model?${accuracy}>0.95"  # 达到目标后跳转
```

---

## 系统架构

### 设计理念

```
┌─────────────────────────────────────────────────────────┐
│                    YAML 配置文件                         │
│          (唯一的控制中心，定义所有训练行为)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ (加载和验证)
┌─────────────────────────────────────────────────────────┐
│                    main.py                               │
│            (主入口，协调各模块执行)                        │
└────┬────────┬──────────┬──────────┬──────────┬──────────┘
     │        │          │          │          │
     ↓        ↓          ↓          ↓          ↓
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Data    │ Models  │Optimizer│ Losses  │Training │
│ Manager │ Builder │ Manager │ Manager │Pipeline │
└─────────┴─────────┴─────────┴─────────┴─────────┘
     ↓        ↓          ↓          ↓          ↓
┌──────────────────────────────────────────────────┐
│          common/utils.py (工具函数库)             │
│   • forward()  • compute_loss()  • backward()    │
└──────────────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────────────┐
│        TensorFlow 2.x / 第三方库                  │
└──────────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **主控制器** | `main.py` | 读取配置、协调模块执行、管理生命周期 |
| **数据管理** | `modules/data_manager.py` | 数据加载、预处理、增强 |
| **模型构建** | `modules/models.py` | 模型创建、层定义、架构管理 |
| **优化器管理** | `modules/optimizers.py` | 优化器配置、学习率调度 |
| **损失函数** | `modules/losses.py` | 损失计算、自定义损失 |
| **训练流程** | `modules/training_pipeline.py` | 训练循环、Bridge 控制、检查点 |
| **模型评估** | `modules/evaluation.py` | 指标计算、模型验证 |
| **模型导出** | `modules/export.py` | 多格式导出、模型优化 |
| **模型部署** | `modules/deployment.py` | 服务部署、API 创建 |
| **公共工具** | `common/common.py` | 日志、反射、配置加载 |
| **训练上下文** | `common/train_context.py` | 状态管理、变量存储 |

---

## 快速开始

### 前置要求

- Python 3.9+
- TensorFlow 2.16.1
- 4GB+ RAM（推荐 8GB+）
- （可选）NVIDIA GPU with CUDA 12.3

### 30 秒快速体验

```bash
# 1. 克隆仓库
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行示例
python main.py config/config_example.yaml
```

### 完整示例：图像分类

```bash
# 1. 准备数据
mkdir -p data/train data/val
# 将图像放入对应目录

# 2. 创建配置文件
cat > config/my_classifier.yaml << 'EOF'
global:
  name: "my_image_classifier"
  version: "v1.0.0"
  seed: 42

training_mode:
  type: "supervised"

models:
  classifier:
    reflection: "tensorflow.keras.Sequential"
    layers:
      - name: "conv1"
        reflection: "tensorflow.keras.layers.Conv2D"
        args: {filters: 32, kernel_size: [3,3], activation: "relu"}
      - name: "pool1"
        reflection: "tensorflow.keras.layers.MaxPooling2D"
        args: {pool_size: [2,2]}
      - name: "flatten"
        reflection: "tensorflow.keras.layers.Flatten"
      - name: "dense1"
        reflection: "tensorflow.keras.layers.Dense"
        args: {units: 128, activation: "relu"}
      - name: "output"
        reflection: "tensorflow.keras.layers.Dense"
        args: {units: 10, activation: "softmax"}

data_manager:
  supervised_source:
    train:
      reflection: "tensorflow.keras.preprocessing.image_dataset_from_directory"
      args:
        directory: "data/train"
        image_size: [128, 128]
        batch_size: 32
        label_mode: "categorical"

optimizers:
  main_optimizer:
    reflection: "tensorflow.keras.optimizers.Adam"
    args:
      learning_rate: 0.001

losses:
  classification_loss:
    reflection: "tensorflow.keras.losses.CategoricalCrossentropy"

training_pipeline:
  supervised:
    loop_config:
      type: "epoch_batch"
      parameters:
        epochs: 50
        batch_size: 32
    step_sequence:
      - name: "forward"
        reflection: "common.utils:forward"
        args: {model: "${classifier}", inputs: "${batch_data}"}
      - name: "loss"
        reflection: "common.utils:compute_loss"
        args: {loss_fn: "${classification_loss}", predictions: "${forward}", targets: "${batch_labels}"}
      - name: "backward"
        reflection: "common.utils:compute_gradients"
        args: {loss: "${loss}", model: "${classifier}"}
      - name: "update"
        reflection: "common.utils:apply_gradients"
        args: {optimizer: "${main_optimizer}", gradients: "${backward}", model: "${classifier}"}

evaluation:
  supervised_eval:
    reflection: "modules.evaluation:evaluate_supervised"
    args:
      model: "${classifier}"
      dataset: "${val_data}"
      metrics: ["accuracy", "precision", "recall"]

export:
  export_onnx:
    model: "${classifier}"
    format: "onnx"
    output_path: "outputs/onnx/classifier.onnx"
EOF

# 3. 开始训练
python main.py config/my_classifier.yaml --export

# 4. 查看结果
ls outputs/onnx/
```

---

## 安装

### 快速安装（最小依赖）

```bash
pip install tensorflow numpy pandas pyyaml colorama requests
```

### 完整安装（所有功能）

```bash
pip install -r requirements.txt
```

### GPU 支持

```bash
# TensorFlow with CUDA
pip install tensorflow[and-cuda]==2.16.1

# 需要安装 CUDA 12.3 和 cuDNN 8.9
```

### 国内镜像加速

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 开发环境

```bash
# 克隆仓库
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 包含测试和代码质量工具

# 运行测试
pytest test/
```

---

## 使用指南

### 命令行参数

```bash
python main.py <config_file> [options]

必需参数:
  config_file          配置文件路径 (YAML)

可选参数:
  --export             训练后导出模型
  --deploy             导出后部署模型
  --deploy-only        仅部署，跳过训练
  --skip-eval          跳过评估阶段
  --checkpoint-dir     检查点目录
  --verbose            详细日志输出
  --dry-run            仅验证配置，不执行训练
```

### 典型工作流

#### 1. 配置验证

```bash
# 验证配置文件正确性
python main.py config/my_config.yaml --dry-run
```

#### 2. 训练模型

```bash
# 标准训练
python main.py config/my_config.yaml

# 训练 + 导出
python main.py config/my_config.yaml --export

# 训练 + 导出 + 部署
python main.py config/my_config.yaml --export --deploy
```

#### 3. 仅部署已有模型

```bash
python main.py config/my_config.yaml --deploy-only
```

#### 4. 调试模式

```bash
python main.py config/my_config.yaml --verbose
```

---

## 配置文件

### 配置文件结构

详细的配置文件说明请参考：[配置文件结构说明文档](docs/配置文件结构说明文档.md)

```yaml
global:           # 全局配置（项目名称、版本、随机种子）
training_mode:    # 训练模式（supervised/reinforcement/unsupervised等）
models:           # 模型定义
data_manager:     # 数据管理
optimizers:       # 优化器配置
losses:           # 损失函数配置
training_pipeline:# 训练流程
evaluation:       # 评估配置（可选）
export:           # 模型导出配置（可选）
deployment:       # 模型部署配置（可选）
```

### 配置模板生成

```python
from common.common import generate_config_template

# 生成监督学习模板
generate_config_template("supervised", "my_supervised_config.yaml")

# 生成强化学习模板
generate_config_template("reinforcement", "my_rl_config.yaml")
```

---

## 支持的训练模式

### 1. 监督学习 (Supervised Learning)

适用于图像分类、目标检测、文本分类等任务。

**示例配置：** `config/supervised_config.yaml`

**特点：**
- Epoch + Batch 训练循环
- 标签数据
- 交叉熵损失

### 2. 强化学习 (Reinforcement Learning)

适用于游戏 AI、机器人控制、自动驾驶等任务。

**示例配置：** `config/reinforcement_config.yaml`

**特点：**
- Episode + Step 训练循环
- 奖励信号
- 策略梯度/Q-Learning

### 3. 无监督学习 (Unsupervised Learning)

适用于聚类、降维、异常检测等任务。

**特点：**
- 迭代训练
- 无标签数据
- 重构损失/聚类损失

### 4. 自监督学习 (Self-Supervised Learning)

适用于对比学习、掩码预测、图像修复等任务。

**示例配置：** `config/self_supervised_config.yaml`

**特点：**
- 对比损失
- 数据增强
- 预训练-微调

### 5. 半监督学习 (Semi-Supervised Learning)

适用于少量标注数据的场景。

**特点：**
- 标注 + 未标注数据
- 伪标签
- 一致性正则化

### 6. 多任务学习 (Multi-Task Learning)

同时训练多个相关任务。

**特点：**
- 共享编码器
- 多个任务头
- 加权损失

### 7. 自定义训练 (Custom Training)

完全自定义训练逻辑。

**特点：**
- 自由定义循环
- 自定义步骤序列
- Bridge 控制流

---

## 模型导出与部署

### 支持的导出格式

| 格式 | 用途 | 文件扩展名 |
|------|------|-----------|
| **SavedModel** | TensorFlow Serving 生产环境 | 目录结构 |
| **ONNX** | 跨平台部署（支持多框架） | `.onnx` |
| **TensorFlow Lite** | 移动端和嵌入式设备 | `.tflite` |
| **H5** | Keras 标准格式 | `.h5` |
| **Weights Only** | 仅保存权重 | `.weights` |

### 导出示例

```yaml
export:
  # SavedModel 格式（TensorFlow Serving）
  export_savedmodel:
    model: "${classifier}"
    format: "savedmodel"
    output_path: "outputs/savedmodel/classifier"

  # ONNX 格式（跨平台）
  export_onnx:
    model: "${classifier}"
    format: "onnx"
    output_path: "outputs/onnx/classifier.onnx"
    args:
      opset_version: 13

  # TFLite 格式（移动端）
  export_tflite:
    model: "${classifier}"
    format: "tflite"
    output_path: "outputs/tflite/classifier.tflite"
    args:
      optimizations: ["DEFAULT"]
```

### 支持的部署方式

| 部署方式 | 说明 | 适用场景 |
|---------|------|---------|
| **REST API** | Flask 服务器 | Web 应用集成 |
| **gRPC** | 高性能 RPC 服务 | 微服务架构 |
| **TensorFlow Serving** | 官方模型服务 | 生产环境 |
| **Docker** | 容器化部署 | 云平台 |
| **自定义** | 反射调用自定义函数 | 特殊需求 |

### 部署示例

```yaml
deployment:
  # REST API 部署
  rest_api:
    type: "rest_api"
    model_path: "${export_paths.classifier}"
    host: "0.0.0.0"
    port: 9000
    endpoints:
      predict: "/api/predict"
      health: "/health"
    performance:
      batch_size: 32
      timeout: 30
      workers: 4
```

### 快速启动服务

```bash
# 训练并部署
python main.py config/my_config.yaml --export --deploy

# 仅部署已有模型
python main.py config/my_config.yaml --deploy-only
```

---

## 项目结构

```
LearnAI/
├── main.py                          # 主入口
├── requirements.txt                 # 依赖列表
├── README.md                        # 本文件
│
├── config/                          # 配置文件目录
│   ├── config_example.yaml         # 示例配置
│   ├── supervised_config.yaml      # 监督学习配置
│   ├── reinforcement_config.yaml   # 强化学习配置
│   └── self_supervised_config.yaml # 自监督学习配置
│
├── common/                          # 公共模块
│   ├── __init__.py
│   ├── common.py                   # 基础功能（日志、反射、配置）
│   ├── utils.py                    # 工具函数
│   ├── train_context.py            # 训练上下文
│   ├── interfaces.py               # 接口定义
│   ├── config_validator.py         # 配置验证
│   └── validators/                 # 验证器子模块
│       ├── structure_validator.py
│       ├── collection_validator.py
│       ├── execution_validator.py
│       ├── bridge_validator.py
│       ├── connection_validator.py
│       └── cross_ref_validator.py
│
├── modules/                         # 核心模块
│   ├── __init__.py
│   ├── data_manager.py             # 数据管理
│   ├── models.py                   # 模型构建
│   ├── optimizers.py               # 优化器管理
│   ├── losses.py                   # 损失函数
│   ├── training_pipeline.py        # 训练流程
│   ├── evaluation.py               # 模型评估
│   ├── export.py                   # 模型导出
│   └── deployment.py               # 模型部署
│
├── lib/                             # 第三方库封装
│   ├── __init__.py
│   └── deployment.py               # 部署工具（Flask/gRPC）
│
├── data/                            # 数据目录
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                     # 检查点目录
├── logs/                           # 日志目录
├── outputs/                        # 输出目录
│   ├── onnx/
│   ├── savedmodel/
│   ├── tflite/
│   └── h5/
│
├── test/                           # 测试代码
│   ├── test_config.py
│   ├── test_modules.py
│   └── config_test.yaml
│
└── docs/                           # 文档目录
    ├── 配置文件结构说明文档.md
    └── AI驱动的自动化机器学习平台演进路线图.md
```

---

## 开发指南

### 添加自定义组件

#### 1. 自定义数据加载器

在 `common/utils.py` 或 `modules/data_manager.py` 中添加：

```python
def load_my_custom_data(file_path: str, batch_size: int = 32):
    """自定义数据加载函数"""
    # 实现数据加载逻辑
    dataset = ...
    return dataset
```

配置文件中使用：

```yaml
data_manager:
  custom_source:
    train:
      reflection: "common.utils:load_my_custom_data"
      args:
        file_path: "data/my_data.txt"
        batch_size: 32
```

#### 2. 自定义损失函数

在 `modules/losses.py` 中添加：

```python
import tensorflow as tf

class MyCustomLoss(tf.keras.losses.Loss):
    """自定义损失函数"""

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # 实现损失计算逻辑
        loss = ...
        return loss
```

配置文件中使用：

```yaml
losses:
  custom_loss:
    reflection: "modules.losses:MyCustomLoss"
    args:
      alpha: 1.5
```

#### 3. 自定义训练步骤

在 `common/utils.py` 中添加：

```python
def my_custom_training_step(model, optimizer, loss_fn, batch_data):
    """自定义训练步骤"""
    inputs, labels = batch_data

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
```

配置文件中使用：

```yaml
training_pipeline:
  custom:
    step_sequence:
      - name: "custom_step"
        reflection: "common.utils:my_custom_training_step"
        args:
          model: "${main_model}"
          optimizer: "${main_optimizer}"
          loss_fn: "${main_loss}"
          batch_data: "${current_batch}"
```

### 代码规范

本项目遵循 PEP 8 代码风格指南。

```bash
# 代码格式化
black .

# 代码检查
flake8 .

# 类型检查
mypy .
```

### 测试

```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/test_modules.py::TestConfigLoading

# 查看测试覆盖率
pytest --cov=. --cov-report=html
```

---

## 常见问题

### Q1: 如何调试配置文件？

**A:** 使用 `--dry-run` 选项验证配置：

```bash
python main.py config.yaml --dry-run
```

如果配置有错误，会显示详细的错误信息和修复建议。

### Q2: 如何使用多 GPU 训练？

**A:** 在配置文件中设置分布式策略：

```yaml
global:
  distributed_strategy:
    type: "MirroredStrategy"
    devices: ["GPU:0", "GPU:1"]
```

### Q3: 如何实现模型热更新？

**A:** 在部署配置中启用自动重载：

```yaml
deployment:
  rest_api:
    auto_reload: true
    reload_interval: 60  # 每60秒检查一次
```

### Q4: 训练过程中如何保存检查点？

**A:** 在训练流程中添加保存步骤：

```yaml
training_pipeline:
  supervised:
    step_sequence:
      # ... 训练步骤

      - name: "save_checkpoint"
        reflection: "common.utils:save_checkpoint"
        args:
          model: "${main_model}"
          epoch: "${current_epoch}"
        bridge: "@skip:save_checkpoint?${epoch}%5!=0"  # 每5轮保存
```

### Q5: 如何集成自己的深度学习框架？

**A:** 通过反射机制调用任何 Python 库：

```yaml
models:
  my_pytorch_model:
    reflection: "my_pytorch_module:MyModel"
    args:
      input_dim: 784
      output_dim: 10
```

只需确保相应的 Python 包已安装。

### Q6: 如何处理不平衡数据集？

**A:** 使用类权重或重采样：

```yaml
losses:
  weighted_loss:
    reflection: "tensorflow.keras.losses.CategoricalCrossentropy"
    args:
      from_logits: false

training_pipeline:
  supervised:
    parameters:
      class_weight: {0: 1.0, 1: 2.0, 2: 3.0}  # 为少数类增加权重
```

### Q7: 如何实现早停（Early Stopping）？

**A:** 使用 Bridge 条件控制：

```yaml
step_sequence:
  - name: "check_early_stop"
    reflection: "common.utils:check_convergence"
    args:
      metric: "${val_loss}"
      patience: 10
    bridge: "@jump:save_and_exit?${converged}==true"
```

---

## 更新日志

### v2.0.0 (2025-11-03)

#### 新增功能
- ✨ 完整的配置验证系统（多层验证器）
- ✨ Bridge 表达式支持（条件控制、循环、分支）
- ✨ 模型导出支持 5 种格式（SavedModel/ONNX/TFLite/H5/Weights）
- ✨ 模型部署支持 5 种方式（REST API/gRPC/TF Serving/Docker/Custom）
- ✨ 统一的训练上下文管理（TrainContext）
- ✨ 完整的测试套件（37 个测试用例）

#### 改进
- 🔧 重构模块架构，清晰的职责划分
- 🔧 改进日志系统，支持彩色输出和文件记录
- 🔧 优化反射机制，支持更灵活的参数传递
- 🔧 增强错误处理和异常信息

#### 文档
- 📚 新增配置文件结构说明文档
- 📚 新增 AI 驱动平台演进路线图
- 📚 完善 README 和 API 文档

### v1.0.0 (2025-10-15)

#### 初始版本
- 🎉 基础配置驱动架构
- 🎉 支持监督学习、强化学习、自监督学习
- 🎉 反射机制实现
- 🎉 基础模型导出功能

---

## 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork 项目**

```bash
git clone https://github.com/your-username/LearnAI.git
```

2. **创建特性分支**

```bash
git checkout -b feature/your-feature-name
```

3. **提交更改**

```bash
git commit -m "Add: 添加某某功能"
```

4. **推送到分支**

```bash
git push origin feature/your-feature-name
```

5. **创建 Pull Request**

在 GitHub 上创建 PR，描述您的更改。

### 提交信息规范

```
Add: 新增功能
Fix: 修复 bug
Docs: 文档更新
Style: 代码格式化
Refactor: 代码重构
Test: 测试相关
Chore: 构建或辅助工具的变动
```

### 代码审查标准

- ✅ 遵循 PEP 8 代码风格
- ✅ 添加必要的测试
- ✅ 更新相关文档
- ✅ 确保所有测试通过
- ✅ 添加清晰的注释

---

## 路线图

### 近期计划（3-6 个月）

- [ ] 微服务架构改造
- [ ] 支持分布式训练（Horovod/Ray）
- [ ] Web UI 控制台
- [ ] 实验追踪系统（MLflow 集成）
- [ ] 容器化部署（Docker + Kubernetes）

### 中期计划（6-12 个月）

- [ ] AutoML 功能（NAS + 超参数优化）
- [ ] 智能数据生成（GAN/Diffusion）
- [ ] 模型压缩和量化
- [ ] 性能预测器
- [ ] A/B 测试支持

### 长期愿景（12-24 个月）

- [ ] LLM 驱动的配置生成
- [ ] 强化学习自动调优
- [ ] 自主任务发现
- [ ] 完全自治的训练系统

详见：[AI驱动的自动化机器学习平台演进路线图](docs/AI驱动的自动化机器学习平台演进路线图.md)

---

## 社区与支持

### 获取帮助

- 📖 [文档](docs/)
- 💬 [GitHub Discussions](https://github.com/qqddtt/LearnAI/discussions)
- 🐛 [问题反馈](https://github.com/qqddtt/LearnAI/issues)
- 📧 Email: support@learnai.org

### 参与社区

- ⭐ 给项目点 Star
- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🤝 提交 Pull Request

---

## 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2025 LearnAI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 致谢

感谢以下开源项目和贡献者：

- [TensorFlow](https://www.tensorflow.org/) - 深度学习框架
- [Keras](https://keras.io/) - 高层 API
- [ONNX](https://onnx.ai/) - 模型交换格式
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [Ray](https://www.ray.io/) - 分布式计算框架
- [MLflow](https://mlflow.org/) - 实验追踪系统

感谢所有为本项目做出贡献的开发者！

---

## 引用

如果您在研究中使用了 LearnAI，请引用：

```bibtex
@software{learnai2025,
  title = {LearnAI: A Configuration-Driven Deep Learning Training Framework},
  author = {LearnAI Team},
  year = {2025},
  url = {https://github.com/qqddtt/LearnAI}
}
```

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！ ⭐**

Made with ❤️ by [LearnAI Team](https://github.com/qqddtt)

[返回顶部](#learnai-深度学习训练框架)

</div>
