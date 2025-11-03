# LearnAI Deep Learning Training Framework

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Configuration-Driven · Modular Design · Multi-Mode Training · Production-Ready**

[Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation) • [Examples](#examples) • [Contributing](#contributing)

</div>

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Configuration Files](#configuration-files)
- [Supported Training Modes](#supported-training-modes)
- [Model Export & Deployment](#model-export--deployment)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [FAQ](#faq)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

**LearnAI** is an enterprise-grade deep learning training framework based on TensorFlow 2.x, featuring a configuration-driven architecture that supports multiple training paradigms including supervised learning, reinforcement learning, and self-supervised learning. The framework defines the entire training process through YAML configuration files, enabling complex deep learning tasks without code modification.

### Why Choose LearnAI?

- ✅ **Zero-Code Training**: Complete model training through YAML configuration alone
- ✅ **Multi-Mode Support**: Supports 7 mainstream training paradigms
- ✅ **Production-Ready**: Built-in model export, deployment, and monitoring
- ✅ **Highly Flexible**: Reflection mechanism supports dynamic loading of any Python component
- ✅ **Enterprise-Grade**: Complete test coverage, logging system, and error handling

---

## Core Features

### 🎯 Configuration-Driven Architecture

```yaml
# A single configuration file defines the complete training process
global:
  name: "image_classifier"
  version: "v1.0.0"

training_mode:
  type: "supervised"

models:
  classifier:
    # Use reflection mechanism to dynamically load models
    reflection: "tensorflow.keras.Sequential"
    layers:
      - name: "conv1"
        reflection: "tensorflow.keras.layers.Conv2D"
        args: {filters: 32, kernel_size: [3,3]}
```

### 🚀 Supported Capabilities

| Feature Category | Supported Content |
|-----------------|-------------------|
| **Training Modes** | Supervised, Reinforcement, Unsupervised, Self-Supervised, Semi-Supervised, Multi-Task, Custom |
| **Data Sources** | CSV, NumPy, Image Directories, TFRecord, Network API, Custom Loaders |
| **Model Architectures** | Keras Sequential, Functional API, Model Subclassing, Pre-trained Models, Custom Models |
| **Optimizers** | Adam, SGD, RMSprop, AdaGrad + all TensorFlow optimizers with learning rate scheduling |
| **Loss Functions** | TensorFlow built-in losses + custom losses (Contrastive, Focal, Multi-task, etc.) |
| **Export Formats** | SavedModel, ONNX, TensorFlow Lite, H5, Weights Only |
| **Deployment Methods** | REST API, gRPC, TensorFlow Serving, Docker, Custom Deployment |

### 🛠️ Reflection Mechanism

Dynamically call any Python class or function through the `reflection` field:

```yaml
# Call TensorFlow components
reflection: "tensorflow.keras.optimizers.Adam"

# Call custom functions
reflection: "modules.custom:my_training_function"

# Call third-party libraries
reflection: "sklearn.preprocessing.StandardScaler"
```

### 📊 Advanced Training Flow Control

Bridge expressions for conditional control:

```yaml
step_sequence:
  - name: "validation"
    reflection: "modules.evaluation:validate"
    bridge: "@skip:validation?${epoch}%10!=0"  # Validate every 10 epochs

  - name: "early_stop"
    reflection: "common.utils:check_convergence"
    bridge: "@jump:save_model?${accuracy}>0.95"  # Jump when target reached
```

---

## System Architecture

### Design Philosophy

```
┌─────────────────────────────────────────────────────────┐
│                    YAML Configuration                    │
│       (Single control center, defines all behavior)      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ (Load and validate)
┌─────────────────────────────────────────────────────────┐
│                    main.py                               │
│        (Main entry point, coordinates modules)           │
└────┬────────┬──────────┬──────────┬──────────┬──────────┘
     │        │          │          │          │
     ↓        ↓          ↓          ↓          ↓
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Data    │ Models  │Optimizer│ Losses  │Training │
│ Manager │ Builder │ Manager │ Manager │Pipeline │
└─────────┴─────────┴─────────┴─────────┴─────────┘
     ↓        ↓          ↓          ↓          ↓
┌──────────────────────────────────────────────────┐
│       common/utils.py (Utility Library)          │
│   • forward()  • compute_loss()  • backward()    │
└──────────────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────────────┐
│        TensorFlow 2.x / Third-party Libraries    │
└──────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Responsibilities |
|--------|------|-----------------|
| **Main Controller** | `main.py` | Read config, coordinate execution, manage lifecycle |
| **Data Management** | `modules/data_manager.py` | Data loading, preprocessing, augmentation |
| **Model Building** | `modules/models.py` | Model creation, layer definition, architecture management |
| **Optimizer Management** | `modules/optimizers.py` | Optimizer config, learning rate scheduling |
| **Loss Functions** | `modules/losses.py` | Loss computation, custom losses |
| **Training Pipeline** | `modules/training_pipeline.py` | Training loop, Bridge control, checkpoints |
| **Model Evaluation** | `modules/evaluation.py` | Metric computation, model validation |
| **Model Export** | `modules/export.py` | Multi-format export, model optimization |
| **Model Deployment** | `modules/deployment.py` | Service deployment, API creation |
| **Common Utilities** | `common/common.py` | Logging, reflection, config loading |
| **Training Context** | `common/train_context.py` | State management, variable storage |

---

## Quick Start

### Prerequisites

- Python 3.9+
- TensorFlow 2.16.1
- 4GB+ RAM (8GB+ recommended)
- (Optional) NVIDIA GPU with CUDA 12.3

### 30-Second Quick Experience

```bash
# 1. Clone repository
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run example
python main.py config/config_example.yaml
```

### Complete Example: Image Classification

```bash
# 1. Prepare data
mkdir -p data/train data/val
# Place images in respective directories

# 2. Create configuration file
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

# 3. Start training
python main.py config/my_classifier.yaml --export

# 4. View results
ls outputs/onnx/
```

---

## Installation

### Quick Install (Minimal Dependencies)

```bash
pip install tensorflow numpy pandas pyyaml colorama requests
```

### Complete Install (All Features)

```bash
pip install -r requirements.txt
```

### GPU Support

```bash
# TensorFlow with CUDA
pip install tensorflow[and-cuda]==2.16.1

# Requires CUDA 12.3 and cuDNN 8.9
```

### China Mirror Acceleration

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### Development Environment

```bash
# Clone repository
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Includes testing and code quality tools

# Run tests
pytest test/
```

---

## Usage Guide

### Command Line Arguments

```bash
python main.py <config_file> [options]

Required:
  config_file          Configuration file path (YAML)

Optional:
  --export             Export model after training
  --deploy             Deploy model after export
  --deploy-only        Deploy only, skip training
  --skip-eval          Skip evaluation phase
  --checkpoint-dir     Checkpoint directory
  --verbose            Verbose logging
  --dry-run            Validate config only, don't execute training
```

### Typical Workflow

#### 1. Configuration Validation

```bash
# Validate configuration file
python main.py config/my_config.yaml --dry-run
```

#### 2. Train Model

```bash
# Standard training
python main.py config/my_config.yaml

# Train + Export
python main.py config/my_config.yaml --export

# Train + Export + Deploy
python main.py config/my_config.yaml --export --deploy
```

#### 3. Deploy Existing Model Only

```bash
python main.py config/my_config.yaml --deploy-only
```

#### 4. Debug Mode

```bash
python main.py config/my_config.yaml --verbose
```

---

## Configuration Files

### Configuration File Structure

For detailed configuration documentation, see: [Configuration File Structure Documentation](docs/配置文件结构说明文档.md)

```yaml
global:           # Global config (project name, version, random seed)
training_mode:    # Training mode (supervised/reinforcement/unsupervised, etc.)
models:           # Model definitions
data_manager:     # Data management
optimizers:       # Optimizer configuration
losses:           # Loss function configuration
training_pipeline:# Training process
evaluation:       # Evaluation configuration (optional)
export:           # Model export configuration (optional)
deployment:       # Model deployment configuration (optional)
```

### Configuration Template Generation

```python
from common.common import generate_config_template

# Generate supervised learning template
generate_config_template("supervised", "my_supervised_config.yaml")

# Generate reinforcement learning template
generate_config_template("reinforcement", "my_rl_config.yaml")
```

---

## Supported Training Modes

### 1. Supervised Learning

For image classification, object detection, text classification, etc.

**Example Config:** `config/supervised_config.yaml`

**Features:**
- Epoch + Batch training loop
- Labeled data
- Cross-entropy loss

### 2. Reinforcement Learning

For game AI, robot control, autonomous driving, etc.

**Example Config:** `config/reinforcement_config.yaml`

**Features:**
- Episode + Step training loop
- Reward signals
- Policy Gradient/Q-Learning

### 3. Unsupervised Learning

For clustering, dimensionality reduction, anomaly detection, etc.

**Features:**
- Iterative training
- Unlabeled data
- Reconstruction/Clustering loss

### 4. Self-Supervised Learning

For contrastive learning, masked prediction, image inpainting, etc.

**Example Config:** `config/self_supervised_config.yaml`

**Features:**
- Contrastive loss
- Data augmentation
- Pre-training + Fine-tuning

### 5. Semi-Supervised Learning

For scenarios with limited labeled data.

**Features:**
- Labeled + Unlabeled data
- Pseudo-labeling
- Consistency regularization

### 6. Multi-Task Learning

Train multiple related tasks simultaneously.

**Features:**
- Shared encoder
- Multiple task heads
- Weighted losses

### 7. Custom Training

Fully custom training logic.

**Features:**
- Free-form loop definition
- Custom step sequences
- Bridge control flow

---

## Model Export & Deployment

### Supported Export Formats

| Format | Purpose | File Extension |
|--------|---------|---------------|
| **SavedModel** | TensorFlow Serving production | Directory structure |
| **ONNX** | Cross-platform deployment (multi-framework) | `.onnx` |
| **TensorFlow Lite** | Mobile and embedded devices | `.tflite` |
| **H5** | Keras standard format | `.h5` |
| **Weights Only** | Save weights only | `.weights` |

### Export Example

```yaml
export:
  # SavedModel format (TensorFlow Serving)
  export_savedmodel:
    model: "${classifier}"
    format: "savedmodel"
    output_path: "outputs/savedmodel/classifier"

  # ONNX format (cross-platform)
  export_onnx:
    model: "${classifier}"
    format: "onnx"
    output_path: "outputs/onnx/classifier.onnx"
    args:
      opset_version: 13

  # TFLite format (mobile)
  export_tflite:
    model: "${classifier}"
    format: "tflite"
    output_path: "outputs/tflite/classifier.tflite"
    args:
      optimizations: ["DEFAULT"]
```

### Supported Deployment Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **REST API** | Flask server | Web application integration |
| **gRPC** | High-performance RPC | Microservices architecture |
| **TensorFlow Serving** | Official model serving | Production environment |
| **Docker** | Containerized deployment | Cloud platforms |
| **Custom** | Reflection-based custom functions | Special requirements |

### Deployment Example

```yaml
deployment:
  # REST API deployment
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

### Quick Start Service

```bash
# Train and deploy
python main.py config/my_config.yaml --export --deploy

# Deploy existing model only
python main.py config/my_config.yaml --deploy-only
```

---

## Project Structure

```
LearnAI/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependency list
├── README.md                        # This file
│
├── config/                          # Configuration directory
│   ├── config_example.yaml         # Example config
│   ├── supervised_config.yaml      # Supervised learning
│   ├── reinforcement_config.yaml   # Reinforcement learning
│   └── self_supervised_config.yaml # Self-supervised learning
│
├── common/                          # Common modules
│   ├── __init__.py
│   ├── common.py                   # Base functions (logging, reflection, config)
│   ├── utils.py                    # Utility functions
│   ├── train_context.py            # Training context
│   ├── interfaces.py               # Interface definitions
│   ├── config_validator.py         # Config validation
│   └── validators/                 # Validator submodules
│
├── modules/                         # Core modules
│   ├── __init__.py
│   ├── data_manager.py             # Data management
│   ├── models.py                   # Model building
│   ├── optimizers.py               # Optimizer management
│   ├── losses.py                   # Loss functions
│   ├── training_pipeline.py        # Training pipeline
│   ├── evaluation.py               # Model evaluation
│   ├── export.py                   # Model export
│   └── deployment.py               # Model deployment
│
├── lib/                             # Third-party library wrappers
│   ├── __init__.py
│   └── deployment.py               # Deployment tools (Flask/gRPC)
│
├── data/                            # Data directory
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                     # Checkpoint directory
├── logs/                           # Log directory
├── outputs/                        # Output directory
│   ├── onnx/
│   ├── savedmodel/
│   ├── tflite/
│   └── h5/
│
├── test/                           # Test code
│   ├── test_config.py
│   ├── test_modules.py
│   └── config_test.yaml
│
└── docs/                           # Documentation
    ├── Configuration_Structure_EN.md
    └── AI_Platform_Roadmap_EN.md
```

---

## Development Guide

### Adding Custom Components

#### 1. Custom Data Loader

Add to `common/utils.py` or `modules/data_manager.py`:

```python
def load_my_custom_data(file_path: str, batch_size: int = 32):
    """Custom data loading function"""
    # Implement data loading logic
    dataset = ...
    return dataset
```

Use in config:

```yaml
data_manager:
  custom_source:
    train:
      reflection: "common.utils:load_my_custom_data"
      args:
        file_path: "data/my_data.txt"
        batch_size: 32
```

#### 2. Custom Loss Function

Add to `modules/losses.py`:

```python
import tensorflow as tf

class MyCustomLoss(tf.keras.losses.Loss):
    """Custom loss function"""

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Implement loss computation
        loss = ...
        return loss
```

Use in config:

```yaml
losses:
  custom_loss:
    reflection: "modules.losses:MyCustomLoss"
    args:
      alpha: 1.5
```

### Code Standards

This project follows PEP 8 style guide.

```bash
# Code formatting
black .

# Code linting
flake8 .

# Type checking
mypy .
```

### Testing

```bash
# Run all tests
pytest test/

# Run specific test
pytest test/test_modules.py::TestConfigLoading

# View test coverage
pytest --cov=. --cov-report=html
```

---

## FAQ

### Q1: How to debug configuration files?

**A:** Use `--dry-run` option to validate config:

```bash
python main.py config.yaml --dry-run
```

### Q2: How to use multi-GPU training?

**A:** Set distributed strategy in config:

```yaml
global:
  distributed_strategy:
    type: "MirroredStrategy"
    devices: ["GPU:0", "GPU:1"]
```

### Q3: How to implement model hot reload?

**A:** Enable auto-reload in deployment config:

```yaml
deployment:
  rest_api:
    auto_reload: true
    reload_interval: 60  # Check every 60 seconds
```

### Q4: How to save checkpoints during training?

**A:** Add save step in training pipeline:

```yaml
training_pipeline:
  supervised:
    step_sequence:
      # ... training steps

      - name: "save_checkpoint"
        reflection: "common.utils:save_checkpoint"
        args:
          model: "${main_model}"
          epoch: "${current_epoch}"
        bridge: "@skip:save_checkpoint?${epoch}%5!=0"  # Save every 5 epochs
```

### Q5: How to integrate my own deep learning framework?

**A:** Use reflection mechanism to call any Python library:

```yaml
models:
  my_pytorch_model:
    reflection: "my_pytorch_module:MyModel"
    args:
      input_dim: 784
      output_dim: 10
```

### Q6: How to handle imbalanced datasets?

**A:** Use class weights or resampling:

```yaml
losses:
  weighted_loss:
    reflection: "tensorflow.keras.losses.CategoricalCrossentropy"

training_pipeline:
  supervised:
    parameters:
      class_weight: {0: 1.0, 1: 2.0, 2: 3.0}  # Increase weight for minority classes
```

### Q7: How to implement early stopping?

**A:** Use Bridge conditional control:

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

## Changelog

### v2.0.0 (2025-11-03)

#### New Features
- ✨ Complete configuration validation system (multi-layer validators)
- ✨ Bridge expression support (conditional control, loops, branches)
- ✨ Model export supports 5 formats (SavedModel/ONNX/TFLite/H5/Weights)
- ✨ Model deployment supports 5 methods (REST API/gRPC/TF Serving/Docker/Custom)
- ✨ Unified training context management (TrainContext)
- ✨ Complete test suite (37 test cases)

#### Improvements
- 🔧 Refactored module architecture with clear responsibilities
- 🔧 Improved logging system with color output and file logging
- 🔧 Optimized reflection mechanism for flexible parameter passing
- 🔧 Enhanced error handling and exception information

#### Documentation
- 📚 Added configuration file structure documentation
- 📚 Added AI-driven platform evolution roadmap
- 📚 Improved README and API documentation

### v1.0.0 (2025-10-15)

#### Initial Release
- 🎉 Basic configuration-driven architecture
- 🎉 Support for supervised, reinforcement, self-supervised learning
- 🎉 Reflection mechanism implementation
- 🎉 Basic model export functionality

---

## Contributing

We welcome all forms of contributions!

### How to Contribute

1. **Fork the project**

```bash
git clone https://github.com/your-username/LearnAI.git
```

2. **Create feature branch**

```bash
git checkout -b feature/your-feature-name
```

3. **Commit changes**

```bash
git commit -m "Add: Add some feature"
```

4. **Push to branch**

```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request**

Create PR on GitHub describing your changes.

### Commit Message Convention

```
Add: New feature
Fix: Bug fix
Docs: Documentation update
Style: Code formatting
Refactor: Code refactoring
Test: Testing related
Chore: Build or auxiliary tool changes
```

### Code Review Standards

- ✅ Follow PEP 8 code style
- ✅ Add necessary tests
- ✅ Update related documentation
- ✅ Ensure all tests pass
- ✅ Add clear comments

---

## Roadmap

### Near-term (3-6 months)

- [ ] Microservices architecture refactoring
- [ ] Distributed training support (Horovod/Ray)
- [ ] Web UI console
- [ ] Experiment tracking system (MLflow integration)
- [ ] Containerized deployment (Docker + Kubernetes)

### Mid-term (6-12 months)

- [ ] AutoML functionality (NAS + hyperparameter optimization)
- [ ] Intelligent data generation (GAN/Diffusion)
- [ ] Model compression and quantization
- [ ] Performance predictor
- [ ] A/B testing support

### Long-term Vision (12-24 months)

- [ ] LLM-driven configuration generation
- [ ] Reinforcement learning auto-tuning
- [ ] Autonomous task discovery
- [ ] Fully autonomous training system

See: [AI-Driven Automated ML Platform Evolution Roadmap](docs/AI驱动的自动化机器学习平台演进路线图.md)

---

## Community & Support

### Get Help

- 📖 [Documentation](docs/)
- 💬 [GitHub Discussions](https://github.com/qqddtt/LearnAI/discussions)
- 🐛 [Issue Tracker](https://github.com/qqddtt/LearnAI/issues)
- 📧 Email: support@learnai.org

### Join Community

- ⭐ Star the project
- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🤝 Submit Pull Requests

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

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

## Acknowledgments

Thanks to the following open source projects and contributors:

- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - High-level API
- [ONNX](https://onnx.ai/) - Model interchange format
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Ray](https://www.ray.io/) - Distributed computing framework
- [MLflow](https://mlflow.org/) - Experiment tracking system

Thanks to all developers who contributed to this project!

---

## Citation

If you use LearnAI in your research, please cite:

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

**⭐ If this project helps you, please give us a Star! ⭐**

Made with ❤️ by [LearnAI Team](https://github.com/qqddtt)

[Back to Top](#learnai-deep-learning-training-framework)

</div>
