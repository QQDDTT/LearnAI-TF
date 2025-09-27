# LearnAI-TF 项目目录


# 项目目录结构

```bash
LearnAI-TF/
├── data/                         # 数据集目录
│   ├── raw/                      # 原始数据（不可修改）
│   ├── processed/                # 处理后的数据
│   └── dataloader.py             # 数据加载 & 预处理逻辑
│
├── models/                       # 模型相关
│   ├── checkpoints/              # 训练中间保存点
│   ├── saved_model/              # 最终导出的模型（部署用）
│   └── model.py                  # 模型结构定义
│
├── config/                       # 配置文件
│   └── config.yaml               # 超参数配置
│
├── train/                        # 训练与评估脚本
│   ├── train.py                  # 主训练脚本
│   ├── evaluate.py               # 测试/验证脚本
│   └── utils.py                  # 公共工具函数（日志、可视化）
│
├── notebooks/                    # Jupyter Notebook 实验
│   └── exploration.ipynb         # 数据探索与实验
│
├── requirements.txt              # Python 依赖清单
├── README.md                     # 项目说明文档
└── main.py                       # 统一入口（可选）
```