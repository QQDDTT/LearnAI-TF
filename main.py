# -*- coding: utf-8 -*-
"""
main.py
统一入口：支持训练和评估
- 使用方式：
    python main.py train --config config/config.yaml
    python main.py evaluate --config config/config.yaml
"""

import yaml
import argparse
import os
import tensorflow as tf

from data.dataloader import load_data
from models.model import build_model


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="LearnAI-TF 项目入口")
    parser.add_argument("task", type=str, choices=["train", "evaluate"],
                        help="运行任务: train / evaluate")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径 (默认: config/config.yaml)")
    return parser.parse_args()


def main():
    # 1️⃣ 解析命令行参数
    args = parse_args()

    # 2️⃣ 读取配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)
    model_cfg = config.get("model", {"hidden_units": 128, "dropout": 0.2})

    # 3️⃣ 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 4️⃣ 构建模型
    model = build_model(model_cfg)

    # 5️⃣ 根据任务选择逻辑
    if args.task == "train":
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # 训练模型
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test)
        )

        # 保存模型
        saved_model_dir = os.path.join("models", "saved_model")
        os.makedirs(saved_model_dir, exist_ok=True)
        model.save(saved_model_dir)
        print(f"✅ 模型已保存到 {saved_model_dir}")

    elif args.task == "evaluate":
        # 加载已保存的模型
        saved_model_dir = os.path.join("models", "saved_model")
        if not os.path.exists(saved_model_dir):
            raise FileNotFoundError("未找到已保存模型，请先运行 train")

        model = tf.keras.models.load_model(saved_model_dir)

        # 评估模型
        loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print(f"📊 评估结果 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
