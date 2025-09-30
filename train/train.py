# -*- coding: utf-8 -*-
"""
train/train.py
训练主入口（TensorFlow 版）
"""

import tensorflow as tf
from common.utils import load_config, call_target, get_logger


def main():
    # 1. 加载配置
    config = load_config()
    logger = get_logger()

    # 2. 加载数据
    data_loader = call_target(config["data"]["class_path"], config["data"]["params"])
    train_dataset, input_dim, num_classes = data_loader.load_data()
    logger.info(f"数据加载成功, input_dim={input_dim}, num_classes={num_classes}")

    # 3. 初始化模型
    model = call_target(config["model"]["class_path"], {
        **config["model"]["params"],
        "input_dim": input_dim,
        "output_dim": num_classes
    })
    logger.info(f"模型加载成功: {model.__class__.__name__}")

    # 4. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["train"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. 训练模型
    history = model.fit(
        train_dataset,
        epochs=config["train"]["epochs"]
    )

    # 6. 保存模型
    model.save("models/saved_model")
    logger.info("模型训练完成并保存到 models/saved_model")


if __name__ == "__main__":
    main()
