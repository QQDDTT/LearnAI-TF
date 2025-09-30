# -*- coding: utf-8 -*-
"""
train/evaluate.py
模型评估脚本
"""

import tensorflow as tf
from common.utils import load_config, get_logger, call_target


def main():
    config = load_config()
    logger = get_logger()

    # 加载数据
    data_loader = call_target(config["data"]["class_path"], config["data"]["params"])
    test_dataset, input_dim, num_classes = data_loader.load_data()

    # 加载模型
    model = tf.keras.models.load_model("models/saved_model")
    logger.info("模型加载完成")

    # 评估
    loss, acc = model.evaluate(test_dataset)
    logger.info(f"评估结果 - Loss: {loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
