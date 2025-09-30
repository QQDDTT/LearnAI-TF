# -*- coding: utf-8 -*-
"""
models/model.py
TensorFlow 模型定义
"""

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=10):
        """
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出类别数
        """
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu", input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)
