# -*- coding: utf-8 -*-
"""
data/dataloader.py
数据加载与预处理
支持本地加载 / Web 加载
"""

import requests
import os
import pandas as pd
import tensorflow as tf


class DataLoader:
    def __init__(self, source_type="local", source_path=None, api_url=None, batch_size=32):
        """
        :param source_type: 数据来源 (local / web)
        :param source_path: 本地数据路径
        :param api_url: Web API 地址
        :param batch_size: 每批大小
        """
        self.source_type = source_type
        self.source_path = source_path
        self.api_url = api_url
        self.batch_size = batch_size

    def load_data(self):
        """根据配置加载数据并转换为 TensorFlow Dataset"""
        if self.source_type == "local":
            df = self._load_local()
        elif self.source_type == "web":
            df = self._load_web()
        else:
            raise ValueError(f"未知数据来源: {self.source_type}")

        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].values.astype("int32")
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(df)).batch(self.batch_size)
        return dataset, X.shape[1], len(set(y))

    def _load_local(self):
        """加载本地 CSV/JSON"""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"文件未找到: {self.source_path}")

        if self.source_path.endswith(".csv"):
            return pd.read_csv(self.source_path)
        elif self.source_path.endswith(".json"):
            return pd.read_json(self.source_path)
        else:
            raise ValueError("仅支持 CSV / JSON")

    def _load_web(self):
        """从 Web API 获取数据"""
        response = requests.get(self.api_url)
        response.raise_for_status()
        return pd.DataFrame(response.json())
