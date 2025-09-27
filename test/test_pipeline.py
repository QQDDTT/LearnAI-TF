import pytest
from data.dataloader import load_data
from models.model import build_model
import tensorflow as tf
import yaml

def test_data_loading():
    """测试数据加载是否正常"""
    (x_train, y_train), (x_test, y_test) = load_data()
    assert x_train.shape[0] > 0, "训练数据为空"
    assert x_test.shape[0] > 0, "测试数据为空"
    assert x_train.max() <= 1.0 and x_train.min() >= 0.0, "数据未归一化"

def test_model_build():
    """测试模型构建是否成功"""
    cfg = {"hidden_units": 64, "dropout": 0.2}
    model = build_model(cfg)
    assert isinstance(model, tf.keras.Model), "模型未正确构建"
    assert model.layers[-1].output_shape[-1] == 10, "输出层类别数应为10"

def test_training_step():
    """测试单步训练是否能跑通"""
    (x_train, y_train), _ = load_data()
    model = build_model({"hidden_units": 32, "dropout": 0.1})
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    
    # 跑一个小 batch 确认训练能运行
    history = model.fit(x_train[:100], y_train[:100], epochs=1, batch_size=32, verbose=0)
    assert "loss" in history.history, "训练未返回 loss"
