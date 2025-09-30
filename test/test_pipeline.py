# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
基于 config/config_test.yaml 的端到端流程测试：
1. 训练流程（train）
2. 评估流程（evaluate）
"""

import os
import pytest
import tensorflow as tf
import yaml
from main import main


@pytest.fixture
def test_config_path():
    """返回测试用的 config 文件路径"""
    return os.path.join("config", "config_test.yaml")


# -----------------------------
# 测试 1️⃣: 训练流程
# -----------------------------
def test_train_pipeline(tmp_path, test_config_path, monkeypatch):
    """
    使用 config_test.yaml 执行训练流程
    """

    # 修改模型保存目录到临时目录，避免污染真实 models/
    save_dir = tmp_path / "saved_model"
    monkeypatch.setenv("MODEL_SAVE_DIR", str(save_dir))

    # 执行 main.py 的 train
    main(["train", "--config", test_config_path])

    # 验证保存目录下是否有模型文件
    assert save_dir.exists()
    assert any(save_dir.iterdir()), "模型未正确保存"


# -----------------------------
# 测试 2️⃣: 评估流程
# -----------------------------
def test_evaluate_pipeline(tmp_path, test_config_path, monkeypatch):
    """
    使用 config_test.yaml 执行评估流程
    """

    # 先跑一遍训练，确保有模型
    save_dir = tmp_path / "saved_model"
    monkeypatch.setenv("MODEL_SAVE_DIR", str(save_dir))
    main(["train", "--config", test_config_path])

    # 执行 main.py 的 evaluate
    main(["evaluate", "--config", test_config_path])

    # 验证模型目录仍存在
    assert save_dir.exists()
