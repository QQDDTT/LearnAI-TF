# -*- coding: utf-8 -*-
"""
generate_mock_data.py - 测试数据生成器
功能：
  - 生成用于测试的模拟数据集
  - 支持CSV和NumPy格式
  - 可配置数据规模和特征
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple
import argparse


def generate_classification_data(
    num_samples: int,
    num_features: int,
    num_classes: int,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成分类任务的模拟数据

    参数：
        num_samples: 样本数量
        num_features: 特征维度
        num_classes: 类别数量
        random_seed: 随机种子

    返回：
        (特征, 标签) 元组
    """
    np.random.seed(random_seed)

    # 生成特征数据（从标准正态分布采样）
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # 为每个类别创建不同的中心点
    class_centers = np.random.randn(num_classes, num_features) * 2

    # 生成标签
    y_labels = np.random.randint(0, num_classes, num_samples)

    # 根据类别调整特征，使数据更有区分性
    for i in range(num_samples):
        label = y_labels[i]
        X[i] += class_centers[label] * 0.5

    # 转换为one-hot编码
    y_onehot = np.eye(num_classes)[y_labels].astype(np.float32)

    return X, y_onehot, y_labels


def save_to_csv(
    X: np.ndarray,
    y_onehot: np.ndarray,
    y_labels: np.ndarray,
    filepath: str
):
    """
    保存数据为CSV格式

    参数：
        X: 特征矩阵
        y_onehot: one-hot标签
        y_labels: 原始标签
        filepath: 保存路径
    """
    # 创建DataFrame
    num_features = X.shape[1]
    feature_cols = [f"feature_{i}" for i in range(num_features)]

    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = y_labels

    # 添加one-hot列
    num_classes = y_onehot.shape[1]
    for i in range(num_classes):
        df[f'class_{i}'] = y_onehot[:, i]

    # 保存
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ 已保存数据到: {filepath}")
    print(f"  形状: {df.shape}")


def save_to_numpy(
    X: np.ndarray,
    y_onehot: np.ndarray,
    y_labels: np.ndarray,
    filepath: str
):
    """
    保存数据为NumPy格式

    参数：
        X: 特征矩阵
        y_onehot: one-hot标签
        y_labels: 原始标签
        filepath: 保存路径（不含扩展名）
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.save(f"{filepath}_X.npy", X)
    np.save(f"{filepath}_y_onehot.npy", y_onehot)
    np.save(f"{filepath}_y_labels.npy", y_labels)

    print(f"✓ 已保存NumPy数据到: {filepath}_*.npy")
    print(f"  X形状: {X.shape}")
    print(f"  y_onehot形状: {y_onehot.shape}")
    print(f"  y_labels形状: {y_labels.shape}")


def generate_all_splits(
    num_train: int = 100,
    num_val: int = 50,
    num_test: int = 50,
    num_features: int = 10,
    num_classes: int = 2,
    output_dir: str = "data",
    format: str = "csv",
    random_seed: int = 42
):
    """
    生成训练、验证和测试集

    参数：
        num_train: 训练样本数
        num_val: 验证样本数
        num_test: 测试样本数
        num_features: 特征维度
        num_classes: 类别数
        output_dir: 输出目录
        format: 保存格式 ('csv' 或 'numpy')
        random_seed: 随机种子
    """
    print("=" * 70)
    print("生成测试数据集")
    print("=" * 70)
    print(f"配置:")
    print(f"  训练样本: {num_train}")
    print(f"  验证样本: {num_val}")
    print(f"  测试样本: {num_test}")
    print(f"  特征维度: {num_features}")
    print(f"  类别数: {num_classes}")
    print(f"  输出目录: {output_dir}")
    print(f"  格式: {format}")
    print("=" * 70)

    # 生成训练集
    print("\n[1/3] 生成训练集...")
    X_train, y_train_onehot, y_train_labels = generate_classification_data(
        num_train, num_features, num_classes, random_seed
    )

    if format == "csv":
        save_to_csv(X_train, y_train_onehot, y_train_labels,
                   os.path.join(output_dir, "train_test.csv"))
    else:
        save_to_numpy(X_train, y_train_onehot, y_train_labels,
                     os.path.join(output_dir, "train_test"))

    # 生成验证集
    print("\n[2/3] 生成验证集...")
    X_val, y_val_onehot, y_val_labels = generate_classification_data(
        num_val, num_features, num_classes, random_seed + 1
    )

    if format == "csv":
        save_to_csv(X_val, y_val_onehot, y_val_labels,
                   os.path.join(output_dir, "val_test.csv"))
    else:
        save_to_numpy(X_val, y_val_onehot, y_val_labels,
                     os.path.join(output_dir, "val_test"))

    # 生成测试集
    print("\n[3/3] 生成测试集...")
    X_test, y_test_onehot, y_test_labels = generate_classification_data(
        num_test, num_features, num_classes, random_seed + 2
    )

    if format == "csv":
        save_to_csv(X_test, y_test_onehot, y_test_labels,
                   os.path.join(output_dir, "test_test.csv"))
    else:
        save_to_numpy(X_test, y_test_onehot, y_test_labels,
                     os.path.join(output_dir, "test_test"))

    print("\n" + "=" * 70)
    print("✓ 所有数据集生成完成!")
    print("=" * 70)

    # 打印数据统计
    print("\n数据统计:")
    print(f"训练集:")
    print(f"  样本数: {len(y_train_labels)}")
    print(f"  类别分布: {np.bincount(y_train_labels)}")
    print(f"  特征范围: [{X_train.min():.2f}, {X_train.max():.2f}]")

    print(f"\n验证集:")
    print(f"  样本数: {len(y_val_labels)}")
    print(f"  类别分布: {np.bincount(y_val_labels)}")

    print(f"\n测试集:")
    print(f"  样本数: {len(y_test_labels)}")
    print(f"  类别分布: {np.bincount(y_test_labels)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成测试用的模拟数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数生成数据
  python generate_mock_data.py

  # 生成更大的数据集
  python generate_mock_data.py --num-train 1000 --num-val 500 --num-test 500

  # 生成多分类数据
  python generate_mock_data.py --num-classes 5

  # 保存为NumPy格式
  python generate_mock_data.py --format numpy
        """
    )

    parser.add_argument(
        "--num-train",
        type=int,
        default=100,
        help="训练样本数 (默认: 100)"
    )

    parser.add_argument(
        "--num-val",
        type=int,
        default=50,
        help="验证样本数 (默认: 50)"
    )

    parser.add_argument(
        "--num-test",
        type=int,
        default=50,
        help="测试样本数 (默认: 50)"
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="特征维度 (默认: 10)"
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="类别数 (默认: 2)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="输出目录 (默认: data)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "numpy"],
        default="csv",
        help="保存格式 (默认: csv)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 生成数据
    generate_all_splits(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_features=args.num_features,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        format=args.format,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
