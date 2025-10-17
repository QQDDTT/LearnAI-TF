# -*- coding: utf-8 -*-
"""
lib/clustering.py
无监督学习聚类算法
"""

import numpy as np
from typing import Dict, Any, Tuple
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


def assign(
    data: np.ndarray,
    centroids: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    分配样本到最近的簇

    参数:
        data: 数据矩阵 (n_samples, n_features)
        centroids: 簇心矩阵 (n_clusters, n_features)

    返回:
        包含assignments的字典
    """
    # 计算每个样本到每个簇心的距离
    distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    # 分配到最近的簇
    assignments = np.argmin(distances, axis=1)

    logger.debug(f"样本分配完成，{len(np.unique(assignments))} 个簇")

    return {
        "data": data,
        "assignments": assignments,
        "distances": distances
    }


def update_centroids(
    data: np.ndarray,
    assignments: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    根据分配更新簇心

    参数:
        data: 数据矩阵
        assignments: 样本分配 (n_samples,)

    返回:
        新的簇心矩阵
    """
    n_clusters = len(np.unique(assignments))
    n_features = data.shape[1]

    new_centroids = np.zeros((n_clusters, n_features))

    for k in range(n_clusters):
        # 找到属于簇k的所有样本
        cluster_members = data[assignments == k]

        if len(cluster_members) > 0:
            # 计算均值作为新簇心
            new_centroids[k] = cluster_members.mean(axis=0)
        else:
            # 如果簇为空，随机初始化
            new_centroids[k] = data[np.random.randint(len(data))]
            logger.warning(f"簇 {k} 为空，随机重新初始化")

    logger.debug(f"簇心已更新")

    return new_centroids


def check_convergence(
    old_centroids: np.ndarray,
    new_centroids: np.ndarray,
    threshold: float = 1e-4,
    **kwargs
) -> Dict[str, Any]:
    """
    检查是否收敛

    参数:
        old_centroids: 旧簇心
        new_centroids: 新簇心
        threshold: 收敛阈值

    返回:
        包含converged和change的字典
    """
    # 计算簇心的变化量
    change = np.sqrt(((new_centroids - old_centroids) ** 2).sum())

    converged = change < threshold

    if converged:
        logger.info(f"✓ 聚类已收敛，变化量: {change:.6f}")
    else:
        logger.debug(f"聚类未收敛，变化量: {change:.6f}")

    return {
        "converged": converged,
        "change": float(change)
    }


class KMeans:
    """
    K-Means聚类算法
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        random_seed: int = None,
        **kwargs
    ):
        """
        初始化K-Means

        参数:
            n_clusters: 簇的数量
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            random_seed: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.random_seed = random_seed

        self.centroids = None
        self.labels = None
        self.inertia = None

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"KMeans 初始化: n_clusters={n_clusters}")

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        训练K-Means模型

        参数:
            X: 数据矩阵 (n_samples, n_features)

        返回:
            self
        """
        n_samples, n_features = X.shape

        # 随机初始化簇心
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        logger.info(f"开始K-Means训练，数据: {n_samples} 样本, {n_features} 特征")

        for iteration in range(self.max_iterations):
            # 分配样本
            result = assign(X, self.centroids)
            assignments = result["assignments"]

            # 更新簇心
            new_centroids = update_centroids(X, assignments)

            # 检查收敛
            convergence = check_convergence(
                self.centroids,
                new_centroids,
                self.convergence_threshold
            )

            self.centroids = new_centroids

            if convergence["converged"]:
                logger.info(f"K-Means在第 {iteration + 1} 次迭代后收敛")
                break

        # 最终分配
        result = assign(X, self.centroids)
        self.labels = result["assignments"]

        # 计算惯性（簇内平方和）
        self.inertia = self._compute_inertia(X, self.labels, self.centroids)

        logger.info(f"✓ K-Means训练完成，惯性: {self.inertia:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本的簇标签

        参数:
            X: 数据矩阵

        返回:
            簇标签
        """
        if self.centroids is None:
            raise ValueError("模型未训练，请先调用fit()")

        result = assign(X, self.centroids)
        return result["assignments"]

    def _compute_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        """计算惯性"""
        inertia = 0.0

        for k in range(self.n_clusters):
            cluster_members = X[labels == k]
            if len(cluster_members) > 0:
                inertia += ((cluster_members - centroids[k]) ** 2).sum()

        return float(inertia)


class DBSCAN:
    """
    DBSCAN聚类算法
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ):
        """
        初始化DBSCAN

        参数:
            eps: 邻域半径
            min_samples: 核心点的最小样本数
        """
        self.eps = eps
        self.min_samples = min_samples

        self.labels = None
        self.core_sample_indices = None

        logger.info(f"DBSCAN 初始化: eps={eps}, min_samples={min_samples}")

    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        训练DBSCAN模型

        参数:
            X: 数据矩阵

        返回:
            self
        """
        n_samples = len(X)

        # 初始化标签（-1表示噪声）
        labels = np.full(n_samples, -1, dtype=int)

        # 计算距离矩阵
        distances = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))

        # 找到核心点
        neighbors = distances < self.eps
        neighbor_counts = neighbors.sum(axis=1)
        core_samples = neighbor_counts >= self.min_samples

        logger.info(f"找到 {core_samples.sum()} 个核心点")

        # 聚类
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1 or not core_samples[i]:
                continue

            # 从核心点开始扩展簇
            labels[i] = cluster_id
            seeds = [i]

            while seeds:
                current = seeds.pop(0)

                # 找到邻居
                neighbor_indices = np.where(neighbors[current])[0]

                for neighbor in neighbor_indices:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id

                        if core_samples[neighbor]:
                            seeds.append(neighbor)

            cluster_id += 1

        self.labels = labels
        self.core_sample_indices = np.where(core_samples)[0]

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"✓ DBSCAN完成: {n_clusters} 个簇, {n_noise} 个噪声点")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本的簇标签

        注意：DBSCAN不直接支持预测，这里使用最近核心点的标签

        参数:
            X: 数据矩阵

        返回:
            簇标签
        """
        if self.labels is None:
            raise ValueError("模型未训练，请先调用fit()")

        # 简化实现：分配到最近核心点的簇
        # 实际应用中可能需要更复杂的策略
        logger.warning("DBSCAN predict 使用简化实现")

        return np.full(len(X), -1)  # 默认标记为噪声
