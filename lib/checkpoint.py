# -*- coding: utf-8 -*-
"""
lib/checkpoint.py
检查点管理器
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


class CheckpointManager:
    """
    检查点管理器

    负责保存和加载模型检查点
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_frequency: str = "epoch",
        keep_last_n: int = 5,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化检查点管理器

        参数:
            save_dir: 保存目录
            save_frequency: 保存频率 (epoch/step)
            keep_last_n: 保留最近N个检查点
            save_best_only: 只保存最佳模型
            monitor: 监控指标
            mode: 优化方向 (min/max)
            enabled: 是否启用
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.keep_last_n = keep_last_n
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.enabled = enabled

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []

        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CheckpointManager 初始化: {self.save_dir}")
            logger.info(f"  监控指标: {monitor} ({mode})")
            logger.info(f"  保留数量: {keep_last_n}")

    def save(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ):
        """
        保存检查点

        参数:
            model: 模型对象
            epoch: 当前epoch
            metrics: 指标字典
        """
        if not self.enabled:
            return

        # 获取监控指标的值
        current_value = metrics.get(self.monitor)

        if current_value is None:
            logger.warning(f"监控指标 '{self.monitor}' 不存在于metrics中")
            return

        # 判断是否需要保存
        should_save = False

        if self.save_best_only:
            if self.mode == 'min' and current_value < self.best_value:
                should_save = True
                self.best_value = current_value
            elif self.mode == 'max' and current_value > self.best_value:
                should_save = True
                self.best_value = current_value
        else:
            should_save = True

        if should_save:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.h5"

            try:
                model.save_weights(str(checkpoint_path))
                logger.info(f"✓ 检查点已保存: {checkpoint_path}")
                logger.info(f"  {self.monitor}: {current_value:.4f}")

                self.checkpoints.append({
                    'path': checkpoint_path,
                    'epoch': epoch,
                    'value': current_value
                })

                # 清理旧检查点
                self._cleanup_old_checkpoints()

            except Exception as e:
                logger.error(f"保存检查点失败: {e}")

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.checkpoints) > self.keep_last_n:
            # 按epoch排序
            self.checkpoints.sort(key=lambda x: x['epoch'])

            # 删除最旧的检查点
            to_remove = self.checkpoints[:-self.keep_last_n]

            for ckpt in to_remove:
                try:
                    os.remove(ckpt['path'])
                    logger.debug(f"删除旧检查点: {ckpt['path']}")
                except Exception as e:
                    logger.warning(f"删除检查点失败: {e}")

            self.checkpoints = self.checkpoints[-self.keep_last_n:]

    def load_latest(self, model: Any) -> Optional[int]:
        """
        加载最新的检查点

        参数:
            model: 模型对象

        返回:
            加载的epoch，如果失败返回None
        """
        if not self.enabled or not self.checkpoints:
            return None

        latest = self.checkpoints[-1]

        try:
            model.load_weights(str(latest['path']))
            logger.info(f"✓ 已加载检查点: {latest['path']}")
            return latest['epoch']

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def load_best(self, model: Any) -> Optional[int]:
        """
        加载最佳检查点

        参数:
            model: 模型对象

        返回:
            加载的epoch，如果失败返回None
        """
        if not self.enabled or not self.checkpoints:
            return None

        # 找到最佳检查点
        if self.mode == 'min':
            best = min(self.checkpoints, key=lambda x: x['value'])
        else:
            best = max(self.checkpoints, key=lambda x: x['value'])

        try:
            model.load_weights(str(best['path']))
            logger.info(f"✓ 已加载最佳检查点: {best['path']}")
            logger.info(f"  {self.monitor}: {best['value']:.4f}")
            return best['epoch']

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
