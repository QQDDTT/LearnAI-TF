# -*- coding: utf-8 -*-
"""
modules/train.py
统一训练模块
支持不同训练类型（GAN / supervised / RL 等）
"""

from typing import Dict
import tensorflow as tf
from datetime import datetime
from modules.utils import Logger, call_target

logger = Logger(__file__)

# -------------------------
# 基础训练类型函数（内部定义，不反射）
# -------------------------
def train_gan_step(generator, discriminator, gen_optimizer, dis_optimizer, gen_loss_fn, dis_loss_fn, x_real, logger=None):
    try:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            z = tf.random.normal([x_real.shape[0], generator.layers[0].input_shape[-1]])
            fake = generator(z, training=True)
            real_output = discriminator(x_real, training=True)
            fake_output = discriminator(fake, training=True)

            gen_loss = gen_loss_fn(tf.ones_like(fake_output), fake_output)
            dis_loss = dis_loss_fn(tf.ones_like(real_output), real_output) + \
                       dis_loss_fn(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        dis_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        logger and logger.info(f"[GAN] Step completed: gen_loss={gen_loss:.4f}, dis_loss={dis_loss:.4f}")
        return gen_loss, dis_loss
    except Exception as e:
        logger and logger.error(f"[GAN] Step failed: {e}")
        raise


def train_supervised_step(model, optimizer, loss_fn, x_batch, y_batch, logger=None):
    try:
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        logger and logger.info(f"[Supervised] Step completed: loss={loss:.4f}")
        return loss
    except Exception as e:
        logger and logger.error(f"[Supervised] Step failed: {e}")
        raise


# -------------------------
# 通用训练入口
# -------------------------
def train_model(train_cfg: dict, model_dict: dict, dataloaders: Dict[str, tf.data.Dataset]):
    """
    通用训练流程，支持不同训练类型（GAN / 监督学习 / RL 等）
    参数:
        train_cfg (dict): YAML 配置中 training 部分
        model_dict (dict): build_model 返回的模型字典
        dataloaders (dict): build_dataloader 返回的训练数据字典
    流程:
        1. 根据 train_type 选择基础训练函数 train_step。
        2. 初始化 TensorFlow 回调（可选）。
        3. 进入 epoch 循环（循环次数由 epochs 指定）。
        4. 对每个 dataloader 执行 train_step:
            - GAN: 同时传入 generator 和 discriminator
            - 监督学习: 传入 model, optimizer, loss_fn
        5. 每个 epoch 开始和结束时打印日志。
        6. train_step 内部会更新梯度、计算损失，并记录日志。
    异常处理:
        - 回调函数构建失败会记录 error，但不中断训练。
        - train_step 内部可使用 try/except 捕获异常，避免单个 batch 崩溃训练。
    """
    train_type = train_cfg["type"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]

    logger.info(f"Training started: type={train_type}, epochs={epochs}, batch_size={batch_size}")
    start_time = datetime.now()

    # 获取训练函数
    if train_type == "gan":
        train_step = train_gan_step
    elif train_type == "supervised":
        train_step = train_supervised_step
    else:
        logger.error(f"Unsupported train_type: {train_type}")
        raise ValueError(f"Unsupported train_type: {train_type}")

    # 回调
    callbacks = []
    for cb_cfg in train_cfg.get("callbacks", []):
        try:
            cb = call_target(cb_cfg["reflection"], cb_cfg.get("args", {}))
            callbacks.append(cb)
            logger.info(f"Callback added: {cb_cfg['reflection']}")
        except Exception as e:
            logger.warning(f"Failed to build callback {cb_cfg}: {e}")

    # 训练循环
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} start...")
        for key, dataset in dataloaders.items():
            logger.info(f"Training on dataset: {key}")
            for step, batch in enumerate(dataset):
                try:
                    if train_type == "gan":
                        x_real, _ = batch
                        gen_loss, dis_loss = train_step(
                            model_dict["generator"], model_dict["discriminator"],
                            model_dict["gen_optimizer"], model_dict["dis_optimizer"],
                            model_dict["gen_loss_fn"], model_dict["dis_loss_fn"],
                            x_real
                        )
                    else:
                        x_batch, y_batch = batch
                        loss = train_step(
                            model_dict["generator"],  # 监督学习使用 generator
                            model_dict["gen_optimizer"],
                            model_dict["gen_loss_fn"],
                            x_batch, y_batch
                        )
                    if step % 10 == 0:
                        logger.info(f"Batch {step} processed")
                except Exception as e:
                    logger.error(f"Batch {step} failed: {e}")

        logger.info(f"Epoch {epoch+1} completed")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Training finished in {duration:.2f}s")
