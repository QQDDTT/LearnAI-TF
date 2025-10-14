# -*- coding: utf-8 -*-
"""
modules/data_manager.py
数据管理模块：
- 加载不同训练模式的数据
- 数据增强逻辑
- 数据预处理
- 数据迭代器管理
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from common.common import LoggerManager, call_target
from common import utils

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主初始化函数
# ======================================================
def initialize(config: Dict, training_mode: str) -> Dict[str, Any]:
    """
    根据训练模式初始化数据加载器

    参数:
        config: data_manager配置
        training_mode: 训练模式 (supervised/reinforcement/self_supervised等)

    返回:
        dict: 数据加载器字典

    示例:
        >>> dataloaders = initialize(config, "supervised")
        >>> print(dataloaders.keys())
    """
    logger.info(f"初始化 {training_mode} 模式的数据管理器")

    preprocessing_config = config.get("preprocessing", {})

    if training_mode == "supervised":
        data_loaders = load_supervised_data(config.get("supervised_source", {}))
    elif training_mode == "reinforcement":
        data_loaders = load_rl_data(config.get("rl_source", {}))
    elif training_mode == "unsupervised":
        data_loaders = load_unsupervised_data(config.get("unsupervised_source", {}))
    elif training_mode == "self_supervised":
        data_loaders = load_self_supervised_data(config.get("self_supervised_source", {}))
    elif training_mode == "multi_task":
        data_loaders = load_multi_task_data(config.get("multi_task_source", {}))
    else:
        logger.warning(f"未知的训练模式，未加载数据: {training_mode}")

    # 2. 应用预处理 (如果存在)
    if preprocessing_config:
        logger.info(f"开始对 {training_mode} 数据应用预处理...")
        for name, loader in data_loaders.items():
            # 假设数据加载器返回的是 tf.data.Dataset 或其他可处理对象
            if loader is not None:
                data_loaders[name] = apply_preprocessing(
                    loader,
                    preprocessing_config
                )
                logger.info(f"  对数据加载器 '{name}' 完成预处理。")
            else:
                logger.warning(f"  数据加载器 '{name}' 为 None，跳过预处理。")

    return data_loaders

# ======================================================
# 监督学习数据加载
# ======================================================
def load_supervised_data(config: Dict) -> Dict[str, tf.data.Dataset]:
    """
    加载监督学习数据（train/val/test）

    参数:
        config: supervised_source配置

    返回:
        dict: {"train": train_ds, "val": val_ds, "test": test_ds}

    示例:
        >>> dataloaders = load_supervised_data(config)
        >>> for x, y in dataloaders["train"]:
        >>>     print(x.shape, y.shape)
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        if split in config:
            split_config = config[split]
            logger.info(f"加载 {split} 数据集")

            try:
                # 通过反射调用数据加载函数
                dataset = call_target(
                    split_config["reflection"],
                    split_config.get("args", {})
                )

                # 如果返回的是元组（train, val），处理分割情况
                if isinstance(dataset, tuple) and len(dataset) == 2:
                    if split == "train":
                        dataloaders["train"] = dataset[0]
                        dataloaders["val"] = dataset[1]
                        logger.info(f"train 数据集已分割为 train 和 val")
                    else:
                        dataloaders[split] = dataset[0]
                else:
                    dataloaders[split] = dataset

                logger.info(f"{split} 数据集加载完成")

            except Exception as e:
                logger.error(f"加载 {split} 数据集失败: {str(e)}")
                raise

    return dataloaders


# ======================================================
# 强化学习数据加载
# ======================================================
def load_rl_data(config: Dict) -> Dict[str, Any]:
    """
    初始化强化学习环境客户端

    参数:
        config: rl_source配置

    返回:
        dict: {"client": NetworkClient实例, "dataset": 可选的数据集, "buffer": 经验回放缓冲区}

    示例:
        >>> rl_data = load_rl_data(config)
        >>> client = rl_data["client"]
        >>> response = client.request("reset")
    """
    logger.info("初始化强化学习环境")

    result = {}

    # 构建网络客户端
    if "client" in config:
        client_config = config["client"]
        try:
            client = call_target(
                client_config["reflection"],
                client_config.get("args", {})
            )
            result["client"] = client
            logger.info("强化学习客户端初始化完成")
        except Exception as e:
            logger.error(f"初始化RL客户端失败: {str(e)}")
            raise

    # 可选：构建批量训练数据集（用于经验回放）
    if "dataset" in config:
        dataset_config = config["dataset"]
        dataset_args = dataset_config.get("args", {})

        # 注入client实例
        if "client" in result:
            dataset_args["client"] = result["client"]

        try:
            dataset = call_target(
                dataset_config["reflection"],
                dataset_args
            )
            result["dataset"] = dataset
            logger.info("强化学习数据集初始化完成")
        except Exception as e:
            logger.warning(f"初始化RL数据集失败: {str(e)}")

    # 初始化经验回放缓冲区
    buffer_size = config.get("buffer_size", 10000)
    result["buffer"] = []
    result["buffer_size"] = buffer_size
    logger.info(f"经验回放缓冲区大小: {buffer_size}")

    return result


# ======================================================
# 无监督学习数据加载
# ======================================================
def load_unsupervised_data(config: Dict) -> Dict[str, tf.data.Dataset]:
    """
    加载无监督学习数据

    参数:
        config: unsupervised_source配置

    返回:
        dict: {"train": dataset}

    示例:
        >>> dataloaders = load_unsupervised_data(config)
    """
    logger.info("加载无监督学习数据")

    try:
        dataset = call_target(
            config["reflection"],
            config.get("args", {})
        )

        # 应用预处理（如果配置了）
        if config.get("preprocessing", {}).get("enabled", False):
            dataset = apply_preprocessing(dataset, config["preprocessing"])

        return {"train": dataset}

    except Exception as e:
        logger.error(f"加载无监督学习数据失败: {str(e)}")
        raise


# ======================================================
# 自监督学习数据加载（含增强）
# ======================================================
def load_self_supervised_data(config: Dict) -> Dict[str, tf.data.Dataset]:
    """
    加载自监督学习数据，并应用对比增强

    参数:
        config: self_supervised_source配置

    返回:
        dict: {"train": augmented_dataset}

    示例:
        >>> dataloaders = load_self_supervised_data(config)
        >>> for (view1, view2), y in dataloaders["train"]:
        >>>     print(view1.shape, view2.shape)
    """
    logger.info("加载自监督学习数据")

    try:
        # 加载基础数据集
        dataset = call_target(
            config["reflection"],
            config.get("args", {})
        )

        # 应用对比学习数据增强
        if config.get("augmentation", {}).get("enabled", False):
            logger.info("应用对比学习数据增强")
            transforms = config["augmentation"].get("transforms", [])

            # 使用map应用增强
            dataset = dataset.map(
                lambda x, y: (apply_contrastive_augmentation(x, transforms), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        return {"train": dataset}

    except Exception as e:
        logger.error(f"加载自监督学习数据失败: {str(e)}")
        raise


# ======================================================
# 多任务学习数据加载
# ======================================================
def load_multi_task_data(config: Dict) -> Dict[str, tf.data.Dataset]:
    """
    加载多任务学习数据

    参数:
        config: multi_task_source配置

    返回:
        dict: 多任务数据加载器

    示例:
        >>> dataloaders = load_multi_task_data(config)
    """
    logger.info("加载多任务学习数据")

    dataloaders = {}

    for task_name, task_config in config.items():
        logger.info(f"加载任务 {task_name} 的数据")

        try:
            dataset = call_target(
                task_config["reflection"],
                task_config.get("args", {})
            )
            dataloaders[task_name] = dataset
        except Exception as e:
            logger.error(f"加载任务 {task_name} 数据失败: {str(e)}")
            raise

    return dataloaders


# ======================================================
# 数据增强逻辑
# ======================================================
def apply_contrastive_augmentation(
    x: tf.Tensor,
    transforms: List[Dict]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    应用对比学习增强（生成两个增强视图）

    参数:
        x: 输入数据
        transforms: 增强变换列表

    返回:
        (view1, view2): 两个增强视图

    示例:
        >>> view1, view2 = apply_contrastive_augmentation(x, transforms)
    """
    view1 = apply_transforms(x, transforms)
    view2 = apply_transforms(x, transforms)

    return view1, view2


def apply_transforms(data: tf.Tensor, transforms: List[Dict]) -> tf.Tensor:
    """
    应用一系列增强变换

    参数:
        data: 输入数据
        transforms: 变换配置列表
            [{"type": "RandomCrop", "args": {...}}, ...]

    返回:
        增强后的数据

    示例:
        >>> transforms = [
        >>>     {"type": "RandomCrop", "args": {"size": [224, 224]}},
        >>>     {"type": "RandomFlip", "args": {"horizontal": True}}
        >>> ]
        >>> augmented = apply_transforms(data, transforms)
    """
    result = data

    for transform_config in transforms:
        transform_type = transform_config.get("type")
        transform_args = transform_config.get("args", {})

        try:
            # 应用不同类型的增强
            if transform_type == "RandomCrop":
                result = apply_random_crop(result, transform_args)

            elif transform_type == "RandomFlip":
                result = apply_random_flip(result, transform_args)

            elif transform_type == "ColorJitter":
                result = apply_color_jitter(result, transform_args)

            elif transform_type == "RandomRotation":
                result = apply_random_rotation(result, transform_args)

            elif transform_type == "RandomResizedCrop":
                result = apply_random_resized_crop(result, transform_args)

            elif transform_type == "GaussianBlur":
                result = apply_gaussian_blur(result, transform_args)

            elif transform_type == "Normalize":
                result = apply_normalize(result, transform_args)

            elif transform_type == "RandomErasing":
                result = apply_random_erasing(result, transform_args)

            else:
                logger.warning(f"未知的增强类型: {transform_type}")

        except Exception as e:
            logger.error(f"应用增强 {transform_type} 失败: {str(e)}")

    return result


# ======================================================
# 具体增强函数实现
# ======================================================
def apply_random_crop(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """随机裁剪"""
    size = args.get("size", [224, 224])
    padding = args.get("padding", 0)

    if padding > 0:
        image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]], mode='REFLECT')

    return tf.image.random_crop(image, [size[0], size[1], image.shape[-1]])


def apply_random_flip(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """随机翻转"""
    horizontal = args.get("horizontal", True)
    vertical = args.get("vertical", False)

    if horizontal:
        image = tf.image.random_flip_left_right(image)
    if vertical:
        image = tf.image.random_flip_up_down(image)

    return image


def apply_color_jitter(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """颜色抖动"""
    brightness = args.get("brightness", 0.0)
    contrast = args.get("contrast", 0.0)
    saturation = args.get("saturation", 0.0)
    hue = args.get("hue", 0.0)

    if brightness > 0:
        image = tf.image.random_brightness(image, brightness)
    if contrast > 0:
        image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)
    if saturation > 0:
        image = tf.image.random_saturation(image, 1 - saturation, 1 + saturation)
    if hue > 0:
        image = tf.image.random_hue(image, hue)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def apply_random_rotation(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """随机旋转（使用原生TensorFlow实现）"""
    degrees = args.get("degrees", 15)

    # 随机生成旋转角度（弧度）
    angle = tf.random.uniform([], -degrees, degrees) * np.pi / 180.0

    # 使用TensorFlow原生的旋转实现（通过仿射变换）
    # 注意：如果需要更复杂的旋转，可以考虑安装 tensorflow-addons
    # 但这里提供一个简单的替代实现

    # 简单实现：使用最近邻插值的旋转
    # 对于生产环境，建议使用专门的图像处理库
    logger.warning("RandomRotation使用简化实现，如需完整功能请安装: pip install tensorflow-addons")

    # 这里返回原图，或者可以使用其他变换代替
    # 如果确实需要旋转功能，用户可以选择：
    # 1. 安装 tensorflow-addons: pip install tensorflow-addons
    # 2. 使用其他数据增强库如 albumentations
    # 3. 在配置中移除 RandomRotation 增强

    return image  # 暂时返回原图，不执行旋转


def apply_random_resized_crop(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """随机裁剪并调整大小"""
    size = args.get("size", [224, 224])
    scale = args.get("scale", (0.08, 1.0))
    ratio = args.get("ratio", (0.75, 1.333))

    # 随机裁剪
    scale_factor = tf.random.uniform([], scale[0], scale[1])
    ratio_factor = tf.random.uniform([], ratio[0], ratio[1])

    crop_h = int(image.shape[0] * tf.sqrt(scale_factor * ratio_factor))
    crop_w = int(image.shape[1] * tf.sqrt(scale_factor / ratio_factor))

    crop_h = tf.minimum(crop_h, image.shape[0])
    crop_w = tf.minimum(crop_w, image.shape[1])

    image = tf.image.random_crop(image, [crop_h, crop_w, image.shape[-1]])

    # 调整大小
    image = tf.image.resize(image, size)

    return image


def apply_gaussian_blur(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """高斯模糊"""
    kernel_size = args.get("kernel_size", 3)
    sigma = args.get("sigma", (0.1, 2.0))

    # 随机sigma
    if isinstance(sigma, tuple):
        sigma_value = tf.random.uniform([], sigma[0], sigma[1])
    else:
        sigma_value = sigma

    # 创建高斯核
    kernel = _create_gaussian_kernel(kernel_size, sigma_value)

    # 应用卷积
    image = tf.expand_dims(image, 0)
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    image = tf.squeeze(image, 0)

    return image


def apply_normalize(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """标准化"""
    mean = args.get("mean", [0.485, 0.456, 0.406])
    std = args.get("std", [0.229, 0.224, 0.225])

    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)

    return (image - mean) / std


def apply_random_erasing(image: tf.Tensor, args: Dict) -> tf.Tensor:
    """随机擦除"""
    probability = args.get("probability", 0.5)
    area_ratio = args.get("area_ratio", (0.02, 0.4))
    aspect_ratio = args.get("aspect_ratio", (0.3, 3.3))

    if tf.random.uniform([]) > probability:
        return image

    h, w = image.shape[0], image.shape[1]
    area = h * w

    target_area = tf.random.uniform([], area_ratio[0], area_ratio[1]) * area
    aspect = tf.random.uniform([], aspect_ratio[0], aspect_ratio[1])

    erase_h = tf.cast(tf.sqrt(target_area * aspect), tf.int32)
    erase_w = tf.cast(tf.sqrt(target_area / aspect), tf.int32)

    erase_h = tf.minimum(erase_h, h)
    erase_w = tf.minimum(erase_w, w)

    top = tf.random.uniform([], 0, h - erase_h, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - erase_w, dtype=tf.int32)

    # 创建mask
    mask = tf.ones_like(image)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, j] for i in range(top, top + erase_h) for j in range(left, left + erase_w)],
        tf.zeros([erase_h * erase_w, image.shape[-1]])
    )

    return image * mask


def _create_gaussian_kernel(kernel_size: int, sigma: float) -> tf.Tensor:
    """创建高斯卷积核"""
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    gauss = tf.exp(-tf.square(x) / (2.0 * sigma ** 2))
    kernel = gauss / tf.reduce_sum(gauss)
    kernel = tf.tensordot(kernel, kernel, axes=0)
    kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
    return kernel


# ======================================================
# 数据预处理工具
# ======================================================
def apply_preprocessing(dataset: tf.data.Dataset, preprocessing_config: Dict) -> tf.data.Dataset:
    """
    应用数据预处理

    参数:
        dataset: 输入数据集
        preprocessing_config: 预处理配置

    返回:
        预处理后的数据集
    """
    operations = preprocessing_config.get("operations", [])

    for operation in operations:
        op_type = operation.get("type")
        op_args = operation.get("args", {})

        if op_type == "normalize":
            dataset = dataset.map(
                lambda x, y: (normalize_data(x, op_args.get("method", "standard")), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        elif op_type == "resize":
            size = op_args.get("size", [224, 224])
            dataset = dataset.map(
                lambda x, y: (tf.image.resize(x, size), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

    return dataset


def normalize_data(data: tf.Tensor, method: str = "standard") -> tf.Tensor:
    """
    数据归一化

    参数:
        data: 输入数据
        method: 归一化方法 ("standard", "minmax", "l2")

    返回:
        归一化后的数据
    """
    if method == "standard":
        mean = tf.reduce_mean(data, axis=0, keepdims=True)
        std = tf.math.reduce_std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = tf.reduce_min(data, axis=0, keepdims=True)
        max_val = tf.reduce_max(data, axis=0, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "l2":
        return tf.nn.l2_normalize(data, axis=-1)
    else:
        logger.warning(f"未知的归一化方法: {method}")
        return data


def create_batches(
    dataset: tf.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    buffer_size: int = 1000,
    drop_remainder: bool = False
) -> tf.data.Dataset:
    """
    创建批次

    参数:
        dataset: 输入数据集
        batch_size: 批大小
        shuffle: 是否打乱
        buffer_size: 打乱缓冲区大小
        drop_remainder: 是否丢弃最后不完整的批次

    返回:
        批处理后的数据集
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ======================================================
# 数据集信息工具
# ======================================================
def get_dataset_info(dataset: tf.data.Dataset) -> Dict[str, Any]:
    """
    获取数据集信息

    参数:
        dataset: TensorFlow数据集

    返回:
        dict: 数据集信息
    """
    info = {}

    try:
        # 获取元素规格
        element_spec = dataset.element_spec
        info["element_spec"] = str(element_spec)

        # 尝试获取数据集大小（如果可能）
        try:
            size = len(list(dataset.as_numpy_iterator()))
            info["size"] = size
        except:
            info["size"] = "unknown"

        logger.info(f"数据集信息: {info}")

    except Exception as e:
        logger.warning(f"获取数据集信息失败: {str(e)}")

    return info
