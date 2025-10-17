# -*- coding: utf-8 -*-
"""
common/common.py
基础公共模块：
- 日志工具类
- 动态反射调用函数/类
- 配置加载函数（集成验证）
"""

import importlib
import logging
import os
from datetime import datetime
import yaml
from colorama import Fore, Style, init
from typing import Dict, Any, List, Optional
import tensorflow as tf


# ======================================================
# 日志工具类
# ======================================================
class LoggerManager:
    """
    全局日志管理器：
    - 控制台彩色输出
    - 文件日志纯文本
    - 每天一个文件，按项目存放
    - 每个 Python 文件调用 get_logger(__file__) 获取 logger
    """
    _loggers = {}      # {logger_name: logger实例}
    _file_handlers = {}  # {project: FileHandler实例}
    _LEVEL_COLOR = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    @classmethod
    def get_logger(cls, filename: str, project="LearnAI"):
        init(autoreset=True)
        classname = os.path.splitext(os.path.basename(filename))[0]
        logger_name = f"{project}.{classname}"

        # 已存在直接返回
        if logger_name in cls._loggers:
            return cls._loggers[logger_name]

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # ---------------- 控制台 Handler ----------------
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s][%(classname)s] %(message)s"))
            ch.addFilter(cls.ColorFilter(classname))
            logger.addHandler(ch)

        # ---------------- 文件 Handler ----------------
        today = datetime.now().strftime("%Y%m%d")
        log_home = os.environ.get("LOG_HOME", "/tmp")
        log_dir = os.path.join(log_home, project)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{project}_{today}.log")

        # 每个项目共用一个 FileHandler
        if project not in cls._file_handlers:
            fh = logging.FileHandler(log_file_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s][%(classname)s] %(message)s"))
            fh.addFilter(cls.PlainFilter(classname))  # 文件纯文本
            cls._file_handlers[project] = fh
            print(f"日志文件输出路径: {log_file_path}")

        # 添加 FileHandler 到 logger（避免重复添加）
        if cls._file_handlers[project] not in logger.handlers:
            logger.addHandler(cls._file_handlers[project])

        cls._loggers[logger_name] = logger
        return logger

    # ---------------- Filter ----------------
    class ColorFilter(logging.Filter):
        """控制台彩色输出"""
        def __init__(self, classname):
            super().__init__()
            self.classname = classname

        def filter(self, record):
            record.classname = f"{Fore.BLUE}{self.classname}{Style.RESET_ALL}"
            color = LoggerManager._LEVEL_COLOR.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            if record.levelno >= logging.WARNING:
                record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
            return True

    class PlainFilter(logging.Filter):
        """文件输出纯文本"""
        def __init__(self, classname):
            super().__init__()
            self.classname = classname

        def filter(self, record):
            record.classname = self.classname
            # 恢复原始等级名
            record.levelname = logging.getLevelName(record.levelno)
            # 确保 msg 纯文本
            if hasattr(record, "msg"):
                # 如果 msg 是 ANSI 彩色的，去掉 ESC 序列
                import re
                ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
                record.msg = ansi_escape.sub("", str(record.msg))
            return True


# ======================================================
# 配置加载（集成验证）
# ======================================================
def load_yaml(file: str, validate: bool = True) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置，并可选地进行验证

    参数:
        file: YAML 配置文件名（相对于 config 目录）
        validate: 是否验证配置结构（默认True）

    返回:
        dict: 配置字典

    异常:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 解析错误
        ValueError: 配置验证失败
    """
    logger = LoggerManager.get_logger(__file__)

    # 构建配置文件路径
    project_path = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(project_path, "config")
    config_file = os.path.join(config_dir, file)

    # 检查文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    logger.info(f"加载配置文件: {config_file}")

    # 加载 YAML
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML 解析错误: {e}")
        raise

    if config is None:
        raise ValueError(f"配置文件为空: {config_file}")

    logger.info(f"✓ 配置文件加载成功")

    # 可选验证
    if validate:
        logger.info("开始验证配置结构...")

        try:
            # 延迟导入，避免循环依赖
            from common.config_validator import validate_config_file

            is_valid, errors, warnings = validate_config_file(config)

            if not is_valid:
                logger.error(f"配置验证失败，发现 {len(errors)} 个错误")
                for error in errors:
                    logger.error(f"  - {error}")
                raise ValueError("配置文件验证失败")

            if warnings:
                logger.warning(f"配置验证通过，但有 {len(warnings)} 个警告")
                for warning in warnings:
                    logger.warning(f"  - {warning}")

        except ImportError:
            logger.warning("配置验证器未找到，跳过验证")

    return config


def load_yaml_without_validation(file: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置（不验证）

    参数:
        file: YAML 配置文件名（相对于 config 目录）

    返回:
        dict: 配置字典
    """
    return load_yaml(file, validate=False)


# ======================================================
# 动态反射调用
# ======================================================
def call_target(target: str, arguments: Dict[str, Any] = None):
    """
    动态反射调用指定函数/类
    支持两种写法:
        1. 模块路径:函数/类名 -> "tensorflow.keras.layers:Dense"
        2. 完整路径 -> "tensorflow.keras.layers.Dense"

    参数:
        target (str): 目标函数/类的路径
            - 冒号分隔: "模块路径:函数名"
            - 点分隔: "完整.模块.路径.函数名"
        arguments (dict): 传递给目标函数的参数

    返回:
        调用结果

    示例:
        >>> call_target("tensorflow.keras.layers:Dense", {"units": 128})
        >>> call_target("tensorflow.keras.layers.Dense", {"units": 128})

    异常:
        ImportError: 模块导入失败
        AttributeError: 函数/类不存在
        ValueError: target 不可调用
    """
    logger = LoggerManager.get_logger(__file__)

    if arguments is None:
        arguments = {}

    try:
        if ":" in target:
            # 冒号分隔
            module_path, func_name = target.split(":")
            module = importlib.import_module(module_path)
            func_or_class = getattr(module, func_name)
        else:
            # 点分隔，取最后一段为类名，其余为模块
            module_path, func_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func_or_class = getattr(module, func_name)
    except ImportError as e:
        logger.error(f"无法导入模块: {target}")
        raise ImportError(f"模块导入失败: {target}") from e
    except AttributeError as e:
        logger.error(f"模块中不存在: {target}")
        raise AttributeError(f"函数/类不存在: {target}") from e

    if not callable(func_or_class):
        raise ValueError(f"{target} is not callable")

    try:
        result = func_or_class(**arguments)
        return result
    except Exception as e:
        logger.error(f"调用 {target} 失败: {e}")
        raise


# ======================================================
# CUDA兼容性检测与硬件优化
# ======================================================
def check_cuda_compatibility() -> Dict[str, Any]:
    """
    检测CUDA兼容性并返回系统硬件信息

    返回:
        dict: 包含硬件信息的字典
            - cuda_available: bool, CUDA是否可用
            - gpu_count: int, GPU数量
            - gpu_names: list, GPU名称列表
            - gpu_memory: list, GPU内存列表(GB)
            - tf_version: str, TensorFlow版本
            - cuda_version: str, CUDA版本
            - cudnn_version: str, cuDNN版本
    """
    logger = LoggerManager.get_logger(__file__)

    hardware_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'tf_version': tf.__version__,
        'cuda_version': None,
        'cudnn_version': None
    }

    # 检测GPU
    gpus = tf.config.list_physical_devices('GPU')
    hardware_info['gpu_count'] = len(gpus)
    hardware_info['cuda_available'] = len(gpus) > 0

    if hardware_info['cuda_available']:
        # 获取GPU详细信息
        for gpu in gpus:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                hardware_info['gpu_names'].append(gpu_name)
            except:
                hardware_info['gpu_names'].append(str(gpu))

        # 获取CUDA和cuDNN版本
        try:
            from tensorflow.python.platform import build_info
            hardware_info['cuda_version'] = build_info.build_info.get('cuda_version', 'Unknown')
            hardware_info['cudnn_version'] = build_info.build_info.get('cudnn_version', 'Unknown')
        except:
            pass

        # 配置GPU内存增长策略（避免占满所有显存）
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("已启用GPU内存动态增长策略")
        except RuntimeError as e:
            logger.warning(f"GPU内存配置失败: {e}")

    return hardware_info


def optimize_config_for_hardware(
    config: Dict[str, Any],
    hardware_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据硬件信息优化训练配置

    参数:
        config: 原始配置字典
        hardware_info: 硬件信息字典

    返回:
        dict: 优化后的配置
    """
    logger = LoggerManager.get_logger(__file__)
    optimized_config = config.copy()

    # 获取global配置
    if 'global' not in optimized_config:
        optimized_config['global'] = {}

    # 根据CUDA可用性设置设备
    if hardware_info['cuda_available']:
        logger.info(f"检测到 {hardware_info['gpu_count']} 个GPU，使用GPU加速")
        optimized_config['global']['device'] = 'GPU'

        # 如果有多个GPU，启用分布式策略
        if hardware_info['gpu_count'] > 1:
            logger.info(f"多GPU环境，建议使用分布式训练策略")
            if 'distributed' not in optimized_config:
                optimized_config['distributed'] = {}
            optimized_config['distributed']['strategy'] = 'mirrored'
            optimized_config['distributed']['num_gpus'] = hardware_info['gpu_count']

        # 启用混合精度训练（在支持的GPU上）
        optimized_config['global']['mixed_precision'] = True
        logger.info("启用混合精度训练以提升性能")

    else:
        logger.warning("未检测到CUDA支持，使用CPU训练")
        optimized_config['global']['device'] = 'CPU'

        # CPU多线程优化
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        optimized_config['global']['num_workers'] = min(cpu_count, 8)
        logger.info(f"CPU环境，使用 {optimized_config['global']['num_workers']} 个worker线程")

        # 禁用混合精度（CPU不支持）
        optimized_config['global']['mixed_precision'] = False

    return optimized_config


def print_hardware_info(hardware_info: Dict[str, Any]):
    """
    打印硬件信息

    参数:
        hardware_info: 硬件信息字典
    """
    logger = LoggerManager.get_logger(__file__)

    logger.info("=" * 60)
    logger.info("硬件环境检测")
    logger.info("=" * 60)
    logger.info(f"TensorFlow版本: {hardware_info['tf_version']}")
    logger.info(f"CUDA可用: {hardware_info['cuda_available']}")

    if hardware_info['cuda_available']:
        logger.info(f"GPU数量: {hardware_info['gpu_count']}")
        for i, gpu_name in enumerate(hardware_info['gpu_names']):
            logger.info(f"  GPU {i}: {gpu_name}")
        if hardware_info['cuda_version']:
            logger.info(f"CUDA版本: {hardware_info['cuda_version']}")
        if hardware_info['cudnn_version']:
            logger.info(f"cuDNN版本: {hardware_info['cudnn_version']}")
    else:
        logger.warning("未检测到GPU，将使用CPU进行训练")
        logger.warning("训练速度可能较慢，建议使用GPU加速")

    logger.info("=" * 60)


# ======================================================
# 配置摘要
# ======================================================
def print_config_summary(config: Dict[str, Any]):
    """
    打印配置摘要

    参数:
        config: 配置字典
    """
    logger = LoggerManager.get_logger(__file__)

    logger.info("=" * 60)
    logger.info("配置摘要")
    logger.info("=" * 60)

    # 全局信息
    if "global" in config:
        logger.info(f"项目名称: {config['global'].get('name', 'Unknown')}")
        logger.info(f"版本: {config['global'].get('version', 'Unknown')}")
        logger.info(f"随机种子: {config['global'].get('seed', 'None')}")

    # 训练模式
    if "training_mode" in config:
        mode = config["training_mode"].get("type", "Unknown")
        subtype = config["training_mode"].get("subtype")
        logger.info(f"训练模式: {mode}" + (f" ({subtype})" if subtype else ""))

    # 模型统计
    if "models" in config:
        model_count = len(config["models"])
        logger.info(f"模型数量: {model_count}")
        for model_name in config["models"].keys():
            logger.info(f"  - {model_name}")

    # 优化器统计
    if "optimizers" in config:
        opt_count = len(config["optimizers"])
        logger.info(f"优化器数量: {opt_count}")

    # 损失函数统计
    if "losses" in config:
        loss_count = len(config["losses"])
        logger.info(f"损失函数数量: {loss_count}")

    # 数据源统计
    if "data_manager" in config:
        source_count = len(config["data_manager"])
        logger.info(f"数据源数量: {source_count}")

    # 训练流程统计
    if "training_pipeline" in config:
        pipeline_count = len(config["training_pipeline"])
        logger.info(f"训练流程数量: {pipeline_count}")

        # 获取当前模式的流程
        mode = config.get("training_mode", {}).get("type")
        if mode and mode in config["training_pipeline"]:
            pipeline = config["training_pipeline"][mode]
            if "step_sequence" in pipeline:
                step_count = len(pipeline["step_sequence"])
                logger.info(f"  当前流程步骤数: {step_count}")
            if "parameters" in pipeline:
                params = pipeline["parameters"]
                if "epochs" in params:
                    logger.info(f"  训练轮数: {params['epochs']}")
                if "episodes" in params:
                    logger.info(f"  训练episode数: {params['episodes']}")

    logger.info("=" * 60)


# ======================================================
# 配置加载的便捷函数
# ======================================================
def load_and_validate_config(
    config_file: str,
    optimize_hardware: bool = True,
    print_summary: bool = True
) -> Dict[str, Any]:
    """
    加载、验证并优化配置文件（一站式函数）

    参数:
        config_file: 配置文件名
        optimize_hardware: 是否根据硬件优化配置
        print_summary: 是否打印配置摘要

    返回:
        dict: 配置字典
    """
    logger = LoggerManager.get_logger(__file__)

    # 加载配置（带验证）
    config = load_yaml(config_file, validate=True)

    # 硬件检测和优化
    if optimize_hardware:
        logger.info("\n开始硬件检测和配置优化...")
        hardware_info = check_cuda_compatibility()
        print_hardware_info(hardware_info)
        config = optimize_config_for_hardware(config, hardware_info)

    # 打印摘要
    if print_summary:
        print_config_summary(config)

    return config


def load_config_simple(config_file: str) -> Dict[str, Any]:
    """
    简单加载配置（无验证、无优化、无摘要）

    参数:
        config_file: 配置文件名

    返回:
        dict: 配置字典
    """
    project_path = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(project_path, "config")
    config_path = os.path.join(config_dir, config_file)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ======================================================
# 配置保存
# ======================================================
def save_yaml(config: Dict[str, Any], file: str):
    """
    保存配置到 YAML 文件

    参数:
        config: 配置字典
        file: 输出文件名（相对于 config 目录）
    """
    logger = LoggerManager.get_logger(__file__)

    project_path = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(project_path, "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, file)

    logger.info(f"保存配置到: {config_file}")

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info("✓ 配置保存成功")
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        raise


# ======================================================
# 配置合并
# ======================================================
def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    合并两个配置字典（override_config 会覆盖 base_config）

    参数:
        base_config: 基础配置
        override_config: 覆盖配置

    返回:
        dict: 合并后的配置
    """
    logger = LoggerManager.get_logger(__file__)

    def _merge_dict(base: dict, override: dict) -> dict:
        """递归合并字典"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dict(result[key], value)
            else:
                result[key] = value

        return result

    logger.info("合并配置文件...")
    merged = _merge_dict(base_config, override_config)
    logger.info("✓ 配置合并完成")

    return merged


# ======================================================
# 配置模板生成
# ======================================================
def generate_config_template(
    training_mode: str,
    output_file: str = "template.yaml"
) -> Dict[str, Any]:
    """
    生成配置文件模板

    参数:
        training_mode: 训练模式
        output_file: 输出文件名

    返回:
        dict: 模板配置
    """
    logger = LoggerManager.get_logger(__file__)

    from common.train_context import TrainingModeConstants, get_default_loop_type

    if training_mode not in TrainingModeConstants.SUPPORTED_MODES:
        raise ValueError(f"不支持的训练模式: {training_mode}")

    logger.info(f"生成 {training_mode} 模式的配置模板...")

    # 基础模板
    template = {
        "global": {
            "name": f"{training_mode}_project",
            "version": "v1.0.0",
            "seed": 42
        },
        "training_mode": {
            "type": training_mode,
            "subtype": None
        },
        "models": {
            "main_model": {
                "type": "Sequential",
                "reflection": "tensorflow.keras.Sequential",
                "args": {},
                "layers": [
                    {
                        "name": "layer1",
                        "reflection": "tensorflow.keras.layers.Dense",
                        "args": {
                            "units": 128,
                            "activation": "relu"
                        }
                    }
                ]
            }
        },
        "optimizers": {
            "main_optimizer": {
                "reflection": "tensorflow.keras.optimizers.Adam",
                "args": {
                    "learning_rate": 0.001
                }
            }
        },
        "losses": {
            "main_loss": {
                "reflection": "tensorflow.keras.losses.MeanSquaredError",
                "args": {}
            }
        },
        "data_manager": {},
        "training_pipeline": {},
        "evaluation": {},
        "checkpointing": {
            "main_checkpoint": {
                "reflection": "lib.checkpoint:CheckpointManager",
                "args": {
                    "save_dir": "checkpoints",
                    "save_frequency": "epoch",
                    "keep_last_n": 5,
                    "save_best_only": True,
                    "monitor": "val_loss",
                    "mode": "min",
                    "enabled": True
                }
            }
        },
        "logging": {
            "file_logger": {
                "reflection": "lib.logger:FileLogger",
                "args": {
                    "log_dir": "logs",
                    "log_frequency": "step",
                    "metrics_to_log": ["loss", "accuracy"],
                    "enabled": True
                }
            },
            "tensorboard_logger": {
                "reflection": "lib.logger:TensorBoardLogger",
                "args": {
                    "log_dir": "logs/tensorboard",
                    "update_freq": "epoch",
                    "enabled": True
                }
            }
        },
        "export": {
            "onnx_exporter": {
                "reflection": "lib.export:ONNXExporter",
                "args": {
                    "model": "main_model",
                    "output_path": "outputs/model.onnx",
                    "opset_version": 16,
                    "enabled": False
                }
            }
        },
        "deployment": {
            "rest_api_service": {
                "reflection": "lib.deployment:RestAPIServer",
                "args": {
                    "model_path": "outputs/model.onnx",
                    "host": "0.0.0.0",
                    "port": 9000,
                    "enabled": False
                }
            }
        }
    }

    # 根据训练模式添加特定配置
    if training_mode == "supervised":
        template["data_manager"]["supervised_source"] = {
            "train": {
                "reflection": "lib.utils:build_csv_dataset",
                "args": {
                    "file_path": "data/train.csv",
                    "batch_size": 64,
                    "shuffle": True
                }
            },
            "val": {
                "reflection": "lib.utils:build_csv_dataset",
                "args": {
                    "file_path": "data/val.csv",
                    "batch_size": 64,
                    "shuffle": False
                }
            }
        }

        template["training_pipeline"]["supervised"] = {
            "loop_type": "epoch_batch",
            "parameters": {
                "epochs": 100,
                "steps_per_epoch": None
            },
            "step_sequence": [
                {
                    "name": "fetch_batch",
                    "reflection": "lib.utils:get_batch",
                    "args": {
                        "loader": "train_loader"
                    },
                    "bridge": None
                },
                {
                    "name": "forward_pass",
                    "reflection": "lib.utils:forward",
                    "args": {
                        "model": "${main_model}",
                        "inputs": "${fetch_batch.x}",
                        "training": True
                    },
                    "bridge": None
                },
                {
                    "name": "compute_loss",
                    "reflection": "lib.utils:compute_loss",
                    "args": {
                        "predictions": "${forward_pass}",
                        "targets": "${fetch_batch.y}",
                        "loss_fn": "${main_loss}"
                    },
                    "bridge": None
                },
                {
                    "name": "backward_pass",
                    "reflection": "lib.utils:compute_gradients",
                    "args": {
                        "loss": "${compute_loss}",
                        "model": "${main_model}"
                    },
                    "bridge": None
                },
                {
                    "name": "update_params",
                    "reflection": "lib.utils:apply_gradients",
                    "args": {
                        "optimizer": "${main_optimizer}",
                        "gradients": "${backward_pass}",
                        "model": "${main_model}"
                    },
                    "bridge": None
                }
            ],
            "loop_condition": {
                "check_type": "epoch_based",
                "max_epochs": 100
            },
            "evaluation": {
                "frequency": "epoch",
                "eval_split": "val"
            }
        }

    elif training_mode == "reinforcement":
        template["data_manager"]["reinforcement_source"] = {
            "type": "interactive",
            "client": {
                "reflection": "lib.utils:NetworkClient",
                "args": {
                    "host": "localhost",
                    "port": 8080,
                    "timeout": 30
                }
            }
        }

        template["training_pipeline"]["reinforcement"] = {
            "loop_type": "episode_step",
            "parameters": {
                "episodes": 1000,
                "steps_per_episode": 500
            },
            "step_sequence": [
                {
                    "name": "observe_state",
                    "reflection": "lib.utils:observe_state",
                    "args": {
                        "client": "${rl_client}",
                        "endpoint": "state"
                    },
                    "bridge": None
                }
            ],
            "loop_condition": {
                "check_type": "episode_based",
                "max_episodes": 1000
            }
        }

    # 保存模板
    save_yaml(template, output_file)

    logger.info(f"✓ 模板生成完成: {output_file}")

    return template


# ======================================================
# 配置差异比较
# ======================================================
def compare_configs(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    path: str = ""
) -> List[str]:
    """
    比较两个配置的差异

    参数:
        config1: 第一个配置
        config2: 第二个配置
        path: 当前路径（用于递归）

    返回:
        差异列表
    """
    differences = []

    # 检查config1中有但config2中没有的键
    for key in config1:
        current_path = f"{path}.{key}" if path else key

        if key not in config2:
            differences.append(f"- {current_path}: 仅在config1中存在")
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            differences.extend(compare_configs(config1[key], config2[key], current_path))
        elif config1[key] != config2[key]:
            differences.append(f"≠ {current_path}: {config1[key]} vs {config2[key]}")

    # 检查config2中有但config1中没有的键
    for key in config2:
        current_path = f"{path}.{key}" if path else key

        if key not in config1:
            differences.append(f"+ {current_path}: 仅在config2中存在")

    return differences
