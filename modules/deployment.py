# -*- coding: utf-8 -*-
"""
modules/deployment.py
模型部署管理器：从配置上下文部署模型到目标平台

导入限制：
- 仅导入 common 文件夹内的函数和 lib 文件夹内的基础类
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from common.interfaces import DeploymentInterface
from common.train_context import TrainContext
from common.common import call_target, LoggerManager

logger = LoggerManager.get_logger(__file__)


class ModelDeploymentManager(DeploymentInterface):
    """
    模型部署管理器

    职责：
    1. 读取 TrainContext 中的部署配置
    2. 使用 lib.deployment 中的基础类部署模型
    3. 配置服务端点（REST API/gRPC/TF Serving）
    4. 健康检查和监控
    5. 生成部署文档和脚本

    标准生命周期：
    initialize() -> validate_deployment() -> prepare_deployment() ->
    deploy() -> health_check() -> finalize()

    支持的部署方式：
    - REST API: 使用 RestAPIServer (Flask)
    - gRPC: 使用 GRPCServer
    - TensorFlow Serving: 使用 TFServingDeployer
    - Docker: 使用 DockerDeployer
    - Custom: 通过反射调用自定义部署函数

    注意：优先使用 lib.deployment 中的类，通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化模型部署管理器

        参数:
            context: 训练上下文
        """
        super().__init__(context)
        self.deployment_configs: Dict[str, Any] = {}
        self.deployment_info: Dict[str, Dict[str, Any]] = {}
        self.deployment_servers: Dict[str, Any] = {}
        self.endpoints: Dict[str, str] = {}

    def initialize(self) -> None:
        """
        从 TrainContext 初始化模块

        职责：
        1. 验证上下文中是否有部署配置
        2. 加载部署配置数据
        3. 初始化内部状态变量

        抛出:
            ValueError: 如果上下文中缺少必需配置
        """
        logger.info("初始化模型部署管理器")

        # 提取部署配置
        self.deployment_configs = self._extract_deployment_configs()

        if not self.deployment_configs:
            logger.warning("未找到部署配置，部署功能将不可用")
            self.deployment_configs = {}

        logger.info(f"找到 {len(self.deployment_configs)} 个部署配置")

        # 初始化状态
        self.deployment_info = {}
        self.deployment_servers = {}
        self.endpoints = {}

        self._initialized = True

    def validate_deployment(self) -> bool:
        """
        验证部署配置

        返回:
            配置是否有效
        """
        logger.info("验证部署配置")

        if not self.deployment_configs:
            logger.warning("部署配置为空，跳过验证")
            return True

        # 验证每个部署配置
        for deploy_name, deploy_config in self.deployment_configs.items():
            if not isinstance(deploy_config, dict):
                logger.error(f"部署配置 '{deploy_name}' 必须是字典")
                return False

            # 检查必需字段
            if 'type' not in deploy_config and 'reflection' not in deploy_config:
                logger.error(
                    f"部署配置 '{deploy_name}' 缺少 'type' 或 'reflection' 字段"
                )
                return False

            # 验证类型
            if 'type' in deploy_config:
                deploy_type = deploy_config['type'].lower()
                supported_types = ['rest_api', 'grpc', 'tf_serving', 'docker', 'custom']
                if deploy_type not in supported_types:
                    logger.error(
                        f"部署配置 '{deploy_name}' 的类型 '{deploy_type}' 不支持，"
                        f"支持的类型: {supported_types}"
                    )
                    return False

        logger.info("部署配置验证通过")
        return True

    def prepare_deployment(self, model_path: str) -> Dict[str, Any]:
        """
        准备部署

        参数:
            model_path: 模型路径

        返回:
            部署准备信息
        """
        logger.info(f"准备部署，模型路径: {model_path}")

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.warning(f"模型路径不存在: {model_path}")
            # 不抛出异常，因为可能是配置阶段

        # 准备部署信息
        prep_info = {
            "model_path": model_path,
            "model_exists": os.path.exists(model_path),
            "model_format": self._detect_model_format(model_path)
        }

        if prep_info["model_exists"]:
            prep_info["model_size"] = self._get_model_size(model_path)

        logger.info(f"部署准备完成: {prep_info}")

        return prep_info

    def deploy(
        self,
        deploy_name: str,
        deploy_config: Any
    ) -> Dict[str, Any]:
        """
        部署单个服务

        参数:
            deploy_name: 部署配置名称
            deploy_config: 部署配置

        返回:
            部署信息
        """
        logger.info(f"开始部署服务: {deploy_name}")

        if not isinstance(deploy_config, dict):
            raise TypeError(f"部署配置必须是字典: {type(deploy_config)}")

        # 获取模型路径
        model_path = self._resolve_model_path(deploy_config)

        # 准备部署
        prep_info = self.prepare_deployment(model_path)

        # 根据类型或反射调用部署
        if 'reflection' in deploy_config:
            # 使用自定义反射调用
            deploy_info = self._deploy_with_reflection(
                deploy_name,
                deploy_config,
                model_path,
                prep_info
            )
        else:
            # 使用内置类型部署
            deploy_type = deploy_config['type'].lower()
            deploy_info = self._deploy_with_type(
                deploy_type,
                deploy_name,
                model_path,
                deploy_config
            )

        # 保存部署信息
        self.deployment_info[deploy_name] = deploy_info

        # 保存服务端点
        if 'endpoint' in deploy_info:
            self.endpoints[deploy_name] = deploy_info['endpoint']

        logger.info(f"服务 '{deploy_name}' 部署成功")

        # 健康检查
        if deploy_config.get('health_check', False) and 'endpoint' in deploy_info:
            is_healthy = self.health_check(deploy_info['endpoint'])
            deploy_info['health_status'] = 'healthy' if is_healthy else 'unhealthy'

        return deploy_info

    def deploy_all(self) -> Dict[str, Any]:
        """
        部署所有配置的服务

        返回:
            部署结果字典 {deploy_name: deployment_info}
        """
        logger.info("开始部署所有服务")

        if not self.deployment_configs:
            logger.warning("没有部署配置，跳过部署")
            return {}

        all_deployments = {}

        for deploy_name, deploy_config in self.deployment_configs.items():
            try:
                deploy_info = self.deploy(deploy_name, deploy_config)
                all_deployments[deploy_name] = deploy_info

            except Exception as e:
                logger.error(f"部署 '{deploy_name}' 失败: {str(e)}", exc_info=True)
                all_deployments[deploy_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        success_count = sum(1 for d in all_deployments.values() if d.get('status') != 'failed')
        logger.info(f"完成部署，成功 {success_count}/{len(self.deployment_configs)} 个")

        return all_deployments

    def health_check(self, endpoint: str) -> bool:
        """
        健康检查（可选）

        参数:
            endpoint: 服务端点

        返回:
            服务是否健康
        """
        logger.info(f"执行健康检查: {endpoint}")

        try:
            import requests

            # 尝试访问健康检查端点
            health_url = f"{endpoint}/health" if not endpoint.endswith('/health') else endpoint
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                logger.info(f"服务健康: {endpoint}")
                return True
            else:
                logger.warning(f"服务不健康: {endpoint}, 状态码: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"健康检查失败: {endpoint}, 错误: {str(e)}")
            return False

    def execute(self) -> Dict[str, Any]:
        """
        执行核心逻辑：部署所有服务

        返回:
            部署结果
        """
        logger.info("开始执行部署流程")

        if not self.validate_deployment():
            raise ValueError("部署配置验证失败")

        # 部署所有服务
        deployment_results = self.deploy_all()

        logger.info("部署流程执行完成")

        return deployment_results

    def finalize(self) -> None:
        """
        清理和保存结果

        职责：
        1. 将部署信息写回 TrainContext
        2. 保存服务端点
        3. 记录日志
        """
        logger.info("完成部署，保存结果")

        # 保存部署信息到上下文
        if self.deployment_info:
            self.context.execution_results['deployment'] = self.deployment_info

        # 保存服务端点
        if self.endpoints:
            self.context.execution_results['service_endpoints'] = self.endpoints

        logger.info(f"部署信息已保存，包含 {len(self.deployment_info)} 个服务")

    # ========================================================================
    # 私有方法：部署实现（使用 lib.deployment 中的类）
    # ========================================================================

    def _deploy_with_type(
        self,
        deploy_type: str,
        deploy_name: str,
        model_path: str,
        deploy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据类型部署服务（使用 lib.deployment 中的类）

        参数:
            deploy_type: 部署类型
            deploy_name: 部署名称
            model_path: 模型路径
            deploy_config: 部署配置

        返回:
            部署信息
        """
        if deploy_type == 'rest_api':
            return self._deploy_rest_api(deploy_name, model_path, deploy_config)

        elif deploy_type == 'grpc':
            return self._deploy_grpc(deploy_name, model_path, deploy_config)

        elif deploy_type == 'tf_serving':
            return self._deploy_tf_serving(deploy_name, model_path, deploy_config)

        elif deploy_type == 'docker':
            return self._deploy_docker(deploy_name, model_path, deploy_config)

        elif deploy_type == 'custom':
            return self._deploy_custom(deploy_name, model_path, deploy_config)

        else:
            raise ValueError(f"不支持的部署类型: {deploy_type}")

    def _deploy_rest_api(
        self,
        deploy_name: str,
        model_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        部署 REST API 服务（使用 lib.deployment.RestAPIServer）

        参数:
            deploy_name: 部署名称
            model_path: 模型路径
            config: 配置

        返回:
            部署信息
        """
        logger.info(f"部署 REST API: {deploy_name} (使用 RestAPIServer)")

        # 准备参数
        host = config.get('host', '0.0.0.0')
        port = config.get('port', 9000)
        endpoints = config.get('endpoints', None)
        performance = config.get('performance', None)
        enabled = config.get('enabled', True)

        # 使用 call_target 创建 RestAPIServer 实例
        server = call_target(
            reflection='lib.deployment:RestAPIServer',
            args={
                'model_path': model_path,
                'host': host,
                'port': port,
                'endpoints': endpoints,
                'performance': performance,
                'enabled': enabled
            }
        )

        # 保存服务器实例
        self.deployment_servers[deploy_name] = server

        # 如果配置要求立即启动
        if config.get('auto_start', False):
            logger.info(f"自动启动 REST API 服务: {deploy_name}")
            # 注意：start() 会阻塞，通常在单独的进程中运行
            # server.start()

        return {
            "type": "rest_api",
            "status": "configured",
            "deploy_name": deploy_name,
            "model_path": model_path,
            "host": host,
            "port": port,
            "endpoint": f"http://{host}:{port}",
            "endpoints": endpoints or {
                "predict": "/api/predict",
                "health": "/health",
                "metrics": "/metrics"
            },
            "instructions": f"使用 server.start() 启动服务，或在单独进程中运行"
        }

    def _deploy_grpc(
        self,
        deploy_name: str,
        model_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        部署 gRPC 服务（使用 lib.deployment.GRPCServer）

        参数:
            deploy_name: 部署名称
            model_path: 模型路径
            config: 配置

        返回:
            部署信息
        """
        logger.info(f"部署 gRPC: {deploy_name} (使用 GRPCServer)")

        # 准备参数
        host = config.get('host', '0.0.0.0')
        port = config.get('port', 50051)
        max_workers = config.get('max_workers', 10)
        enabled = config.get('enabled', True)

        # 使用 call_target 创建 GRPCServer 实例
        server = call_target(
            reflection='lib.deployment:GRPCServer',
            args={
                'model_path': model_path,
                'host': host,
                'port': port,
                'max_workers': max_workers,
                'enabled': enabled
            }
        )

        # 保存服务器实例
        self.deployment_servers[deploy_name] = server

        return {
            "type": "grpc",
            "status": "configured",
            "deploy_name": deploy_name,
            "model_path": model_path,
            "host": host,
            "port": port,
            "endpoint": f"{host}:{port}",
            "max_workers": max_workers,
            "instructions": "使用 server.start() 启动 gRPC 服务"
        }

    def _deploy_tf_serving(
        self,
        deploy_name: str,
        model_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        部署 TensorFlow Serving（使用 lib.deployment.TFServingDeployer）

        参数:
            deploy_name: 部署名称
            model_path: 模型路径
            config: 配置

        返回:
            部署信息
        """
        logger.info(f"部署 TensorFlow Serving: {deploy_name} (使用 TFServingDeployer)")

        # 准备参数
        model_name = config.get('model_name', deploy_name)
        port = config.get('port', 8501)
        rest_api_port = config.get('rest_api_port', 8500)
        enabled = config.get('enabled', True)

        # 使用 call_target 创建 TFServingDeployer 实例
        deployer = call_target(
            reflection='lib.deployment:TFServingDeployer',
            args={
                'model_path': model_path,
                'model_name': model_name,
                'port': port,
                'rest_api_port': rest_api_port,
                'enabled': enabled
            }
        )

        # 保存部署器实例
        self.deployment_servers[deploy_name] = deployer

        # 如果配置要求立即部署
        if config.get('auto_deploy', False):
            logger.info(f"自动部署到 TensorFlow Serving: {deploy_name}")
            deployer.deploy()

        return {
            "type": "tf_serving",
            "status": "configured",
            "deploy_name": deploy_name,
            "model_name": model_name,
            "model_path": model_path,
            "grpc_port": port,
            "rest_api_port": rest_api_port,
            "endpoint": f"http://localhost:{rest_api_port}",
            "grpc_endpoint": f"localhost:{port}",
            "instructions": "使用 deployer.deploy() 查看 Docker 命令"
        }

    def _deploy_docker(
        self,
        deploy_name: str,
        model_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Docker 容器化部署（使用 lib.deployment.DockerDeployer）

        参数:
            deploy_name: 部署名称
            model_path: 模型路径
            config: 配置

        返回:
            部署信息
        """
        logger.info(f"Docker 容器化部署: {deploy_name} (使用 DockerDeployer)")

        # 准备参数
        dockerfile_template = config.get('dockerfile_template', 'deploy/Dockerfile.template')
        image_name = config.get('image_name', f'{deploy_name}:latest')
        container_port = config.get('container_port', 9000)
        host_port = config.get('host_port', 9000)
        enabled = config.get('enabled', True)

        # 使用 call_target 创建 DockerDeployer 实例
        deployer = call_target(
            reflection='lib.deployment:DockerDeployer',
            args={
                'model_path': model_path,
                'dockerfile_template': dockerfile_template,
                'image_name': image_name,
                'container_port': container_port,
                'host_port': host_port,
                'enabled': enabled
            }
        )

        # 保存部署器实例
        self.deployment_servers[deploy_name] = deployer

        # 如果配置要求自动构建
        if config.get('auto_build', False):
            logger.info(f"自动构建 Docker 镜像: {deploy_name}")
            deployer.build()

        # 如果配置要求自动运行
        if config.get('auto_run', False):
            logger.info(f"自动运行 Docker 容器: {deploy_name}")
            deployer.run()

        return {
            "type": "docker",
            "status": "configured",
            "deploy_name": deploy_name,
            "image_name": image_name,
            "container_port": container_port,
            "host_port": host_port,
            "endpoint": f"http://localhost:{host_port}",
            "instructions": "使用 deployer.build() 构建镜像，deployer.run() 运行容器"
        }

    def _deploy_custom(
        self,
        deploy_name: str,
        model_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        自定义部署（通过配置中的自定义逻辑）

        参数:
            deploy_name: 部署名称
            model_path: 模型路径
            config: 配置

        返回:
            部署信息
        """
        logger.info(f"自定义部署: {deploy_name}")

        # 获取自定义配置
        custom_config = config.get('custom_config', {})

        return {
            "type": "custom",
            "status": "configured",
            "deploy_name": deploy_name,
            "model_path": model_path,
            "custom_config": custom_config,
            "instructions": "请根据 custom_config 手动部署"
        }

    def _deploy_with_reflection(
        self,
        deploy_name: str,
        deploy_config: Dict[str, Any],
        model_path: str,
        prep_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用反射调用部署

        参数:
            deploy_name: 部署名称
            deploy_config: 部署配置
            model_path: 模型路径
            prep_info: 准备信息

        返回:
            部署信息
        """
        logger.info(f"使用反射调用部署服务: {deploy_name}")

        reflection = deploy_config['reflection']
        args = deploy_config.get('args', {})

        # 添加默认参数
        if 'model_path' not in args:
            args['model_path'] = model_path
        if 'prep_info' not in args:
            args['prep_info'] = prep_info
        if 'deploy_name' not in args:
            args['deploy_name'] = deploy_name

        # 解析参数中的变量引用
        resolved_args = self._resolve_arguments(args)

        try:
            # 调用部署函数
            result = call_target(
                reflection=reflection,
                args=resolved_args
            )

            # 如果返回字典，直接使用
            if isinstance(result, dict):
                if 'status' not in result:
                    result['status'] = 'deployed'
                return result

            # 如果返回对象，保存并返回信息
            self.deployment_servers[deploy_name] = result

            return {
                "status": "deployed",
                "deploy_name": deploy_name,
                "result": str(result)
            }

        except Exception as e:
            logger.error(f"反射调用部署失败: {str(e)}", exc_info=True)
            raise

    # ========================================================================
    # 私有方法：辅助功能
    # ========================================================================

    def _extract_deployment_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        从上下文中提取部署配置

        返回:
            部署配置字典
        """
        configs = {}

        # 1. 从 training_pipeline 的参数中查找
        training_mode = self.context.training_mode
        if training_mode in self.context.training_pipelines:
            pipeline = self.context.training_pipelines[training_mode]
            if hasattr(pipeline, 'parameters') and 'deployment' in pipeline.parameters:
                configs = pipeline.parameters['deployment']
                logger.debug("从 training_pipeline 中找到部署配置")

        # 2. 从全局配置中查找
        if hasattr(self.context, 'global_config'):
            global_config = self.context.global_config
            if isinstance(global_config, dict) and 'deployment' in global_config:
                configs.update(global_config['deployment'])

        # 3. 从自定义字段查找
        if hasattr(self.context, 'deployment'):
            configs.update(self.context.deployment)

        return configs

    def _resolve_model_path(self, deploy_config: Dict[str, Any]) -> str:
        """
        解析模型路径

        参数:
            deploy_config: 部署配置

        返回:
            模型路径
        """
        # 1. 从配置中获取
        if 'model_path' in deploy_config:
            model_path = deploy_config['model_path']
            if isinstance(model_path, str) and model_path.startswith('$'):
                return str(self._resolve_variable(model_path[1:]))
            return model_path

        # 2. 从导出结果中获取
        if 'export_paths' in self.context.execution_results:
            export_paths = self.context.execution_results['export_paths']
            model_name = deploy_config.get('model', list(self.context.container.models.keys())[0])
            if model_name in export_paths:
                return export_paths[model_name]

        # 3. 使用默认路径
        model_name = deploy_config.get('model', 'default_model')
        return f'./exports/{model_name}'

    def _resolve_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析参数中的变量引用

        参数:
            args: 原始参数字典

        返回:
            解析后的参数字典
        """
        resolved_args = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith('$'):
                var_name = value[1:]
                resolved_args[key] = self._resolve_variable(var_name)
            elif isinstance(value, dict):
                resolved_args[key] = self._resolve_arguments(value)
            elif isinstance(value, list):
                resolved_args[key] = [
                    self._resolve_variable(v[1:]) if isinstance(v, str) and v.startswith('$') else v
                    for v in value
                ]
            else:
                resolved_args[key] = value

        return resolved_args

    def _resolve_variable(self, var_name: str) -> Any:
        """
        解析变量引用

        参数:
            var_name: 变量名称

        返回:
            变量值
        """
        # 支持点号访问
        if '.' in var_name:
            parts = var_name.split('.')
            obj = self.context

            # 尝试从特定位置访问
            if parts[0] == 'models':
                obj = self.context.container.models
                parts = parts[1:]
            elif parts[0] == 'export_paths':
                obj = self.context.execution_results.get('export_paths', {})
                parts = parts[1:]

            # 逐层访问
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part)
                elif hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"无法解析变量: ${var_name}")

            return obj

        # 直接变量名
        if hasattr(self.context.container, 'models') and var_name in self.context.container.models:
            return self.context.container.models[var_name]

        if 'export_paths' in self.context.execution_results and var_name in self.context.execution_results['export_paths']:
            return self.context.execution_results['export_paths'][var_name]

        if var_name in self.context.execution_results:
            return self.context.execution_results[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${var_name}")

    def _get_model_size(self, model_path: str) -> int:
        """获取模型大小（字节）"""
        total_size = 0

        if os.path.isfile(model_path):
            total_size = os.path.getsize(model_path)
        elif os.path.isdir(model_path):
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

        return total_size

    def _detect_model_format(self, model_path: str) -> str:
        """检测模型格式"""
        if not os.path.exists(model_path):
            return 'unknown'

        if os.path.isfile(model_path):
            ext = os.path.splitext(model_path)[1].lower()
            format_map = {
                '.h5': 'h5',
                '.onnx': 'onnx',
                '.tflite': 'tflite',
                '.pb': 'pb'
            }
            return format_map.get(ext, 'unknown')
        elif os.path.isdir(model_path):
            # 检查是否是 SavedModel 格式
            if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                return 'savedmodel'

        return 'unknown'


# ============================================================================
# 便捷函数
# ============================================================================

def deploy_model_from_context(
    context: TrainContext,
    deploy_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    从训练上下文部署模型的便捷函数

    参数:
        context: 训练上下文
        deploy_name: 部署配置名称（可选）

    返回:
        部署结果字典

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> # ... 配置 context
        >>> results = deploy_model_from_context(context)
    """
    manager = ModelDeploymentManager(context)
    return manager.run()


def create_deployment_manager(context: TrainContext) -> ModelDeploymentManager:
    """
    创建并初始化部署管理器的便捷函数

    参数:
        context: 训练上下文

    返回:
        ModelDeploymentManager 实例

    示例:
        >>> from common.train_context import TrainContext
        >>> context = TrainContext()
        >>> manager = create_deployment_manager(context)
        >>> results = manager.deploy_all()
    """
    manager = ModelDeploymentManager(context)
    manager.initialize()
    return manager


def deploy_single_service(
    context: TrainContext,
    model_name: str,
    deploy_type: str,
    model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    部署单个服务的便捷函数

    参数:
        context: 训练上下文
        model_name: 模型名称
        deploy_type: 部署类型 ('rest_api', 'grpc', 'tf_serving', 'docker')
        model_path: 模型路径（可选）
        **kwargs: 额外配置

    返回:
        部署信息

    示例:
        >>> info = deploy_single_service(
        ...     context,
        ...     "classifier",
        ...     "rest_api",
        ...     model_path="./exports/classifier",
        ...     port=9000
        ... )
    """
    manager = ModelDeploymentManager(context)
    manager.initialize()

    # 如果没有提供模型路径，尝试从导出结果获取
    if model_path is None:
        if 'export_paths' in context.execution_results:
            export_paths = context.execution_results['export_paths']
            model_path = export_paths.get(model_name, f'./exports/{model_name}')
        else:
            model_path = f'./exports/{model_name}'

    # 构造部署配置
    deploy_config = {
        'type': deploy_type,
        'model_path': model_path,
        'model': model_name,
        **kwargs
    }

    return manager.deploy('temp_deploy', deploy_config)


def get_service_endpoints(context: TrainContext) -> Dict[str, str]:
    """
    获取所有服务端点的便捷函数

    参数:
        context: 训练上下文

    返回:
        服务端点字典 {deploy_name: endpoint}

    示例:
        >>> endpoints = get_service_endpoints(context)
        >>> print(f"REST API: {endpoints.get('rest_api')}")
    """
    if 'service_endpoints' in context.execution_results:
        return context.execution_results['service_endpoints']

    # 如果没有保存的端点，尝试从部署信息中提取
    if 'deployment' in context.execution_results:
        deployment_info = context.execution_results['deployment']
        endpoints = {}
        for deploy_name, info in deployment_info.items():
            if 'endpoint' in info:
                endpoints[deploy_name] = info['endpoint']
        return endpoints

    return {}


def get_deployment_server(
    context: TrainContext,
    deploy_name: str
) -> Optional[Any]:
    """
    获取部署服务器实例的便捷函数

    参数:
        context: 训练上下文
        deploy_name: 部署名称

    返回:
        服务器实例（RestAPIServer/GRPCServer等）

    示例:
        >>> manager = create_deployment_manager(context)
        >>> manager.deploy_all()
        >>> server = get_deployment_server(context, 'rest_api')
        >>> if server:
        ...     server.start()  # 启动服务器
    """
    if 'deployment' not in context.execution_results:
        logger.warning("未找到部署信息")
        return None

    # 从管理器中获取服务器实例
    # 注意：这需要管理器实例仍然存在
    logger.warning("服务器实例获取功能需要保持管理器引用")
    return None


def start_rest_api_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 9000,
    **kwargs
) -> None:
    """
    快速启动 REST API 服务器的便捷函数

    参数:
        model_path: 模型路径
        host: 主机地址
        port: 端口号
        **kwargs: 额外参数

    示例:
        >>> # 在单独的脚本中使用
        >>> start_rest_api_server(
        ...     "./exports/classifier",
        ...     host="0.0.0.0",
        ...     port=9000
        ... )
    """
    from lib.deployment import RestAPIServer

    server = RestAPIServer(
        model_path=model_path,
        host=host,
        port=port,
        **kwargs
    )

    logger.info(f"启动 REST API 服务器: http://{host}:{port}")
    server.start()


def start_grpc_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 50051,
    **kwargs
) -> None:
    """
    快速启动 gRPC 服务器的便捷函数

    参数:
        model_path: 模型路径
        host: 主机地址
        port: 端口号
        **kwargs: 额外参数

    示例:
        >>> # 在单独的脚本中使用
        >>> start_grpc_server(
        ...     "./exports/classifier",
        ...     host="0.0.0.0",
        ...     port=50051
        ... )
    """
    from lib.deployment import GRPCServer

    server = GRPCServer(
        model_path=model_path,
        host=host,
        port=port,
        **kwargs
    )

    logger.info(f"启动 gRPC 服务器: {host}:{port}")
    server.start()


def generate_deployment_docs(
    context: TrainContext,
    output_file: str = './deployment/README.md'
) -> str:
    """
    生成部署文档的便捷函数

    参数:
        context: 训练上下文
        output_file: 输出文件路径

    返回:
        文档内容

    示例:
        >>> doc = generate_deployment_docs(context)
        >>> print(doc)
    """
    if 'deployment' not in context.execution_results:
        return "# 部署文档\n\n暂无部署信息"

    deployment_info = context.execution_results['deployment']

    # 生成 Markdown 文档
    doc_lines = [
        "# 模型部署文档",
        "",
        f"**项目名称**: {getattr(context, 'project_name', 'Unknown')}",
        f"**训练模式**: {context.training_mode}",
        "",
        "---",
        "",
        "## 部署服务概览",
        ""
    ]

    # 添加概览表格
    doc_lines.extend([
        "| 服务名称 | 类型 | 状态 | 端点 |",
        "|---------|------|------|------|"
    ])

    for deploy_name, info in deployment_info.items():
        deploy_type = info.get('type', 'unknown')
        status = info.get('status', 'unknown')
        endpoint = info.get('endpoint', 'N/A')
        doc_lines.append(f"| {deploy_name} | {deploy_type} | {status} | {endpoint} |")

    doc_lines.append("")

    # 添加每个服务的详细信息
    doc_lines.extend([
        "---",
        "",
        "## 详细配置",
        ""
    ])

    for deploy_name, info in deployment_info.items():
        doc_lines.extend([
            f"### {deploy_name}",
            ""
        ])

        # 基本信息
        doc_lines.extend([
            "**基本信息**:",
            "",
            f"- **类型**: `{info.get('type', 'unknown')}`",
            f"- **状态**: `{info.get('status', 'unknown')}`",
        ])

        if 'model_path' in info:
            doc_lines.append(f"- **模型路径**: `{info['model_path']}`")

        if 'endpoint' in info:
            doc_lines.append(f"- **服务端点**: `{info['endpoint']}`")

        doc_lines.append("")

        # 配置详情
        if info.get('type') == 'rest_api':
            doc_lines.extend([
                "**REST API 端点**:",
                ""
            ])
            endpoints = info.get('endpoints', {})
            for ep_name, ep_path in endpoints.items():
                doc_lines.append(f"- **{ep_name}**: `{info.get('endpoint', '')}{ep_path}`")
            doc_lines.append("")

        elif info.get('type') == 'grpc':
            doc_lines.extend([
                "**gRPC 配置**:",
                "",
                f"- **最大工作线程**: {info.get('max_workers', 'N/A')}",
                ""
            ])

        elif info.get('type') == 'tf_serving':
            doc_lines.extend([
                "**TensorFlow Serving 配置**:",
                "",
                f"- **模型名称**: `{info.get('model_name', 'N/A')}`",
                f"- **gRPC 端点**: `{info.get('grpc_endpoint', 'N/A')}`",
                f"- **REST API 端点**: `{info.get('endpoint', 'N/A')}`",
                ""
            ])

        elif info.get('type') == 'docker':
            doc_lines.extend([
                "**Docker 配置**:",
                "",
                f"- **镜像名称**: `{info.get('image_name', 'N/A')}`",
                f"- **容器端口**: `{info.get('container_port', 'N/A')}`",
                f"- **主机端口**: `{info.get('host_port', 'N/A')}`",
                ""
            ])

        # 使用说明
        if 'instructions' in info:
            doc_lines.extend([
                "**使用说明**:",
                "",
                "```bash",
                info['instructions'],
                "```",
                ""
            ])

        doc_lines.append("")

    # 添加健康检查说明
    doc_lines.extend([
        "---",
        "",
        "## 健康检查",
        "",
        "访问以下端点进行健康检查:",
        ""
    ])

    endpoints = get_service_endpoints(context)
    if endpoints:
        for deploy_name, endpoint in endpoints.items():
            health_endpoint = f"{endpoint}/health" if not endpoint.endswith('/health') else endpoint
            doc_lines.append(f"- **{deploy_name}**: `{health_endpoint}`")
    else:
        doc_lines.append("暂无可用端点")

    doc_lines.extend([
        "",
        "---",
        "",
        "## 使用示例",
        "",
        "### REST API 调用示例",
        "",
        "```python",
        "import requests",
        "",
        "# 发送预测请求",
        "response = requests.post(",
        "    'http://localhost:9000/api/predict',",
        "    json={'inputs': [[1, 2, 3, 4]]}",
        ")",
        "",
        "predictions = response.json()['predictions']",
        "print(predictions)",
        "```",
        "",
        "### cURL 调用示例",
        "",
        "```bash",
        "# 健康检查",
        "curl http://localhost:9000/health",
        "",
        "# 预测请求",
        "curl -X POST http://localhost:9000/api/predict \\",
        "  -H 'Content-Type: application/json' \\",
        "  -d '{\"inputs\": [[1, 2, 3, 4]]}'",
        "```",
        "",
        "---",
        "",
        f"*文档生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ""
    ])

    # 生成文档内容
    doc_content = "\n".join(doc_lines)

    # 保存到文件
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    logger.info(f"部署文档已生成: {output_file}")

    return doc_content


def list_available_deployers() -> Dict[str, str]:
    """
    列出所有可用的部署器

    返回:
        部署器字典 {deployer_name: description}

    示例:
        >>> deployers = list_available_deployers()
        >>> for name, desc in deployers.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        "rest_api": "REST API 服务器 (Flask) - 适用于 Web 应用集成",
        "grpc": "gRPC 服务器 - 适用于高性能微服务",
        "tf_serving": "TensorFlow Serving - 适用于生产环境的 TF 模型",
        "docker": "Docker 容器 - 适用于容器化部署",
        "custom": "自定义部署 - 通过反射调用自定义函数"
    }


def get_deployment_example_config() -> Dict[str, Any]:
    """
    获取部署配置示例

    返回:
        示例配置字典

    示例:
        >>> example = get_deployment_example_config()
        >>> import yaml
        >>> print(yaml.dump(example))
    """
    return {
        "deployment": {
            "rest_api": {
                "type": "rest_api",
                "model_path": "$export_paths.classifier",
                "host": "0.0.0.0",
                "port": 9000,
                "endpoints": {
                    "predict": "/api/predict",
                    "health": "/health",
                    "metrics": "/metrics"
                },
                "performance": {
                    "batch_size": 32,
                    "timeout": 30,
                    "workers": 4
                },
                "enabled": True,
                "auto_start": False,
                "health_check": True
            },
            "grpc": {
                "type": "grpc",
                "model_path": "$export_paths.classifier",
                "host": "0.0.0.0",
                "port": 50051,
                "max_workers": 10,
                "enabled": True
            },
            "tf_serving": {
                "type": "tf_serving",
                "model_path": "$export_paths.classifier",
                "model_name": "classifier",
                "port": 8501,
                "rest_api_port": 8500,
                "enabled": True,
                "auto_deploy": False
            },
            "docker": {
                "type": "docker",
                "model_path": "$export_paths.classifier",
                "image_name": "my-model:latest",
                "container_port": 9000,
                "host_port": 9000,
                "enabled": True,
                "auto_build": False,
                "auto_run": False
            }
        }
    }
