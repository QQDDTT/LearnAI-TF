# -*- coding: utf-8 -*-
"""
lib/deployment.py
模型部署器
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


class RestAPIServer:
    """
    REST API服务器
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 9000,
        endpoints: Optional[Dict[str, str]] = None,
        performance: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化REST API服务器

        参数:
            model_path: 模型文件路径
            host: 主机地址
            port: 端口号
            endpoints: API端点配置
            performance: 性能配置
            enabled: 是否启用
        """
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.endpoints = endpoints or {
            "predict": "/api/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
        self.performance = performance or {
            "batch_size": 32,
            "timeout": 30,
            "workers": 4
        }
        self.enabled = enabled

        if self.enabled:
            logger.info(f"RestAPIServer 初始化")
            logger.info(f"  地址: http://{host}:{port}")
            logger.info(f"  模型: {model_path}")

    def start(self, **kwargs):
        """启动服务器"""
        if not self.enabled:
            return

        logger.info("启动REST API服务器...")

        try:
            from flask import Flask, request, jsonify

            app = Flask(__name__)

            # 加载模型
            model = self._load_model()

            @app.route(self.endpoints["predict"], methods=["POST"])
            def predict():
                data = request.json
                result = self._predict(model, data)
                return jsonify(result)

            @app.route(self.endpoints["health"], methods=["GET"])
            def health():
                return jsonify({"status": "healthy"})

            @app.route(self.endpoints["metrics"], methods=["GET"])
            def metrics():
                return jsonify({"requests": 0, "latency": 0})

            app.run(
                host=self.host,
                port=self.port,
                threaded=True
            )

        except ImportError:
            logger.error("Flask未安装，REST API服务器不可用")
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")

    def _load_model(self):
        """加载模型"""
        logger.info(f"加载模型: {self.model_path}")

        if self.model_path.suffix == ".onnx":
            import onnxruntime
            return onnxruntime.InferenceSession(str(self.model_path))
        else:
            import tensorflow as tf
            return tf.keras.models.load_model(str(self.model_path))

    def _predict(self, model, data):
        """执行预测"""
        import numpy as np

        inputs = np.array(data.get("inputs", []))

        if hasattr(model, 'run'):  # ONNX模型
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: inputs})
            predictions = outputs[0]
        else:  # Keras模型
            predictions = model.predict(inputs)

        return {
            "predictions": predictions.tolist()
        }


class GRPCServer:
    """
    gRPC服务器
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化gRPC服务器

        参数:
            model_path: 模型文件路径
            host: 主机地址
            port: 端口号
            max_workers: 最大工作线程数
            enabled: 是否启用
        """
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.enabled = enabled

        if self.enabled:
            logger.info(f"GRPCServer 初始化")
            logger.info(f"  地址: {host}:{port}")
            logger.info(f"  模型: {model_path}")

    def start(self, **kwargs):
        """启动gRPC服务器"""
        if not self.enabled:
            return

        logger.info("启动gRPC服务器...")

        try:
            import grpc
            from concurrent import futures

            model = self._load_model()

            server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))

            server.add_insecure_port(f"{self.host}:{self.port}")
            server.start()

            logger.info(f"✓ gRPC服务器已启动: {self.host}:{self.port}")
            server.wait_for_termination()

        except ImportError:
            logger.error("grpcio未安装，gRPC服务器不可用")
        except Exception as e:
            logger.error(f"启动gRPC服务器失败: {e}")

    def _load_model(self):
        """加载模型"""
        logger.info(f"加载模型: {self.model_path}")

        if self.model_path.suffix == ".onnx":
            import onnxruntime
            return onnxruntime.InferenceSession(str(self.model_path))
        else:
            import tensorflow as tf
            return tf.keras.models.load_model(str(self.model_path))


class TFServingDeployer:
    """
    TensorFlow Serving部署器
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "model",
        port: int = 8501,
        rest_api_port: int = 8500,
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化TF Serving部署器

        参数:
            model_path: SavedModel路径
            model_name: 模型名称
            port: gRPC端口
            rest_api_port: REST API端口
            enabled: 是否启用
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.port = port
        self.rest_api_port = rest_api_port
        self.enabled = enabled

        if self.enabled:
            logger.info(f"TFServingDeployer 初始化")
            logger.info(f"  模型: {model_path}")
            logger.info(f"  gRPC端口: {port}")
            logger.info(f"  REST端口: {rest_api_port}")

    def deploy(self, **kwargs):
        """部署到TensorFlow Serving"""
        if not self.enabled:
            return

        logger.info("部署到TensorFlow Serving...")

        docker_cmd = f"""
docker run -p {self.rest_api_port}:8501 -p {self.port}:8500 \\
  --mount type=bind,source={self.model_path.absolute()},target=/models/{self.model_name} \\
  -e MODEL_NAME={self.model_name} -t tensorflow/serving
"""

        logger.info("使用以下命令启动TensorFlow Serving:")
        logger.info(docker_cmd)


class DockerDeployer:
    """
    Docker容器部署器
    """

    def __init__(
        self,
        model_path: str,
        dockerfile_template: str = "deploy/Dockerfile.template",
        image_name: str = "ml-model:latest",
        container_port: int = 9000,
        host_port: int = 9000,
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化Docker部署器

        参数:
            model_path: 模型文件路径
            dockerfile_template: Dockerfile模板路径
            image_name: Docker镜像名称
            container_port: 容器端口
            host_port: 主机端口
            enabled: 是否启用
        """
        self.model_path = Path(model_path)
        self.dockerfile_template = Path(dockerfile_template)
        self.image_name = image_name
        self.container_port = container_port
        self.host_port = host_port
        self.enabled = enabled

        if self.enabled:
            logger.info(f"DockerDeployer 初始化")
            logger.info(f"  镜像: {image_name}")
            logger.info(f"  端口: {host_port}:{container_port}")

    def build(self, **kwargs):
        """构建Docker镜像"""
        if not self.enabled:
            return

        logger.info(f"构建Docker镜像: {self.image_name}")

        dockerfile_content = self._generate_dockerfile()

        dockerfile_path = Path("Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        build_cmd = f"docker build -t {self.image_name} ."
        logger.info(f"执行: {build_cmd}")

    def run(self, **kwargs):
        """运行Docker容器"""
        if not self.enabled:
            return

        logger.info(f"运行Docker容器: {self.image_name}")

        run_cmd = f"docker run -d -p {self.host_port}:{self.container_port} {self.image_name}"
        logger.info(f"执行: {run_cmd}")

    def _generate_dockerfile(self) -> str:
        """生成Dockerfile内容"""
        return f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY {self.model_path} /app/model/

COPY serve.py .

EXPOSE {self.container_port}

CMD ["python", "serve.py"]
"""


class LambdaDeployer:
    """
    AWS Lambda部署器
    """

    def __init__(
        self,
        model_path: str,
        function_name: str = "ml-inference",
        runtime: str = "python3.9",
        memory_size: int = 512,
        timeout: int = 30,
        enabled: bool = True,
        **kwargs
    ):
        """
        初始化Lambda部署器

        参数:
            model_path: 模型文件路径
            function_name: Lambda函数名称
            runtime: 运行时环境
            memory_size: 内存大小（MB）
            timeout: 超时时间（秒）
            enabled: 是否启用
        """
        self.model_path = Path(model_path)
        self.function_name = function_name
        self.runtime = runtime
        self.memory_size = memory_size
        self.timeout = timeout
        self.enabled = enabled

        if self.enabled:
            logger.info(f"LambdaDeployer 初始化")
            logger.info(f"  函数: {function_name}")
            logger.info(f"  运行时: {runtime}")

    def deploy(self, **kwargs):
        """部署到AWS Lambda"""
        if not self.enabled:
            return

        logger.info(f"部署到AWS Lambda: {self.function_name}")

        try:
            import boto3

            lambda_client = boto3.client('lambda')

            zip_path = self._create_deployment_package()

            with open(zip_path, 'rb') as f:
                response = lambda_client.create_function(
                    FunctionName=self.function_name,
                    Runtime=self.runtime,
                    Role='arn:aws:iam::account-id:role/lambda-role',
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': f.read()},
                    MemorySize=self.memory_size,
                    Timeout=self.timeout
                )

            logger.info(f"✓ Lambda函数已部署: {response['FunctionArn']}")

        except ImportError:
            logger.error("boto3未安装，Lambda部署器不可用")
        except Exception as e:
            logger.error(f"部署到Lambda失败: {e}")

    def _create_deployment_package(self) -> Path:
        """创建部署包"""
        zip_path = Path("lambda_deployment.zip")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(self.model_path, arcname="model/" + self.model_path.name)

            handler_code = """
import json
import numpy as np

def lambda_handler(event, context):
    inputs = np.array(event['inputs'])

    return {
        'statusCode': 200,
        'body': json.dumps({'predictions': []})
    }
"""
            zipf.writestr("lambda_function.py", handler_code)

        logger.info(f"部署包已创建: {zip_path}")
        return zip_path
