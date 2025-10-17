# -*- coding: utf-8 -*-
"""
modules/deployment.py
模型部署模块：从配置上下文部署训练好的模型

导入限制：
- 仅导入 common 文件夹内的函数
- modules 中的函数通过反射机制调用
"""

from typing import Dict, Any, List, Optional, Callable
from common.train_context import (
    TrainContext,
    DeploymentConfig
)
from common.common import call_target


class ModelDeployer:
    """
    模型部署器

    职责：
    1. 读取 TrainContext 中的部署配置
    2. 部署模型到不同的目标平台
    3. 支持 TF Serving、REST API、gRPC 等
    4. 生成部署脚本和配置文件
    5. 健康检查和监控

    注意：所有部署操作通过 call_target 动态调用
    """

    def __init__(self, context: TrainContext):
        """
        初始化模型部署器

        参数:
            context: 训练上下文
        """
        self.context = context
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.deployment_info: Dict[str, Dict[str, Any]] = {}

    def setup_deployments(self):
        """设置所有部署配置"""
        if not self.context.deployment:
            return

        for deploy_name, deploy_config in self.context.deployment.items():
            self.deployment_configs[deploy_name] = deploy_config

    def deploy_all(self) -> Dict[str, Any]:
        """
        部署所有配置的服务

        返回:
            部署结果字典 {deploy_name: deployment_info}
        """
        if not self.deployment_configs:
            self.setup_deployments()

        results = {}

        for deploy_name, deploy_config in self.deployment_configs.items():
            try:
                deploy_info = self.deploy(deploy_name, deploy_config)
                results[deploy_name] = deploy_info
            except Exception as e:
                raise RuntimeError(f"部署服务 '{deploy_name}' 失败: {e}")

        return results

    def deploy(
        self,
        deploy_name: str,
        deploy_config: Optional[DeploymentConfig] = None
    ) -> Dict[str, Any]:
        """
        部署单个服务

        参数:
            deploy_name: 部署配置名称
            deploy_config: 部署配置（可选）

        返回:
            部署信息
        """
        if deploy_config is None:
            if deploy_name not in self.deployment_configs:
                raise ValueError(f"部署配置 '{deploy_name}' 不存在")
            deploy_config = self.deployment_configs[deploy_name]

        # 准备参数
        args = self._prepare_deploy_args(deploy_config.args)

        # 使用 call_target 执行部署
        try:
            result = call_target(
                reflection=deploy_config.reflection,
                args=args
            )

            # 构建部署信息
            deploy_info = {
                "deploy_name": deploy_name,
                "status": "success",
                "result": result
            }

        except Exception as e:
            deploy_info = {
                "deploy_name": deploy_name,
                "status": "failed",
                "error": str(e)
            }
            raise

        # 记录部署信息
        self.deployment_info[deploy_name] = deploy_info

        return deploy_info

    def _prepare_deploy_args(self, args_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备部署参数（解析变量引用）

        参数:
            args_config: 参数配置

        返回:
            解析后的参数
        """
        prepared_args = {}

        for key, value in args_config.items():
            # 解析变量引用 ${variable}
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                var_name = value[2:-1]
                resolved_value = self._resolve_variable(var_name)
                prepared_args[key] = resolved_value

            elif isinstance(value, dict):
                prepared_args[key] = self._prepare_deploy_args(value)

            elif isinstance(value, list):
                prepared_args[key] = [
                    self._prepare_deploy_args(item) if isinstance(item, dict) else item
                    for item in value
                ]

            else:
                prepared_args[key] = value

        return prepared_args

    def _resolve_variable(self, var_name: str) -> Any:
        """
        解析变量引用

        参数:
            var_name: 变量名

        返回:
            变量值
        """
        # 支持点号访问
        if '.' in var_name:
            parts = var_name.split('.')
            base_name = parts[0]
            field_path = parts[1:]

            if base_name in self.context.instantiated_models:
                value = self.context.instantiated_models[base_name]
            elif base_name in self.context.execution_results:
                value = self.context.execution_results[base_name]
            else:
                raise ValueError(f"无法解析变量: ${{{var_name}}}")

            for field in field_path:
                if isinstance(value, dict):
                    value = value.get(field)
                elif hasattr(value, field):
                    value = getattr(value, field)
                else:
                    raise ValueError(f"无法访问 '{var_name}'")

            return value

        # 直接变量名
        if var_name in self.context.instantiated_models:
            return self.context.instantiated_models[var_name]

        if hasattr(self.context, var_name):
            return getattr(self.context, var_name)

        raise ValueError(f"无法解析变量: ${{{var_name}}}")

    def deploy_tf_serving(
        self,
        model_name: str,
        model_path: str,
        serving_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        部署到 TensorFlow Serving

        参数:
            model_name: 模型名称
            model_path: 模型路径（SavedModel 格式）
            serving_config: Serving 配置

        返回:
            部署信息
        """
        import json
        import os

        if serving_config is None:
            serving_config = {}

        # 生成 model config
        model_config = {
            "model_config_list": {
                "config": [
                    {
                        "name": model_name,
                        "base_path": model_path,
                        "model_platform": "tensorflow"
                    }
                ]
            }
        }

        # 添加版本策略
        if "version_policy" in serving_config:
            model_config["model_config_list"]["config"][0]["model_version_policy"] = \
                serving_config["version_policy"]

        # 保存配置文件
        config_dir = serving_config.get("config_dir", "./deployment/tf_serving")
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, "models.config")
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)

        # 生成启动脚本
        script_content = self._generate_tf_serving_script(
            config_file,
            serving_config
        )

        script_file = os.path.join(config_dir, "start_serving.sh")
        with open(script_file, 'w') as f:
            f.write(script_content)

        # 使脚本可执行
        os.chmod(script_file, 0o755)

        return {
            "type": "tf_serving",
            "model_name": model_name,
            "model_path": model_path,
            "config_file": config_file,
            "script_file": script_file,
            "port": serving_config.get("port", 8501)
        }

    def _generate_tf_serving_script(
        self,
        config_file: str,
        serving_config: Dict[str, Any]
    ) -> str:
        """
        生成 TensorFlow Serving 启动脚本

        参数:
            config_file: 配置文件路径
            serving_config: Serving 配置

        返回:
            脚本内容
        """
        port = serving_config.get("port", 8501)
        rest_api_port = serving_config.get("rest_api_port", 8501)
        grpc_port = serving_config.get("grpc_port", 8500)

        script = f"""#!/bin/bash
# TensorFlow Serving 启动脚本

echo "启动 TensorFlow Serving..."

docker run -d \\
  --name tf_serving \\
  -p {rest_api_port}:8501 \\
  -p {grpc_port}:8500 \\
  -v "$(pwd)/{config_file}:/models/models.config" \\
  -v "$(pwd)/models:/models" \\
  -e MODEL_NAME=default \\
  tensorflow/serving \\
  --model_config_file=/models/models.config \\
  --rest_api_port=8501 \\
  --grpc_port=8500

echo "TensorFlow Serving 已启动"
echo "REST API: http://localhost:{rest_api_port}"
echo "gRPC: localhost:{grpc_port}"
"""

        return script

    def deploy_rest_api(
        self,
        model_name: str,
        model_path: str,
        api_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        部署 REST API 服务

        参数:
            model_name: 模型名称
            model_path: 模型路径
            api_config: API 配置

        返回:
            部署信息
        """
        import os

        if api_config is None:
            api_config = {}

        # 生成 Flask/FastAPI 应用代码
        app_code = self._generate_rest_api_code(
            model_name,
            model_path,
            api_config
        )

        # 保存应用代码
        deploy_dir = api_config.get("deploy_dir", "./deployment/rest_api")
        os.makedirs(deploy_dir, exist_ok=True)

        app_file = os.path.join(deploy_dir, "app.py")
        with open(app_file, 'w') as f:
            f.write(app_code)

        # 生成 requirements.txt
        requirements = self._generate_requirements(api_config)
        req_file = os.path.join(deploy_dir, "requirements.txt")
        with open(req_file, 'w') as f:
            f.write(requirements)

        # 生成 Dockerfile
        dockerfile = self._generate_dockerfile(api_config)
        docker_file = os.path.join(deploy_dir, "Dockerfile")
        with open(docker_file, 'w') as f:
            f.write(dockerfile)

        return {
            "type": "rest_api",
            "model_name": model_name,
            "deploy_dir": deploy_dir,
            "app_file": app_file,
            "dockerfile": docker_file,
            "port": api_config.get("port", 8000)
        }

    def _generate_rest_api_code(
        self,
        model_name: str,
        model_path: str,
        api_config: Dict[str, Any]
    ) -> str:
        """
        生成 REST API 应用代码

        参数:
            model_name: 模型名称
            model_path: 模型路径
            api_config: API 配置

        返回:
            应用代码
        """
        framework = api_config.get("framework", "fastapi")
        port = api_config.get("port", 8000)

        if framework == "fastapi":
            code = f"""
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn

# 加载模型
model = tf.keras.models.load_model('{model_path}')

# 创建 FastAPI 应用
app = FastAPI(title="{model_name} API")

class PredictionRequest(BaseModel):
    data: list

class PredictionResponse(BaseModel):
    predictions: list

@app.get("/")
def read_root():
    return {{"message": "Model API is running", "model": "{model_name}"}}

@app.get("/health")
def health_check():
    return {{"status": "healthy", "model": "{model_name}"}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # 转换输入
        input_data = np.array(request.data)

        # 预测
        predictions = model.predict(input_data)

        # 返回结果
        return {{"predictions": predictions.tolist()}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
"""

        elif framework == "flask":
            code = f"""
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

# 加载模型
model = tf.keras.models.load_model('{model_path}')

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({{"message": "Model API is running", "model": "{model_name}"}})

@app.route("/health")
def health():
    return jsonify({{"status": "healthy", "model": "{model_name}"}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取输入数据
        data = request.json.get("data")
        input_data = np.array(data)

        # 预测
        predictions = model.predict(input_data)

        # 返回结果
        return jsonify({{"predictions": predictions.tolist()}})

    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port={port})
"""

        else:
            raise ValueError(f"不支持的框架: {framework}")

        return code

    def _generate_requirements(self, api_config: Dict[str, Any]) -> str:
        """生成 requirements.txt"""
        framework = api_config.get("framework", "fastapi")

        if framework == "fastapi":
            requirements = """tensorflow>=2.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
numpy>=1.19.0
"""
        else:  # flask
            requirements = """tensorflow>=2.0.0
flask>=2.0.0
numpy>=1.19.0
"""

        return requirements

    def _generate_dockerfile(self, api_config: Dict[str, Any]) -> str:
        """生成 Dockerfile"""
        port = api_config.get("port", 8000)

        dockerfile = f"""FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 复制模型
COPY model /app/model

# 暴露端口
EXPOSE {port}

# 启动应用
CMD ["python", "app.py"]
"""

        return dockerfile

    def deploy_cloud(
        self,
        model_name: str,
        model_path: str,
        cloud_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        部署到云平台

        参数:
            model_name: 模型名称
            model_path: 模型路径
            cloud_config: 云平台配置

        返回:
            部署信息
        """
        provider = cloud_config.get("provider", "gcp")

        if provider == "gcp":
            return self._deploy_to_gcp(model_name, model_path, cloud_config)
        elif provider == "aws":
            return self._deploy_to_aws(model_name, model_path, cloud_config)
        elif provider == "azure":
            return self._deploy_to_azure(model_name, model_path, cloud_config)
        else:
            raise ValueError(f"不支持的云平台: {provider}")

    def _deploy_to_gcp(
        self,
        model_name: str,
        model_path: str,
        cloud_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """部署到 Google Cloud Platform"""
        # 生成部署脚本
        project_id = cloud_config.get("project_id")
        region = cloud_config.get("region", "us-central1")

        script = f"""#!/bin/bash
# GCP AI Platform 部署脚本

# 上传模型到 GCS
gsutil -m cp -r {model_path} gs://{project_id}-ml-models/{model_name}/

# 创建模型版本
gcloud ai-platform models create {model_name} \\
    --regions={region}

# 部署版本
gcloud ai-platform versions create v1 \\
    --model={model_name} \\
    --origin=gs://{project_id}-ml-models/{model_name}/ \\
    --runtime-version=2.8 \\
    --framework=tensorflow \\
    --python-version=3.9

echo "模型已部署到 GCP AI Platform"
"""

        return {
            "type": "cloud",
            "provider": "gcp",
            "model_name": model_name,
            "script": script
        }

    def _deploy_to_aws(
        self,
        model_name: str,
        model_path: str,
        cloud_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """部署到 AWS SageMaker"""
        # 简化实现
        return {
            "type": "cloud",
            "provider": "aws",
            "model_name": model_name,
            "message": "AWS SageMaker 部署配置已生成"
        }

    def _deploy_to_azure(
        self,
        model_name: str,
        model_path: str,
        cloud_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """部署到 Azure ML"""
        # 简化实现
        return {
            "type": "cloud",
            "provider": "azure",
            "model_name": model_name,
            "message": "Azure ML 部署配置已生成"
        }

    def health_check(self, deploy_name: str) -> Dict[str, Any]:
        """
        健康检查

        参数:
            deploy_name: 部署名称

        返回:
            健康状态
        """
        if deploy_name not in self.deployment_info:
            return {"status": "unknown", "message": "部署不存在"}

        deploy_info = self.deployment_info[deploy_name]

        # 简化的健康检查
        return {
            "deploy_name": deploy_name,
            "status": deploy_info.get("status", "unknown"),
            "type": deploy_info.get("type", "unknown")
        }

    def get_deployment_info(self, deploy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取部署信息

        参数:
            deploy_name: 部署名称（None 表示所有）

        返回:
            部署信息
        """
        if deploy_name is not None:
            return self.deployment_info.get(deploy_name, {})

        return self.deployment_info.copy()


def create_deployer(context: TrainContext) -> ModelDeployer:
    """
    创建模型部署器的便捷函数

    参数:
        context: 训练上下文

    返回:
        ModelDeployer 实例
    """
    deployer = ModelDeployer(context)
    deployer.setup_deployments()
    return deployer


def deploy_model(
    context: TrainContext,
    model_name: str,
    deploy_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    部署模型的便捷函数

    参数:
        context: 训练上下文
        model_name: 模型名称
        deploy_type: 部署类型 ('tf_serving', 'rest_api', 'cloud')
        **kwargs: 额外参数

    返回:
        部署信息
    """
    deployer = ModelDeployer(context)

    model_path = kwargs.get("model_path", f"./exports/{model_name}")

    if deploy_type == "tf_serving":
        return deployer.deploy_tf_serving(model_name, model_path, kwargs)

    elif deploy_type == "rest_api":
        return deployer.deploy_rest_api(model_name, model_path, kwargs)

    elif deploy_type == "cloud":
        return deployer.deploy_cloud(model_name, model_path, kwargs)

    else:
        raise ValueError(f"不支持的部署类型: {deploy_type}")
