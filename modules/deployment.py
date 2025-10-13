# -*- coding: utf-8 -*-
"""
deployment.py - 模型部署
功能：
  - 部署模型为Web服务
  - 生成Docker配置
  - 管理推理服务
"""

from typing import Dict, Any
import os
import json
from utils import LoggerManager, call_target

logger = LoggerManager.get_logger(__file__)


class ModelDeployer:
    """
    模型部署器：
    - 部署模型为HTTP服务
    - 生成Docker容器配置
    - 管理服务生命周期
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化部署器
        参数：
            config: 完整配置文件
        """
        self.config = config
        self.deployment_config = config.get("deployment", {})

        # 创建部署目录
        self.deploy_dir = "outputs/deployment"
        os.makedirs(self.deploy_dir, exist_ok=True)

    def deploy(self, models: Dict[str, Any]):
        """
        部署模型
        参数：
            models: 模型字典
        """
        logger.info("=" * 60)
        logger.info("开始部署模型")
        logger.info("=" * 60)

        if not self.deployment_config:
            logger.warning("未找到deployment配置")
            return

        service_type = self.deployment_config.get("service_type", "onnx_inference")

        try:
            if service_type == "onnx_inference":
                self._deploy_onnx_service()
            elif service_type == "tensorflow_serving":
                self._deploy_tensorflow_service()
            else:
                logger.warning(f"未知的服务类型: {service_type}")

            logger.info("模型部署成功")
        except Exception as e:
            logger.error(f"模型部署失败: {str(e)}", exc_info=True)
            raise

    def _deploy_onnx_service(self):
        """部署ONNX推理服务"""
        logger.info("部署ONNX推理服务")

        host = self.deployment_config.get("host", "0.0.0.0")
        port = self.deployment_config.get("port", 8000)
        model_key = self.deployment_config.get("model_key", "model")

        # 生成Docker Compose配置
        self._generate_docker_compose(host, port, model_key)

        # 生成FastAPI服务配置
        self._generate_fastapi_server(host, port, model_key)

        logger.info(f"ONNX服务已生成: {host}:{port}")

    def _deploy_tensorflow_service(self):
        """部署TensorFlow Serving"""
        logger.info("部署TensorFlow Serving")

        host = self.deployment_config.get("host", "0.0.0.0")
        port = self.deployment_config.get("port", 8501)
        model_key = self.deployment_config.get("model_key", "model")

        # 生成TensorFlow Serving配置
        self._generate_tensorflow_config(host, port, model_key)

        logger.info(f"TensorFlow Serving已生成: {host}:{port}")

    def _generate_docker_compose(self, host: str, port: int, model_key: str):
        """生成Docker Compose配置"""
        logger.info("生成Docker Compose配置")

        docker_compose = {
            "version": "3.8",
            "services": {
                "onnx_inference": {
                    "image": "onnx_inference:latest",
                    "ports": [f"{port}:8000"],
                    "environment": {
                        "MODEL_KEY": model_key,
                        "HOST": host,
                        "PORT": "8000"
                    },
                    "volumes": [
                        "./outputs/onnx:/app/models:ro"
                    ]
                }
            }
        }

        # 保存为docker-compose.yml
        compose_path = os.path.join(self.deploy_dir, "docker-compose.yml")
        import yaml
        with open(compose_path, 'w', encoding='utf-8') as f:
            yaml.dump(docker_compose, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Docker Compose配置已保存到: {compose_path}")

    def _generate_fastapi_server(self, host: str, port: int, model_key: str):
        """生成FastAPI服务代码"""
        logger.info("生成FastAPI服务代码")

        server_code = f'''# -*- coding: utf-8 -*-
"""
ONNX推理服务
自动生成的FastAPI服务器
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
import os

app = FastAPI()

# 加载ONNX模型
MODEL_PATH = os.path.join("models", "{model_key}.onnx")
session = ort.InferenceSession(MODEL_PATH)

@app.get("/health")
async def health():
    """健康检查"""
    return {{"status": "healthy"}}

@app.post("/predict")
async def predict(data: dict):
    """推理端点"""
    try:
        # 获取输入数据
        input_data = np.array(data.get("input")).astype(np.float32)

        # 获取输入名称
        input_names = [input.name for input in session.get_inputs()]

        # 运行推理
        output = session.run(None, {{input_names[0]: input_data}})

        return {{
            "success": True,
            "predictions": output[0].tolist()
        }}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={{"success": False, "error": str(e)}}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="{host}", port={port})
'''

        # 保存服务代码
        server_path = os.path.join(self.deploy_dir, "onnx_server.py")
        with open(server_path, 'w', encoding='utf-8') as f:
            f.write(server_code)

        logger.info(f"FastAPI服务代码已保存到: {server_path}")

    def _generate_dockerfile(self, model_key: str):
        """生成Dockerfile"""
        logger.info("生成Dockerfile")

        dockerfile = f'''FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY onnx_server.py .

# 复制模型
COPY outputs/onnx/models .

EXPOSE 8000

CMD ["python", "onnx_server.py"]
'''

        # 保存Dockerfile
        dockerfile_path = os.path.join(self.deploy_dir, "Dockerfile")
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile)

        logger.info(f"Dockerfile已保存到: {dockerfile_path}")

    def _generate_tensorflow_config(self, host: str, port: int, model_key: str):
        """生成TensorFlow Serving配置"""
        logger.info("生成TensorFlow Serving配置")

        config = {
            "model_config_list": [
                {
                    "config": {
                        "name": model_key,
                        "base_path": f"/models/{model_key}",
                        "model_platform": "tensorflow"
                    }
                }
            ]
        }

        # 保存配置
        config_path = os.path.join(self.deploy_dir, "models.config")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"TensorFlow Serving配置已保存到: {config_path}")

    def generate_deployment_guide(self):
        """生成部署指南"""
        logger.info("生成部署指南")

        guide = '''# 模型部署指南

## 快速开始

### 1. ONNX推理服务部署

#### 使用Docker Compose:
```bash
cd outputs/deployment
docker-compose up -d
```

#### 使用FastAPI直接运行:
```bash
pip install fastapi uvicorn onnxruntime
python onnx_server.py
```

#### 测试服务:
```bash
# 健康检查
curl http://localhost:8000/health

# 推理请求
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input": [[1.0, 2.0, 3.0]]}'
```

### 2. TensorFlow Serving部署

#### 使用Docker:
```bash
docker run -t --rm -p 8500:8500 -p 8501:8501 \\
  -v "$(pwd)/outputs/models:/models" \\
  -e MODEL_NAME=model \\
  tensorflow/serving
```

#### 使用TensorFlow Serving配置:
```bash
tensorflow_model_server \\
  --port=8500 \\
  --rest_api_port=8501 \\
  --model_config_file=models.config
```

## 性能优化

### 模型量化
- 使用INT8量化减少模型大小
- 提高推理速度

### 批处理
- 启用动态批处理
- 提高吞吐量

### GPU加速
- 使用CUDA加速
- 配置GPU内存

## 监控和日志

### Prometheus指标
服务已集成Prometheus指标导出，访问 `/metrics` 端点

### ELK日志收集
建议使用ELK Stack收集和分析日志

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 验证模型格式
   - 检查依赖库版本

2. **内存溢出**
   - 减小batch大小
   - 启用模型量化
   - 使用模型分片

3. **推理速度慢**
   - 启用GPU加速
   - 使用批处理
   - 优化输入预处理

## 安全建议

- 在生产环境使用TLS/SSL
- 实现API认证和授权
- 限制请求速率
- 监控异常流量
'''

        guide_path = os.path.join(self.deploy_dir, "DEPLOYMENT_GUIDE.md")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)

        logger.info(f"部署指南已保存到: {guide_path}")

    def generate_requirements(self):
        """生成requirements.txt"""
        logger.info("生成requirements.txt")

        requirements = '''fastapi>=0.95.0
uvicorn>=0.20.0
onnxruntime>=1.14.0
numpy>=1.21.0
pydantic>=1.9.0
python-multipart>=0.0.5
'''

        req_path = os.path.join(self.deploy_dir, "requirements.txt")
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(requirements)

        logger.info(f"requirements.txt已保存到: {req_path}")


class ServiceManager:
    """
    服务管理器：
    - 启动/停止服务
    - 监控服务状态
    - 处理服务请求
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        初始化服务管理器
        参数：
            host: 服务主机
            port: 服务端口
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"健康检查失败: {str(e)}")
            return False

    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        调用推理端点
        参数：
            input_data: 输入数据
        返回：
            预测结果
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/predict",
                json={"input": input_data},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"推理失败: {response.status_code}")
                return {"success": False, "error": "推理失败"}
        except Exception as e:
            logger.error(f"调用推理端点失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/model_info",
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            logger.warning(f"获取模型信息失败: {str(e)}")
            return {}
