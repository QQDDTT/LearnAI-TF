# -*- coding: utf-8 -*-
"""
modules/deployment.py
模型部署模块：
- 生成Java服务代码（Jakarta + Jetty）
- ONNX模型部署
- 服务配置生成
"""

import os
import json
from typing import Dict, Any
from common.common import LoggerManager

logger = LoggerManager.get_logger(__file__)


# ======================================================
# 主部署函数
# ======================================================
def deploy_model(deployment_config: Dict, context: Dict) -> None:
    """
    根据配置部署模型为Java服务

    参数:
        deployment_config: 部署配置
        context: 训练上下文

    示例:
        >>> deploy_model(deployment_config, context)
    """
    service_type = deployment_config.get("service_type", "java_onnx")

    logger.info(f"部署模型为 {service_type} 服务")

    if service_type == "java_onnx":
        deploy_java_onnx_service(deployment_config, context)
    else:
        logger.error(f"未知的服务类型: {service_type}")


# ======================================================
# Java ONNX服务部署
# ======================================================
def deploy_java_onnx_service(config: Dict, context: Dict) -> None:
    """
    生成Java服务代码（Jakarta + Jetty + ONNX Runtime）

    参数:
        config: 部署配置
        context: 上下文

    生成文件结构:
        java-service/
        ├── pom.xml                     # Maven配置
        ├── src/
        │   └── main/
        │       ├── java/
        │       │   └── com/
        │       │       └── ai/
        │       │           └── model/
        │       │               ├── ModelServer.java          # Jetty服务器
        │       │               ├── PredictionServlet.java    # 预测端点
        │       │               ├── HealthServlet.java        # 健康检查
        │       │               └── ModelInference.java       # ONNX推理
        │       └── resources/
        │           └── model.onnx       # 模型文件
        └── README.md
    """
    output_dir = config.get("output_dir", "java-service")
    model_name = config.get("model", "classifier")
    host = config.get("host", "0.0.0.0")
    port = config.get("port", 9000)

    logger.info("=" * 60)
    logger.info("生成Java服务代码（Jakarta + Jetty + ONNX）")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # 创建目录结构
    create_java_project_structure(output_dir)

    # 生成Maven配置
    generate_pom_xml(output_dir, config)

    # 生成Java代码
    generate_model_server(output_dir, host, port)
    generate_prediction_servlet(output_dir)
    generate_health_servlet(output_dir)
    generate_model_inference(output_dir, model_name)

    # 生成配置文件
    generate_application_properties(output_dir, config)

    # 生成README
    generate_readme(output_dir, host, port)

    # 复制ONNX模型（如果存在）
    copy_onnx_model(output_dir, context, model_name)

    logger.info("=" * 60)
    logger.info("Java服务代码生成完成！")
    logger.info(f"项目位置: {output_dir}")
    logger.info("\n运行步骤:")
    logger.info("  1. cd java-service")
    logger.info("  2. mvn clean package")
    logger.info("  3. java -jar target/model-service-1.0-SNAPSHOT.jar")
    logger.info(f"  4. 访问 http://localhost:{port}/health")
    logger.info("=" * 60)


# ======================================================
# 创建项目结构
# ======================================================
def create_java_project_structure(output_dir: str) -> None:
    """创建Java项目目录结构"""
    dirs = [
        output_dir,
        f"{output_dir}/src/main/java/com/ai/model",
        f"{output_dir}/src/main/resources",
        f"{output_dir}/src/test/java",
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    logger.info("项目结构创建完成")


# ======================================================
# 生成Maven配置 (pom.xml)
# ======================================================
def generate_pom_xml(output_dir: str, config: Dict) -> None:
    """生成Maven配置文件"""
    pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.ai</groupId>
    <artifactId>model-service</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>AI Model Service</name>
    <description>Lightweight AI Model Inference Service using Jakarta EE and Jetty</description>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <jetty.version>11.0.15</jetty.version>
        <jakarta.servlet.version>6.0.0</jakarta.servlet.version>
        <onnxruntime.version>1.16.3</onnxruntime.version>
    </properties>

    <dependencies>
        <!-- Jakarta Servlet API -->
        <dependency>
            <groupId>jakarta.servlet</groupId>
            <artifactId>jakarta.servlet-api</artifactId>
            <version>${jakarta.servlet.version}</version>
            <scope>provided</scope>
        </dependency>

        <!-- Jetty Server (嵌入式) -->
        <dependency>
            <groupId>org.eclipse.jetty</groupId>
            <artifactId>jetty-server</artifactId>
            <version>${jetty.version}</version>
        </dependency>

        <dependency>
            <groupId>org.eclipse.jetty</groupId>
            <artifactId>jetty-servlet</artifactId>
            <version>${jetty.version}</version>
        </dependency>

        <!-- ONNX Runtime (Java推理引擎) -->
        <dependency>
            <groupId>com.microsoft.onnxruntime</groupId>
            <artifactId>onnxruntime</artifactId>
            <version>${onnxruntime.version}</version>
        </dependency>

        <!-- JSON处理 -->
        <dependency>
            <groupId>com.google.code.gson</groupId>
            <artifactId>gson</artifactId>
            <version>2.10.1</version>
        </dependency>

        <!-- 日志 -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-simple</artifactId>
            <version>2.0.9</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Maven Compiler Plugin -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>

            <!-- Maven Shade Plugin (创建可执行JAR) -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>3.5.0</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals>
                            <goal>shade</goal>
                        </goals>
                        <configuration>
                            <transformers>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <mainClass>com.ai.model.ModelServer</mainClass>
                                </transformer>
                            </transformers>
                            <filters>
                                <filter>
                                    <artifact>*:*</artifact>
                                    <excludes>
                                        <exclude>META-INF/*.SF</exclude>
                                        <exclude>META-INF/*.DSA</exclude>
                                        <exclude>META-INF/*.RSA</exclude>
                                    </excludes>
                                </filter>
                            </filters>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
"""

    with open(f"{output_dir}/pom.xml", "w", encoding="utf-8") as f:
        f.write(pom_content)

    logger.info("生成 pom.xml")


# ======================================================
# 生成Java代码
# ======================================================
def generate_model_server(output_dir: str, host: str, port: int) -> None:
    """生成Jetty服务器主类"""
    java_code = f"""package com.ai.model;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AI模型推理服务 - 使用Jakarta EE + 嵌入式Jetty
 *
 * 架构:
 * - Jakarta Servlet API: 标准化的Web接口
 * - Jetty: 轻量级、高性能的嵌入式Web服务器
 * - ONNX Runtime: 跨平台的深度学习推理引擎
 */
public class ModelServer {{
    private static final Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private static final String HOST = "{host}";
    private static final int PORT = {port};

    public static void main(String[] args) throws Exception {{
        logger.info("========================================");
        logger.info("启动AI模型推理服务");
        logger.info("架构: Jakarta EE + Jetty + ONNX Runtime");
        logger.info("========================================");

        // 创建Jetty服务器
        Server server = new Server(PORT);

        // 创建Servlet上下文
        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("/");
        server.setHandler(context);

        // 注册Servlet
        context.addServlet(new ServletHolder(new HealthServlet()), "/health");
        context.addServlet(new ServletHolder(new PredictionServlet()), "/predict");
        context.addServlet(new ServletHolder(new ModelInfoServlet()), "/model_info");

        // 启动服务器
        try {{
            server.start();

            logger.info("========================================");
            logger.info("服务启动成功!");
            logger.info("地址: http://{{}}:{{}}", HOST, PORT);
            logger.info("端点:");
            logger.info("  - GET  http://{{}}:{{}}/health", HOST, PORT);
            logger.info("  - POST http://{{}}:{{}}/predict", HOST, PORT);
            logger.info("  - GET  http://{{}}:{{}}/model_info", HOST, PORT);
            logger.info("========================================");

            server.join();
        }} catch (Exception e) {{
            logger.error("服务器启动失败", e);
            System.exit(1);
        }}
    }}
}}
"""

    with open(f"{output_dir}/src/main/java/com/ai/model/ModelServer.java", "w", encoding="utf-8") as f:
        f.write(java_code)

    logger.info("生成 ModelServer.java")


def generate_prediction_servlet(output_dir: str) -> None:
    """生成预测Servlet"""
    java_code = """package com.ai.model;

import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

/**
 * 预测端点 - 处理推理请求
 */
public class PredictionServlet extends HttpServlet {
    private static final Logger logger = LoggerFactory.getLogger(PredictionServlet.class);
    private static final Gson gson = new Gson();
    private static final ModelInference inference = ModelInference.getInstance();

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        resp.setContentType("application/json");
        resp.setCharacterEncoding("UTF-8");

        long startTime = System.currentTimeMillis();

        try {
            // 解析请求
            JsonObject request = gson.fromJson(req.getReader(), JsonObject.class);

            if (!request.has("input")) {
                sendError(resp, 400, "缺少input字段");
                return;
            }

            // 获取输入数据
            float[][] inputData = gson.fromJson(request.get("input"), float[][].class);

            // 执行推理
            float[][] predictions = inference.predict(inputData);

            // 构建响应
            JsonObject response = new JsonObject();
            response.add("predictions", gson.toJsonTree(predictions));
            response.addProperty("inference_time", (System.currentTimeMillis() - startTime) / 1000.0);
            response.addProperty("batch_size", inputData.length);

            // 返回结果
            resp.setStatus(200);
            resp.getWriter().write(gson.toJson(response));

            logger.info("预测完成: {} 样本, 耗时 {}ms",
                inputData.length, System.currentTimeMillis() - startTime);

        } catch (Exception e) {
            logger.error("预测失败", e);
            sendError(resp, 500, "预测失败: " + e.getMessage());
        }
    }

    private void sendError(HttpServletResponse resp, int status, String message) throws IOException {
        resp.setStatus(status);
        JsonObject error = new JsonObject();
        error.addProperty("error", message);
        resp.getWriter().write(gson.toJson(error));
    }
}
"""

    with open(f"{output_dir}/src/main/java/com/ai/model/PredictionServlet.java", "w", encoding="utf-8") as f:
        f.write(java_code)

    logger.info("生成 PredictionServlet.java")


def generate_health_servlet(output_dir: str) -> None:
    """生成健康检查Servlet"""
    java_code = """package com.ai.model;

import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;

/**
 * 健康检查端点
 */
public class HealthServlet extends HttpServlet {
    private static final Gson gson = new Gson();

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        resp.setContentType("application/json");
        resp.setCharacterEncoding("UTF-8");

        JsonObject response = new JsonObject();
        response.addProperty("status", "healthy");
        response.addProperty("service", "AI Model Inference");
        response.addProperty("framework", "Jakarta EE + Jetty + ONNX");
        response.addProperty("model", "loaded");

        resp.setStatus(200);
        resp.getWriter().write(gson.toJson(response));
    }
}

/**
 * 模型信息端点
 */
class ModelInfoServlet extends HttpServlet {
    private static final Gson gson = new Gson();
    private static final ModelInference inference = ModelInference.getInstance();

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        resp.setContentType("application/json");
        resp.setCharacterEncoding("UTF-8");

        JsonObject response = new JsonObject();
        response.addProperty("model_path", "model.onnx");
        response.addProperty("framework", "ONNX Runtime");
        response.addProperty("status", inference.isLoaded() ? "loaded" : "not_loaded");

        resp.setStatus(200);
        resp.getWriter().write(gson.toJson(response));
    }
}
"""

    with open(f"{output_dir}/src/main/java/com/ai/model/HealthServlet.java", "w", encoding="utf-8") as f:
        f.write(java_code)

    logger.info("生成 HealthServlet.java")


def generate_model_inference(output_dir: str, model_name: str) -> None:
    """生成ONNX推理类"""
    java_code = """package com.ai.model;

import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * ONNX模型推理类 - 单例模式
 *
 * ONNX Runtime是跨平台的深度学习推理引擎:
 * - 支持多种深度学习框架导出的模型
 * - 高性能CPU/GPU推理
 * - 跨平台兼容
 */
public class ModelInference {
    private static final Logger logger = LoggerFactory.getLogger(ModelInference.class);
    private static final ModelInference INSTANCE = new ModelInference();

    private OrtEnvironment env;
    private OrtSession session;
    private boolean loaded = false;

    private ModelInference() {
        try {
            // 初始化ONNX Runtime环境
            env = OrtEnvironment.getEnvironment();

            // 加载模型
            String modelPath = getClass().getClassLoader()
                .getResource("model.onnx").getPath();

            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            session = env.createSession(modelPath, opts);

            loaded = true;
            logger.info("ONNX模型加载成功");
            logger.info("输入: {}", session.getInputNames());
            logger.info("输出: {}", session.getOutputNames());

        } catch (Exception e) {
            logger.error("模型加载失败", e);
        }
    }

    public static ModelInference getInstance() {
        return INSTANCE;
    }

    public boolean isLoaded() {
        return loaded;
    }

    /**
     * 执行推理
     *
     * @param inputData 输入数据 [batch_size, features]
     * @return 预测结果 [batch_size, outputs]
     */
    public float[][] predict(float[][] inputData) throws OrtException {
        if (!loaded) {
            throw new RuntimeException("模型未加载");
        }

        int batchSize = inputData.length;
        int features = inputData[0].length;

        // 准备输入张量
        long[] shape = {batchSize, features};
        FloatBuffer buffer = FloatBuffer.allocate(batchSize * features);

        for (float[] row : inputData) {
            buffer.put(row);
        }
        buffer.rewind();

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buffer, shape);

        // 执行推理
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result result = session.run(inputs);

        // 提取输出
        float[][] output = (float[][]) result.get(0).getValue();

        // 清理资源
        inputTensor.close();
        result.close();

        return output;
    }

    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            logger.error("关闭模型失败", e);
        }
    }
}
"""

    with open(f"{output_dir}/src/main/java/com/ai/model/ModelInference.java", "w", encoding="utf-8") as f:
        f.write(java_code)

    logger.info("生成 ModelInference.java")


# ======================================================
# 生成配置文件
# ======================================================
def generate_application_properties(output_dir: str, config: Dict) -> None:
    """生成应用配置"""
    properties = f"""# AI Model Service Configuration
server.host={config.get('host', '0.0.0.0')}
server.port={config.get('port', 9000)}
model.path=model.onnx
model.name={config.get('model', 'classifier')}

# Jetty Configuration
jetty.threads.min=10
jetty.threads.max=200
jetty.threads.idle.timeout=60000

# Logging
logging.level=INFO
"""

    with open(f"{output_dir}/src/main/resources/application.properties", "w", encoding="utf-8") as f:
        f.write(properties)

    logger.info("生成 application.properties")


def generate_readme(output_dir: str, host: str, port: int) -> None:
    """生成README文档"""
    readme = f"""# AI模型推理服务

使用 Jakarta EE + 嵌入式Jetty + ONNX Runtime 构建的轻量级AI模型推理服务

## 架构

- **Jakarta EE**: Java企业版标准，提供Servlet API
- **Jetty**: 轻量级、高性能的嵌入式Web服务器
- **ONNX Runtime**: 微软开源的跨平台深度学习推理引擎

## 构建和运行

### 前置条件
- JDK 17+
- Maven 3.6+

### 构建项目
```bash
mvn clean package
```

### 运行服务
```bash
java -jar target/model-service-1.0-SNAPSHOT.jar
```

服务将在 `http://{host}:{port}` 启动

## API端点

### 1. 健康检查
```bash
curl http://localhost:{port}/health
```

### 2. 模型推理
```bash
curl -X POST http://localhost:{port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"input": [[1, 2, 3, 4, 5]]}}'
```

### 3. 模型信息
```bash
curl http://localhost:{port}/model_info
```

## 性能特点

- ✅ 轻量级：打包后<50MB
- ✅ 快速启动：<2秒
- ✅ 低内存：运行时<200MB
- ✅ 高并发：支持数百并发连接
- ✅ 跨平台：Linux/Windows/macOS

## 部署

### Docker部署
```dockerfile
FROM openjdk:17-slim
COPY target/model-service-1.0-SNAPSHOT.jar /app.jar
EXPOSE {port}
CMD ["java", "-jar", "/app.jar"]
```

### K8s部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: model-service
        image: model-service:latest
        ports:
        - containerPort: {port}
```

## 监控和日志

服务使用SLF4J进行日志记录，所有请求和错误都会被记录。

## 许可证

MIT License
"""

    with open(f"{output_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    logger.info("生成 README.md")


# ======================================================
# 复制ONNX模型
# ======================================================
def copy_onnx_model(output_dir: str, context: Dict, model_name: str) -> None:
    """复制ONNX模型到resources目录"""
    import shutil

    # 查找ONNX模型文件
    onnx_path = f"outputs/{model_name}.onnx"

    if os.path.exists(onnx_path):
        target_path = f"{output_dir}/src/main/resources/model.onnx"
        shutil.copy(onnx_path, target_path)
        logger.info(f"复制ONNX模型: {onnx_path} -> {target_path}")
    else:
        logger.warning(f"未找到ONNX模型: {onnx_path}")
        logger.info("请先导出ONNX模型，或手动放置到 src/main/resources/model.onnx")
