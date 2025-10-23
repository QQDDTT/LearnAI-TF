"""
强化学习网络客户端
用于与远程强化学习环境服务进行交互，动态加载训练数据
"""

import requests
import json
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod


class BaseRLClient(ABC):
    """强化学习客户端基类"""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        pass

    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作，返回 (next_state, reward, done, info)"""
        pass

    @abstractmethod
    def close(self):
        """关闭连接"""
        pass


class NetworkClient(BaseRLClient):
    """
    基于HTTP的强化学习网络客户端
    用于与远程RL环境服务进行交互
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        timeout: int = 10,
        max_retries: int = 3,
        endpoints: Optional[Dict[str, str]] = None,
        state_space: Optional[Dict] = None,
        action_space: Optional[Dict] = None,
        use_https: bool = False,
        headers: Optional[Dict[str, str]] = None,
        lazy_connect: bool = True
    ):
        """
        初始化网络客户端

        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            endpoints: API端点配置
            state_space: 状态空间配置
            action_space: 动作空间配置
            use_https: 是否使用HTTPS
            headers: 自定义HTTP头
            lazy_connect: 是否延迟连接（True则在初始化时不连接服务器）
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.lazy_connect = lazy_connect

        # 构建基础URL
        protocol = "https" if use_https else "http"
        self.base_url = f"{protocol}://{host}:{port}"

        # 端点配置
        self.endpoints = endpoints or {
            "reset": "/reset",
            "step": "/step",
            "close": "/close",
            "info": "/info"
        }

        # 空间配置
        self.state_space = state_space or {"shape": [4], "dtype": "float32"}
        self.action_space = action_space or {"type": "discrete", "n": 2}

        # HTTP头
        self.headers = headers or {"Content-Type": "application/json"}

        # 会话管理
        self.session = None
        self.session_id = None
        self._session_initialized = False

        # 日志
        self.logger = logging.getLogger(__name__)
        # 不在这里设置logging配置，避免覆盖用户配置

        # 统计信息
        self.total_requests = 0
        self.failed_requests = 0
        self.total_episodes = 0
        self.current_episode_steps = 0

        # 如果不是延迟连接，立即初始化会话
        if not lazy_connect:
            self._init_session()

        self.logger.debug(f"NetworkClient 初始化: {self.base_url}")

    def _init_session(self):
        """初始化HTTP会话"""
        if not self._session_initialized:
            self.session = requests.Session()
            self.session.headers.update(self.headers)
            self._session_initialized = True
            self.logger.debug("HTTP会话已初始化")

    def _ensure_session(self):
        """确保会话已初始化"""
        if not self._session_initialized:
            self._init_session()

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict:
        """
        发送HTTP请求

        Args:
            endpoint: API端点
            method: HTTP方法
            data: 请求数据
            retry_count: 当前重试次数

        Returns:
            响应数据字典
        """
        # 确保会话已初始化
        self._ensure_session()

        url = f"{self.base_url}{endpoint}"

        try:
            self.total_requests += 1

            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(
                    url,
                    json=data,
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            # 检查响应状态
            response.raise_for_status()

            # 解析响应
            result = response.json()

            # 保存会话ID
            if "session_id" in result:
                self.session_id = result["session_id"]

            return result

        except requests.exceptions.Timeout as e:
            self.logger.warning(f"请求超时: {url} - {e}")
            self.failed_requests += 1

            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # 指数退避
                self.logger.info(f"等待 {wait_time}s 后重试 ({retry_count + 1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._make_request(endpoint, method, data, retry_count + 1)
            else:
                raise Exception(f"请求失败，已达到最大重试次数: {url}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求错误: {url} - {e}")
            self.failed_requests += 1

            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                self.logger.info(f"等待 {wait_time}s 后重试 ({retry_count + 1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._make_request(endpoint, method, data, retry_count + 1)
            else:
                raise Exception(f"请求失败: {url} - {e}")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            self.failed_requests += 1
            raise Exception(f"服务器响应格式错误: {e}")

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            初始状态
        """
        endpoint = self.endpoints.get("reset", "/reset")

        request_data = {
            "session_id": self.session_id
        }

        try:
            response = self._make_request(endpoint, method="POST", data=request_data)

            # 提取状态
            state = response.get("state")
            if state is None:
                raise ValueError("服务器响应中缺少 'state' 字段")

            # 转换为numpy数组
            state = np.array(state, dtype=self.state_space.get("dtype", "float32"))

            # 验证状态形状
            expected_shape = tuple(self.state_space.get("shape", []))
            if state.shape != expected_shape:
                self.logger.warning(
                    f"状态形状不匹配: 期望 {expected_shape}, 实际 {state.shape}"
                )

            # 更新统计
            self.total_episodes += 1
            self.current_episode_steps = 0

            self.logger.info(f"环境已重置 (Episode {self.total_episodes})")

            return state

        except Exception as e:
            self.logger.error(f"重置环境失败: {e}")
            raise

    def step(
        self,
        action: Union[int, np.ndarray, List]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 要执行的动作

        Returns:
            (next_state, reward, done, info) 元组
        """
        endpoint = self.endpoints.get("step", "/step")

        # 处理动作格式
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, (int, float)):
            action = int(action)

        # 验证动作
        self._validate_action(action)

        request_data = {
            "action": action,
            "session_id": self.session_id
        }

        try:
            response = self._make_request(endpoint, method="POST", data=request_data)

            # 提取响应数据
            next_state = response.get("state")
            reward = response.get("reward")
            done = response.get("done", False)
            info = response.get("info", {})

            if next_state is None or reward is None:
                raise ValueError("服务器响应中缺少必要字段 ('state' 或 'reward')")

            # 转换数据类型
            next_state = np.array(
                next_state,
                dtype=self.state_space.get("dtype", "float32")
            )
            reward = float(reward)
            done = bool(done)

            # 更新统计
            self.current_episode_steps += 1

            if done:
                self.logger.info(
                    f"Episode {self.total_episodes} 完成, "
                    f"步数: {self.current_episode_steps}, "
                    f"最终奖励: {reward}"
                )

            return next_state, reward, done, info

        except Exception as e:
            self.logger.error(f"执行步骤失败: {e}")
            raise

    def _validate_action(self, action: Union[int, List]):
        """验证动作是否有效"""
        action_type = self.action_space.get("type", "discrete")

        if action_type == "discrete":
            n = self.action_space.get("n", 2)
            if not isinstance(action, int) or action < 0 or action >= n:
                raise ValueError(f"动作必须是 [0, {n}) 范围内的整数，得到: {action}")

        elif action_type == "continuous":
            shape = self.action_space.get("shape", [1])
            if isinstance(action, list):
                if len(action) != shape[0]:
                    raise ValueError(
                        f"动作维度不匹配: 期望 {shape[0]}, 得到 {len(action)}"
                    )
            low = self.action_space.get("low", -np.inf)
            high = self.action_space.get("high", np.inf)
            # 可以添加更多连续动作空间的验证

    def get_info(self) -> Dict:
        """
        获取环境信息

        Returns:
            环境信息字典
        """
        endpoint = self.endpoints.get("info", "/info")

        try:
            response = self._make_request(endpoint, method="GET")
            return response
        except Exception as e:
            self.logger.warning(f"获取环境信息失败: {e}")
            return {}

    def get_statistics(self) -> Dict:
        """
        获取客户端统计信息

        Returns:
            统计信息字典
        """
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests
            if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "total_episodes": self.total_episodes,
            "current_episode_steps": self.current_episode_steps,
            "session_id": self.session_id
        }

    def close(self):
        """关闭连接"""
        endpoint = self.endpoints.get("close", "/close")

        try:
            if self.session_id and self._session_initialized:
                request_data = {"session_id": self.session_id}
                self._make_request(endpoint, method="POST", data=request_data)

            if self._session_initialized and self.session:
                self.session.close()

            self.logger.debug("NetworkClient 已关闭")

        except Exception as e:
            self.logger.warning(f"关闭连接时出现警告: {e}")
        finally:
            self.session_id = None
            self._session_initialized = False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass


class WebSocketClient(BaseRLClient):
    """
    基于WebSocket的强化学习客户端
    用于需要实时通信的场景
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        timeout: int = 10,
        max_retries: int = 3,
        state_space: Optional[Dict] = None,
        action_space: Optional[Dict] = None
    ):
        """
        初始化WebSocket客户端

        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 连接超时时间
            max_retries: 最大重试次数
            state_space: 状态空间配置
            action_space: 动作空间配置
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries

        self.state_space = state_space or {"shape": [4], "dtype": "float32"}
        self.action_space = action_space or {"type": "discrete", "n": 2}

        self.ws = None
        self.connected = False

        self.logger = logging.getLogger(__name__)

        # 注意：实际使用需要安装 websocket-client 库
        try:
            import websocket
            self.websocket = websocket
        except ImportError:
            self.logger.warning(
                "未安装 websocket-client 库，WebSocket客户端不可用。"
                "请运行: pip install websocket-client"
            )
            self.websocket = None

    def connect(self):
        """建立WebSocket连接"""
        if self.websocket is None:
            raise ImportError("需要安装 websocket-client 库")

        ws_url = f"ws://{self.host}:{self.port}"

        try:
            self.ws = self.websocket.create_connection(
                ws_url,
                timeout=self.timeout
            )
            self.connected = True
            self.logger.info(f"WebSocket 已连接: {ws_url}")

        except Exception as e:
            self.logger.error(f"WebSocket 连接失败: {e}")
            raise

    def _send_receive(self, message: Dict) -> Dict:
        """发送消息并接收响应"""
        if not self.connected:
            self.connect()

        try:
            # 发送消息
            self.ws.send(json.dumps(message))

            # 接收响应
            response = self.ws.recv()
            return json.loads(response)

        except Exception as e:
            self.logger.error(f"WebSocket 通信错误: {e}")
            self.connected = False
            raise

    def reset(self) -> np.ndarray:
        """重置环境"""
        message = {"type": "reset"}
        response = self._send_receive(message)

        state = np.array(
            response["state"],
            dtype=self.state_space.get("dtype", "float32")
        )
        return state

    def step(
        self,
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        if isinstance(action, np.ndarray):
            action = action.tolist()

        message = {
            "type": "step",
            "action": action
        }

        response = self._send_receive(message)

        next_state = np.array(
            response["state"],
            dtype=self.state_space.get("dtype", "float32")
        )
        reward = float(response["reward"])
        done = bool(response["done"])
        info = response.get("info", {})

        return next_state, reward, done, info

    def close(self):
        """关闭连接"""
        if self.connected and self.ws:
            try:
                message = {"type": "close"}
                self.ws.send(json.dumps(message))
                self.ws.close()
                self.logger.info("WebSocket 已关闭")
            except Exception as e:
                self.logger.warning(f"关闭WebSocket时出现警告: {e}")
            finally:
                self.connected = False
                self.ws = None


class BatchNetworkClient:
    """
    批量网络客户端
    支持并行请求多个环境实例
    """

    def __init__(
        self,
        num_clients: int,
        host: str = "localhost",
        port: int = 8888,
        **kwargs
    ):
        """
        初始化批量客户端

        Args:
            num_clients: 客户端数量
            host: 服务器地址
            port: 服务器端口
            **kwargs: 传递给NetworkClient的其他参数
        """
        self.num_clients = num_clients
        self.clients = []

        for i in range(num_clients):
            client = NetworkClient(host=host, port=port, **kwargs)
            self.clients.append(client)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BatchNetworkClient 初始化: {num_clients} 个客户端")

    def reset(self) -> np.ndarray:
        """批量重置所有环境"""
        states = []
        for client in self.clients:
            state = client.reset()
            states.append(state)
        return np.array(states)

    def step(
        self,
        actions: Union[List, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """批量执行动作"""
        if len(actions) != self.num_clients:
            raise ValueError(
                f"动作数量 ({len(actions)}) 与客户端数量 ({self.num_clients}) 不匹配"
            )

        next_states = []
        rewards = []
        dones = []
        infos = []

        for client, action in zip(self.clients, actions):
            next_state, reward, done, info = client.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.array(next_states),
            np.array(rewards),
            np.array(dones),
            infos
        )

    def close(self):
        """关闭所有客户端"""
        for client in self.clients:
            client.close()
        self.logger.info("所有客户端已关闭")

