"""DeepSeek Chat Completions 封装；工具的执行由 Agent 层负责。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import dotenv_values
from openai import OpenAI
from openai.types.chat import ChatCompletion


@dataclass
class ToolCall:
    """模型请求的一次工具调用，由 Agent 根据名称和参数执行。

    Attributes:
        id: API 返回的单次工具调用标识，回传工具结果时用作 tool_call_id。
        name: 模型选择的工具名称。
        arguments: 从工具调用参数 JSON 解析得到的字典。
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """统一表示用户、模型或工具消息。

    Attributes:
        role: 消息角色，例如 system、user、assistant 或 tool。
        content: 消息正文，工具调用时可能为 None。
        tool_calls: 模型请求的工具调用列表，每条消息使用独立的列表。
        tool_call_id: 工具结果对应的调用标识，非工具消息通常为 None。
        metadata: 推理内容、结束原因、token 用量等附加信息。
    """

    role: str
    content: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    """ field用来配置数据类中的成员变量,default_factory 的设计就是为了让每个实例获得独立的默认对象。"""

class Trip_Client:
    """封装 DeepSeek 的非流式询问和响应解析，工具列表由调用方传入。

    工具执行和对话历史由调用方管理。支持 with 语句，退出时关闭连接。
    """

    def __init__(self, env_file: str | Path | None = None) -> None:
        """加载配置并创建 SDK 客户端，不在初始化时发送请求。

        Args:
            env_file: 配置文件路径，默认使用 client.py 同目录下的 .env。
                同名系统环境变量优先于文件配置。

        Raises:
            ValueError: API Key 为空或超时时间无法转换为数字。
        """
        config = {
            **dotenv_values(env_file or Path(__file__).with_name(".env")),
            **os.environ,
        }
        api_key = (config.get("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("请在 .env 或系统环境变量中设置 DEEPSEEK_API_KEY")

        self.model = config.get("DEEPSEEK_MODEL") or "deepseek-flash"
        self._api = OpenAI(
            api_key=api_key,
            base_url=config.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            timeout=float(config.get("DEEPSEEK_TIMEOUT") or "60"),
            max_retries=0,
        )

    def ask(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        """
        str | dict[str, Any]相当于union,
        值	含义
        "auto"	模型自主决定是否调用工具
        "none"	不调用工具
        "required"	必须调用工具
        """
        """发送一次非流式请求，直接返回 SDK 响应，不修改或保存对话历史。

        Args:
            messages: OpenAI 兼容格式的消息序列，由调用方维护完整历史。
            tools: ToolExecutor.get_all_tools() 返回的工具描述列表，可省略。
            tool_choice: 工具选择策略，仅在传入非空工具列表时发送给 API。
            temperature: 可选采样温度，None 表示采用服务端默认值。
            max_tokens: 可选的最大输出 token 数，None 表示采用服务端默认值。

        Returns:
            OpenAI SDK 返回的原始 ChatCompletion 响应，由调用方决定何时解析。

        网络及 API 错误直接向调用方抛出。
        """
        options: dict[str, Any] = {}
        if tools:
            options["tools"] = list(tools)
            options["tool_choice"] = tool_choice
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens

        response = self._api.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **options,
        )
        return response

    @staticmethod
    def parse_response(response: ChatCompletion) -> Message:
        """
        SDK返回的是对象(类)而不是dict
        将 SDK 响应的第一个候选消息解析为 Message。

        Args:
            response: OpenAI SDK 返回的非流式 ChatCompletion 对象。

        Returns:
            统一的 Message，工具参数转换为 Python 字典，其他信息放入 metadata。

        Raises:
            json.JSONDecodeError: 工具调用参数不是合法 JSON。
            ValueError: 工具调用参数的 JSON 顶层不是对象。
        """
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        tool_calls = []
        for call in message.get("tool_calls", []):
            function = call["function"]
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError(f"工具 {function['name']} 的参数必须为 JSON 对象")
            tool_calls.append(ToolCall(call["id"], function["name"], arguments))

        return Message(
            role=choice.message.role,
            content=choice.message.content,
            tool_calls=tool_calls,
            metadata={
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage is not None else None,
            },
        )

    def close(self) -> None:
        """关闭底层 SDK 客户端，释放 HTTP 连接资源。"""
        self._api.close()

    def __enter__(self) -> Trip_Client:
        """进入 with 上下文并返回当前客户端。"""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """退出 with 上下文时关闭连接，上下文中的异常继续向外传播。

        Args:
            exc_type: 上下文中的异常类型，无异常时为 None。
            exc_value: 上下文中的异常实例，无异常时为 None。
            traceback: 异常的回溯对象，无异常时为 None。
        """
        self.close()
