"""统计消息 token，并缓存由服务端用量校准的当前上下文总量。"""

import json
from typing import Any, Callable, Sequence


class Tokens:
    """提供单条消息计数、实时上下文总量及压缩阈值判断。"""

    def __init__(
        self, compression_limit: int | None = None,
        counter: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        """初始化缓存；compression_limit 为压缩阈值，None 表示暂不设置。"""
        self.total_tokens = 0
        self.compression_limit = compression_limit
        self.counter = counter

    def count_value(self, value: dict[str, Any]) -> int:
        """统一计数接口，可注入 Tokenizer；默认使用四字节估算。"""
        return self.counter(value) if self.counter is not None else self.count(value)

    def count_request(
        self, messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        **injected: Any,
    ) -> int:
        """统计实际有效请求，包含系统消息、工具 Schema 及额外注入字段。"""
        request = {"messages": list(messages), **injected}
        if tools:
            request["tools"] = list(tools)
            request.setdefault("tool_choice", "auto")
        return self.count_value(request)

    def measure_request(
        self, messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None, **injected: Any,
    ) -> int:
        """准备请求时重新计量当前有效上下文，不累计历史请求用量。"""
        self.total_tokens = self.count_request(messages, tools, **injected)
        return self.total_tokens

    def update(
        self,
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> int:
        """追加一条上下文消息后更新并返回缓存总量。

        有服务端用量时以 total_tokens 覆盖缓存，校准已有上下文和本次输出。
        本次输出中的 tool-call 已包含在内，不再次累加。
        无服务端用量时按四字节算法估算新增消息，并累加到缓存。
        """
        if usage is not None:
            self.total_tokens = usage["total_tokens"]
        else:
            self.total_tokens += self.count_value(message)
        return self.total_tokens

    @property
    def needs_compression(self) -> bool:
        """返回缓存总量是否达到配置阈值，只判断，不执行压缩。"""
        return self.compression_limit is not None and self.total_tokens >= self.compression_limit

    def reset(self) -> None:
        """清零上下文 token 缓存，保留配置的压缩阈值。"""
        self.total_tokens = 0

    @staticmethod
    def count(
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> int:
        """返回单条消息的 token 数，不累计整个请求或会话的用量。

        Args:
            message: OpenAI 兼容的原始消息字典，可包含正文、推理内容和工具调用。
            usage: 该 assistant 消息对应的服务端用量，使用 completion_tokens。
                用户、工具等本地消息不传此参数。

        Returns:
            有服务端用量时返回输出 token 数；否则将消息序列化为紧凑 JSON，
            按 UTF-8 字节数除以 4 向上取整，作为近似 token 数。
        """
        if usage is not None:
            return usage["completion_tokens"]
        content = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        return (len(content.encode("utf-8")) + 3) // 4
