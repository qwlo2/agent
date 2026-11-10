"""组合历史存储和模型上下文，提供会话级操作。"""

from typing import Any, Callable, Sequence

from compact import Compact, CompactResult
from context_builder import ContextBuilder, SurfaceNode
from history import HistoryStore
from tokens import Tokens


class Session:
    """持有 History、ContextBuilder、Tokens 和 Compact，管理会话的完整生命周期。"""

    def __init__(
        self,
        history: HistoryStore,
        context: ContextBuilder,
        system_prompt: str,
        compact: Compact,
    ) -> None:
        """加载历史消息到上下文，并恢复最近的 turn 编号。

        Args:
            history: 提供事件持久化及 seq 编号的历史存储。
            context: 用于存储消息的空上下文对象。
            system_prompt: 新建或清空会话时使用的系统身份提示词。
            compact: 当前会话拥有的压缩对象，调用时由 Session 提供上下文和历史。
        """
        self.history = history
        self.context = context
        self.system_prompt = system_prompt
        self.compact = compact
        self.tokens = Tokens(compression_limit=compact.threshold)
        self.tools_provider: Callable[[], Sequence[dict[str, Any]]] | None = None
        self._projection_valid = True
        events = history.load(system_prompt)
        self._replay(events)

    def _replay(self, events: list[dict[str, Any]]) -> None:
        """按日志顺序重放 Append 和 Replace，构建当前有效消息，不按 Surface seq 排序。"""
        self.turn = max(event["turn"] for event in events)
        for event in events:
            if event["type"] == "message":
                self.append_context(event["message"], usage=event.get("usage"), seq=event["seq"])
            elif event["type"] == "compaction/summary":
                self.apply_summary(event)

    def append_history(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """调用 HistoryStore.append() 写入事件并更新最近 turn，返回已编号的事件。"""
        if not self._projection_valid:
            raise RuntimeError("摘要已提交但消息投影未完成，请重启从日志重建")
        records = self.history.append(events)
        for event in records:
            self.turn = max(self.turn, event["turn"])
        return records

    def append_context(
        self,
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
        *,
        seq: int,
    ) -> None:
        """追加上下文消息并更新 token 缓存，usage 用于服务端总量校准。"""
        if not self._projection_valid:
            raise RuntimeError("消息投影未完成，请重启从日志重建")
        self.context.append(message, seq)
        self.tokens.update(message, usage=usage)

    def get_tools(self) -> Sequence[dict[str, Any]]:
        """统一获取当前工具 Schema，尚未绑定提供函数时返回空列表。"""
        return self.tools_provider() if self.tools_provider is not None else []

    def count_context_tokens(self, messages: Sequence[dict[str, Any]] | None = None) -> int:
        """计算消息及当前工具 Schema 的成本，可用于评估候选上下文，不更新缓存。"""
        if messages is None:
            messages = self.context.build()
        return self.tokens.count_request(messages, self.get_tools())

    def measure_context_tokens(self, messages: Sequence[dict[str, Any]] | None = None) -> int:
        """统一计量实际上下文并更新总 token 缓存。"""
        self.tokens.total_tokens = self.count_context_tokens(messages)
        return self.tokens.total_tokens

    def compact_if_needed(self) -> CompactResult:
        """调用所属 Compact，自动判断当前完整上下文是否需要压缩。"""
        return self.compact.compact_if_needed(self)

    def compact_now(self, *, source: str = "manual") -> CompactResult:
        """手动压缩及 LLM 工具入口，向所属 Compact 临时传入当前会话。"""
        return self.compact.compact_now(self, source=source)

    def apply_summary(self, event: dict[str, Any], *, persist: bool = False) -> dict[str, Any]:
        """验证并应用一条完整摘要 Replace；persist 时先追加单条提交事件再替换。

        相同入口用于日志重放和运行时提交，原始消息及摘要来源均保留在 History。
        日志提交后即使进程在内存替换前退出，重启也能重放同一次 Replace。
        """
        if event["type"] != "compaction/summary":
            raise ValueError("Replace 必须由 compaction/summary 事件提交")
        data, operation = event["data"], event["surfaceOp"]
        if operation["op"] != "replace" or data["shadowedRange"] != {
            "startSeq": operation["startSeq"], "endSeq": operation["endSeq"],
        }:
            raise ValueError("摘要与 Replace 边界不一致")
        if not isinstance(data["summary"], str) or not data["summary"].strip():
            raise ValueError("摘要正文为空或不是文本")
        if event["message"] != {"role": "user", "content": data["summary"]}:
            raise ValueError("模型可见摘要与持久化正文不一致")
        start, end = self.context.locate_range(
            operation["startSeq"], operation["endSeq"], data["shadowedSeqs"],
        )
        for seq in data["shadowedSeqs"]:
            if self.history.get(seq)["type"] not in {"message", "compaction/summary"}:
                raise ValueError("Replace 来源不是模型可见消息")
        if persist:
            event = self.append_history([event])[0]
            self._projection_valid = False
        self.context.replace(start, end, SurfaceNode(event["seq"], event["message"]))
        self._projection_valid = True
        self.measure_context_tokens()
        return event

    def clear(self) -> None:
        """清空历史、上下文、token 及压缩缓存，重新写入系统消息并重置 turn。"""
        self.history.clear()
        self.context.clear()
        self.tokens.reset()
        self.compact.balance_cache = None
        self.compact.last_result = None
        self._projection_valid = True
        events = self.history.load(self.system_prompt)
        self._replay(events)
