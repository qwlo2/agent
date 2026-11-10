"""存储消息并提供模型请求使用的上下文。"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SurfaceNode:
    """具有 History seq 身份的模型可见消息节点。"""

    seq: int
    message: dict[str, Any]


class ContextBuilder:
    """维护内存消息列表，提供追加、获取和清空操作。"""

    def __init__(self) -> None:
        """初始化空的消息列表，消息由 Session 或调用方追加。"""
        self._nodes: list[SurfaceNode] = []
        self.replace_generation = 0

    def __len__(self) -> int:
        """返回当前有效消息节点数，供缓存判断是否有新增节点。"""
        return len(self._nodes)

    def append(self, message: dict[str, Any], seq: int) -> None:
        """按当前 Surface 顺序追加节点，不增加 Replace generation。"""
        if any(node.seq == seq for node in self._nodes):
            raise ValueError(f"重复的消息 seq：{seq}")
        if not self._nodes and (seq != 0 or message.get("role") != "system"):
            raise ValueError("首个上下文节点必须是 seq=0 的系统消息")
        self._nodes.append(SurfaceNode(seq, deepcopy(message)))

    def nodes(self, start: int = 0, end: int | None = None) -> tuple[SurfaceNode, ...]:
        """返回指定位置范围的节点快照，外部修改不会影响有效消息或配对缓存。"""
        return tuple(deepcopy(self._nodes[start:end]))

    def locate_range(self, start_seq: int, end_seq: int, seqs: list[int]) -> tuple[int, int]:
        """按 Surface 位置验证连续的替换范围，返回左闭右开的索引。"""
        index = {node.seq: position for position, node in enumerate(self._nodes)}
        if start_seq not in index or end_seq not in index:
            raise ValueError("Replace 目标不在当前有效消息中")
        start, end = index[start_seq], index[end_seq] + 1
        if start >= end or start == 0 or 0 in seqs:
            raise ValueError("Replace 范围顺序错误或试图替换 seq=0")
        if [node.seq for node in self._nodes[start:end]] != seqs:
            raise ValueError("Replace 来源与当前连续消息范围不一致")
        return start, end

    def replace(self, start: int, end: int, node: SurfaceNode) -> None:
        """在已验证的位置替换节点，保持逻辑顺序并递增 generation。"""
        if not 0 < start < end <= len(self._nodes) or node.seq == 0:
            raise ValueError("无效的 Replace 位置或受保护的 seq=0")
        if any(existing.seq == node.seq for existing in self._nodes):
            raise ValueError("摘要 seq 已存在，不能重复应用 Replace")
        self._nodes[start:end] = [deepcopy(node)]
        self.replace_generation += 1

    def build(self) -> list[dict[str, Any]]:
        """返回独立的有效消息副本，可直接传入 Client.ask()。"""
        return [deepcopy(node.message) for node in self._nodes]

    def clear(self) -> None:
        """清空内存消息列表。"""
        self._nodes.clear()
        self.replace_generation += 1
