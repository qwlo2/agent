"""单会话 JSONL 事件存储，记录消息及问答、步骤的开始和结束。"""

import json
import os
import shutil
import time
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

from tokens import Tokens


class TurnEndReason(str, Enum):
    """问答结束原因：正常完成、执行异常或达到步骤上限。"""

    COMPLETED = "completed"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class HistoryStore:
    """追加带全局 seq 的历史事件，seq=0 固定为系统身份提示词。"""

    def __init__(self, path: str | Path | None = None) -> None:
        """指定事件 JSONL 文件，默认使用 data/history.jsonl。"""
        history_dir = Path(__file__).parent / "data"
        self.path = Path(path) if path is not None else history_dir / "history.jsonl"
        self._next_seq = 0
        self._events: dict[int, dict[str, Any]] = {}
        self._write_failed = False

    def load(self, system_prompt: str) -> list[dict[str, Any]]:
        """读取事件并接续 seq；空历史先写入 seq=0、turn=0 的系统消息。

        未完成的末行不视为已提交事件，备份后截去；完整行损坏则明确报错。
        """
        if self.path.exists():
            content = self.path.read_bytes()
            committed_end = content.rfind(b"\n") + 1
            if committed_end != len(content):
                backup = self.path.with_name(f"{self.path.name}.incomplete.{time.time_ns()}.bak")
                shutil.copy2(self.path, backup)
                with self.path.open("r+b") as target:
                    target.truncate(committed_end)
                    target.flush()
                    os.fsync(target.fileno())
            records = [json.loads(line) for line in content[:committed_end].splitlines()]
        else:
            records = []

        if records:
            if any(record["seq"] != index for index, record in enumerate(records)):
                raise ValueError("History seq 必须从 0 开始连续递增，不能重复或乱序")
            self._events = {record["seq"]: deepcopy(record) for record in records}
            self._next_seq = records[-1]["seq"] + 1
            self._write_failed = False
            return records

        system_event = {
            "type": "message", "turn": 0,
            "message": {"role": "system", "content": system_prompt},
        }
        self._next_seq = 0
        self._events.clear()
        self._write_failed = False
        return self.append([system_event])

    def append(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """给新增事件分配全局递增的 seq，并追加到 JSONL，返回已编号的事件。

        message 事件增加 tokens 字段；有 usage 时使用服务端输出用量，否则估算。
        调用方传入 type、turn、step 等事件字段；编号在写入成功后递增。
        每次写入后 flush 并 fsync，不重写已有事件。
        """
        if self._write_failed:
            raise RuntimeError("历史提交曾失败，请重启加载日志后再写入")
        records = []
        for index, event in enumerate(events):
            record = {**event, "seq": self._next_seq + index}
            if event["type"] in {"message", "compaction/summary"}:
                record["tokens"] = Tokens.count(event["message"], event.get("usage"))
            records.append(record)
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        except OSError:
            self._write_failed = True
            raise
        self._next_seq += len(records)
        self._events.update({record["seq"]: deepcopy(record) for record in records})
        return records

    def get(self, seq: int) -> dict[str, Any]:
        """按 seq 查询包括被压缩来源在内的原始事件，返回独立副本。"""
        if seq not in self._events:
            raise ValueError(f"History 中不存在来源 seq={seq}")
        return deepcopy(self._events[seq])

    def clear(self) -> None:
        """清空事件文件并重置 seq。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._next_seq = 0
        self._events.clear()
        self._write_failed = False
