"""Compact 的离线验收测试，所有 LLM 响应均由内存桩提供。"""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from agent import GuideAgent
from client import Trip_Client
from compact import Compact
from context_builder import ContextBuilder, SurfaceNode
from history import HistoryStore
from main import load_compact_settings
from session import Session
from tool_executor import ToolExecutor


def response(content="摘要", *, calls=None, finish="stop"):
    """创建具有真实 SDK 类型的离线响应。"""
    return ChatCompletion.model_validate({
        "id": "test", "object": "chat.completion", "created": 0, "model": "mock",
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": content, "tool_calls": calls,
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    })


def tool_call(call_id, name="echo"):
    """创建模型可见的函数调用。"""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


class FakeClient:
    """区分摘要调用和普通调用的同步 LLM 桩，不访问网络。"""

    def __init__(self):
        """初始化可定制响应及调用记录。"""
        self.requests = []
        self.model = "mock"
        self.summary_response = response()
        self.normal_responses = []
        self.on_summary = None

    def ask(self, messages, tools=None, **options):
        """记录请求，摘要响应可模拟异常和生成期间的上下文变化。"""
        self.requests.append((deepcopy(messages), deepcopy(tools), options))
        if tools is None:
            if self.on_summary is not None:
                self.on_summary()
            if isinstance(self.summary_response, Exception):
                raise self.summary_response
            return self.summary_response
        return self.normal_responses.pop(0)

    parse_response = staticmethod(Trip_Client.parse_response)


class CompactTests(unittest.TestCase):
    """验证阈值、配对缓存、压缩事务、日志重放和 Agent 工具边界。"""

    def setUp(self):
        """每个测试使用独立临时日志与模拟模型。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.jsonl"
        self.client = FakeClient()
        self.compact_settings = {
            "context_window": 2000, "max_output_tokens": 100,
            "threshold_ratio": 0.3, "retain_ratio": 0.05,
        }
        self.compact = Compact(self.client, **self.compact_settings)
        self.session = Session(HistoryStore(self.path), ContextBuilder(), "system", self.compact)

    def add(self, message):
        """持久化并追加一条具有 seq 身份的上下文消息。"""
        event = self.session.append_history([{"type": "message", "turn": 1, "message": message}])[0]
        self.session.append_context(message, seq=event["seq"])
        return event["seq"]

    def populate(self):
        """生成超过默认阈值、但可安全缩减的上下文。"""
        for role in ("user", "assistant", "user", "assistant"):
            self.add({"role": role, "content": "x" * 600})

    def reopen(self):
        """仅依赖日志重建新的会话，不复用任何投影或缓存。"""
        compact = Compact(FakeClient(), **self.compact_settings)
        return Session(HistoryStore(self.path), ContextBuilder(), "unused", compact)

    def test_threshold_and_configuration(self):
        """预算严格使用 W、O、r、k，配置缺失及无效值明确失败。"""
        self.assertEqual(self.compact.threshold, 600)
        self.assertEqual(self.compact.retain_tokens, 95)
        settings = {
            "DEEPSEEK_CONTEXT_WINDOW": "2000", "DEEPSEEK_MAX_OUTPUT_TOKENS": "100",
            "COMPACT_THRESHOLD_RATIO": "0.3", "COMPACT_RETAIN_RATIO": "0.05",
        }
        self.assertEqual(load_compact_settings(settings), self.compact_settings)
        with self.assertRaisesRegex(ValueError, "T"):
            load_compact_settings({**settings, "DEEPSEEK_CONTEXT_WINDOW": "200"})
        for window, output in ((0, 1), (10, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                load_compact_settings({
                    **settings, "DEEPSEEK_CONTEXT_WINDOW": str(window),
                    "DEEPSEEK_MAX_OUTPUT_TOKENS": str(output),
                })
        with self.assertRaises(ValueError):
            load_compact_settings({**settings, "COMPACT_RETAIN_RATIO": "0.5"})
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_CONTEXT_WINDOW"):
            load_compact_settings({})
        with self.assertRaisesRegex(ValueError, "COMPACT_THRESHOLD_RATIO"):
            load_compact_settings({**settings, "COMPACT_THRESHOLD_RATIO": ""})

    def test_below_threshold(self):
        """低压力请求不调用摘要模型，也不写日志。"""
        original = self.path.read_bytes()
        result = self.session.compact_if_needed()
        self.assertEqual(result.reason, "below_threshold")
        self.assertFalse(self.client.requests)
        self.assertEqual(original, self.path.read_bytes())

    def test_threshold_trigger_and_position(self):
        """达到 T 后按 K 留下尾部，摘要原位替换且保护 seq0。"""
        self.populate()
        original = self.session.context.nodes()
        original_log = self.path.read_bytes()
        result = self.session.compact_if_needed()
        self.assertEqual(result.status, "compacted")
        nodes = self.session.context.nodes()
        self.assertEqual(nodes[0], original[0])
        self.assertEqual(nodes[1].seq, result.summary_seq)
        self.assertEqual(nodes[-1], original[-1])
        self.assertLess(result.after_tokens, result.before_tokens)
        self.assertTrue(self.path.read_bytes().startswith(original_log))
        event = self.session.history.get(result.summary_seq)
        self.assertEqual(event["data"]["shadowedSeqs"], [node.seq for node in original[1:-1]])
        for seq in event["data"]["shadowedSeqs"]:
            self.assertEqual(self.session.history.get(seq)["type"], "message")
        self.assertIsNone(self.client.requests[0][1])
        self.assertEqual(self.client.requests[0][2]["tool_choice"], "none")

    def test_request_measurement_includes_tools(self):
        """工具 Schema 参与请求计量，非模型日志不进入当前上下文。"""
        before = self.session.count_context_tokens()
        tools = [{"type": "function", "function": {"name": "large", "description": "x" * 3000}}]
        self.session.tools_provider = lambda: tools
        after = self.session.count_context_tokens()
        self.assertGreater(after, before + 600)
        self.session.tools_provider = None
        self.session.append_history([{"type": "tool/call", "turn": 1, "payload": "x" * 10000}])
        self.assertEqual(self.session.count_context_tokens(), before)
        self.assertEqual(self.session.compact_if_needed().reason, "below_threshold")

    def test_parallel_tools_move_boundary_backward(self):
        """并行调用按数量平衡，无论结果顺序如何都不拆开整组工具消息。"""
        user_seq = self.add({"role": "user", "content": "x" * 600})
        call_seq = self.add({"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]})
        self.add({"role": "tool", "tool_call_id": "b", "content": "x" * 600})
        self.add({"role": "tool", "tool_call_id": "a", "content": "x" * 600})
        cache = self.compact.ensure_balance_cache(self.session)
        self.assertEqual(cache.in_progress_tool_calls, 0)
        self.assertEqual(cache.cut_balanced, [True, True, True, False, False, True])
        selection = self.compact.select_compactable_range(self.session)
        self.assertEqual([node.seq for node in selection.nodes], [user_seq])
        self.assertEqual(self.session.context.nodes(selection.end)[0].seq, call_seq)

    def test_cache_incremental_and_generation(self):
        """缓存无变化时复用，Append 扩展，Replace 后才重建。"""
        cache = self.compact.ensure_balance_cache(self.session)
        self.assertIs(cache, self.compact.ensure_balance_cache(self.session))
        with patch.object(self.compact, "event_delta", wraps=self.compact.event_delta) as scan:
            self.add({"role": "user", "content": "x" * 600})
            self.assertIs(cache, self.compact.ensure_balance_cache(self.session))
            self.assertEqual(scan.call_count, 1)
            self.compact.ensure_balance_cache(self.session)
            self.assertEqual(scan.call_count, 1)
        self.populate()
        old_generation = cache.generation
        self.assertEqual(self.session.compact_now().status, "compacted")
        rebuilt = self.compact.ensure_balance_cache(self.session)
        self.assertIsNot(rebuilt, cache)
        self.assertGreater(rebuilt.generation, old_generation)
        self.assertEqual(len(rebuilt.cut_balanced), len(self.session.context) + 1)

    def test_pending_and_no_safe_range(self):
        """没有非空安全范围时跳过，不因工具调用而引入延后调度。"""
        self.add({"role": "user", "content": "x" * 1000})
        self.assertEqual(self.session.compact_now().reason, "no_safe_range")
        self.add({"role": "assistant", "tool_calls": [tool_call("a")]})
        self.assertEqual(self.session.compact_now().reason, "no_safe_range")
        self.assertFalse(self.client.requests)

    def test_summary_error_propagates(self):
        """摘要请求异常直接传播，已有历史和消息保持不变。"""
        self.populate()
        original_nodes = self.session.context.nodes()
        original_log = self.path.read_bytes()
        self.client.summary_response = TimeoutError("offline")
        with self.assertRaises(TimeoutError):
            self.session.compact_now()
        self.assertEqual(self.session.context.nodes(), original_nodes)
        self.assertEqual(self.path.read_bytes(), original_log)

    def test_changed_range_aborts_commit(self):
        """摘要目标已被替换时由 Session 投影层报错，不转换失败状态。"""
        self.populate()
        self.client.on_summary = lambda: self.session.context.replace(
            1, 2, SurfaceNode(999, {"role": "user", "content": "changed"}),
        )
        original_log = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.session.compact_now()
        self.assertEqual(self.path.read_bytes(), original_log)

    def test_multiple_compactions_and_replay(self):
        """摘要可再压缩，来源 seq 非数值连续时仍按 Surface 顺序重放。"""
        self.populate()
        first = self.session.compact_now()
        for role in ("user", "assistant", "user"):
            self.add({"role": role, "content": "y" * 600})
        second = self.session.compact_now()
        self.assertEqual(second.status, "compacted")
        source = self.session.history.get(second.summary_seq)["data"]["shadowedSeqs"]
        self.assertIn(first.summary_seq, source)
        self.assertNotEqual(source, sorted(source))
        self.assertEqual(self.reopen().context.nodes(), self.session.context.nodes())
        self.assertEqual(self.reopen().context.replace_generation, 2)
        self.assertIn("摘要", self.client.requests[-1][0][1]["content"])

    def test_append_during_summary_is_compatible(self):
        """摘要生成期间仅尾部追加消息，不改变目标范围时仍可提交。"""
        self.populate()
        self.client.on_summary = lambda: self.add({"role": "user", "content": "new" * 1000})
        result = self.session.compact_now()
        self.assertEqual(result.status, "compacted")
        self.assertEqual(self.session.context.nodes()[-1].message["content"], "new" * 1000)
        self.assertEqual(self.reopen().context.nodes(), self.session.context.nodes())

    def test_replay_rejects_invalid_range_and_seq0(self):
        """日志中的无效目标、逆序范围及 seq0 替换不会被重放。"""
        self.populate()
        result = self.session.compact_now()
        good = self.path.read_text()
        records = [json.loads(line) for line in good.splitlines()]
        for source in ([0, 1], [2, 1], [999]):
            with self.subTest(source=source):
                bad = deepcopy(records)
                event = bad[-1]
                event["data"]["shadowedSeqs"] = source
                boundary = {"startSeq": source[0], "endSeq": source[-1]}
                event["data"]["shadowedRange"] = boundary
                event["surfaceOp"] = {"op": "replace", **boundary}
                self.path.write_text("".join(json.dumps(item) + "\n" for item in bad))
                with self.assertRaises(ValueError):
                    self.reopen()
        self.path.write_text(good)
        duplicate = self.session.history.get(result.summary_seq)
        self.session.append_history([duplicate])
        with self.assertRaises(ValueError):
            self.reopen()

    def test_crash_after_log_commit_replays_replace(self):
        """日志已提交而内存替换尚未完成时，重启按提交记录完成投影。"""
        self.populate()
        before = self.session.context.nodes()
        with patch.object(self.session.context, "replace", side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError):
                self.session.compact_now()
        self.assertEqual(self.session.context.nodes(), before)
        with self.assertRaisesRegex(RuntimeError, "投影"):
            self.session.append_history([{"type": "turn/start", "turn": 2}])
        restored = self.reopen()
        self.assertLess(len(restored.context), len(before))
        self.assertEqual(restored.context.nodes()[0].seq, 0)

    def test_durability_failure_blocks_further_writes(self):
        """持久化失败后禁止用过期 seq 继续写入，重启以实际日志为准。"""
        event = {"type": "message", "turn": 1, "message": {"role": "user", "content": "saved"}}
        with patch("history.os.fsync", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.session.append_history([event])
        with self.assertRaisesRegex(RuntimeError, "重启"):
            self.session.append_history([event])
        restored = self.reopen()
        self.assertEqual(restored.context.nodes()[-1].message["content"], "saved")

    def test_incomplete_commit_is_not_replayed(self):
        """写到一半的摘要末行保留备份但不成为有效提交。"""
        self.populate()
        self.session.compact_now()
        committed = self.path.read_bytes()
        with self.path.open("ab") as target:
            target.write(b'{"seq":99,"type":"compaction/summary"')
        restored = self.reopen()
        self.assertEqual(restored.context.nodes(), self.session.context.nodes())
        self.assertEqual(self.path.read_bytes(), committed)
        self.assertTrue(list(self.path.parent.glob("*.incomplete.*.bak")))

    def test_external_message_mutation_cannot_invalidate_cache(self):
        """外部修改消息副本不会绕过 generation 影响配对。"""
        self.add({"role": "assistant", "tool_calls": [tool_call("a")]})
        cache = self.compact.ensure_balance_cache(self.session)
        self.session.context.build()[-1]["tool_calls"].clear()
        self.session.context.nodes()[-1].message["tool_calls"].clear()
        self.assertIs(cache, self.compact.ensure_balance_cache(self.session))
        self.assertEqual(cache.in_progress_tool_calls, 1)

    def test_manual_tool_directly_calls_compact_now(self):
        """LLM 工具直接执行公共压缩函数，未完成的整组调用保留在尾部。"""
        self.populate()
        compact = Compact(
            self.client, context_window=10000, max_output_tokens=200,
            threshold_ratio=0.9, retain_ratio=0.01,
        )
        self.session = Session(HistoryStore(self.path), ContextBuilder(), "unused", compact)
        executor = ToolExecutor()
        executor.register_tool("echo", "测试", {"type": "object", "properties": {}}, lambda: "z" * 600)
        self.client.normal_responses = [
            response(None, calls=[tool_call("compact", "compact_context"), tool_call("echo")], finish="tool_calls"),
            response("完成"),
        ]
        observed_pending = []
        self.client.on_summary = lambda: observed_pending.append(compact.ensure_balance_cache(self.session).in_progress_tool_calls)
        agent = GuideAgent(self.client, executor, self.session)
        self.assertEqual(executor._tools["compact_context"][1], self.session.compact_now)
        self.assertEqual(agent.ask("请压缩上下文"), "完成")
        self.assertEqual(observed_pending, [2])
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        summary = next(item for item in records if item["type"] == "compaction/summary")
        results = [item for item in records if item["type"] == "message" and item["message"]["role"] == "tool"]
        self.assertTrue(all(item["seq"] > summary["seq"] for item in results))
        self.assertEqual(json.loads(results[0]["message"]["content"])["status"], "compacted")
        self.assertEqual(summary["data"]["source"], "manual")
        self.assertEqual(compact.ensure_balance_cache(self.session).in_progress_tool_calls, 0)
        self.assertEqual(self.reopen().context.nodes(), self.session.context.nodes())

    def test_agent_auto_compacts_before_request(self):
        """Agent 在普通请求之前自动摘要，并发送已替换的消息。"""
        self.populate()
        self.client.normal_responses = [response("回答")]
        agent = GuideAgent(self.client, ToolExecutor(), self.session)
        self.assertEqual(agent.ask("继续"), "回答")
        self.assertIsNone(self.client.requests[0][1])
        self.assertIsNotNone(self.client.requests[1][1])
        self.assertTrue(any(item.get("content") == "摘要" for item in self.client.requests[1][0]))


if __name__ == "__main__":
    unittest.main()
