# Trip Agent

Python 3.10+。中文导游问答 Agent，支持高德工具、多轮对话和可重放的上下文压缩。

## 配置

在项目 `.env` 中填写 `DEEPSEEK_API_KEY` 和高德地图的 `AMAP_API_KEY`。
高德 Key 需选择 Web 服务类型，并具有对应搜索及路线服务的权限。
`.env.example` 为可提交的配置模板。系统环境变量优先于 `.env`；
`DEEPSEEK_TIMEOUT` 和 `AMAP_TIMEOUT` 单位均为秒。

压缩配置为必填，不内置模型容量或比例默认值：

| 配置项 | 含义 |
| --- | --- |
| `DEEPSEEK_CONTEXT_WINDOW` | 当前 `DEEPSEEK_MODEL` 的 Context Window，即 W |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | 当前模型支持的最大输出容量，同时用作 O 和 B |
| `COMPACT_THRESHOLD_RATIO` | 触发比例 r，介于 0 和 1 之间 |
| `COMPACT_RETAIN_RATIO` | 近期保留比例 k，介于 0 和 r 之间 |

模型容量须按实际模型配置填写；空值、无效容量、非正 T 或不满足 `0<K<T` 都会明确报错。
策略配置采用 DSH 的 `thresholdRatio`、`retainRatio` 形式，
预算按本项目要求计算为 `T=floor(min(W*r,W-O-B))`、`K=floor((W-O)*k)`，`O=B=maxOutputTokens`。
参考 [DSH 压缩配置实现](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/compaction/compaction-basic/src/config.ts)。

创建项目虚拟环境并安装依赖（Ubuntu / Bash）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

VS Code 的默认 Python 解释器已配置为 `.venv/bin/python`。若仍提示无法解析
`openai.types.chat`，通过 `Python: Select Interpreter` 选择该解释器。
终端中可用 `source .venv/bin/activate` 激活环境后再运行 Python。

默认模型为 `deepseek-flash`，可通过 `DEEPSEEK_MODEL` 修改。
接口格式参考 [DeepSeek 文档](https://api-docs.deepseek.com/)，
工具定义参考 [Tool Calls 文档](https://api-docs.deepseek.com/guides/tool_calls/)。

## 使用

填写 `.env` 并安装依赖后，启动命令行问答：

```bash
.venv/bin/python main.py
```

可以提问“杭州有哪些适合亲子游的景点？”，再追问“帮我查一下西湖附近的餐厅”。
输入 `/compact` 立即请求压缩，`/clear` 清空历史，`exit` 或 `quit` 退出。
也可以要求导游压缩上下文，由模型调用 `compact_context` 工具。

Agent 每轮自动询问模型、执行模型请求的所有工具、回传结果，直到生成回复。
每个问题最多请求模型 6 次，可通过 `GuideAgent(..., max_steps=...)` 调整。
工具和 API 错误直接抛出；达到次数上限仍未完成时也会抛出异常，进度保留。

历史保存到项目目录下的 `data/history.jsonl`，每行一个事件：

- `seq`：整个文件内从 0 递增，消息和边界事件均占用一个编号，重启后接续。
- `seq=0`：固定为 LLM 的系统身份提示词，使用 `turn=0`；此提示也用于后续请求。
- `turn`：每次用户提问开启一轮，从 1 递增。
- `step`：一次 LLM 请求及其产生的工具调用为一步，每轮从 1 递增。
- `type`：消息、turn/step 边界事件，以及包含 Replace 操作的 `compaction/summary`。

一轮不需要工具的问答记录示例：

```jsonl
{"type":"message","turn":0,"message":{"role":"system","content":"你是一名中文旅游导游……"},"seq":0}
{"type":"turn/start","turn":1,"seq":1}
{"type":"message","turn":1,"message":{"role":"user","content":"你好"},"seq":2}
{"type":"step/start","turn":1,"step":1,"seq":3}
{"type":"message","turn":1,"step":1,"message":{"role":"assistant","content":"你好，有什么旅游问题？"},"seq":4}
{"type":"step/end","turn":1,"step":1,"seq":5}
{"type":"turn/end","turn":1,"reason":"completed","seq":6}
```

`turn/end.reason` 为 `TurnEndReason`：`completed`、`error` 或 `max_steps`。
发生异常时记录当前 `step/end` 和 `turn/end`，异常继续向调用方传播。
所有事件立即追加并执行 flush、fsync，不等待整轮问答完成。
重启时依次重放消息 Append 和摘要 Replace，仅将当前有效消息提供给 LLM。
`HistoryStore.load(system_prompt)` 初始化日志和编号，`append()` 追加事件，
`clear()` 清空日志；Session 清空后会重新写入 seq=0 的系统消息并重置 turn。

`Session(history, context, system_prompt, compact)` 持有
`HistoryStore`、`ContextBuilder`、`Tokens` 和 `Compact`，加载消息上下文并恢复 token 缓存，
维护最近的 turn 编号，提供：

- `append_history(events)`：封装 `HistoryStore.append()`，写入事件并更新 turn。
- `append_context(message, usage=None, seq=...)`：封装 `ContextBuilder.append()`，
  向内存追加消息并更新 token 缓存；模型响应可传入服务端 usage。
- `get_tools()`：通过当前 `tools_provider` 统一获取工具 Schema。
- `count_context_tokens(messages=None)`：统计当前或候选上下文及工具 Schema 的成本。
- `measure_context_tokens(messages=None)`：统计实际上下文并更新 token 缓存。
- `compact_if_needed()`：向所属 Compact 传入当前会话，自动检查并压缩。
- `compact_now(source="manual")`：工具和命令行入口，转发给所属 Compact。
- `clear()`：清空历史、上下文、token 及压缩缓存，再重新初始化系统消息。

`ContextBuilder` 使用 `SurfaceNode(seq, message)` 维护当前有效消息，支持
`append(message, seq)`、`build()`、`nodes()`、`locate_range()`、`replace()` 和 `clear()`。
公开获取的消息为独立副本，不能绕过 Replace 修改工具配对。
Agent 先用 `Session.append_history()` 获取实际 seq，再用 `append_context()` 追加消息。
普通 Append 不改变 `replace_generation`；Replace 和清空操作使结构版本递增。
turn、step 等日志事件不进入模型请求。

消息 token 由 `tokens.py` 的 `Tokens.count(message, usage=None)` 计算，
写入新增 `message` 事件的 `tokens` 字段：

- LLM 返回 `usage` 时，使用 `completion_tokens` 作为该 assistant 消息的 token 数，
  完整 `usage` 同时保存在事件顶层，便于查看请求输入、输出和总用量。
- 系统、用户、工具消息，以及没有 `usage` 的模型消息，按原始消息的紧凑 JSON
  UTF-8 字节数除以 4 向上取整估算：`(byte_count + 3) // 4`。
- 估算包含消息自身的角色、正文、推理内容和工具调用等字段，
  不计事件的 `turn`、`step`、`seq` 等字段。边界事件不计 token。

消息级 `tokens` 之和不代表 API 请求累计用量：请求会重复发送已有上下文。
统计实际请求用量时应累计每次响应的 `usage.total_tokens`，避免与消息估算相加。
已有历史保持原样，只有新增记录写入 token 数据。

`session.tokens.total_tokens` 缓存当前上下文用量，按以下规则实时更新：

- 追加系统、用户或 tool result 消息时，累加四字节算法估算的消息 token 数。
- 收到 LLM 响应时，使用 `usage.total_tokens` 覆盖缓存，校准已有上下文和本次输出。
  tool-call 已包含在服务端输出用量中，不再重复累加；没有 usage 时估算完整模型消息。
- 每个工具结果追加后立即更新，不等待下一次模型响应。
- 重启时按历史消息和 usage 重建缓存；清空会话时先清零，再计入系统提示词。

每次准备 LLM 请求时，`Session.measure_context_tokens()` 会重新统计
当前实际请求，包含系统提示词、有效消息、摘要、工具 Schema 及其他注入内容。
被 Replace 的旧消息不再计入。可通过 `Tokens(counter=...)` 注入 Tokenizer 计数接口，
默认仍为四字节估算；Provider usage 用于响应后的校准，不累加为当前上下文长度。
`Session` 使用所属 Compact 的 T 初始化 `tokens.compression_limit`。

## Compact 接口及执行流程

`Compact(client, context_window=..., max_output_tokens=..., threshold_ratio=..., retain_ratio=...)`
使用现有同步 Client，由 Session 持有。Compact 不保存 Session，执行时临时接收 session 参数。
配置统一放在 `.env`，由 `main.py` 的 `load_compact_settings()`
读取并校验后作为关键字参数传入：

- 初始化时直接计算并保存 `threshold`（T）、`retain_tokens`（K）。
- `compact_if_needed(session)`：通过 Session 计量实际请求，达到 T 后启动压缩。
- `compact_now(session, source="manual")`：自动、手动共用的压缩核心，由 Session 转发调用。
- `ensure_balance_cache(session)`：按需统计工具调用与结果的数量平衡，缓存每个切分位置。
- `select_compactable_range(session)`：反向累加至 K，再向前移动至最近安全边界。
- `summarize()`、`commit_compaction()`：独立生成摘要，单条提交并原位替换。

结果为 `CompactResult(status, reason, before_tokens, after_tokens, summary_seq)`：
`compacted` 表示提交成功，`skipped` 表示低于阈值或无安全范围。
Compact 不捕获异常、不主动抛错，也不转换 `failed` 状态；底层异常直接传播。
配置校验统一在启动入口完成。

Agent 通过 `session.compact_if_needed()` 检查自动压缩，达到阈值时调用所属 Compact 的核心函数。
用户主动要求压缩时，LLM 调用已注册的 `compact_context`，其 Handler 为
`session.compact_now()`，转发给同一个 Compact 核心函数。工具执行时立即压缩已闭合的历史前缀，
当前未闭合工具调用所在的整组尾部保留，待执行完成后再追加结果。
工具正常返回实际 `compacted` 或 `skipped` 状态，不设置调度标记。
Session 通过 `get_tools()` 统一获取工具 Schema，自动、手动和替换后的计量均由 Session 完成，
Compact 系列函数不再接收 `tools` 参数。Agent 发送请求时也使用 `session.get_tools()`。
摘要生成调用不提供业务工具，也不递归调用 Agent。
Compact 直接使用摘要正文，记录替换前后的 Token 成本，不额外校验截断或压缩收益。
持久化事件和 Replace 范围的检查由现有 Session、ContextBuilder 负责。

`BalanceCache` 保存 generation、切分位置平衡数组、seq 到位置的映射和未完成调用数，
仅按调用与结果数量判断边界，不校验调用 ID。首次或 generation 改变时重建，Append 时仅扫描新增节点，
无变化时复用。工具执行日志不重复计入模型可见调用。

## 摘要提交与重启恢复

压缩保持原始 History 追加写入。成功时只新增一条 `compaction/summary`，
包含摘要自身的新 seq、模型可见消息、原节点 `shadowedSeqs`、边界 `shadowedRange`、
来源、UTC 时间、模型与 Token 元数据，以及同条记录中的 `surfaceOp.replace`。
摘要在当前范围的位置 Replace，seq0 和近期消息保留；来源 seq 不要求数值连续。

例如 `[0,1,2,3,4]` 压缩 `1,2` 后为 `[0,5,3,4]`；再次压缩 `5,3` 后为 `[0,6,4]`。
原始节点及旧摘要均可通过 `HistoryStore.get(seq)` 查询，来源链不会丢失。

重启时 `Session._replay()` 按日志顺序逐条处理 Append 和 Replace，验证目标存在、
范围连续且顺序正确、来源事件可查询、seq0 未受影响、Replace 未重复应用。
摘要只作为同条提交中的一个有效节点插入一次，BalanceCache 无需持久化。

摘要生成失败或生成期间退出不会改变原消息；完整提交后、内存 Replace 前退出，
重启会完成相同投影。记录以换行作为提交边界，末尾未完整写入的一行会先备份为
`history.jsonl.incomplete.<时间戳>.bak` 再移除，不视为有效事件。
完整行损坏、seq 乱序或 Replace 来源异常会报错，不静默忽略。
提交持久化失败，或日志已提交但内存投影失败时，当前实例停止后续写入，
要求重启按实际日志重建，避免使用过期 seq 或旧投影继续记录。

当前不提供中断任务的续跑功能。如果中断留下不完整的工具调用消息，
可先用 `/clear` 清空历史再开始新问答。上述恢复针对历史投影及压缩提交，不续跑业务工具。
历史存储只支持当前事件格式，不提供旧消息记录的导入或转换。
当前为单会话，模型只接收压缩后的有效上下文，History 保留全部原始记录。

组件划分：

- `client.py`：基础 LLM 客户端 `Trip_Client`，负责询问与响应解析。
- `tool_executor.py`：`ToolExecutor`，负责工具注册和调用。
- `history.py`：`HistoryStore` 和 `TurnEndReason`，负责事件编号、追加、读取与清空。
- `tokens.py`：`Tokens`，统计单条消息、缓存上下文总量并提供压缩阈值判断。
- `context_builder.py`：`ContextBuilder`，存储消息并提供给模型。
- `compact.py`：`Compact`、`BalanceCache`、`CompactRange` 和 `CompactResult`，负责安全摘要压缩。
- `session.py`：`Session`，持有历史、上下文和 Compact，封装追加、压缩及会话初始化、清空。
- `agent.py`：`GuideAgent`，负责导游提示、工具调用循环及 turn、step 的执行边界。
- `main.py`：命令行入口及环境配置的集中读取、校验。
- `tests/test_compact.py`：离线验收测试，覆盖阈值、配对、缓存、提交及重放。

运行离线测试，不发送真实 API 请求：

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

## 单独调用 Client 与工具

```python
import json

from client import Trip_Client
from tool_executor import ToolExecutor

executor = ToolExecutor()
tools = executor.get_all_tools()

with Trip_Client() as client:
    messages = [{"role": "user", "content": "帮我查一下杭州的博物馆和西湖附近的酒店。"}]
    response = client.ask(messages, tools=tools)
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    result = client.parse_response(response)
    print(result.content)

    # 一次响应可以包含多个工具调用，全部执行并回传后再询问模型。
    if result.tool_calls:
        for call in result.tool_calls:
            data = executor.call_tool(call.name, call.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(data, ensure_ascii=False),
            })
        next_response = client.ask(messages, tools=tools)
        messages.append(next_response.choices[0].message.model_dump(exclude_none=True))
        next_result = client.parse_response(next_response)
        print(next_result.content)
```

此底层接口示例只演示一次工具调用批次；完整循环由 `GuideAgent` 实现。
例如先搜索地点取得坐标，再通过周边搜索或路线查询补充信息。

## 工具执行器

`ToolExecutor` 统一管理工具定义和实际 Python 函数：

- `register_tool(name, description, parameters, handler)`：注册描述、JSON Schema
  和同步实现；`handler` 参数名需与 Schema 字段一致，结果应可序列化为 JSON。
  同名注册抛出 `ValueError`。
- `get_all_tools()`：获取所有 OpenAI 兼容的工具描述，不包含 Python 函数。
- `call_tool(name, arguments)`：按名称执行工具并返回结果，未知名称抛出 `KeyError`，
  工具异常直接向调用方传播。JSON Schema 用于模型侧描述，不进行本地 Schema 校验。

内置高德工具如下，均返回高德原始 JSON：

| 工具名 | 作用 | 主要参数 |
| --- | --- | --- |
| `amap_search_places` | 按城市搜索景点、餐厅、酒店等，获取地址和坐标 | `keywords`、`city` |
| `amap_search_nearby` | 搜索地点附近的配套设施，按距离排序 | `location`、`keywords`、`radius` |
| `amap_plan_route` | 查询步行、驾车、公交路线及预计距离、耗时 | `origin`、`destination`、`mode` |

搜索支持 `page`、`page_size` 分页。坐标均为高德 GCJ-02 坐标，格式为
`经度,纬度`，应从高德搜索结果取得。公交模式 `mode="transit"` 需要 `city`，
跨城公交还需 `destination_city`。工具不提供景点实时库存或预订能力。

高德接口参考 [POI 搜索文档](https://lbs.amap.com/api/webservice/guide/api-advanced/search)
及[路线查询文档](https://lbs.amap.com/api/webservice/guide/api/direction)。

新增其他工具时，只需使用 `executor.register_tool()` 注册，无需修改 Client。
未填写高德 Key 时仍可注册和使用其他工具，实际调用高德工具时才检查 Key。

## LLM 响应

`Trip_Client.ask(messages, tools=None)` 接收外部工具列表，省略时为普通对话。

`ask()` 发送一次请求，直接返回 SDK 原始 `ChatCompletion` 响应，不自动解析。
`GuideAgent.ask()` 先将 `response.choices[0].message.model_dump(exclude_none=True)`
通过 `Session.append_history()` 和 `Session.append_context()`
分别追加到文件历史和内存消息列表，
再调用 `parse_response(response)` 得到 `Message`：

- `role`：消息角色，LLM 返回的消息通常为 `assistant`。
- `content`：回复文本，工具调用时可能为空。
- `tool_calls`：工具调用列表，包含 `id`、`name` 和解析为字典的 `arguments`。
- `tool_call_id`：工具结果对应的调用标识，非工具消息默认为 `None`。
- `metadata`：包含 `reasoning_content`（推理内容）、`finish_reason`（结束原因）
  和 `usage`（token 使用信息）。

历史和后续请求直接使用 SDK 原始消息字典，保留原始工具参数与推理内容。
`Message` 用于解析后判断是否执行工具及读取回复正文。

Trip_Client 不执行工具、不维护对话历史、不自动重试。Agent 记录原始消息后，
执行工具并追加带对应 `tool_call_id` 的
`role="tool"` 消息，随后再次调用 `ask()`。`metadata["finish_reason"]` 为
`length` 表示输出被截断，
Agent 应检查结束原因；无效的工具参数 JSON 会直接抛出解析异常。
网络和 API 错误保留 SDK 原始异常，由调用方处理。

如已有 SDK `ChatCompletion` 对象，可直接调用 `Trip_Client.parse_response(response)`。
