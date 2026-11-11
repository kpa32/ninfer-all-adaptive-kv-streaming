---
id: kv-stream-context-cache
type: architecture-design
title: KV 流式的上下文缓存：与 ninfer 基线契约的对齐设计
status: draft
parent: kv-streaming-design
references:
  - kv-streaming-verification
  - kv-streaming-measurements
tags:
  - kv-stream
  - context-cache
  - design
  - plan
---

# KV 流式的上下文缓存：与 ninfer 基线契约的对齐设计

**本文的地位**：把"流式 run 里跨请求复用上下文"作为**一个功能**做整体评估，而不是逐次修 bug。
历史与逐次实测在 `bench/results/kv-stream/ledger.md`；几何/换形缺陷在
`bench/results/kv-stream/layout-rename-defect.md`；接手指针在 `bench/results/kv-stream/handoff.md`。

**本文的立场**：基线（`f6928e00`）的上下文缓存契约已经完整、正确，并且**早已覆盖流式需要的全部情形**
（取消保留、长 prompt 锚点、turn 边界复用、计划式降级）。流式的任务**不是重新设计缓存**，而是让
"流式 KV 的物理承载"满足这份既有契约。**现状未满足，这就是四次崩溃的战略原因。**

---

## 1. 功能认知：缓存只影响速度，不影响接纳与结果

1. 有可复用前缀 → 快；没有 → 从头算。
2. 缓存条目作废了 → 丢掉、继续算。
3. **绝不因为缓存状态报错、拒绝请求或崩溃。**
4. **输出与缓存无关**：命中与否必须逐位相同（基线 §1 亦明确：不同数值路径不要求 logits 相同，
   但同一路径的命中与未命中必须一致——本项目的验收取更强的一条：逐位）。

代价（实测约 1.5 k tok/s prefill，命中 TTFT 百毫秒级）：

| 上下文 | 全量 prefill（估） | 命中 |
|---|---:|---:|
| 16 k | ≈ 11 s | ~0.1 s |
| 37 k | ≈ 25 s | ~0.1 s |
| 190 k | ≈ 2 min | ~0.1 s |

⇒ 判据是"**能复用的必须复用**"，任何"有疑问就重算"都等于删掉缓存。

---

## 2. 基线契约（`f6928e00`，权威文本 `docs/maintainer/resource-scheduling-and-context-cache.md`）

这一节是本次设计的地基，逐条给出出处。

### 2.1 复用是"完整 continuation"，不是"部分 cache hit"（§4.1）

> 某个 frontier 可以复用，当且仅当 Program 能证明：①存在该 frontier 的**完整 StateImage**；
> ②Main KV 满足 target 定义的 typed coverage；③selected backend 的 KV 与 fixed state 满足同一
> continuation；④token、position、Vision 和 mode identity 与 incoming prompt 精确一致。
> **只有 token match、KV bytes 或 page match 时，缺少的是完整 continuation，不是一次部分 cache hit。**

**含义（重要，纠正本文旧稿的一个错误方向）**：Qwen3.5 有不能无损回退的 recurrent state（GDN），
所以**不存在"任意最长前缀复用"**——复用点必须是**一条带完整 StateImage 的 checkpoint frontier**。
"同一个 prompt 再发一次"要命中，靠的不是截断到匹配长度，而是**在 prompt 末尾存在一个 checkpoint**
（§4.3 的 `TurnClosure`）。旧稿的"F2 最长前缀命中"违反这条，已废弃。

### 2.2 checkpoint 种类就是"可复用点"（§4.3）

| Kind | 语义 | 对本任务的作用 |
|---|---|---|
| `SessionEndpoint` | 已完成请求的最新可继续状态 | 续聊命中 |
| `TurnClosure` | assistant 可变后缀之前的稳定状态 | **同一 prompt 重发 / 追加 user 轮** 命中 |
| `ResponseReplay` | generation opener 之前的稳定状态 | 重生成 |
| `LongAnchor` | retention 选择的较早长上下文恢复点 | 长上下文下的回退点 |
| `SharedStablePrefix` | 多条历史共同的不可变前缀 | 跨会话共享 |

### 2.3 取消**已经**保留（§9.5）

> `ProgramImpl::abort` 先尝试 **salvage**：把 active state 作为 endpoint 发布在 prefill cursor
> （需要至少 1024 tokens；MTP 还需要 tail hidden state，即只有最后一个 chunk 之后才行）。
> salvage 不成立时，`retain_prefill_progress` 发布该 lane 已 capture 的 checkpoint（long anchor 或
> rewrite checkpoint）……以 Disposable retention 进入 catalog。
> `ContextCacheOptions::progress_anchor_tokens`（`--progress-anchor-tokens N`，默认 16384）让
> Frontend 在 prompt 内部按绝对 token 栅格提出 `PrivateLongAnchor` opportunities，**使没有 message
> boundary 的超长单条 prompt 被取消时也有 checkpoint 可保留**。

**这就是"取消不作废"的现成机制。** 流式路径没有用它，而是 `reset()` 掉整个 arena。

### 2.4 Terminal 只有两种，降级必须被计划并记录（§6.2、§9.4）

- Finish = 发布完整 immutable checkpoint；Discard = 释放整个 active continuation；
  **只有 Finish 无法形成合法保留终态时才允许 Discard**。
- 降级/驱逐是 `ResourcePlan` 的一部分，绑定**确切的 `CheckpointRef` 删除集合**
  （"同数量但不同 checkpoint 的替换会在任何 catalog mutation 前失败"）——**不允许静默消失**。

### 2.5 多 owner 模型（§4.2、§6.1）

> Private continuation 表示一条可继续演化的会话历史：
> `PrivateContinuation ├── exact target/token history ├── Main KV address space
> ├── optional backend KV address space └── one or more immutable checkpoints`

**每个 owner 有自己完整的 KV address space 和若干 checkpoint。** 缓存容量 = 能同时容纳多少个 owner。

### 2.6 两条我们踩过的判据

- **§2.2 owner edge ≠ capability**：active request 的保护是"持续到 terminal settlement 的逻辑 lease"，
  不是 catalog capability 的镜像。**把保留的上下文当成租约占住槽位**就是违反这条（缺陷 #11）。
- **§5.1 StateImage 是完整迁移单位**：流式把 State 留在 host 时，StateImage 的语义不能丢。

---

## 3. 流式路径相对基线的四处偏差（战略层面）

| # | 基线 | 流式现状 | 后果 |
|---|---|---|---|
| **D1** | 每个 owner 一个 address space，可并存 N 个 | **整个 Program 一个 `StreamedKVSpace`**，`stream_space_` 单一指针；流式 address 的 `page_count = 0`、entitlement = 0（"claims no pool page at all"） | 缓存只剩"一条世系"；第二个会话一开始，第一个的 KV 被覆盖 |
| **D2** | checkpoint 有独立 frontier 与 required coverage，彼此共享页 | arena 只有一个 `active_` 计数，页 = token 位置（线性），`enter_request(tokens)` 直接把 `active_` 设小 | 任何更短的请求都会**覆盖**更深的 checkpoint 所需页；不能并存两个 frontier |
| **D3** | 取消 → salvage / `retain_prefill_progress`，默认保留 | `deactivate`：`retained() ? retain() : reset()` ⇒ 无 endpoint 的取消**清空 arena** | 一份 16 k 上下文被一次取消销毁；下一次规划到它 ⇒ 声明与字节脱节 ⇒ 崩溃（缺陷 #14） |
| **D4** | 锚点/闭包按 `--progress-anchor-tokens` 发布（启动日志 `anchors N`） | serve 启动日志 `context cache | ... | anchors 0`，且流式不发布 TurnClosure | **同一 prompt 重发也 miss**（实测 `r2 reuse=0` ⇒ 多花 ≈11 s） |

**一句话**：流式把基线的**多 owner 缓存**压成了**单 arena + 线性映射**，于是基线的四条不变式
（complete continuation、checkpoint 独立 frontier、取消保留、计划式降级）全部无处落地。
四次崩溃不是四个 bug，是**这四个偏差的必然产物**。

---

## 4. 缺陷编年（同族，供验收对照）

| # | 现象 | 根因（映射到 §3） |
|---|---|---|
| 11 | `KV activation reservation is stale` → 进程 abort | D1 + 把保留当租约（违反 §2.6） |
| 12 | `context cache exhausted`（429） | D1：唯一私有槽被保留占用，规划无解 |
| 13 | `streamed KV frontier is beyond the space's context` → worker crash（无取消） | D2：规划期不校验空间覆盖 |
| 14 | 同上（**取消之后**） | D3 + D2：取消清空 arena，目录条目仍声称有页 |

---

## 5. 设计目标：让 arena 满足基线契约（不是重新设计缓存）

### 5.0 总原则：流式只是**存储层**，不是缓存策略

产品需求只有一条：**上下文更长，其余体验与原版 ninfer 完全一致。**

它在代码里有一个可检验的形式：**缓存决策层不允许知道自己在流式。**
判据：`ResourceManager`、`Program`、`capture/commit` 里**不出现** `streaming()` / `stream_space_` /
`stream_can_serve` 之类的分支；它们看到的是与驻留模式**同一套接口**——
“这个 owner 的地址空间有哪些页、哪些 frontier 有完整 checkpoint、需要多少 coverage”。

由此得到的架构要求：流式 arena 必须**实现同一个地址空间/页池合同**，只是字节住在 pinned host 而不是
device（设备侧是移动窗口）。它能容纳多少个 owner、缓存哪些 checkpoint、取消时保留什么，
**全部由既有的容量公式与既有策略决定**，与驻留模式逐条相同。

**这条判据也解释了今天的局面**：现状把流式泄漏到了准入（`stream_can_serve`）、规划
（`inspect_lane`）、截断、`deactivate` 四处，等于在缓存策略层插入了“我在流式”的分支——
每插一处就多一簇 bug。修复方向是**把这些分支删掉**，而不是继续在分支里加校验。

> 一句话：**驻留模式的缓存行为是本设计的唯一规格；流式只许换“字节住在哪”，不许换“缓存怎么决策”。**

### 5.1 目标不变量（全部来自基线，不新增语义）

- **N1｜一个 owner 一份地址空间**：arena 内为每个可缓存 owner 保留 **contiguous region**（每层内一段
  连续页），owner 的 checkpoint 只引用自己 region 内的页；**没有第二个 owner 能覆盖它**。
- **N2｜checkpoint 是带 StateImage 的 frontier**，允许任意合法 frontier（endpoint / turn closure /
  long anchor），**不要求与当前 `active_` 相同**；agent 的 required coverage 必须被自己 region 覆盖。
- **N3｜取消按基线处理**：salvage（≥1024 tokens 且 MTP tail 可用）否则 `retain_prefill_progress`；
  `--progress-anchor-tokens` 的锚点机会对流式同样生效。**不再 reset。**
- **N4｜降级/驱逐是计划步骤**：删除某 owner 的 region 必须作为 `ResourcePlan` 的显式结果记录
  （沿用基线 §9.4 的 `CheckpointRef` 删除集合），**不允许静默覆盖**。
- **N5｜声明在使用时以 arena 为准**（护栏）：任何前沿在使用那一刻向 arena 取；不够长则按能服务的长
  服务；为 0 才冷启动。**永不抛出。**

### 5.2 可行性：region 化在现有结构里是局部改动（关键）

现状的线性映射是**唯一**把多 owner 挡在外面的东西：

- `StreamedKVSpace` 已持有 `layers()`、`pages_per_layer()`、`split_.run_pages`、`window_start(layer)`；
- arena 是"每层一段 stride、段内连续页"的布局，device window 是段内连续 run（`layer_copy`/`runs_per_layer`）；
- 因此给 owner 一个 `region_base(layer)`（每层内的页偏移），把 pointer/`window_start` 的算术改成
  `region_base(layer) + offset` 即可；**run 的连续性在 region 内仍然成立**，注意力/拷贝路径的形状不变。

⇒ **不需要 page table，也不需要复制页**；代价是**容量**：`per_owner_pages = 总页 / N`，
即 N 个 owner 各能用的上下文长度按 1/N 缩。这是基线的同一本账（每个 private continuation 都占自己的页），
不是流式引入的退化。

### 5.3 容量账（必须由所有者拍板）

serve 启动实测（27B IQ3_S / rk8v4 / `--max-context 192512` / 16 GB 卡）：

```
capacity | KV 192,512 tokens, rk8v4, auto | pages 36,567/48,128 | runtime 4.07 GiB | free 51.5 MiB
context cache | 1 active + 0 cached device states | host 1 states, 0 B KV | private 1 | shared 0 | anchors 0
```

- 192 K 上下文 ≈ 4 GiB arena ⇒ **每个 owner ≈ 4 GiB 宿主内存**（N=2 ⇒ 8 GiB，N=3 ⇒ 12 GiB）。
- 若不接受"每 owner 4 GiB"，替代是把 owner 数降到 1 并**只保证单会话的完整契约**（N3/N4/N5 仍然要做），
  这属于**明确的范围声明**，不是技术退化。

### 5.4 与基线代码的衔接（复用，不重写）

| 基线机制 | 流式如何满足 |
|---|---|
| `ProgramImpl::abort` salvage / `retain_prefill_progress`（§9.5） | 流式 `deactivate` 改走同一条路：保留 arena 的 region，发布 checkpoint；**删除 `reset()` 分支** |
| `--progress-anchor-tokens`（§9.5） | 让 Frontend 的锚点机会对流式生效（`anchors` 计数应 > 0） |
| `TurnClosure`（§4.3） | 流式 finish 时按同一规则发布 prompt-末尾闭包（同一 prompt 重发 ⇒ 命中） |
| `ResourcePlan` 的删除集合（§9.4） | arena region 释放进入同一份计划；`CheckpointRef` 逐项核对 |
| `inspect` 只读（§2.2） | 不变式保留：驱逐只发生在放置循环里 |

---

## 6. 实施顺序（不可颠倒）

| 步 | 内容 | 出口判据 |
|---|---|---|
| 1 | **N3 取消不再清空**：流式 `deactivate` 走 salvage/retain；`reset()` 只在"从未发布且无 checkpoint"时 | C18（取消后继续命中）+ 进程存活 |
| 2 | **N4 降级计划化**：region 释放进入 `ResourcePlan`；`stream_can_serve` 降级为断言 | 计划审计：无静默覆盖 |
| 3 | **N2 TurnClosure/锚点**：流式 finish 发布与基线同规则的 checkpoint；`anchors > 0` | C17（同一 prompt 重发命中）+ `anchors` 计数 |
| 4 | **N1 region 化**：owner 独占 region，多 owner 并存 | A→B→A 两会话都命中；`private N` 与 region 数一致 |
| 5 | **N5 使用时刻钳位**（护栏，若前面全绿则触发 0 次） | 触发计数 = 0 |
| 6 | 删除历史补丁里的"丢弃"逻辑；文档收口（verification C17/C18、measurements、handoff） | 全绿 |

**第 4 步是“多会话复用”的唯一入口**；它不在“做/不做”之间选，而是由 §5.0 的总原则强制：
原版能缓存几个，流式就必须能缓存几个（本机驻留实测 `private 1`，所以流式也必须至少 1，
预算允许就与驻留模式同公式增加）。

---
## 7. 验收矩阵

| 检查 | 判据 |
|---|---|
| 逐位等价 | `check --suite equiv` 4/4，ids 与 logprobs 逐位 |
| C13 跨请求复用 | `reused == retained_frontier`，与冷跑逐位 |
| C14 几何不变 | 两种池形状 ids/logprobs 逐位相同 |
| C15 保留即缓存 | 任意下一请求都被接纳 |
| **C17 假未命中（新）** | 同一 prompt 连续两次 ⇒ 第二次 `reused > 0`；`anchors > 0` |
| **C18 取消不作废（新）** | 长会话 → 续聊 → 客户端断开 → 再续聊：`reused > 0`、进程存活 |
| serve 端到端 | `A→B→A`、`A→A2` 全 200；`worker crash`/`terminate`/`FATAL engine` = 0 |
| 性能 | 251 K `round_ms_median` ≤ 41.4 ms +1%；`upload_bytes` 不显著变化 |
| 冲突检查 | `[stream] request starts` 中 `space_pages < base` 的次数 = 0 |

诊断开关：`NINFER_STREAM_DEBUG=1`、`NINFER_ADMISSION_DEBUG=1`、`NINFER_STREAM_VERIFY=1`。

---

## 8. 明确不做

- **不重写、不硬分叉**：难点是簿记一致性；基线的机制已经对，缺的是流式承载满足它。
- **不做"任意最长前缀复用"**：违反 §2.1（GDN 不可任意回退）。复用点必须是带 StateImage 的 checkpoint。
- **不把冷启动当修复**：只用于 `space_pages == 0`。
- **不在只读路径改状态**，**不在一次放置内重试**，**不并存多条同 arena 世系**（region 化之前）。

---

## 9. 由总原则推出的答案（不是选择题）

上一版把三件事写成“需要所有者拍板”，这是本文的错误：**它们不是产品选择，而是“流式只换存储”的必然结论。**

| 问题 | 与“不破坏 ninfer 体验”的关系 | 推出的答案 |
|---|---|---|
| 允许几个 owner？ | 原版能缓存几个，流式就必须能缓存几个；否则“第二个会话变冷跑”本身就是体验变化 | **与同一内存预算下驻留模式的 `private N` 相同**（本机实测 `private 1`；想要更多是**加宿主内存**的配置问题，与驻留模式加内存等价），不是流式侧的新参数 |
| 取消时是否发布 checkpoint？ | 原版会保留（§2.3），不保留就是体验变差 | **照基线**：salvage 成立就发布，否则 `retain_prefill_progress`；条件完全相同，不新增开关 |
| 251 K 长会话 与 多会话切换 是否二选一？ | 原版两者都能做，受内存限制 | **不二选一**：同一策略在容量允许范围内都成立；唯一区别是限额来自**宿主内存**而不是显存 |

**唯一真正需要所有者决定的是资源额度**：给流式 arena 多少宿主内存（`--kv-capacity` 一族）。
这与“给驻留模式多少显存/宿主 KV”是同一类配置，不改变任何缓存语义。

### 9.1 由此确定的实施顺序

与 §6 相同，但判据改为“上层无流式分支”：

1. `deactivate` 走基线的 salvage/retain（删除 `reset()` 分支、删除 §4.1 的丢弃逻辑）；
2. region 化 owner（每个 owner 一份地址空间），使 `private N` 与驻留模式一致；
3. checkpoint（endpoint / TurnClosure / 锚点）按基线发布，`anchors` 计数 > 0；
4. 把 `stream_can_serve`、`inspect_lane` 的流式分支、截断钳位**降级为断言并删除**；
5. 验收：C13–C18 + equiv + serve 两会话 + 251 K 不退化。
