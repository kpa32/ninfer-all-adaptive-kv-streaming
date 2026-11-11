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

### 3.5 影响面：所有触达 KV 地址空间的功能（逐个核过）

“不破坏原版体验”不能只看前缀命中，必须把所有**碰 KV 地址空间**的功能列全。

| 功能 | 驻留模式 | 流式现状 | 要求 |
|---|---|---|---|
| 前缀复用（endpoint / TurnClosure / 锚点） | 有 | **无 TurnClosure、`anchors 0`** | 照基线发布 |
| 多会话保留（`private N`） | `private 1` | 只有 1 个世系 | 同基线同预算 |
| 取消保留（salvage / retain_progress） | 有 | **`reset()` 整块清空** | 照基线 |
| 淘汰/降级 | 计划 + 记录删除集合 | 静默覆盖 | 进入 `ResourcePlan` |
| **磁盘会话保存/恢复（`--slot-save-path`）** | 有：`POST /slots/{id}?action=save\|restore\|erase` | **保存抛错**：`save_continuation` 用 `mapped_pages(address)`，而流式 address 的 `page_count == 0` ⇒ `retained session Text KV coverage is inconsistent` | arena 实现页导出/导入 |
| **淘汰时自动 spill 到槽文件** | 有：`spill_catalog_slot`（best-effort） | **静默失效**：上面那个抛错被 `catch (...) {}` 吞掉 ⇒ 绑定槽文件的会话被淘汰时一声不响地丢上下文 | 同一次修复 |
| KV 容量/缓存报告（`capacity`、`context cache` 行） | 真实页数/字节 | `0 B KV`（字节在 arena，未计入） | 报告真实占用 |
| Hybrid prefix cache | 有 | 显式拒绝（`--kv-stream` 与 hybrid 不共存） | 已是明确范围，保持 |
| 会话恢复后的冷启动续聊 | 有 | 无（先要保存能成功） | C19 |

**结论**：磁盘会话不是“另一个功能”，而是**同一个地址空间契约的另一半**——
“页能被寻址”包含“页能被拷出/拷回”。漏掉它，流式在“重启后继续对话”这个最基本的使用方式上就是不可用的。

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

### 5.5 磁盘会话：arena 必须能导出/导入页（新增阻碍）

- 保存侧：`session_snapshot.cpp` 计算 `text_pages = kv_pages_for_frontier(text_committed)`，再要求
  `text_pages <= text_kv_addresses->mapped_pages(text_address)`；后者返回 `address.page_count`，
  流式分支把它固定为 0 ⇒ **恒为假** ⇒ 抛错。
- 自动 spill：`engine_core.h::spill_catalog_slot` 用 `catch (...) {}` 包裹 ⇒ **错误被吞**，
  且注释明确“a failed spill never blocks the eviction”。所以流式下槽文件会话被淘汰 = **静默丢失**。
- 要求（与 §5.0 同源）：arena 必须回答与页池相同的四个问题——
  **深度**（这份地址空间现在多少页）、**拷出**（按页拷到快照字节）、**拷入**（恢复时写入 region）、
  **释放**（淘汰时归还 region）。返回字节的布局继续用 `plan_host_kv_page_layout`，快照格式不变。
- 容量事实：192 K 上下文 ≈ 4.07 GiB，所以一个 192 K 的槽文件也约 4 GiB（用户须知）。

### 5.6 为什么会变成“两套存储”，以及如何并回一套

先澄清一个常被误会的事实：**字节格式没有变。** 两边都是 64 token 一页、同样的
`plan_host_kv_page_layout`、同样的量化布局；快照格式也是同一个。变的是**页的身份、所有权与寻址**：

| 维度 | 基线 | 流式现状 |
|---|---|---|
| 页身份 | 逻辑页 → `membership` → 任意物理页 | **逻辑页号 = 物理槽号 = token 位置**（恒等映射） |
| 所有权 | 每个地址空间持有自己的页（`page_count`） | 地址空间 `page_count = 0`，arena 拥有全部 |
| 设备寻址 | 页表间接寻址 | 算术窗口（`active_` + split → 连续 run） |
| 宿主层 | 本来就有：`HostKVPageLayout`、页级 promotion/demotion、`copy_address_pages` | 整个上下文就是一块宿主 arena |

**关键事实：基线本来就有“KV 放在宿主内存、按需搬进设备”的能力。** 所以流式本可以做成
**既有页池的一种放置策略**（字节住在宿主，设备持一段有界窗口），而不需要另造一套存储模型。
现实里流式绕过了页池：`flat_slots` + 恒等映射 + `page_count = 0`。

**代价就是这个“第二套存储”**：凡是依赖“页”的功能——前缀复用、保留、取消、淘汰、磁盘会话——
在流式下都失去了可依托的对象，只能逐个打特例。这就是“和 ninfer 对抗”的真实含义：
不是语义冲突，而是**多养了一套不完全的存储**。

**并回一套的形态（即 §5.1 的 region 化，换个说法）**：

- 页仍然由 `LogicalKVPageStore` 拥有，地址空间仍然持有自己的页与 frontier；
- 这些页的**权威副本在宿主**（已经是基线能力），设备只持一段**有界窗口**；
- 窗口取该 owner 的**连续 run**（region 内连续，所以拷贝仍是多 MB 大块，不退化为 64 token 散拷）；
- 页表、引用计数、COW、checkpoint、快照、spill **全部继续由基线代码负责**。

**收益**：§3.5 那张 11 项影响表**不需要新分支就全部成立**（包括磁盘会话与自动 spill）。
**代价与风险**：流式路径要重接页表/引用计数（删掉恒等映射与 `page_count = 0` 特例），
且窗口不能跨两个地址空间（本来也不跨）；251 K 的拷贝效率必须**重测**（同一次 campaign 对比）。

### 5.7 规划层的教训（写入本文，供以后的特性评审用）

新增一个“超出显存的上下文”能力时，正确的问题是：

> **这个能力是既有存储层的一种放置策略，还是一个新后端？**

- 若是**放置策略**：所有权、页身份、checkpoint、快照全部沿用，新代码只负责“字节在哪、什么时候搬”。
- 若是**新后端**：必须同时实现同一套可寻址合同（深度/拷出/拷入/释放），否则上层每用一个功能就多一簇特例。

本次规划选了后者却没支付后者的账，于是四次崩溃、四次补丁、一个坏的磁盘路径。这是**规划错误**，
不是实现不力。

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
| **C19 磁盘会话（新）** | 流式 + `--slot-save-path`：save → erase → restore → 续聊命中且与不中断逐位相同；淘汰 spill 真实落盘（文件存在且非空） |
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
| 磁盘会话要不要支持？ | 原版支持（`--slot-save-path` + 淘汰 spill），是“重启后继续对话”的唯一手段 | **必须支持**，且与流式组合正是本项目的目标场景（16 GB 卡 + 190 K 上下文 + 跨重启） |

**唯一真正需要所有者决定的是资源额度**：给流式 arena 多少宿主内存（`--kv-capacity` 一族）。
这与“给驻留模式多少显存/宿主 KV”是同一类配置，不改变任何缓存语义。

### 9.1 由此确定的实施顺序

与 §6 相同，但判据改为“上层无流式分支”：

1. `deactivate` 走基线的 salvage/retain（删除 `reset()` 分支、删除 §4.1 的丢弃逻辑）；
2. region 化 owner（每个 owner 一份地址空间），使 `private N` 与驻留模式一致；
3. checkpoint（endpoint / TurnClosure / 锚点）按基线发布，`anchors` 计数 > 0；
4. 把 `stream_can_serve`、`inspect_lane` 的流式分支、截断钳位**降级为断言并删除**；
5. arena 实现页导出/导入（`save_continuation` / `restore_continuation` / spill）⇒ C19；
6. 验收：C13–C19 + equiv + serve 两会话 + 251 K 不退化。

---

## 10. 计划重审（读完代码后的修正，取代 §6 与 §9）

§6/§9 是在没有读 `startup.cpp` 的守卫之前写的，**范围判断错了**。以下是按代码事实重写的版本。

### 10.1 事实：`--kv-stream` 是一个被显式限制的形状（`startup.cpp:1380-1412`）

```cpp
// --kv-stream 拒绝：hybrid prefix cache
// --kv-stream 拒绝：host KV tier / disk KV tier / session slot（host_kv_capacity_bytes、
//                   host_cache_budget_bytes、disk_kv_path、disk_kv_restore、disk_kv_directstorage）
// --kv-stream 拒绝：shared prefixes 与 long anchors（max_long_anchors_per_continuation 必须为 0）
// --kv-stream 拒绝：device state slots；要求 host_state_slots >= 1
```

⇒ 结论一：**磁盘 KV（`--disk-kv-path`）不是"漏掉的路径"，而是被显式排除的范围。** 本文 §3.5 之前把
`save_continuation` 的抛错当成缺陷，方向不完整：真正的第一道墙是这条守卫。

⇒ 结论二：**长锚点被禁止** ⇒ 基线的 `retain_prefill_progress`（取消时保留长 prompt 的进度）
在流式下**没有 checkpoint 可发布**。所以"取消后能接着聊"在当前范围内**做不到**，不是接线问题。

⇒ 结论三：`TurnClosure`（`install_private_capture` 的 `group.rewrite`）**不受锚点守卫限制**，
理论上可以在流式下捕获。**是否真的捕获、状态货架是否给得出第二个镜像，尚未验证**——这是当前范围内
唯一可能让"同一 prompt 再次发送"命中的杠杆。

### 10.2 由此得到的按范围分层的计划

| 层 | 内容 | 需要改的范围 | 验收 |
|---|---|---|---|
| **S0 正确性（不动守卫）** | 身份戳（已完成）、规划期按身份拒绝、截断钳位、取消不清空、**不再崩、不再错答** | 只动簿记 | 取消序列不崩；`equiv 4/4` 逐位；C13/C15 |
| **S0.5 先测现状**（不改代码） | 查流式是否已捕获 TurnClosure：若是，"同一 prompt 再发"本就该命中；若否，看 `[stream] request starts` 的 `reuse/base` | **0 代码** | 同一 prompt 两次的 `reused`；serve 日志 `anchors/rewrite` 计数 |
| **S1 放宽锚点守卫** | 取消后可续聊（长 prompt 也有 checkpoint 可留） | `startup.cpp` 的守卫 + 状态货架容量 | C18 |
| **S2 region 化** | 多会话并存（`private N`） | arena region + 窗口基址 | A→B→A 双命中；251 K 不退化 |
| **S3 流式下的磁盘/会话发布** | 重启后继续 | **要新决定**：是否为流式放宽 §10.1 的 disk 守卫；arena 需导出/导入 | C19 |

**顺序依赖**：S0 → S0.5（先测，避免为不存在的问题做 S1）→ 由所有者选 S1/S2/S3。

### 10.3 关于"丢条目"补丁（§4.3 的说法已作废）

它只在 `engine_core.h:2862` 的**无放置方案**分支里调用，作用是腾出唯一私有槽；删掉它会恢复
`context cache exhausted`（429）。假未命中的原因是**缺少 prompt 末尾的 checkpoint**，不是它。
因此：**保留**，并把 §4.3 的判据改为"驱逐合法，缺 checkpoint 才是 miss 的来源"。

### 10.4 已作废的旧表述（保留在此以免被再次引用）

- "流式下磁盘会话保存抛错是一个缺陷" ⇒ 先看守卫（§10.1 结论一）。
- "F2 最长前缀命中" ⇒ 违反基线 §4.1，早已撤回。
- "先删驱逐补丁，再补校验" ⇒ 顺序反了，驱逐是合法语义。

---

## 11. 压缩前快照（谁读到这里就该从这里接手）

### 11.1 分支与提交

| 项 | 值 |
|---|---|
| 工作分支 | `context-cache-tier`（从 `kv-stream-27b-16g` = `ff23cbe0` 分出） |
| 已提交、**尚未编译** | `7adffbe9` 身份戳 + 取消不清空（7 文件，+81/−48，`src/ops` 改动 0） |
| 计划重审 | `c3b56512`（本文 §10、§11） |
| 剪裁版 | `pruned-qwen38-27b-16gb` = `bcbd38f5`（未动） |
| 测试机 | `ssh debian`，仓库同分支，构建树 `build-verify`，模型 `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` |

### 11.2 已经改了什么（代码）

- `StreamedKVSpace` 新增 `owner_`（字节属于哪条 continuation）+ `can_serve(owner, tokens)`；
  `enter_request(owner, tokens)`：**不是本人或更浅 ⇒ 倒回并换主人**；`reset()` 与 `retained_` 已删除。
- `ProgramImpl::owner_stamp(sequence)` = 槽位 + `ContinuationSlot::generation`（退役即 +1，永不重复）。
- `KVStore::stream_can_serve(owner, tokens)`；`deactivate` 流式分支**不再 reset**。
- `prefill.cpp` 用 owner 进入，调试行新增 `owner/space_owner/retained`。
- `request_plan.cpp` 按身份拒绝复用（共享前缀在流式下无身份 ⇒ 拒绝）。
- `kv_store.h` 两处截断的钳位保留。

### 11.3 还没做的（下一步就是它）

1. **编译**（测试机）：`CCACHE_DISABLE=1 cmake --build build-verify -j 16`；
2. **S0 验收**：取消序列（长会话→续聊→断开→再续聊）不崩、`equiv 4/4` 逐位、C13/C15；
3. **S0.5 观测**（0 代码）：`NINFER_STREAM_DEBUG=1` 看 `reuse/base/space_pages/owner`，
   并查流式是否捕获 `TurnClosure`（若捕获，"同一 prompt 再发"本就该命中）；
4. 之后由所有者选 S1（放宽锚点守卫）/ S2（region 化）/ S3（磁盘或会话发布，需新决定）。

### 11.4 已定的裁决（不要再推翻）

- **保留**驱逐补丁（`drop_continuations_the_head_cannot_use`）：删掉会恢复 429。
- **不做**"任意最长前缀复用"（违反基线 §4.1：GDN 不能任意回退，复用点是带 StateImage 的 checkpoint）。
- **不把冷启动当修复**：只用于 `space_pages == 0`。
- **流式 = 存储层**：`ResourceManager`/`Program`/`capture` 里不应出现流式分支（现状 4 处泄漏是历史债）。
- 磁盘 KV（`--disk-kv-path`）与长锚点**当前被守卫排除**，动它们先改守卫（范围变更，需所有者决定）。

### 11.5 复现与诊断（现成命令）

```bash
# 取消后崩溃的序列（脚本在测试机 /tmp，仓库内 tools/bench/kv_stream/README.md 有驱动说明）
bash /tmp/serve_cancel.sh            # A(16k) → A → 取消 → B → A
# 关键日志
grep -E "\[stream\] request starts|worker crash|FATAL engine" /tmp/kvlogs/serve_cancel.log
```

判据：`r1..r5` 全 200（或仅取消那条 499/中断）、`worker crash`=0、`FATAL engine`=0，
且任何复用的 `owner` 与 `space_owner` 相同、`space_pages >= base`。

---

## 12. 所有者的原话（需求与裁决的原文，不得改写）

这一节是**需求的第一手材料**。上面各节是我的分析与方案；**当两者冲突时，以本节为准**。
引用保留原作者的分句与空格，仅去掉了与语义无关的换行。

### 12.1 目标与边界

> 「流式不应该影响ninfer原有的功能，出现问题是战略失败。」

> 「我们明明是只是增加上下文长度， 真正的好的体验是，不破坏原有的ninfer使用体验和流程。」

> 「为什么要和ninfer对抗，人家都写好能用的， 不能巧妙的融入吗」

> 「难点在于我们的上下文格式变了？？ 和ninfer原始的不同了还是什么情况， 听你改动这个功能跟便秘一样，
> 到底如何彻底改善呢， 从最开始的规划来说 有没有能规避的设计呢」

⇒ 需求归纳（我据原话得出，若与原话冲突按原话）：**流式只扩上下文；缓存行为必须与驻留模式一致；
不得另造与 ninfer 对抗的第二套语义。**

### 12.2 设计与证据方法

> 「我的要求，git看原始的ninfer流程，搜索这块设计有没有留下文档，参考功能要求，
> 重新评估分析我们的特性和这个如何融洽的对接」

> 「从全局角度出发设计，而不是一次临时补丁，不接受降级方案，把难点和阻碍说出来，分析可行性方案。」

> 「读取你需要的代码，看完后 重新审视计划， 什么都不知道你计划是怎么写出来的。」

> 「我准备压缩了，必须谨慎对待这个修复工作，不是一次意外， 而是一个功能的整体性评估和设计。」

⇒ 归纳：**先读基线代码与文档，再写计划；不接受"降级/兜底"当修复；不接受逐次打补丁。**

### 12.3 性能与"已验证部分"

> 「前缀缓存不就是为了用户体验吗， 缓存未命中的代价很高， 每次全量运算，长的上下文一次接近1分钟？？
> 别出馊主意，正确的认识问题 认识功能 评估方案」

> 「你这条路径 速度会变慢吗」

> 「对已经完成的流式核心功能不会有影响吧？ 代码改不到那边是不是？ 之前测试了很久」

⇒ 归纳：**can't-hit 的代价是分钟级 prefill，不可接受；改地址簿可以，改核心算术要有中性证明与 A/B；
`src/ops` 必须零改动。**

### 12.4 协作规则（对我的约束）

> 「根据计划开发，忠实落实， 如果你发现任何需要做决定或者调整计划的东西，立即停下来汇报，我来决定。」

⇒ 归纳：范围/计划发生变化**必须先停下汇报**，由所有者决定；不得自行扩大范围。

> 「把我的原话需求都加上，我说了什么很重要」

⇒ 归纳：本节必须常驻，压缩后不能被删减或改写。

### 12.5 压缩前已交代的分支操作（已完成）

> 「把当前layout 合并到 kv-stream-27b 这个 不是fork那个分支」

> 「layout分支怎么还在啊， layout合并到刚才说的 kv-stream-27b-16g 然后layout不要了，
> 在 kv-stream-27b-16g合并后 在从他身上开个 context-cache-tier」

> 「fork-qwen38-27b-16gb 这个改个名字 就叫剪裁版本 或者精简版本」

> 「内容都没丢吧？」

⇒ 结果（可核对，见 §11.1）：`layout` 分支已删（提交由 tag `kv-stream/layout-switch-before-removal`
与 `kv-stream-27b-16g` 双路可达）；特性分支 `context-cache-tier` 从 `kv-stream-27b-16g` 分出；
`fork-qwen38-27b-16gb` → `pruned-qwen38-27b-16gb`（提交仍 `bcbd38f5`，树 `ad9e13d3` 不变）。

### 12.6 更早的约束（交接记录，非逐字引用，仍然有效）

- 动手前先说明、等回话；不租 pod；Conventional Commits；**没跑就写"未跑"**。
- 源码同步只经 `commit → push → git pull`；测试机不得有未提交的源码改动。
- 全量编译 `-j 16`、用全新构建目录；日志只写 `/tmp/kvlogs/`；同一张卡一次只跑一个 GPU 任务；
  全量 ctest 用 `-j 1`。
- 先跑特性层（流式开），全绿后才跑共享面全量 ctest。
- 证据标准：**分支文档不作证据**，只用 base 代码（`f6928e00`）、上游原始提交与代码本身。
- 只有 1–2% 速度收益却带来灾难的优化可以砍，砍前留回退点（tag/分支）+ 缺陷档案。
- 修复不能让项目"越跑越远"；牵连面过大就退回"只写文档、不删代码"。

---

## 13. 按代码重审（取代 §10.2 的分层与顺序）

§10 仍是在"守卫 = 不可动的范围"这个前提下写的；把代码逐处读完，这个前提本身是错的。
以下每条都可核对，行号为 `context-cache-tier` @ `7adffbe9` 之后的树。

### 13.1 事实

**F1 流式地址空间不是"没有实现协议"，而是"按单 owner 形状回答协议"**（`context.cpp:692-900`）：

| 协议问题 | 流式回答 | 单 owner 形状下是否正确 |
|---|---|---|
| `device_kv_prefix_pages` | `required`（整段可恢复） | **正确**：每个页都在 pinned host |
| `shared_kv_prefix_pages` | `0` | 正确：没有共享页 |
| `shared_device_kv_prefix_pages` | `0` | 正确 |
| `missing_shared_device_kv_prefix_pages` | `0` | 正确 |
| `partial_tail_cow_required` | `false` | 正确：单 owner 无需 COW |
| `host_kv_prefix_bytes` | `0`（arena 自记） | 正确 |
| `mapped_pages` | `page_count` = 0 | **错误**：它是"这份地址空间存在多少页"的答案，流式下应为 arena 深度 |

⇒ 这些常量**是单 owner 形状的正确实现**，不是占位符。真正缺的只有两件事：**(a) 谁拥有这些字节**（缺陷 #15，已在 `7adffbe9` 修）；**(b) `mapped_pages` 撒谎**（返回 0 而不是 arena 深度）。

**F2 有页表，不是恒等映射**（`kv_stream_space.h`：`stream_slots_` + `publish_slot` + `slot_of_page`）：
每层一张块表，逻辑页 → 窗口槽。所以"第二套存储"的差别**不在寻址**，而在**所有权/共享/设备驻留/磁盘**四项。
§5.6 里"恒等映射、无页表"的说法**错误，此处更正**。

**F3 基线的宿主层本来就存在**：`HostKVArena`（`src/core/host_kv_arena.h`，280 行）+ `HostKVExtentStore`
（`host_kv_store.h`，715 行，带逻辑页 membership），被 `host_kv_store`、`disk_tier` **和** `kv_stream_space`
共同使用。即：流式复用了同样的宿主分配器，只是没复用它上面那层策略。

**F4 取消保留的机制是完整的，而且流式够得着**（`commit.cpp:757` `salvage_continuation`）：
它在 `Lifecycle::Prefilling` 下用 `request.prefill->cursor` 发布 endpoint，条件是
`cursor >= kSalvageMinFrontier`、`state.text_kv_valid == cursor`、MTP 需要 `tail_hidden_valid` 且
`mtp_kv_valid + 1 >= frontier`；不成立才退到 `retain_prefill_progress`（`commit.cpp:829`），
后者发布**已捕获的 checkpoint**（锚点 / rewrite checkpoint）。

⇒ **流式的"取消后续聊"能力 = 基线同一个函数**。它不需要新机制；它需要 (a) 不再 `reset()`（已做），
(b) 锚点不被禁（F5），(c) 回退时把更深的 checkpoint 按计划作废（P1）。

**F5 是守卫，不是技术限制**（`startup.cpp:1380-1412`）：`--kv-stream` 主动拒绝
`--progress-anchor-tokens` 非 0（"retains one continuation: no shared prefixes and no long anchors"）、
拒绝 host/disk KV tier、拒绝 device state slot。

⇒ 对单 owner 形状，**锚点是可实现的**：锚点 frontier ≤ `active_`，其下所有页都在 host arena；
复用锚点 = 倒回 `active_` 再往前写（`enter_request` 已经是这个动作）。**唯一的缺口是作废语义**：
倒回到 F 之后，所有 frontier > F 的 checkpoint 都已失效，必须作为 `ResourcePlan` 的删除集合丢弃（基线 §9.4）。

**F6 `private N` 与 region 无关**：基线在本机驻留模式下也是 `private 1`（serve 日志）。
所以"多会话"**不是"与 ninfer 一致"的必要条件**；region 化只在所有者要把 `private` 提到 2+ 时才需要。

**F7 流式分支的真实数量是 ~22 处**（`pressure.cpp` 9、`context.cpp` 7、`checkpoint_recovery.cpp` 2、
`request_plan.cpp` 2、`prefill.cpp` 1、`kv_store.h` 2），不是 §5.0 写的 4 处。

### 13.2 由事实推出的修订计划

| 步 | 内容 | 规模 | 证据 |
|---|---|---|---|
| **P0 收尾正确性** | 编译 `7adffbe9`；取消序列不崩；`equiv 4/4`；C13/C15；`mapped_pages` 改为返回 arena 深度（F1b，一处，但需先确认没有调用方依赖 0） | 小 | 取消序列日志、equiv 逐位 |
| **P1 锚点（真缺口）** | 放开守卫 F5 + 回退时按计划作废更深 checkpoint（F5 的唯一缺口）+ 验证"取消 16 k 后续聊 `reused > 0`" | 中 | C18；`anchors > 0` |
| **P2 TurnClosure** | 先测：流式 finish 时前端是否提出 rewrite 边界。提了 ⇒ 同一 prompt 重发本就该命中（若仍 miss，看 `[stream] request starts` 的 `reuse/base`）；没提 ⇒ 改的是前端 offer，不是 store | 小/零 | C17 |
| **P3 磁盘与会话** | arena 导出/导入 + 放开 disk 守卫 | **大** | C19 |
| **P4 多 owner region** | 仅当 `private N > 1`（F6 说明这不是"对齐 ninfer"所必需） | 大 | A→B→A |

顺序：**P0 → P1 → P2 →（所有者决定）P3/P4**。P1 是唯一"用户能感觉到、且现在做不到"的缺口；
P2 可能是零成本；P3/P4 是范围问题，不是缺陷修复。

### 13.3 需要所有者决定的两件事

1. **是否放开流式下的锚点（P1）**：放开 = 与基线同能力（取消后续聊），代价是每 owner 多几个
   host state image（每镜像 ~百 MB 级，需实测）；不放开 = 长 prompt 取消后必须重算（分钟级）。
2. **磁盘/会话（P3）与多 owner（P4）是否在本交付内**：两者都大，且都不影响"单会话 192 K"这一主场景。

### 13.4 立即可做、不需要决定的

P0（编译 + 取消序列 + equiv + C13/C15）与 P2 的观测（0 代码）。这两件先做完，再回答 13.3。

### 13.5 §13 的三处更正（再读代码后，均以代码为准）

**C1 §13.2 的 P0 里"把 `mapped_pages` 改成返回 arena 深度"是错的，撤回。**
流式地址背后没有 `LogicalKVPageStore` 的成员页，`logical_page(address, page)` 对它没有意义；
让 `mapped_pages` 报出页数，会使所有"`required <= mapped_pages` 后用逻辑页取页"的非流式分支
（`session_snapshot.cpp:533`、`pressure.cpp:156`、`checkpoint_recovery.cpp` 等）去解引用不存在的页。
所以 `mapped_pages` 保持 0，直到真正有页成员关系（即 P3/P4）。它属于**新存储合同**的工作，不是收尾。

**C2 §13.1 的 F6 错："本机驻留模式也是 private 1"。** 那个 `private 1` 是**流式 serve 的日志**。
基线默认（`model_instance.cpp:443-477`，context cache 启用、Legacy）：

```cpp
cache.max_private_continuations = value_or(2 * concurrency);          // concurrency=1 ⇒ 2
cache.max_shared_prefixes       = value_or(max(concurrency, K)) + grafts;
cache.max_long_anchors_per_continuation =
    value_or(cache.automatic_long_anchors ? 4U : 2U);                  // 默认 2，不是 0
if (*cache.max_long_anchors_per_continuation == 0) {                   // 锚点容量为 0 才关
    cache.endpoint_anchors = false; cache.progress_anchor_tokens = 0; }
```

⇒ **基线的默认是"2 个私有 continuation + 每个 2 个长锚点 + progress 锚点开"**，流式的
`private 1 / anchors 0` 是**收窄**，不是"与基线一致"。§9 那句"本机驻留实测 private 1"作废。

**C3 §13.2 的 P1 只说"放开守卫"，不够。** `progress_anchor_tokens` 在锚点容量为 0 时被强制清零
（同上），宿主状态槽数由 `(2 + anchors) * private + shared` 决定（`startup.cpp:1790-1840`）。
所以恢复锚点 = 三处一起改（守卫、`max_long_anchors_per_continuation`、`host_state_slots` 预算），
代价是每个 owner 多 `A` 个宿主 StateImage（单个镜像字节数需实测）。

**确认成立的部分**（这几条读到了，可以当事实用）：

- **salvage 在流式下可达**：`ProgramImpl::abort`（`commit.cpp:966`）先 `salvage_continuation`
  再 `retain_prefill_progress`，**没有流式分支**；流式只影响 `deactivate` 之后的 `retain`。
- F1（协议常量）、F2（有块表）、F3（共用 `HostKVArena`/`HostKVExtentStore`）、F7（~22 处分支）成立。

**仍未读、因而仍未进入计划的部分**（不得当成已知）：

- 回退时更深 checkpoint 的作废路径（`checkpoint_recovery.cpp:126/387`、`pressure.cpp` 的删除集合）；
- finish 时 TurnClosure 的 capture offer 序列（`frontend.cpp:628-840` 的 fetch/rewrite 组）；
- `session_snapshot` 的 restore 侧（保存侧只读到 `:533` 的覆盖检查）；
- 单个宿主 StateImage 的实际字节数（决定 P1 的代价）。

⇒ 修订后的执行顺序：**P0（编译 + 取消序列 + equiv + C13/C15，不含 `mapped_pages`）→
P2 观测（零代码）→ P1（C3 的三处 + 上述作废路径，读完再动手）→ 所有者决定 P3/P4。**

---

## 14. 读完全部 kv 文档后的最终判断：声明范围 vs 所有者要求

这一节取代 §10–§13 的**计划口径**（事实仍然有效）。触发它的三份文件：
`kv-streaming-strategy.md`（status: active，**计划的唯一权威**，P0–P8 + §10 门禁）、
`kv-streaming-design.md`（§4.9 与 §4.11 是流式形状的结构性决定）、
`kv-streaming-verification.md`（C1–C16 的验收登记）。

### 14.1 这些文档**已经声明**了流式的形状（不是缺陷，是决定）

| 声明 | 出处 | 含义 |
|---|---|---|
| "流式 run 是单序列服务端模式：一个 continuation（host 权威 + 一张 host StateImage + 前沿）在请求间保留，同前缀的下一个请求只 prefill 增量" | strategy §0.2、§9.5 | **单 continuation 是设计**，不是遗漏 |
| "端点必须在 host shelf … 并存一个 device StateImage 槽时 … target 从第 3 个 decode 轮起读到旧 state，所以 `device_state_slots` 必须为 0" | design §4.9 | device 槽禁令是**实测结论** |
| "流式模式下 text pool 用 `KVStreamSpace` 取代前两层 … `ProgramImpl` 的 `text_kv_pages`/`text_kv_addresses` 在流式下为空；所有使用它们的路径必须逐一处理：**retention/checkpoint/prefix 路径拒绝**" | design §4.11 | 保留/checkpoint 最初被整段拒绝，P6 只把"一个 host 端点"放回来 |
| "多序列/并发 … M1 限制 `--max-concurrency 1`，M4 再按序列数分摊环" | design §4.9 | 多会话是**后续里程碑**，不是当前缺陷 |
| 守卫拒绝 hybrid / host KV tier / disk tier / session slot / shared prefix / long anchor | `startup.cpp:1380-1412` | 与 design §4.11 一致 |

⇒ 结论：我此前把它们称作"四处偏差/漏掉的路径"（§3、§10）**口径过重**。它们是**声明的取舍**。
真正属于缺陷的只有一类：**声明的形状下，声明了的行为不成立**——
identity（#15）、规划期与使用期的窗口（#13/#14）、截断钳位、"保留被当租约"（#11/#12）。

### 14.2 与所有者要求的真实冲突（必须由所有者裁决）

> 「我们明明是只是增加上下文长度， 真正的好的体验是，不破坏原有的ninfer使用体验和流程。」

| 能力 | 基线（不流式，默认值） | 流式（声明形状） | 关闭流式是否恢复 |
|---|---|---|---|
| 续聊命中（endpoint） | 有 | **有**（C13 PASS） | 是 |
| 同一 prompt 重发命中（TurnClosure） | 有（anchors 默认 2） | **未验**（C17） | 是 |
| 取消后续聊（salvage / 锚点） | 有 | **仅 salvage 可达**；无锚点 ⇒ 长 prompt 取消即丢（C18） | 是 |
| 多会话并存（`private 2*N`） | 有 | 无（`private 1`） | 是 |
| 磁盘会话 / 淘汰 spill（`--slot-save-path`） | 有 | 无（arena 不可导出） | 是 |
| host KV tier / disk KV / hybrid | 有 | 拒（守卫） | 是 |

两种解读，结论相反：

- **解读 A（opt-in 即可）**：`--kv-stream` 是显式开关，其边界写在 `docs/cli.md`/`docs/serving.md`，
  关掉即恢复到完整基线 ⇒ 当前声明**不违反**"不破坏原有流程"；本交付只需**把声明的形状做对**。
- **解读 B（开着也要一样）**：打开流式后，上述能力必须与基线相同 ⇒ 声明的形状本身不满足需求，
  需要逐项加回（锚点 → 多 owner → 磁盘），每一项都要付状态镜像/宿主内存/契约实现的账。

**这是产品范围决定，不是技术选择。**我不能替所有者选；以下给出两条路线各自的具体工作量。

### 14.3 两条路线

**路线 A（把声明的形状做对）— 小，可立即完成**

| 步 | 内容 | 验收 |
|---|---|---|
| A0 | 编译 `7adffbe9`（identity + 取消不清空 + 截断钳位 + 规划期按身份拒绝） | 编译通过 |
| A1 | 取消序列（长会话→续聊→断开→再聊）：不崩、不静默错答 | C18 的"不崩/不错"半边；`equiv 4/4`；C13/C15/C16 不回归 |
| A2 | 观测 TurnClosure（0 代码） | 同一 prompt 两次的 `reused`；决定 C17 是"已工作"还是"要接线" |
| A3 | 文档写清边界（`docs/cli.md`、`docs/serving.md`、`--help`）——strategy §10 门禁本来就要 | 门禁项 |

**路线 B（开着流式也与基线同能力）— 大，按项付费**

| 步 | 内容 | 代价 | 依据 |
|---|---|---|---|
| B1 锚点 + TurnClosure | 放开守卫（3 处，见 §13.5 C3）+ 回退时按计划作废更深 checkpoint + 状态槽 `(2+A)*private+shared` | 每 owner A 个 host StateImage（**字节数未测**） | `model_instance.cpp:443-477`、`startup.cpp:1790-1840` |
| B2 多 owner（`private N`） | arena region + 每 owner 一套前沿/checkpoint（= strategy 的 M4） | 每 owner ≈ 4 GiB 宿主（192 K） | design §4.9 的 M4 |
| B3 磁盘/会话 | arena 导出/导入 + 放开 disk 守卫 + `session_snapshot` restore 侧 | 大；快照 ~4 GiB/192 K | `session_snapshot.cpp:533` |

### 14.4 与 strategy 的 P 系列如何对齐

- A0–A2 = **P6 的收尾**（P6 自己写着"未得：retained 臂的 CTest 未跑"）；取消崩溃属于 P6 交付的缺陷。
- B1 = **P6 的延伸**（把 host shelf 从 1 个端点扩到"端点 + 锚点"）。
- B2 = strategy 的 **M4**（多序列），本来就是后续里程碑。
- B3 = strategy **未列**（磁盘/会话不在 P0–P8 任何一期）⇒ 若要，是新增范围。
- strategy §10 的未过门禁（cli 文档、span tuner 注册、sanitizer、跨架构、交错测量）是**特性交付门**，
  与本文的缺陷修复并行，不冲突。

### 14.5 仍未验证（不得当已知）

1. 单个 host StateImage 的字节数（决定 B1 的代价）；
2. 回退时更深 checkpoint 的作废路径是否存在（`checkpoint_recovery.cpp:126/387`、`pressure.cpp`）；
3. finish 时 TurnClosure 的 capture offer 是否对流式生效（`frontend.cpp:628-840`）；
4. `session_snapshot` 的 restore 侧。

⇒ **可立即执行、不需要裁决的**：A0 + A1 + A2。**B1/B2/B3 需所有者选择路线后，先读完 14.5 再动手。**
