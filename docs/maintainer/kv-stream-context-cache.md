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

---

## 15. 路线 B 定案：开着流式，能力与基线相同（所有者裁决）

> 所有者裁决（原文）：「必须开着也要一样， kv流式就是解决上下文不足的，我后面用肯定全程开着，
> 要不破坏原有的ninfer使用体验和流程。」

⇒ 规格从此是**基线（驻留模式）的行为**，流式只改"字节住在哪"。§14.2 的解读 A 作废。

### 15.1 三项缺口与它们的真实成本（读完代码后）

| # | 缺口 | 关键代码事实 | 规模 |
|---|---|---|---|
| **B1** | checkpoint 库存只有 1 个 host 端点：无锚点、无 TurnClosure ⇒ 同一 prompt 重发 miss、长 prompt 取消即丢 | 锚点由前端按 `progress_anchor_tokens` 的绝对栅格生成，**与流式无关**（`frontend.cpp:710-838`）；`progress_anchor_tokens` 在锚点容量 0 时被清零（`model_instance.cpp:452`）；库存 `(2+A)*private+shared`（`startup.cpp:1790-1840`）；基线的 `save_continuation` 靠**每页引用计数 + COW** 保住 checkpoint 的页，流式 arena 两者都没有 ⇒ 回退必须**按计划作废**更深的 checkpoint | 小–中 |
| **B2** | `private 1`：第二个会话一来，第一个的字节被覆盖 | 基线默认 `max_private_continuations = 2 * concurrency`（`model_instance.cpp:443`）；流式守卫强制 `--max-concurrency 1` ⇒ **同一时刻只有一个 active decode**，环/窗口只服务当前请求；空闲 owner 只需要宿主页 + StateImage | 中 |
| **B3** | 磁盘会话/淘汰 spill：`mapped_pages` 恒 0 ⇒ 保存抛错、spill 被 `catch (...)` 吞掉 | 保存与恢复两侧都用 `mapped_pages`（`session_snapshot.cpp:534/1020`，恢复还要求 `mapped_pages == page_count`）；流式地址没有逻辑页成员 | 中–大 |

### 15.2 B2 的关键简化（由 `--max-concurrency 1` 推出，不是折衷）

基线 M4 说"多序列同时 decode 会争环，M4 再按序列数分摊环"。但**流式被强制单并发**，
所以"多个缓存的会话"与"多个并发的 decode"是两件事：
**空闲 owner 不需要任何设备窗口**，宿主页就是它的全部状态。

于是 B2 = **宿主 arena 按 owner 分区 + 设备窗口/环在请求进入时绑定到该 owner**：
- 宿主：N 个连续 region，每层 `region_base(layer)`，owner 的页只落在自己的 region（等价于基线的
  "每个 private continuation 有自己的地址空间"）；
- 设备：窗口/环的算术不变，只是它对当前 owner 的页生效；切换 owner 时**不拷贝**（宿主是权威），
  该 owner 的页在 `advance` 里按需上传，块表随之发布（`publish_slot` 已有）；
- 目录：每个 owner 的条目携带 `region + 深度`，`can_serve(owner, tokens)` 增加"region 归属"判据；
- 预算：每 owner 的容量 = 宿主预算 / N；N 取基线的 `2 * concurrency`（本机单人使用 ⇒ 2）。

### 15.3 分步计划（每步单独可回退，验收不过不进下一步）

| 步 | 内容 | 出口判据 |
|---|---|---|
| **B0** | 量出 `state_images.host.image_bytes`（一个打印持久布局的小驱动，不改产品代码） | 得到一个字节数，写进 §15.4 |
| **B1a** | 放开锚点：守卫 + `max_long_anchors_per_continuation`（≥2，基线默认）+ `host_state_slots = (2+A)*private+shared` | 启动日志 `anchors 2`（或 A）；`equiv 4/4` 逐位不回归 |
| **B1b** | 回退作废：倒回 frontier F 时，把 frontier > F 的 checkpoint 作为 `ResourcePlan` 的删除集合丢弃（`CheckpointRef` 逐项） | 无静默覆盖；计划审计可查 |
| **B1c** | TurnClosure：先测（流式 finish 是否已有 `group.rewrite`），有 ⇒ 验 C17；无 ⇒ 在前端 offer 里接上 | **C17**：同一 prompt 两次，第二次 `reused > 0` 且逐位 |
| **B1d** | 取消：salvage 可达已确认（`commit.cpp:966`）；锚点就绪后验 | **C18**：取消 16 k 后续聊 `reused > 0`、进程存活 |
| **B2a** | 宿主按 owner 分区 + 窗口绑定（零行为变化：N=1 时 region_base≡0） | `equiv 4/4`、251 K `round_ms_median ≤ 41.4×1.01` |
| **B2b** | `private N = 2*concurrency`：目录/准入/报告接入 | **A→B→A 两个会话都命中**、`private 2` 出现在启动行 |
| **B3a** | 流式地址的页合同：`mapped_pages` = 真实深度、页可拷出/拷入（沿用 `plan_host_kv_page_layout`） | save→erase→restore→续聊逐位 |
| **B3b** | 淘汰 spill 真实落盘（`spill_catalog_slot` 不再吞错） | spill 文件存在且非空 |
| **B4** | 收口：`docs/cli.md`/`docs/serving.md`/`--help`、verification C17–C19、measurements、strategy §10 门禁 | 门禁项逐条 |

顺序依据：**B1 先**（它让"同一 prompt / 取消"当天可用，且不动设备布局）；**B2 次**（宿主分区，
N=1 时中性可证）；**B3 最后**（需要 B2 的 region 只是方便，不依赖它——B3 依赖的是"页可寻址"）。

### 15.4 预算（B0 之后填实数）

- 每 owner 宿主 KV：192 K ≈ 4.07 GiB（实测启动行）⇒ N=2 ⇒ ≈ 8.1 GiB。
- 每 owner StateImage：`(2 + A)` 张；A=2、N=2 ⇒ 8 张 + shared；单张字节数待 B0 实测。
- 上限：`resolve_host_cache_budget` 要求 `state_bytes ≤ host_cache_budget / 2`，否则启动即报错并给出算式。

### 15.5 不做的事（保持）

- 不重写流式核心：`src/ops` 零改动、`text.cpp` 拷贝编排不变；B2 只改 region 基址与目录判据。
- 不引入第二套缓存策略：B1/B2 的删除与保留都走 `ResourcePlan`，与基线同一套计划语义。
- 不用冷启动掩盖缺口：唯一的冷启动条件是"该 owner 没有可用 checkpoint"。

---

## 16. 基线能力的代码清单（取代一切按文档罗列的能力表）

**基线 = `KVAddressSpaceStore` + `LogicalKVPageStore` 现在能做的事**。下面是它们的合同面
（`kv_store.h`，行号相对文件），流式必须逐条满足，才叫"开着也一样"。

### 16.1 地址空间合同（`KVAddressSpaceStore`）

| 组 | 方法（`kv_store.h`） | 语义 |
|---|---|---|
| 生命周期 | `create_active`(901)、`create_inactive`(915)、`activate`(953)、`deactivate`(1095)、`release`(1065)、`release_after_deactivate`(1057)、`can_release*` | owner 的创建/激活/退役 |
| 页物化与配额 | `prepare_activation`(970)、`commit_activation`(1031)、`page_reservation`(1023)、`materializable_pages`(1012)、`ensure_mapped_to_tokens`(1700)、`can_resize_entitlement`(1666)、`resize_entitlement`(1675)、`release_growth_entitlement`(1692) | 页按需物化成设备槽；配额可增长/收缩 |
| 前缀 fork | `prepare_prefix_fork`(1127)、`settle_prefix_fork_tail_source`(1209)、`complete_prefix_fork_after_tail_release`(1224)、`commit_prefix_fork`(1240)、`abort_prefix_fork`(1303)、页级 `prepare/commit/abort_page_prefix_fork`(1323/1388/1441) | 共享前缀：页引用计数 + 尾部 COW 源结算 |
| 活体快照 | `active_snapshot_shape`(1459)、`prepare_active_snapshot`(1511)、`commit_active_snapshot`(1572)、`abort_active_snapshot`(1644) | 把 active 地址的页物化成 checkpoint/移植目标 |
| 前沿与保护 | `commit_frontier`(1747)、`set_checkpoint_requirement`(1908)、`rebuild_checkpoint_protection`(2110) | checkpoint 保护列：页被 checkpoint 引用则不可回收/覆盖 |
| 截断族 | `destructive_truncate`(1775)、`destructive_truncate_inactive`(1865)、`can_truncate_inactive_prefix`(1924)、`can_deactivate_and_truncate_prefix`(1942)、`truncate_inactive_prefix`(1956) | 四种回退语义，各自的前置条件 |
| 查询 | `mapped_pages`(1973)、`entitlement`(1977)、`committed_frontier`(1981)、`physical_page`(2002)、`content_epoch`(2011)、`logical_page`(2020)、`has_active_reference`(2029) | 深度、配额、前沿、物理槽、内容纪元、逻辑页 |
| 页表发布 | `publish_membership`(2195) | 逻辑页 → 物理槽，一次发布 |

### 16.2 页模型（`LogicalKVPageStore`，同一文件）

一页同时可有：`device_replica`（设备槽副本）、`host_replica`（宿主副本）、`pending_device_replica`、
`content_epoch`、`committed_columns`/`protected_columns`、`references`/`writer_references`/`source_pins`/
`destination_pinned`（`kv_store.h:319-460`）。

**关键事实**：`device_resident()` 与 `host_resident()` 是两个独立可选项，`demote` 路径会把
`device_replica` 清空而保留 `host_replica`（`kv_store.h:417/537/742/809`）。
⇒ **"页住宿主、设备只留窗口"本来就是这个页模型的一个合法状态**，
不需要另造存储。design §4.11 说它"假设每个 occupied 逻辑页都有 device replica"——那是对
`materialize` 的**按需语义**的描述，不是数据模型的限制。

### 16.3 结论：`--kv-stream` 必须是这套 store 的一种放置策略

要"开着也一样"，流式地址必须能回答 16.1 的全部问题。两条路：

| 路 | 做法 | 代价 |
|---|---|---|
| **(a) 在 arena 里重造** | 在 `StreamedKVSpace` 里实现 fork/snapshot/protection/entitlement/truncate 族 | 复制整套页语义，约 15 个方法 + 3 个 tier；仍有 22 处分支；最贵且最容易再出缺陷 |
| **(b) 让流式回到 store 里**（推荐） | 页仍由 `LogicalKVPageStore` 拥有、**宿主副本覆盖全部页**、设备只持窗口；`materialize` 向 `StreamedKVSpace` 要一个环槽；fork/snapshot/protection/truncate/snapshot 全部原样生效 | 反转 design §4.11 的结构决定；动 `text.cpp` 的分发钩子（行为不变）；重测性能 |

(b) 之后：`mapped_pages` 真、`private N` 真、锚点/闭包真、磁盘会话真、spill 真、22 处分支**全部删除**。
B1/B2/B3 从"三个项目"变成"三个验证里程碑"。

### 16.4 (b) 的触点与风险（必须让所有者知道）

- **`text.cpp` 的分发钩子会改**：现在它从 `stream_space_` 取窗口/环；改后从 store 取"该 owner 的
  窗口与环槽"。**拷贝编排的形状（每层 run、事件顺序）不变**，`src/ops` 仍零改动。
- **`kv_stream_space` 缩小**为窗口/环/事件/块表管理器（它现在也承担"拥有全部槽与整块宿主"）。
- **性能必须重测**：按需物化不能退化成逐页拷贝，必须保持"按 run 顺序预取"；251 K 的
  `round_ms_median ≤ 41.4 ms × 1.01` 是硬判据。
- **风险**：这是 1298 行设计文档里 §4.11 的**反转**。它的原理由是"把流式塞进 store 会继承按需分配的
  语义，而那正是要避开的"——现在由所有者的要求推翻：按需语义正是基线行为，必须保留。

### 16.5 需要所有者批准的两项

1. **反转 design §4.11**（流式从"取代两层存储"改为"store 的放置策略"）。
2. **允许修改 `text.cpp` 的分发钩子**（拷贝编排与 `src/ops` 不变，行为不变）。

不批准则只剩 §16.3(a)：在 arena 里重造约 15 个协议方法，工作量与风险都更大。

---

## 17. 融入方案（最终形态）：流式是页池的放置策略

> 所有者原话：「不要和ninfer对抗， 好的功能要学会融入ninfer的规范里， 不是在里面胡搅蛮缠。」

### 17.1 目标形态（一句话）

`--kv-stream` 不再替换存储层，而是给**既有页池**换一条放置规则：
**每页的权威副本在宿主（`host_replica`），设备池只持窗口（`device_replica` 的一个子集）**。
其余全部由既有代码回答：`mapped_pages`、`entitlement`、fork、snapshot、checkpoint 保护列、
四种截断、session 快照、spill、`private N`。

### 17.2 代码映射：每个流式概念落到既有机制上

| 流式概念 | 现在（并行存储） | 融入后（既有机制） |
|---|---|---|
| 宿主权威副本 | `StreamedKVSpace::arena_`（自己的 `HostKVArena`） | `HostKVExtentStore` 的 host replica（`--host-kv-capacity` 覆盖全部页） |
| 设备窗口 + 环 | `StreamedKVSpace` 拥有池的**全部**槽 | 地址空间 `entitlement = 窗口页数`；`prepare_activation`/`commit_activation` 批量物化 |
| 页表 | 每层 `stream_slots_` | `publish_membership`（同一条） |
| 每层 run 拷贝 | `upload(layer, copy)` + ready/consumed 事件 | 同一套（`src/ops/kv_stream` 不动），只是由 store 的物化结果驱动 |
| `mapped_pages = 0` | 特例 | 真值：地址空间已映射的页数 |
| 保留/checkpoint | 一张 host StateImage + 前沿 | 既有 catalog：`TurnClosure`/`LongAnchor`/`endpoint`，保护列生效 |
| 取消 | `retain()`/`reset()` | 基线 `salvage` / `retain_prefill_progress`（`commit.cpp:966`） |
| 磁盘/槽文件 | 不可用 | `session_snapshot`（保存/恢复都已按页写） |
| 22 处 `streaming()` 分支 | 特例 | 全部删除；守卫只保留真实冲突（hybrid 的块级语义、并发环） |

### 17.3 顺序（每步编译通过、可回退、先验收后推进）

| 步 | 内容 | 出口 |
|---|---|---|
| **R0** | **一致性验收清单先落地**（不改产品代码）：把"开着也要一样"逐条写成可执行的检查——锚点、TurnClosure 重发、取消续聊、A→B→A、槽会话 save/erase/restore、spill、shared prefix、`mapped_pages`/报告 | 清单存在且**当前会红**（这就是缺口事实） |
| **R1** | 放置策略：宿主 tier 覆盖全部页 + `entitlement = 窗口` + 窗口物化走 `prepare/commit_activation` + 删除第一批 `streaming()` 分支（`mapped_pages`、`host_kv_prefix_bytes`、`checkpoint_recovery`、`request_plan`） | `equiv 4/4` 逐位；251 K `round_ms_median ≤ 41.4×1.01`；C13/C15 不回归 |
| **R2** | checkpoint：放开锚点/闭包守卫，验 C17/C18（此时应"自动成立"，因为保护列与 catalog 都在） | C17/C18 |
| **R3** | 会话与磁盘：save/erase/restore、spill 真实落盘（`session_snapshot` 现成） | C19 |
| **R4** | 收尾：清空剩余分支与守卫、`private N`、报告字段、`docs/cli.md`/`serving.md`/`--help`、strategy §10 门禁 | 门禁逐条 |

### 17.4 明确保留（不碰）

- `src/ops/kv_stream/*`（内核、环、事件、几何、调优器）**零改动**。
- `text.cpp` 的 run 分发**形状与事件顺序**不变；它从"空间的窗口"改成"store 的窗口"，行为相同。
- 性能判据不变：251 K `round_ms_median ≤ 41.4 ms×1.01`、`upload_bytes` 变化 ≤1%、`runs_per_layer` 不碎。

### 17.5 如果 R1 撞上真实冲突

只允许在"功能冲突、无法融入"时保留特例，并且必须写进本文：冲突是什么、基线的哪条语义无法表达、代价。
不允许以"快一点""简单一点"为理由保留分支。

---

## 18. R0 实测（`context-cache-tier` @ 编译通过，2026-11-11，RTX 4080 SUPER 16 GB）

命令：`python3 tools/bench/kv_stream/parity.py --model <Qwen3.8-27B…ninfer> --bin build-verify/apps/ninfer-serve`
（日志 `/tmp/kvlogs/parity/`）。

| 检查 | 结果 | 证据 |
|---|---|---|
| **P1 库存** | **FAIL** | `host states=1 host_kv=0B private=1 shared=0 anchors=0`；而 serve 自己的 `--help` 写基线默认：`device-state=max-concurrency, private=2x concurrency, shared=max(max-concurrency,7), anchors=2 (4 with --auto-long-anchors), Host state=8 slots, Host KV=8192 MiB` |
| **P2 同一 prompt 两次** | **PASS（今天就已命中）** | `reuse=2 base=1427 valid=1427`、`cache 1,427 (99.7%, turn closure)`、TTFT **45 ms**。⇒ 本文 §3 的 D4"同一 prompt 重发也 miss"**作废**；TurnClosure 在流式下已经工作 |
| **P3 取消后继续** | **FAIL** | 取消那条把空间倒回自己的前缀，下一条 `reuse=0 base=0`、`cache 0 (0.0%)`；进程存活（无崩溃） |
| **P4 A→B→A** | **FAIL** | A 命中一次后，B 冷跑；A 再来 `reuse=0`（`space_owner` 已换成 B）⇒ 单 owner 覆盖 |
| **P5 槽会话** | **拒绝（不是崩溃）** | serve 直接拒绝：`--kv-stream keeps its context in the streamed arena: --slot-save-path is not available` ⇒ §3.5 的"保存抛错"在 serve 路径上不可达，真实形态是**显式拒绝** |
| **P6 进程存活** | **PASS** | 全部检查后 `alive=True`、无 `worker crash`/`FATAL engine`/`terminate` |

### 18.1 由此得到的真实缺口（取代 §15.1 的估计）

1. **取消后继续**（P3）：需要锚点（基线 `anchors=2` 默认）或可用的 salvage frontier；取消当前把空间倒回，
   之后没有 checkpoint 可复用。
2. **多会话**（P4）：单 arena 单 owner，B 覆盖 A。
3. **槽会话/磁盘**（P5）：显式拒绝，需要 arena 的页导出/导入才能打开。
4. **库存/报告**（P1）：流式只报 1 个端点、0 B 宿主 KV。

**已经工作、不要再修**：同一 prompt 重发（turn closure，99.7%、45 ms）、跨请求 endpoint 复用（C13）、
取消后不崩溃（P6，identity 修复的结果）。

### 18.2 Engine 级复用用例（新测试 `test_engine_kv_stream_reuse.cpp`，5 例全过）

`ninfer_tests ninfer_qwen3_5_kv_stream_reuse_real_test`（RTX 4080 SUPER，`rk8v4`，1.4 k token 提示）：

```
reuse: repeated prompt reused 0 tokens
reuse: unrelated after cancel reused 0 tokens
reuse: prompt after cancel reused 0 tokens
reuse: A -> B -> A reused 0 tokens
reuse: five cases admitted; upload 0 B, download 0 B, verify 0/0
```

五个用例的**正确性**全部成立（每个用例都与无缓存引擎逐 token 相同、全部被接纳、无崩溃），
但复用计数全为 0。两个原因，分开记录：

1. **原始 token 提示没有消息结构**（`engine.h:101-106` 的 `anchor_prompt_end` 注释即此意）：
   要让同一 prompt 重发命中，必须 `prepare_tokens(..., anchor_prompt_end=true)`，而它要的
   正是一个 anchor checkpoint —— 流式当前**禁止锚点**（守卫要求
   `max_long_anchors_per_continuation == 0`），于是请求了也不生效。**这是 §15.1 B1 的具体表现**，
   且与 serve 路径不矛盾：serve 走 chat 模板，TurnClosure 可用，实测 99.7% 命中（§18.1 P2）。
2. 提示只有 1.4 k token，池的窗口装得下整段 ⇒ `upload/download 0 B`，这条用例测的不是流式搬运，
   而是复用判据本身（也正因如此它便宜、可进 CTest）。

### 18.3 由 18.2 得出的 R2 判据（写给实现）

- `prepare_tokens(anchor_prompt_end=true)` 在流式下必须与基线同样生效 ⇒ 锚点容量 ≥1；
- 复用命中后 ids 必须逐 token 相同（本测试已强制）；
- 取消后的请求必须被接纳且与冷跑逐 token 相同（本测试已强制）。

---

## 19. R2 第一刀落地：流式恢复锚点（`startup.cpp` 守卫）

改动：`--kv-stream` 不再拒绝 `max_long_anchors_per_continuation > 0`；共享前缀仍拒绝；宿主状态库存按
基线的同一公式校验（`2 * private + anchors`），不足在启动期报错而不是让接纳失败。测试相应给
`host_state_slots = 4`、`anchors = 2`（serve 默认 private=2、anchors=2、host state=8 自动满足）。

### 19.1 实测（同一测试、同一产物）

| 用例 | 改前 | 改后 |
|---|---|---|
| 同一 prompt 两次（`anchor_prompt_end=true`） | `reused 0` | **`reused 900`**，ids 仍与冷跑逐 token 相同 |
| 取消后另一个会话 | 0 | 0 |
| 取消后同一 prompt | 0 | 0 |
| A→B→A | 0 | 0 |

⇒ **锚点在流式下本来就能工作**，此前只是被守卫挡住。这也证实 §15.1 B1 的判断：
需要的不是新机制，而是让库存存在。

### 19.2 仍为 0 的两类，原因不同（写给下一步）

1. **取消后**：锚点来自 prompt 结构（`anchor_prompt_end` 在**提示末尾**）或
   `progress_anchor_tokens` 的绝对栅格。取消发生在 prefill 中途时，末尾锚点尚未产生，
   所以需要**进度锚点**（`progress_anchor_tokens`，基线默认 16384）。测试要变成
   "16 k 提示 + 中途取消 + 再次发送"，才能进入 C18 的真实路径。
2. **A→B→A**：单 arena 单 owner，与锚点无关；这是 R3（宿主按 owner 分区）的判据。

### 19.3 复用深度的事实

`reused 900`（提示约 1 432 token）——命中的是锚点 frontier，不是提示末尾；
所以"同一 prompt 重发"省的是锚点之后的 prefill，而不是全部。验收应写成
`reused == 该锚点 frontier` 且 ids 逐 token 相同，不写成"等于提示长度"。

### 19.4 取消保留仍为 0：已排除与待查

测试改成"约 20 k token 的单一提示、在 16 384 进度锚点之后取消、再发同一提示"后，`reused` 仍是 0。
按代码逐条核对（`commit.cpp:829-900`）：

**已排除**（这几条对 `--kv-stream` 不成立，不是原因）：
- `can_deactivate_and_truncate_prefix`（`kv_store.h:1942`）在流式下 `page_count == 0` ⇒ 循环为空，
  只要 `frontier <= committed_frontier` 就返回真；取消发生在 16 k 之后，锚点 frontier 满足该条件；
- `retain_prefill_progress` 要求 `state.long_anchors` 非空或 `rewrite_checkpoint.valid` ——
  进度锚点会填 `long_anchors`（未被锚点容量挡住，因为容量已为 2）。

**待查**（下一枪，未验证，别当结论）：`retain_prefill_progress` 的前置条件里还剩三条只读检查：
`state.endpoint_valid`、`state.state.fork_pending`、`can_release(state.write)`，以及
`state.text_kv_valid` 与 `retained.main_frontier` 的关系。这四种任一为真都会让保留直接 `return false`
（随后 `clear_lane_strict` 丢掉整条 lane），表现正是"取消后复用为 0 且不崩"。
**办法**：在这三个 `return false` 前各加一行 `[retain] refused: <原因>` 的调试输出（只在
`NINFER_STREAM_DEBUG` 下），跑一次同场景，就能定位是哪一条，再谈修改。

### 19.5 找到根因：身份戳绑在"请求的槽"上，而不是"目录条目"上

在干净的引擎上先跑取消用例，`NINFER_STREAM_DEBUG=1` 的 trace 直接给出了答案：

```
[stream] request starts: owner=65536  reuse=0 base=0     valid=0     space_pages=0   space_owner=0
[stream] retained endpoint frontier=19924 ledger=19925 kv_valid=19924      ← 取消的 prefill 确实保留了端点
[stream] request starts: owner=131072 reuse=0 base=0     valid=0     space_pages=312 space_owner=65536 retained=19924
```

三件事同时成立：

1. **取消确实保留了上下文**：`retained endpoint frontier=19924`，`space_pages=312` = 19924 token 的页数
   ⇒ `retain_prefill_progress` 走通了（所以没有 `[retain] refused` 输出）；
2. 下一条同 prompt 的请求**没有被判为可复用**：`reuse=0`；
3. **空间的主人（65536）与新请求的身份（131072）不同** ⇒ 是我在 `7adffbe9` 加的身份判据把它挡住了。

原因：`owner_stamp(sequence)` 取的是**请求当前占用槽位**的 stamp。取消后该 lane 的 continuation
被置为 `Catalogued` 并退役（generation 递增），下一个请求分配到的是另一个槽 ⇒ stamp 不同，
`can_serve` 正确地拒绝了。**身份语义本身没错（它挡住了缺陷 #15），错在它算在了错误的粒度上**：
字节属于的是"**目录里那一条 continuation**"，不是"当时正在跑的那条请求"。

修法（一处语义调整，不是打补丁）：把 stamp 附着在**目录条目**上——发布 checkpoint 时把
`owner_` 设为被发布 continuation 的 `(slot, generation)`，复用时由**源条目**带过来（`request_plan`
已经在拿 `source`，`owner_stamp(*source)` 就是它），`enter_request` 用这个源 stamp，而不是新请求的槽。
这样：同一会话续聊（stamp 相同）、取消后续聊（源条目 stamp）、A→B→A（B 进入时换主人、A 回来时源条目
的 stamp 与空间不符 ⇒ 正确地冷跑）三种情况都落在同一条规则上。

**这是个语义决定，按纪律先停下报告**：它改变了"谁能用这些字节"的判据来源（请求槽 → 目录条目），
并需要 `request_plan.cpp`/`prefill.cpp`/`kv_store.h` 三处一起改。

### 19.6 身份已改到"目录条目"（已提交），取消用例仍未命中：下一处不是身份

改动与验证（`NINFER_STREAM_DEBUG=1`，测试全过，rc=0）：

- 发布 checkpoint 时把空间的 `owner_` 设为**该目录条目的** `(slot, generation)`；复用请求带**源条目**的
  stamp 进入（`request_plan.has_source` → `continuation_stamp(source_index, source_generation)`）。
- 证据：同一 prompt 两次的 trace 现在显示 `source=196608 reuse=4 base=900`（此前 `source=0`），说明
  源条目身份确实传到了空间侧。

但**取消后的请求仍然 `reuse=0 source=0`**：

```
[stream] retained endpoint frontier=19924 ledger=19925 kv_valid=19924        ← 端点已发布
[stream] request starts: owner=131072 source=0 reuse=0 base=0 space_pages=312 space_owner=65536 retained=19924
```

`source=0` 意味着 **规划器根本没有给出候选**（不是身份拒绝——身份拒绝会先选中再 `return std::nullopt`）。
所以下一处要查的是**候选选择**：salvage 出来的端点被目录记为可复用条目了吗、
`matched_prefix_tokens`/候选筛选是否把"取消留下的端点"排除在外。
这条线索比继续改身份更可能命中，下一步就从 `request_plan` 的候选构建读起。

### 19.7 身份已不是瓶颈：取消后的端点根本没进候选（已在代码里定位到层）

开了 `NINFER_MAT_DEBUG`（`cdbg_log` 的门槛）后，取消场景**没有任何 `[candgen]` 行**：

```
[stream] retained endpoint frontier=19924 ledger=19925 kv_valid=19924
[stream] request starts: owner=131072 source=0 reuse=0 base=0 space_pages=312 space_owner=65536 retained=19924
```

`[candgen]` 只在 `resource_manager.h` 的私有候选循环里打印（`inspect_admission` 之后）。没有它意味着
**候选循环没有为该请求产出任何候选键**：`candidate_keys` 由
`base.context_cache().opportunities` 生成（`resource_manager.h:419-426`），而 `prefix_index_` 的每一项
以 checkpoint 的 `shortlist_key` 索引（`rebuild_prefix_index`，`:2427-2460`）。

⇒ 结论：取消留下的端点 frontier（19924，**不在任何栅格上**）与请求侧的机会栅格
（`anchor_prompt_end` 的 prompt_end-1，或 `progress_anchor_tokens` 的整倍数）**没有共同的键**，
因此连候选都构不成。这不是身份问题，也不是深度问题；是**查找键的构造方式**问题。

**下一枪的判据**（两种情况，二选一即可定案）：
1. 若 `[candgen]` 在高频机会（`anchor_prompt_end`）下出现而这次没有 ⇒ 机会集合为空/不含该 frontier；
2. 若给同一请求加 `anchor_prompt_end` 后仍无 `[candgen]` ⇒ 键不匹配（短名单键是按 frontier 精确匹配）。

**修法方向（未实施，待判据）**：请求侧的候选键不能只来自"机会栅格"，还要能表达"我的前缀在某处结束"——
即按**提示前缀**去索引（条目携带其前缀的 shortlist key，请求按自身前缀在所有条目 frontier 处求键）。
这正是 §2.1"完整 continuation"允许的：条目仍然是带 StateImage 的 checkpoint，变的只是**如何找到它**。

---

## 20. R3（多 owner）实施设计：宿主分区 + 设备池共享

这一步的目标只有一条：**A→B→A 的第二次 A 命中**，同时不改变单会话的任何字节路径。
已确认的代码事实（`kv_store.h` 与 `kv_stream_layout.h`）：

- 流式地址的页号就是 token 位置，设备槽号由 `kv_stream_window_slot(layer, page, split)` 算出
  （窗口在前、环在后），宿主页视图是 `host_pages(layer)` 的 `subview(dropped, n)`；
- 所有几何都从"当前深度"派生（`kv_stream_window_start(active_pages, split, layer)`、
  `kv_stream_runs_per_layer`），所以**"深度"是唯一需要变成 per-owner 的量**；
- 设备池的分配在 `StreamedKVSpace` 构造时一次取完；请求之间只改块表内容（`publish_slot`）。

### 20.1 设计决定

1. **设备池不分区**：`--max-concurrency 1` 下同一时刻只有一条 decode，环与窗口只服务当前 owner；
   空闲 owner 只占宿主页与 StateImage。⇒ 槽号算术、拷贝形状、事件顺序**全部不变**。
2. **宿主 arena 按 owner 分区**：每层切成 R 段连续页，`region_base(layer, r) = r * region_pages`。
   页的宿主索引从 `offset` 变成 `region_base + offset`；这是唯一需要改的寻址。
3. **深度 per owner**：`std::vector<std::uint32_t> region_depth_`；`owner_` 变成"当前 region 的主人"，
   另加 `stamp → region` 的小表（R≤4，线性查找即可）。
4. **进入请求**（`enter_request(owner, tokens)`）：
   - stamp 已知 ⇒ 选中其 region，深度取 `min(tokens, depth)`，只前进不重写（页还在宿主）；
   - stamp 未知 ⇒ 选一个空/最久不用的 region，深度归零、写入新 stamp；
   - 设备侧：`advance()` 从 0 到该 region 的深度会按需重发块表（`publish_slot` 已做），
     **不搬字节**（宿主是权威）；窗口里被顶掉的页本来就有宿主副本。
5. **作废语义**：region 被新 owner 占用时，该 region 里所有旧 checkpoint 必须作为
   `ResourcePlan` 的删除集合丢弃（§19.3 的判据），不得静默覆盖。

### 20.2 触点清单（按改动顺序）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `kv_stream_space.h/.cpp` | `regions_`、`region_pages_`、`region_depth_`、`stamp→region`；`host_pages` 的页偏移加 region base；`advance` 接受 region |
| 2 | `kv_store.h` | `enter_request` 传 owner 不变量不变；`mapped_pages`/`entitlement` 返回该 owner 的深度页数 |
| 3 | `request_plan.cpp` | 复用门用"该 owner 的深度"，不是全局深度（`can_serve` 已按 owner 查） |
| 4 | `prefill.cpp` | 不变（仍按 owner 进入） |
| 5 | `startup.cpp` | `host_state_slots` 与宿主 KV 预算按 R 分配；`max_private_continuations` 放开到基线公式 |
| 6 | `pressure.cpp` | 驱逐某 region 时产出计划化的 checkpoint 删除集合 |

### 20.3 验收（顺序即门）

| 门 | 判据 |
|---|---|
| G1 中性 | R=1 时 `equiv 4/4`、`kv_stream_reuse`/`eviction` 全过、251 K `round_ms_median ≤ 41.4×1.01 ms` |
| G2 两会话 | A→B→A：第三次请求 `reused > 0` 且 ids 与冷跑逐 token 相同 |
| G3 作业化 | region 被占用时旧 checkpoint 出现在删除集合里（`NINFER_ADMISSION_DEBUG` 可查） |
| G4 不退化 | `upload_bytes` 变化 ≤1%，`runs_per_layer` 与 R=1 相同 |

### 20.4 风险

- `host_pages(layer)` 的 view 被 `verify_streamed_page` 与 `upload` 共用，偏移漏一处就会读到别的 region
  （`NINFER_STREAM_VERIFY=1` 的字节 oracle 是唯一的兜底，必须在该模式下跑一次 G2）；
- 宿主内存 = R × 4 GiB（192 K），启动检查必须先算账再分配。

### 20.5 第一片已落地（region 视图，R=1 中性，已验证）

`StreamedKVSpace` 现在有 `regions_` / `region_pages_per_layer_` 与 `region_pages(layer)` 视图；
`upload`、`advance` 的降级拷贝、`verify_streamed_page`、`verify_after_attention` 全部改为读
**region 视图**而不是整层视图。R=1 时 region 覆盖整层、偏移为 0，所以这是**改名**而不是行为变化：
`kv_stream_reuse` + `kv_stream_eviction` 2/2 通过（146 s / 92 s）。

### 20.6 下一片的两个已知难点（读代码后的结论，不是猜测）

1. **块表在 owner 切换时必须重发**：`publish_slot` 写的是 `(layer, logical_page)` 的一项，
   而设备窗口/环由当前 owner 独占。新 owner 的页 `[0, depth)` 里，凡是没有在本次 `advance` 里
   "长出来"的页，其表项仍指向**上一个 owner 的槽**。⇒ 切换时必须为该 region 重发全部表项。
2. **窗口要按 region 上传**：`upload()` 只搬"窗口之外的 streamed 页"（`[0, window_start)`），
   因为同一 owner 的窗口本来就在设备上。新 owner 的窗口**一页都不在**，需要一条
   `upload_window(layer)` 把 `[window_start, depth)` 搬进窗口槽。
   代价可算：192 K 时窗口 ≈ 2 905 页/层 × 16 层 × 104 KB ≈ **4.8 GB**，按实测 ~25 GB/s ≈ **190 ms**，
   发生在请求进入时（TTFT），不是每轮。

⇒ 下一片的实现顺序：先写 `upload_window`（单 owner 下也能用：进入时的"全冷"路径等价于它），
用 `NINFER_STREAM_VERIFY=1` 验证字节；再让 `enter_request` 在 owner 不同时调用"重发表项 + upload_window"，
最后才把 `regions_` 从 1 提到 `max_private_continuations`。

### 20.7 region 分区已落地（R=2 可配），A→B→A 仍为 0：下一个瓶颈是**目录容量**，不是 region

已实现并编译通过：

- 空间持有 `region_states_`（每个 region：`owner` / `depth` / `used`）；
- `enter_request`：已知 owner ⇒ 复用其 region 与深度；未知 ⇒ 取空余或最久未用的 region 并回退深度，
  此时 `window_dirty_` 触发 `upload_window`；
- `can_serve` 按**该 owner 的 region 深度**判断；
- region 数来自 `max_private_continuations`（`program_impl.cpp` 构造空间时传入）；
- 修掉一处：`request_plan.cpp` 对"保留源"的 entitlement 扣减在流式下无意义（流式不占池页），
  否则两个私有 continuation 时抛 `retained prefix exceeds its active KV entitlement`。

**实测（`max_private_continuations = 2`，rc=0，无崩溃）**：取消后复用 23 424、重发 900 均保持；
A→B→A 仍 0。trace 指出原因：

```
request starts: owner=131072 source=0 ... space_owner=65536
request starts: owner=196608 source=0 ... space_owner=131072
request starts: owner=327680 source=0 ... space_owner=262144
```

**每个请求的 owner stamp 都是"slot 0 + 递增的 generation"**——即每条新请求都在退役并重占**同一个**
continuation 槽。说明 Program 的 `continuation_capacity` 实际仍是 1（`max_private_continuations = 2`
没有传导到它），于是 A 的目录条目在 B 到来时被退役，A 回来时无条目可查。

⇒ 下一枪：查 `continuation_capacity` 的推导（`program_impl` 构造 `continuation_states` /
`continuation_slots` 时用的容量），确认 `ContextCacheOptions::max_private_continuations` 是否只在
`host_cache_budget` 路径被使用、以及 Engine 侧 `context_cache` 规范化（`model_instance.cpp:443-477`）
是否把它改回 `2 * concurrency`（本机 concurrency=1 ⇒ 2，本应足够）。region 侧不需要再改。

### 20.8 A→B→A 在"全新目录"上仍是 0：候选没被提出，region 与空间都对

把 A→B→A 挪到引擎的第一个动作（目录空、两个 private continuation、两个 region），trace：

```
[stream] request starts: owner=65536  ... space_pages=0   space_owner=0       retained=0     (A)
[stream] retained endpoint frontier=904 ledger=905 kv_valid=904
[stream] request starts: owner=131072 ... space_pages=15  space_owner=65536  retained=904    (B)
[stream] retained endpoint frontier=244 ledger=245 kv_valid=244
[stream] request starts: owner=196608 ... space_pages=15  space_owner=131072 retained=904    (A again)
        └─ source=0, reuse=0
```

- **空间侧正确**：A 的区仍在（`space_pages=15`、`space_owner` 是 B 的区），`retained=904` 也还在；
- **规划器侧 `source=0`**：根本没有提出 A 的条目 ⇒ 不是 region、不是深度、不是身份，而是
  **候选/目录这一层**（A 的条目在 B 发布后不可见，或 `inspect_admission` 被拒且没打印）。

⇒ 下一步（下一 turn 的第一件事）：在 `resource_manager` 的候选循环里打印
`catalog_` 全部条目的 `(slot, state, session, endpoint.frontier, handle?)` 与 `prefix_index_` 的
`(occupied, frontier, slot)`，跑同一场景；这会一次分清"条目被驱逐"与"键没命中"。
在那之前不再改 region 侧代码——它们已经是对的。

### 20.9 A→B→A：A 回来时**候选生成根本没运行**（目录已被清空）

`NINFER_MAT_DEBUG=1` 的 dump（新加的 `[candgen] CAT/IDX`）给出新事实：

```
A:        request starts ... space_pages=0  space_owner=0      （无 [candgen] 行）
B:        request starts ... space_pages=15 space_owner=65536  （无 [candgen] 行）
A again:  request starts ... space_pages=15 space_owner=131072 （无 [candgen] 行）→ source=0
```

- 三次请求**都没有 `[candgen] === episode`**，说明管理器在进入候选循环前就短路了（目录里没有可查的条目）；
- 之后（取消用例）的 episode 打印显示 `CAT slot=0 state=1 endpoint=1`、`IDX slot=0 frontier=904`，
  即目录在**有**条目时是正常的、索引也对。

⇒ 因此 A 的条目在 **B 被接纳时就被清掉了**，而 `private_catalog_capacity` 本应是 2
（`make_resource_manager` 用 `options.context_cache.max_private_continuations`，测试给 2）。
下一步的判据：在管理器的"无可查条目→冷计划"短路点打一行原因
（`catalog_count_`、`occupied`、`free`、是否 `cache_enabled_`、是否本次请求带 `publish_continuation`），
以及 B 接纳时是否触发了 supersede/recency 驱逐。**在拿到这两行之前不改容量或驱逐逻辑。**

### 20.10 真凶找到：**B 的接纳把 A 的条目驱逐了**，尽管目录还有空槽

`[admission] inspect` 一行（新加）把整个 A→B→A 链条显示清楚：

```
A:        enabled=1 catalog=2 ... opportunities=1   → CAT slot=0 endpoint=1 anchors=1 (frontier 904 / anchor 900)
B:        enabled=1 catalog=2 ... opportunities=0   → （同一 episode 里 CAT 还是 A 的条目，说明 B 进来时 A 在）
A again:  enabled=1 catalog=2 ... opportunities=1   → CAT slot=0 state=0, slot=1 state=0   ← 两个槽全空
```

即：**B 被接纳时 A 的条目被释放**，而 `catalog_count_ = 2`、槽 1 一直空着、`host_state_slots=4` 也够。
这只能是"**冷请求无放置方案 ⇒ 丢条目**"那条补丁（`drop_continuations_the_head_cannot_use`，
§10.3）在**还有空槽时**就触发了。

⇒ 结论（重要）：A→B→A 的失败**与 region、身份、键都无关**，而是那条**驱逐补丁过于激进**。
它当时是为了修 C15/429（唯一私有槽被占用时冷请求进不来）而加的，条件是"这条请求根本没有放置方案"，
而现在的现象说明：即使有空槽，规划仍判为"无放置方案"（原因未打印）。

**下一枪**：在 `engine_core.h` 的 no-plan 分支打印`inspect` 的 readiness/blocked 原因与
`catalog` 占用情况，确认是"Program 侧无 placement"还是"manager 侧无 slot"，再决定：
- 若是 manager 侧误判 ⇒ 修判据（有空槽就不该丢）；
- 若是 Program 侧（后续槽位/状态账） ⇒ 修账，而不是丢缓存。

**在这条链条上还有一条已验证的好消息**：取消用例（prefill 中途取消）在 region=2 下依然
`reused 23424`、逐 token 正确，说明 region 与 `upload_window` 没有破坏既有能力。

### 20.11 排除驱逐补丁：根计划正常，A 的条目仍消失

在管理器里加了 `[admission] root plan refused` 之后，整条 A→B→A 链条**一次都没有打印**它：

```
A:        enabled=1 catalog=2 opportunities=1
B:        enabled=1 catalog=2 opportunities=0      ← 根计划正常，能冷跑
A again:  enabled=1 catalog=2 opportunities=1      ← 此时 CAT 里两个槽都是空
```

⇒ **驱逐补丁没有参与**（它只在"完全无放置方案"时才跑）。A 的条目是在 B 的**接纳过程中**
（`plan_materialization` / 压力规划）被释放的，而这不是"无放置"分支。

**下一枪（唯一未打印的环节）**：在 `plan_materialization` 与压力规划的释放点打印
"释放 slot/owner、原因、释放后的 free 统计"。判据：
- 若是**内存确实不够**（A 的 StateImage + arena 的 region 与 B 的需求冲突）⇒ 这不是 bug，
  而是宿主预算不足，需要按 `host_state_slots`/region 数把账抬高；
- 若是**判据误算**（有空槽/足够状态槽却驱逐）⇒ 修判据。

这条一旦定案，A→B→A 要么直接变成"加预算即可"，要么是一个小修。region 与身份侧不需要再改。

### 20.12 定案：就是"无放置方案 ⇒ 丢条目"补丁驱逐了 A

给三条释放路径各加一行（`[evict] drop-unusable` / `release-owner`）后，A→B→A 的 trace 直接指认：

```
A:        enabled=1 catalog=2 opportunities=1
B:        enabled=1 catalog=2 opportunities=0
[evict] drop-unusable slot=0 frontier=904      ← B 的接纳把 A 丢了
A again:  enabled=1 catalog=2 opportunities=1   ← 目录已空
```

`drop-unusable` 只在 `engine_core.h` 的"**本次请求没有任何可选放置**且没有活跃 lane"分支里触发
（`drop_continuations_the_head_cannot_use`），丢弃后再 inspect 一次。因此：

- **目录容量不是原因**（B 有根计划、槽 1 空着）；
- **补丁本身按设计工作**，它被触发说明"B 完全没有可选放置方案"；
- 唯一还没打印的是 **为什么 B 没有可选放置**（`inspect` 返回的 readiness / choice 数）。

⇒ 下一枪：在 `engine_core.h` 的 no-choice 分支打印 `readiness` 与候选数，
在 `resource_manager` 的选择过滤处打印每个候选被拒的理由（成本/可行性）。
判据：若是"流式地址在 Program 侧被判定不可放置"（例如仍按池页记账），那是 Program 侧的一处特例；
若只是"需要状态槽/宿主状态镜像"，则抬预算即可。

**已验证不受影响**：取消用例仍 `reused 23 424`；region=2 下无崩溃。

### 20.13 B 的结果是 `readiness=2`（TemporarilyBlocked），不是"计划失败"

```
[admit] no choice: readiness=2 active=0     ← B
[evict] drop-unusable slot=0 frontier=904
[admit] ready: choice=1                     ← 丢掉 A 之后同一个 B 就能进了
```

- 冷请求 B **有根计划**（§20.11 已证），但管理器给出的是 **TemporarilyBlocked**，且当时没有活跃 lane
  ⇒ 永远不会自己解除 ⇒ 引擎走"丢条目再试"的分支；
- 丢掉 A 之后，**同一个 B 立刻 `ready`**。

⇒ 这说明阻塞来自**"已有条目 + 新请求"的组合**（状态镜像/传输/放置的账），不是目录容量、不是 region。
下一步（最后一处未打印）：在 `inspect` 的候选过滤里打印每个候选的
`MaterializationPhysicalStatus` 与阻塞原因（`TemporarilyBlocked` 的判据），定位到具体是
"没有可用状态槽/镜像"还是"传输需求被排队"。

**已验证不受影响**：取消用例 `reused 23 424`；region=2 无崩溃；两个 fast 用例全过。

### 20.14 定案的数字：端点 StateImage 赖在**唯一的设备状态槽**上

`[mat-debug] peak / held / limits` 三行（`build-verify`，`private=2`、`host_state_slots=4`、
`device_state_slots=0`）：

```
peak  (要的): lanes=1 states=1 main=0 backend=0 host_kv=0
held  (已占): lanes=0 states=1 main=0 backend=0 host_states=1 host_kv=0
limits(上限): lanes=1 states=1 main=47403 backend=0 host_states=4 host_kv=0
```

推理链（每一步都有数字）：A 目录化后仍占 `device.state_slots = 1`；流式形状的上限
`limits.device.state_slots = max_concurrency + device_state_slots = 1 + 0 = 1`；B 的 peak 要 1 个
⇒ `1 + 1 > 1` ⇒ `physical_peak_fits` 为假 ⇒ `revalidate` 返回 stale ⇒ `inspect` 返回
`TemporarilyBlocked` ⇒ 引擎"无放置且无活跃 lane"⇒ 丢 A ⇒ B 立刻 `Ready`。

⇒ **流式形状下，已目录化的端点必须住 host shelf**，不能占设备槽：
- `host_state_slots = 4` 已经配了，`host_states = 1` 也在计数（说明 host 副本存在）；
- 但设备那份没被释放，于是把唯一的设备槽占死（那 1 个槽是给**在飞请求**的 active 镜像的）。

**修法（下一处代码）**：给 `StateImageStore` 加一个 `demote_checkpoint_to_host(handle)`
（分配 host slot → 设备→宿主拷贝 → 释放设备槽 → residency 变 HostOnly），并在
`publish_active_continuation`/`retain_prefill_progress` 的 `freeze` 之后、当
"设备槽没有余量给下一个请求"时调用它。

**为什么这条路安全**：HostOnly checkpoint 在恢复路径上已有支持——
`checkpoint_recovery.cpp:115` 对 `HostOnly` 会追加一条 `HostToDevice` 传输需求；
`adopt_host_image`（`state_store.h:265`）已经能造出同形状的镜像（session restore 用的就是它）。

**另一条路（不推荐）**：把 `limits.device.state_slots` 抬到 `max_concurrency + private`。
它会让"设备镜像与流式窗口并存"，而 C13 已实测该组合从第 3 个 decode 轮起读到旧 state
（`mtp accepted 0/9`、第 3 轮 argmax 变 EOS）——正是守卫 `device_state_slots == 0` 的由来。

### 20.15 demote 的第一次落点失败（已回滚），正确落点在管理器侧

实现过的内容（已 `git revert`，`3a89a5ea`）：
- `StateImageStore::demote_checkpoint_to_host(handle, streams)`：分配 host slot → `device_->copy_to_host`
  → `return_device_slot` → residency 变 HostOnly（与 `drop_device_replica` 同一套归还路径）；
- 在 `publish_active_continuation` 的 `freeze` 之后调用。

实测 `rc=1`：`FATAL: KV truncate frontier is beyond the sequence`。

⇒ 落点太早：发布事务在 `freeze` 之后仍要用**设备副本**完成前沿/截断收尾
（`truncate_inactive_prefix`、`rebuild_work`、`refresh_state_views` 的顺序），设备槽被提前归还即失败。

**正确的落点（下一片）**：把"目录条目住 host"放在**拥有一条目录条目的那一层**——
`resource_manager` 在把条目写入目录之后（`publish_continuation` / `publish_salvaged_continuation`
的尾部）调用一次 `program.demote_catalogued_state(handle)`；Program 内部实现为
"仅流式形状 + role==CheckpointImmutable + 无 pin → demote"，非流式形状是 no-op。

好处：发布事务不感知流式；demote 发生时该条目已经无人再写，只等被恢复读取；
HostOnly 恢复路径已存在（`checkpoint_recovery.cpp:115` 的 `HostToDevice` 需求）。

### 20.16 两次落点都失败，结论：HostOnly 端点需要改状态库的恢复/截断路径

| 落点 | 结果 |
|---|---|
| `publish_active_continuation` 的 `freeze` 之后 | `FATAL: KV truncate frontier is beyond the sequence` |
| 管理器侧（条目写入目录之后，`demote_catalogued_state`） | 同上 |

⇒ 不是时序问题：**回退前沿时走的 `truncate_inactive_prefix` 仍需要设备副本**，与"条目住哪"无关。

因此 A→B→A 的正确修法是三选一，且都必须改状态库而不是调用点：

1. **让 HostOnly 端点可恢复/可截断**：`demote` 之后，恢复路径先用 `HostToDevice` 上传再截断
   （`checkpoint_recovery.cpp:115` 已有这条需求，缺的是"恢复后仍按 HostOnly 记账"）；
2. **给流式形状一个额外的设备状态槽**（`limits.device.state_slots = max_concurrency + private`），
   但 C13 实测该叠加会让 target 从第 3 个 decode 轮读到旧 state —— **已否决**；
3. **端点留在宿主但恢复时临时占槽**：`demote` 在恢复前不释放设备槽，只在"空闲且无人引用"时释放，
   即让设备槽成为一个**按需租约**而不是常驻。

当前树已恢复到最后一个绿色状态（`ddeacbeb`，编译通过、复用/驱逐用例通过）。

### 20.17 失败点的精确读法（下一次带 demote 重跑要打印什么）

两处抛出都是同一条断言（`kv_store.h:1782`）：

```cpp
if (frontier > address.committed_frontier)
    throw std::invalid_argument("KV truncate frontier is beyond the sequence");
```

即"要截断到的前沿超过了该地址已提交的前沿"。demote 本身不碰 `committed_frontier`，所以要么
(a) **恢复路径**（HostOnly → active）没有把目标地址的前沿推到 reuse base，随后 `trim_sequence_kv` 拿
base 去截断一个 `committed_frontier == 0` 的新地址；要么 (b) demote 之后 `state.state.read` 的角色/前沿
在摘要里读到 0，导致下游按 0 记账。

**分辨方法（一次运行即可）**：在该断言前打印
`frontier`、`address.committed_frontier`、`address.page_count`、`stream_space_->active_pages()`、
以及调用栈来源（prefill 的 `trim_sequence_kv` 还是 finish 的收尾）。三者一对就能定 (a)/(b)。

在拿到这三个数之前不再改代码——前两次都是"看起来该在这里"的调用点，实际缺的是别处的记账。

### 20.18 两个 region 时失败的具体数字（修复入口）

```
[stream] request starts: owner=65537 source=0 reuse=0 base=0 space_pages=15 space_owner=65536 retained=904
[stream] truncate beyond: frontier=900 committed=256 pages=0 space_pages=4
```

- 请求以 `source=0`（规划器判为冷）进入，**stamp 与空间当前主人不同** ⇒ 我的 `enter_request` 取一个空
  region、深度归 0（`space_pages=4` = 第一块 256 token）；
- 随后上游仍按"已恢复到 900"的路径截断 ⇒ `900 > committed 256` ⇒ 抛。

⇒ 入口语义缺一条：**带着非零 `tokens` 进入的请求，必须拿到一个深度至少到 `pages(tokens)` 的 region**
（或者与它的源条目同 region），否则深度会被清零、前沿与实际不符。
这条要在"规划器给出 source 时"才安全（否则等于把别人的字节让给新主人），所以先解决
`source=0`：当目录里存在匹配的 checkpoint 时，规划器为什么没给出 `source`（§20.12/20.13 的
`opportunities=0` 与峰值账已能看到一部分）。

### 20.19 多会话复用的最新定位：走的是"已恢复"分支，状态由宿主镜像填充

两 region 下的 trace（一次运行）：

```
[stream] anchor restore: base=900 valid=900 fork_required=1 checkpoint=4@900 role=2 residency=2
[stream] activate: fork_required=1 residency=2 refs=1
[stream] window upload: region=0 active=15 start=0 count=15 resident=2874
[reuse] second A reused 900 tokens
[reuse] first difference at 0: got 220 cold 248046
```

- `role=2 CheckpointImmutable`、`residency=2 HostOnly`：锚点镜像确实被 demote 到宿主；
- `activate` 走进的是 `transaction.state_restored` 分支（因此**没有** `begin_fork` 打印），
  即目的地由"恢复路径"（`checkpoint_recovery` 的 HostToDevice 需求）填充，而不是 fork 拷贝；
- KV 侧已排除：窗口上传搬了 15 页（=900 token），且 `start=0`（全部在窗口内，没有走环）；
- **第一个生成 token 就不同** ⇒ 问题在"从 HostOnly 镜像恢复到设备"的那条路径（分段拷贝/布局），
  不在 KV、不在 region、不在窗口。

**下一步（唯一未验证的假设）**：`state_restored` 分支填充的目标镜像内容与源镜像不一致
（分段 `copy_from_host_segments` 的段序/布局，或恢复了部分段）。验证方式：在该分支后立刻把
目标镜像拷回宿主，与源镜像逐字节比较（`memcmp`），不一致则打印第一个不同的偏移与段号。

### 20.20 状态已逐字节核对：不是镜像问题

```
[stream] anchor restore: base=900 valid=900 fork_required=1 checkpoint=4@900 role=2 residency=2
[stream] restored state matches host: 1        ← 目标设备镜像与宿主源镜像 memcmp 相同
[stream] window upload: region=0 active=15 start=0 count=15
[reuse] second A reused 900 tokens
[reuse] first difference at 0: got 220 cold 248046
```

⇒ 已排除：锚点镜像内容、KV 窗口内容（15 页全部回迁）、region 归属、身份戳。
第一个 token 就不同 ⇒ 差值来自**镜像与 KV 之外**的某个上下文量。下一批候选（按可能性排序）：

1. 复用请求的 **positions / rope** 是否从 0 开始（而不是从 base 起）；
2. `sequence.ledger` / `prefix_digests` 在 `resize(base)` 后的内容；
3. 生成首步的 **采样输入**（hidden）是否被前一次会话的 `tail_hidden` 污染；
4. 流式的 `content_epoch` / 表项发布是否让首轮读到旧槽（窗口已回迁，但**环**在 active < resident 时不参与，
   所以这一项只在 active > resident 时成立）。

排查方式：在 decode 的第一步打印 `positions[0..3]`、`ledger.size()`、`tail_hidden_valid`，
并在两臂各打印一次，一次运行即可定案。

### 20.21 复用请求携带的上下文量全部一致，差值只能在注意力输入

```
[stream] setup: reuse=4 base=900 prompt=901 text_kv=900 ledger=900 tail_hidden=0 rope_delta=0 space=15
（对照冷跑：  reuse=0 base=0   prompt=901 text_kv=0   ledger=0   tail_hidden=0 rope_delta=0 space=0）
```

- `text_kv=base=900`、`ledger=900`、`space=15` 全部自洽；
- 状态镜像逐字节等于宿主源（§20.20）；
- 窗口 15 页全部回迁（§20.20）；
- 环不参与（active 15 < resident 2874）。

⇒ 剩下的只能是**注意力/前向的输入**本身：同一份状态 + 同一份 KV + 同样的位置，却给出不同 logits。
下一批探针（一次构建全加）：

1. decode 第一步打印 `hidden[0..3]` 与 `logits` 的前 4 个值（两臂对比，定位是 hidden 还是采样）；
2. 打印该步读到的页范围与槽号（`slot_of_page`）前 3 项，确认没有读到别的槽；
3. 打印 `invariant_attention` 下的 key 分段边界，确认分段与冷跑一致。

### 20.22 首步 logits 大幅不同（不是数值噪声）

有序 dump（`cudaMemcpyAsync` + `cudaStreamSynchronize`，否则读到的是上一步的旧值——第一次 dump 就因为没排序
而出现"两个不同请求打印相同 logits"的假象）：

```
A 首次  decode: hidden=3f923ea5,…  logits=40347402,40bdca2d,402f3f05,4043b86e   (≈ +2.8, +5.9, +2.7, +3.1)
A 复用  decode: hidden=bf853e96,…  logits=bf370290,c0570252,bf71e3a4,c01ed772   (≈ -0.7, -3.4, -0.9, -2.5)
```

符号与量级都不同 ⇒ 不是精度问题，而是**前向拿到的量本身不对**（而状态镜像已逐字节等于宿主源）。
下一个分叉点：复用的这一请求里，`kv_rows`（执行表行）与 `state_sources/destinations`
（状态槽）是否指向了正确的槽。**最可疑**：demote 把端点的设备镜像还给了池，锚点恢复的目标槽
可能与 `ordinary.state_sources`（从槽号读 state）不一致 ⇒ 前向读到一个未初始化/别的内容的槽。

探针（下一构建）：在 decode 首步打印 `state_sources[0]`、`state_destinations[0]`、`kv_rows[0]`
的原始值（两臂对比），并打印 `state_store` 中该槽的 `role/residency`。

### 20.23 槽号一致、前沿一致，差值仍在 KV 内容或时序

```
A 复用  setup: reuse=4 base=900 prompt=901 text_kv=900 space=15
        decode: state_source=0 state_destination=0 kv_row=0 logits=bf370290,…（负值）
A 冷跑  decode: state_source=0 state_destination=0 kv_row=0 logits=40347402,…（正值）
```

两臂的槽号、行号完全相同，状态镜像逐字节相同，前沿相同（prefill 末尾 `text_kv_valid = prompt_tokens`）。
⇒ 差值只剩两处：

1. **KV 内容**：窗口回迁的字节是否真的落到了这 15 个槽（上传在 `compute_streams`，decode 在
   `device.stream`；若两者不是同一条流，首步会读到回迁前的旧字节）；
2. **页 14 的 committed columns**：锚点在 900（页 14 只有 4 列有效），而 A 后来把 token 900 写进了第 5 列；
   恢复时若掩码按"5 列有效"计算，注意力的可见长度就比冷跑多一个 token。

下一枪（两件事一起做）：
- 在 decode 首步前 `cudaStreamSynchronize(device.stream)`，排除上传/解码的时序竞争；
- 打印 `committed_frontier` 与页 14 的 `committed_columns`（两臂对比）。

### 20.24 窗口回写已实现，但 region 记账自相矛盾（下一步：单一真源）

新增 `flush_window`：换上下文前把**出方**窗口 `[window_start, depth)` 回写到它自己的 region
（窗口是缓存，页只有跌出窗口才落宿主；否则新上下文一写就把唯一副本覆盖了）。这条是**必要的**，
实测也确实触发：

```
[stream] window flush: region=0 depth=15
[stream] window flush: region=0 depth=4     ← 同一 region 深度先 15 后 4
[stream] window flush: region=1 depth=15    ← region 1 本该只有 4 页
```

⇒ `active_` 与 `region_states_[current_region_].depth` **两份真源**在互相打架，导致 flush 用错深度、
把 A 的 15 页覆盖成 B 的 4 页。答案因此仍未对上（首 token 从 220 变成 357，方向变了但不对）。

**下一步（明确的重构，不再打补丁）**：删掉 `active_`，让
`active_pages() == region_states_[current_region_].depth` 成为**唯一真源**；
`advance()` 只写 region 的 depth；`upload_window`/`flush_window`/`window_start` 全部从它派生。
触点约 12 处（`kv_stream_space.cpp` 内），改完再跑 A→B→A。

---

## 21. 攻前提：单窗口换不上两个上下文（按 ub-mode 记录）

### 21.1 前提与失衡计数

**前提（一句话）**：*一个设备窗口可以在多个上下文之间换收换放，从而支持多 owner。*

**失衡计数**：两次修法（`window_dirty_` 延迟标志；`window_source_/switch_target_` 单条记录）死在同一道门——
换上下文后回来的会话首 token 与冷跑不同。持有"窗口装的是谁"这一事实的角色有 5 个：
进入（prefill）、逐块 `advance`、decode 读取、发布（demote）、截断（trim 重入）。
只有"进入"知道意图，其余四者读写同一状态且都能改它 ⇒ **该事实没有单一写入者**。

**移除分配**：不是继续修交换，而是让"多 owner + 单窗口"**不可表达**。

### 21.2 动作与结果

- 删除换收换放机制（`flush_window`/`switch_window`/`window_source_`/`switch_target_`），空间回到
  `a0134d82` 的单窗口行为（`active_` + `window_dirty_` + `upload_window`）；
- `--kv-stream` 在启动期拒绝 `max_private_continuations > 1`：
  "--kv-stream keeps one device window: set --max-private-continuations 1"；
- 验收：`nvfer_qwen3_5_kv_stream_reuse_real_test` + `..._eviction_real_test` **2/2 通过**（150 s + 92 s）。

### 21.3 多 owner 的正确形态（下一单元，不是补丁）

**每个 region 自己的窗口与环**：设备池按 region 切分，`plan_kv_stream_split(budget / regions)` 各算一次，
`slot_of_page(layer, page)` 用**本 region 的 split**；块表按 region 分别发布。
这样任何时刻都不需要"换"，也就不存在 flush/load、`window_source_`、`window_dirty_` 这些状态。

代价（先算清再动手）：
- 每个 region 的环至少要 `ring_runs ≥ 1`（`run_pages=128` ⇒ 128 页/区），R=2 时窗口各少 128 页/层；
- `--kv-capacity auto` 的池必然够，但**每 region 的窗口变小**，251 K 的每轮字节会上升；验收要量。

### 21.4 本轮动作日志（ub-mode: Keep a Log）

| # | 动作 | 原因 |
|---|---|---|
| 1 | 回滚到 `a0134d82` 的空间实现 | 移除"窗口换上下文"这条错误分配 |
| 2 | 新增私有 continuation 上限守卫 | 让非法形状不可表达 |
| 3 | 测试回到 `private=1` | 只跑被证明的配置 |
| 4 | 复用/驱逐两用例复跑 | 证明回滚没有破坏既有能力 |
| 5 | 记录 21.1–21.3 | 前提、计数、下一单元设计与代价 |

### 21.5 一次被回滚的尝试（记录，避免重犯）

把 `mapped_pages` 改成"返回 arena 深度"⇒ 两个用例同时失败（`kv_stream_reuse`、`kv_stream_eviction`）。

**根因**：`mapped_pages` 回答的是"这份地址空间**拥有**多少逻辑页"（流式为 0，因为页归 arena），
而我想让它回答"上下文**覆盖**到多深"。这是**两个不同的问题**，共用一个访问器 ⇒ 所有既有调用方
（`logical_page` 遍历、checkpoint 覆盖检查、快照写入）立即读到不存在的页。

**正确的域建模**（下一单元）：保持 `mapped_pages` 语义不变，新增独立访问器
`covered_tokens(handle)` / `covered_pages(handle)`：流式返回 arena 深度，驻留返回 `committed_frontier`。
会话导出/导入用它，而不是改写 `mapped_pages`。已被 `git revert`（`ce78f131`）。

### 21.6 会话单元的第一条事实：没有公开的 Engine 会话 API

`save_continuation` / `restore_continuation` 只存在于 `Program`（`program.h`），**不在** `include/ninfer/engine.h`。
会话的唯一入口是 serve 的 `POST /slots/{id}?action=save|restore|erase`，而它今天对 `--kv-stream` **直接拒绝**
（`--kv-stream keeps its context in the streamed arena: --slot-save-path is not available`）。

⇒ 因此该单元的失败测试**已经是现成的**：`tools/bench/kv_stream/parity.py` 的 **P5**
（save → erase → restore → 续聊，要求 ids 与不中断逐 token 相同），今天会以"拒绝"这一明确原因失败。

**下一单元的执行顺序（ub-mode: 先失败测试，再最小改动）**：

1. 保留 P5 作为判据（它已能跑，只是红在拒绝上）；
2. 去掉 serve 的拒绝，改为真正的校验（`--kv-stream` + 槽路径应被接受）；
3. `session_snapshot` 写侧：新增 `covered_tokens(handle)`（独立访问器，**不动** `mapped_pages`），
   流式分支按 `region_pages(layer)[0, pages)` 拷出，布局仍用 `plan_host_kv_page_layout`；
4. 读侧：把快照写回 arena 的对应 region，并把 depth 设为快照前沿；
5. 跑 P5，要求 save→erase→restore→续聊逐 token 相同。

第 3–4 步是这条链上唯一的新代码，其余是接线；每一步之后都要跑 P5 与两个 fast 用例。

### 21.7 会话单元：红在预期的地方了

去掉 serve 的拒绝后，`parity.py --only P5` 逐步收敛到真正的原因（每一步都是先修"测试发错了请求"）：

| 现象 | 原因 | 处置 |
|---|---|---|
| `invalid_slot: slot 1 is outside this server's 1 slots` | 槽号零基，harness 用了 1 | 改用 `/slots/0` |
| `invalid_filename` | 槽动作要求 `filename` | 请求体带 `{"filename": "parity-session.ninfer"}` |
| **`retained session Text KV coverage is inconsistent`** | `session_snapshot` 用 `mapped_pages`（流式为 0）判覆盖 | **这就是本单元要修的** |

⇒ TDD 到达"见它按预期失败"。下一处改动明确：

1. 新增 `KVAddressSpaceStore::covered_tokens(handle)`（流式 = arena 深度；驻留 = `committed_frontier`）；
2. `session_snapshot` 写侧：覆盖判据改用 `covered_tokens`，流式分支按 `region_pages(layer)[0, pages)`
   拷出，布局仍用 `plan_host_kv_page_layout`；
3. 读侧：写回 arena 的该 region，并把 depth 设为快照前沿；
4. 再跑 P5，要求 save→erase→restore→续聊逐 token 相同。

### 21.8 导出侧的结构事实：快照是**页主序**，不能按层整段拷

读 `session_snapshot` 的写侧后确认：

- 快照页载荷是 `page * text_page_stride` 一页一条记录，而一页记录里包含**全部层**的 K/V/scale 平面
  （`plan_host_kv_page_layout` 的 planes 覆盖所有层）；
- 因此流式导出**不能**按 `region_pages(layer)` 整段 memcpy（我试过一版，是错的，已回退）；
  必须对每个 page、每层，按 `layout.planes[]` 的平面偏移把该层该页的字节拷进页记录。

正确形状（下一段代码）：

```cpp
for (page : [0, text_pages))
  for (layer : layers)
    for (plane : layout.planes_of(layer))
      memcpy(page_record + plane.offset, region_pages(layer).subview(page,1) + plane.local_offset,
             plane.bytes);
```

- 覆盖判据：`text_pages <= text_stream_space->active_pages()`（而不是 `mapped_pages`）；
- 读侧对称：把页记录拆回各层 region，并把该 region 的 depth 设为快照前沿；
- 验收仍是 P5（save→erase→restore→续聊逐 token 相同）。

这一步是"小步但结构正确"的典型：先把**数据形状**写对（页主序 + 平面偏移），再写循环。

### 21.9 一个必须先用测量回答的问题（不再靠读代码猜）

快照的文本载荷是 `text_page_stride * text_pages`，而 `copy_address_pages` 按 `logical_page(address, p)`
逐页走。要写流式导出，必须先确定**一条页记录覆盖一层还是全部层**。代码里有两种迹象：

- `startup.cpp:348-362` 的 DFlash 几何把每层平面**追加**进同一个 `KVPageGeometry` ⇒ 池的一页记录含全部层；
- 但 M1 的实测账（`rk8v4` 页字节 104448、262144 上下文 ≈ 6.38 GiB = 16 × 4096 × 104448）说明
  **104448 是一层的页**，流式空间的 `host_pages(layer)` 也按层切片。

⇒ 两者不可能同时成立，必须以一次**测量**定案（而不是继续读）：

```
printf("%zu %zu %zu\n", text_layout.page_stride, text_stream_space->host_pages(0).page_count(),
       text_stream_space->layers());
```

- 若 `page_stride == 104448`（一层）⇒ 快照必然是 **layer-major 拼接**（每层一段 `pages × stride`），
  流式导出就是按层连续拷，我先前那版整段 memcpy 其实方向对、只是偏移算错；
- 若 `page_stride ≈ 16 × 104448` ⇒ 页记录含全部层，导出必须 page × plane 拼装。

下一轮第一件事：加这三行打印并跑一次（≈1 分钟），按结果写导出/导入循环。

### 21.10 测量结果与一个仍未解的矛盾（阻塞会话单元）

两次实测：

```
[stream] layout: page_stride=71680  planes=4 layers=16 pages_per_layer=4096   (rk4v4)
[stream] save:   text_pages=23 mapped=0 committed=1447 stride=104448 layers=16 (rk8v4)
```

- **页记录 = 一层的页**（71680 / 104448 与 M1 的每层页字节一致，`planes=4` 是该层的 K/V/scale）；
- 快照载荷 = `text_page_stride * text_pages` = 23 条一层页记录，**按此读法只覆盖一层**。

这与"驻留模式的会话保存/恢复是可用功能"相矛盾。两种可能，必须由**读侧**判定（它才是载荷的消费者）：

1. 读侧对每层各调一次、偏移按层推进（写侧的 `copy_address_pages` 我可能只看到了文本平面的一部分）；
2. 或 `logical_page(address, p)` 的 p 其实是**跨层**编号，`text_pages` 只是逻辑前沿换算的一半。

⇒ **下一步不是继续写导出，而是读 `restore_continuation` 的载荷消费**（`session_snapshot.cpp:247` 之后），
用它的读法反推写法的层结构；在此之前不动导出代码（本轮已两次因形状猜错而回退）。

### 21.11 读侧给出的形状，与仍差的一环

读侧的消费方式（`session_snapshot.cpp:1031-1045`）：

```cpp
destinations = { addresses.physical_page(*address, page) for page in [0, page_count) };
pool.copy_from_host(payload, layout, destinations, streams);
```

- 载荷是 `page_count` 条 `layout.page_stride` 记录，逐条写进该地址的**物理页**；
- 实测 `layout.page_stride = 104448`（rk8v4）与 M1 的**每层页字节**一致，`planes=4` 是该层 K/V/scale；
- 因此"一条记录 = 一层的页"在读侧同样成立。

⇒ 若地址的物理页是"一层的页"，会话就只覆盖一层；若物理页是"页组（含全部层）"，`page_stride` 就不该是
一层字节。**二者必有一错，判定点在 `DeviceKVPagePool`**：`physical_page(handle)` 返回的到底是
"一层的页"还是"含 16 层的页组"，以及 `copy_from_host` 是否在内部循环层。

**这是会话单元的阻塞项**，下一步只做一件事：读 `DeviceKVPagePool::copy_from_host` 与
`physical_page` 的字节含义（必要时加一行打印 `pool.geometry().planes.size()` 与单页字节数），
在那之前不写导出/导入代码——本轮已因形状猜错回退两次。

### 21.12 当前树的验证（本轮末）

- `ninfer_qwen3_5_kv_stream_reuse_real_test`：**Passed**（126.5 s）
- `ninfer_qwen3_5_kv_stream_eviction_real_test`：**Passed**（76 s）
- 本轮产品代码改动：仅新增两处调试打印（布局形状、保存覆盖数字）；serve 放行槽目录（上一轮）。

### 21.13 池一侧的读法，与一个一分钟能定的实验

`DeviceKVPagePool::copy_from_host`（`paged_kv_cache.cpp:967-990`）要求 `host.geometry == geometry()`，
并按 `planes_.size()` 循环平面；`copy_host_run` 用 `host.page_stride` 步进。实测该几何的
`page_stride = 104448`、`planes = 4` ⇒ **池的一页就是一层的一页**。

于是出现一个硬矛盾：一个 1447 token 的上下文，`text_pages = 23`、载荷 = 23 × 104448 ≈ **2.4 MB**，
而它的真实 KV 总量 ≈ 1447 × 1632 B/token × 16 层 ≈ **37.8 MB**。

**一分钟定案的实验**（不依赖再读代码）：用**驻留模式**（不带 `--kv-stream`）跑一次
`POST /slots/0?action=save`，然后 `ls -l` 那个文件：

- 文件 ≈ **2.4 MB** ⇒ 会话载荷只有一层，驻留模式的会话恢复本身不完整（那是一个独立的、更严重的缺陷，
  要先报告而不是继续做流式导出）；
- 文件 ≈ **38 MB** ⇒ 载荷含全部层，`text_pages` 的语义是"跨层页数"，流式导出按 page 记录逐层填即可。

在拿到文件大小之前，会话单元**不动代码**（本轮已因形状猜错回退两次，按纪律不再猜第三次）。
