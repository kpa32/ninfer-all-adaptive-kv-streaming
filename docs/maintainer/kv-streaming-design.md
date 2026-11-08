---
id: kv-streaming-design
type: architecture-design
title: 自适应 KV 流式：原型设计（16 GB 上的 262 144 上下文）
status: draft
parent: ninfer-kv-streaming
depends-on:
  - paged-kv-cache
  - kv-stream-op
references:
  - adaptive-kv-streaming-inputs
  - qwen3-5-model
---

# 自适应 KV 流式：原型设计

本文件是**施工前的完整设计**：目标、参考实现解剖、NInfer 现状核实、数据模型、每轮执行
序列、容量算术、自适应控制器、失败语义、指标、里程碑。实现必须按本文件的合同走；任何偏离
先改本文件。

外部的两篇资料在 [adaptive-kv-streaming-inputs](adaptive-kv-streaming-inputs.md)（参考文章与
CUDA Graph 说明）。参考实现是作者在 llama.cpp 上的 61 个提交、10 519 行改动。

---

## 0 结论摘要

1. 262 144 在 16 GB 上**不可能靠提高 KV 精度装下**：`rk8v4` 要 7.33 GB，`int8` 要 9.90 GB，
   而带 MTP3 时 KV 预算只有 **3.07 GB**。
2. 可行的做法是**页粒度**的“常驻 + 环”：host 是权威副本，设备池 = 每层若干常驻页 + 一个
   **跨层共享**的槽位环，每轮把每层的流式段拷回槽位。
3. 每轮搬运字节 = `(1 − f) × 总 KV`，**与切分形状无关**；形状只决定环的显存开销。
4. NInfer 有一个参考实现没有的优势：**KV 通过 block table 间接寻址**。参考实现因此必须做
   “分段注意力 + 偏 softmax 合并”，而且在流式生效时**关掉 CUDA Graph**；NInfer 可以只改表，
   两个都不需要。
5. 但今天的设备池是**每层一个 plane**，页索引只能指向本层内部，环就无法跨层复用。由此可得一个
   硬结果：`16(R + S) ≥ 16P`，**在 per-layer plane 的池里，页粒度流式在数学上不可能省内存**。
   所以第一个结构性决定是新建一个**扁平槽池**（一个槽 = 一层一页）+ **每层一张表**（D1，§4.1）。
6. 262 144 时的真实上界：MVP（单流、无重叠）约 **15.3 tok/s**；加分段流水后约 **18.9 tok/s**
   （`rk8v4` + MTP3，已用实测 25.7 GB/s 重算）。`int8` 约 10.6 / 12.6。见 §5。

---

## 0.1 可行性判定（施工前必须知道）

**结论：能实现，无决定性硬件/API 阻碍。** 逐项依据：

| 检查项 | 结论 | 依据 |
|---|---|---|
| PCIe 4.0 x16 带宽 | **实测 25.7 GB/s H2D / 25.4 D2H**，且拷贝引擎与计算**完全独立** | 512 MiB × 30 次，2D 同速；并发 spin kernel 时 20.803 ms vs 串行 20.832 ms |
| 拷贝与计算并行 | 可以 | 拷贝引擎独立于 SM；`copy_from_host` 走 `cudaMemcpy2DAsync` |
| pinned host | 已有 | `HostKVArena` / `HostKVExtentStore` 就是 pinned 页分配器 |
| 异步 H2D + 事件 | 已有 | `PhysicalKVPagePool::copy_from_host(span)`；`hybrid_cache` 的 `submit_restore/land_restore` |
| 图内等外部事件 | 有先例 | `program/structured_round.cpp:136` 的 `cudaEventWaitExternal` |
| 图内改页映射 | **支持，且是 NInfer 的优势** | attention 吃 `(pages, block table)`，表是显存数据，不是图参数 |

**必须改的一处架构**：今天 per-(layer,rank,kind) plane，页索引指不出本层 → `16(R+S) ≥ 16P`，
流式不可能省内存。换扁平槽池 + 每层表（§4.1），**内核签名与数学不变**（已核实：
`kv_cache_append.cpp:52-56` 用 `k_pages` 的 4D 形状 + 基址 + 物理页号；
`text.cpp:890` 的 attention 显式收 `table`）。

**3 条硬约束**：M1 单序列（环是共享资源，N 序列要 N 倍环）；M1 拒绝 Hybrid prefix cache 与
Legacy checkpoint（设备页不再代表完整 KV）；M1 不支持 TP/流水（`page_stride/page_residue`）。

**预算修正（M0-4 推翻旧假设）**：MTP 的池与主池**同尺寸**（`mtp_physical_pages = physical_pages +
1`，`mtp_layers = 1`），262 144 `rk8v4` → **428 MB**；而 3.07 GB 的预算来自“带 MTP3 的实测运行时
容量”，**已包含这 428 MB** → 流式池只有 **2.64 GB（25 300 页）**。所有容量与上界已按此重算。

**2 个待核实（会影响做法，不会变成做不到）**：块表在重放间的地址稳定性；
`cudaEventWaitExternal` 重播语义。退路：失图改 eager（参考实现就这样）或
`cudaGraphExecKernelNodeSetParams`。

---

## 1 目标与非目标

**目标**：`--kv-stream` 打开后，Qwen3.8-27B Dense 在 RTX 4080 SUPER 上以 `rk8v4`（最低档）
或 `int8` 跑满 262 144 原生上下文；同一格式下 tok/s 与 logprobs 与全驻留逐 token 一致；
默认关闭；关闭时零行为变化。

**非目标**：删模型族/删架构；UVM 路线（参考文章已证明页面抖动会毁掉 decode）；降低 KV 质量换
容量；预填充阶段的分块流式（本设计只管 decode 的按层流式，预填充走既有 chunk 路径，页在
预填充时正常写入 host 与设备）。

---

## 2 参考实现解剖（llama.cpp fork）

### 2.1 数据放置

- **权威副本在 pinned host**：KV 文本张量分配在自定义 CPU buffer type 上；设备侧只有一个
  **有界池**（`--kv-stream-stage-mib`），池内分两段：`scratch`（传输槽）与 `resident`（常驻页）。
- **页 = 256 token**。池分配：`resident_pages_per_layer × layer_count + ring_slots` 个页。
  `llama_kv_stream_pool_layout_make()` 算常驻页数，`llama_kv_stream_plan_make()` 把 region
  列表（每层每页一个 region，带 `residency_priority`、`pinned`）切成 resident / streamed 两组。
- 常驻的是每层**最前面**若干页（`region.token_begin` 从小到大），流式段是每层的尾部。
- 写入：`ggml_set_rows` 要么经 **mirror** 映射直接写进常驻页（`online_write` 类型），要么落
  host 再重传；`mark_dirty_rows` 记录被写的行。

### 2.2 调度与同步

- 每次图执行前 `ggml_cuda_kv_stream_prepare_graph()` 扫节点：`graph_begin` → 每个
  flash-attn 节点 `graph_add_attention()`（登记“这一层这一页要读”）→ `graph_finalize()`。
- 执行到某层的 attention 时，把**未来层**的页顺着空闲槽投递到一条**独立的非阻塞拷贝流**：
  每槽一对 `ready` / `consumed` 事件 + 一份设备侧 `ready_flags`；消费者等 `ready`；
  过期/完成的槽等 `consumed` 才能被复用。
- 记 `deadline_samples` / `deadline_misses`（页是否在需要前到达）、`copy_engine_busy_ratio`、
  `ring_peak_occupancy`、`hand_to_device_bytes` 等。
- 注意力有两种形态：**整段**（拷齐再算）与**分段流水**（拷一段算一段，用偏 softmax 按
  `max_logit` 重新缩放合并；`llama-kv-stream-softmax`）。用在线调参器（`kv_stream_span_tuner`，
  16 次采样、0.5% 最小增益、布局变化即复位）选更快的那个。

### 2.3 自适应控制器（`llama_kv_stream_partition_adapt`）

- 目标：`ring ≥ 1.10 × 一层要流的页数`（`target_ring_working_set_ratio`）。
- 扩环（每层降级一页）：`miss_ratio > 1%` **且** `copy_busy < 80%`，连续 3 次评估。
- 收环（每层升级一页）：`miss_ratio ≤ 1%` **且** `copy_busy < 50%` **且** `ring_peak < 50%`，
  连续 8 次。
- 两次重分区之间冷却 64 次评估；进入 decode 布局时直接跳到目标；环超配则退回目标。
- `extent`（每层活跃页数）按 `live + reserve` 页对齐增长，收缩有滞后。

### 2.4 三个硬限制（必须知道）

1. **流式生效时会关闭 CUDA Graph**：`ggml_cuda_graph_set_enabled` 里，只要有任一 attention
   读到非常驻页，`use_cuda_graph = false`。全部常驻时才保留图。
2. **只支持目标上下文**：MTP/草稿上下文不允许流式（配置校验直接报错）。
3. 只支持 Qwen3.5、单序列、Flash Attention、GPU KV offload、`stage ≥ 一页`。

### 2.5 可以借 / 不可以照搬

| 借 | 不照搬 |
|---|---|
| 页粒度 + 公平份额（每层降一页即扩一个环页） | 关闭 CUDA Graph（NInfer 有 block table，不需要） |
| 三指标反馈（miss / copy busy / ring peak）+ hysteresis + 冷却 | 分段注意力 + 偏 softmax 合并（NInfer 改表即可，MVP 不需要） |
| 每槽 `ready`/`consumed` 事件对 + 设备侧 flag | 256-token 页（NInfer 是 64） |
| “mutable tail 不可预取” | 把 MTP 上下文排除在外（NInfer 的 MTP KV 单独算，页数很小，可一并处理） |

---

## 3 NInfer 现状（已核实的代码事实）

| # | 事实 | 位置 |
|---|---|---|
| 1 | 页 = **64 token** | `src/core/paged_kv_cache.h:20` |
| 2 | 每 (layer, rank, kind) 一个 plane（K/V/scale 分开） | `state/decoder_state.cpp:200-210` |
| 3 | **块表每序列一行，所有层共用**（同一页在每层同号） | `state/decoder_state.cpp:223`、`paged-kv-cache.md` |
| 4 | 批量视图有 `block_tables` 矩阵形式 | `src/core/paged_kv_cache.h:44` |
| 4b | 地址公式 = 本层 plane 基址 + `local_page × page_bytes`，**页索引不能指向别的层** | `src/ops/kernel/paged_kv_address.cuh:11-20` |
| 5 | 页搬运原语已存在：`copy_page`（D2D）、`copy_to_host`（D2H，按 span）、`copy_from_host`（H2D，按 span） | `src/core/paged_kv_cache.h:332-339` |
| 6 | host 页布局与分配已存在：`plan_host_kv_page_layout`、`HostKVArena`、`HostKVExtentStore` | `src/core/host_kv_arena.h:26-40`、`program/storage/host_kv_store.h` |
| 7 | `TransferWork` 只是代价模型（字节数/拷贝次数），不是执行器 | `src/core/transfer_work.h` |
| 8 | 图捕获/更新：`DecodeGraphDefinition::capture` + `DecodeGraphExecutable::update` | `src/core/decode_graph.h` |
| 9 | 图内等外部事件的先例：`cudaEventWaitExternal` | `program/structured_round.cpp:136-141` |
| 10 | 已有的流式 KV 前例：qwen4_exp 的稀疏层把 host 页当权威、设备池当工作集、`kv_stream_resolve` 在 GPU 内搬页并改表 | `src/models/qwen4_exp/executor.cpp:3444-3500` |
| 11 | 已有的异步 host↔device 往来引擎：hybrid prefix cache 的 `open_restore / restore_block / submit_restore / land_restore(consumer) / transfers_pending / drain` | `program/prefix/hybrid_cache.h:179-198` |
| 12 | `ops/kv_stream` 是可捕获的“页→槽”拷贝 + 表改写（网格固定、位置来自设备 I32） | `src/ops/kv_stream/SPEC.md` |
| 13 | 16 个 full-attention 层（3、7、…、63），4 个 KV 头，head_dim 256；另外 48 层线性注意力状态不随上下文增长 | artifact 实测 |
| 14 | 带 MTP3 的 KV 预算 **3.07 GB**；`rk8v4` 每层每页 **104 448 B**（64 token × 4 头 × 408 B） | 本机实测 + 几何 |

**由 3 推出的第一个结构性结论（D1）**：块表一行为全层共用，所以同一逻辑页在所有层必须落在
同一物理页。流式要“每层各自决定哪些页在环里”，就必须**每层一张表**。设备上多 16 张
`[max_pages] I32` 表（262 K 时 16 × 4096 × 4 B = 256 KB），可以放在一块缓冲里按层切行，
`PagedKVLayerView.block_table` 本来就是 Tensor，按行取即可——不需要改消费端合同。

---

## 4 设计

### 4.1 第一个结构性决定：必须有**扁平槽池**（不是每层 plane）

这里有一个必须先算清楚的算术。今天的设备池是**每（layer, rank, kind）一个 plane**
（`plane_index()`），attention 的地址是 `plane(layer) + page × page_bytes`（`paged_kv_address.cuh`）。
在这种布局下：页索引只能指向**本层 plane 内部**的位置，所以环必须嵌在每层的 plane 里，
而且**不能被别的层复用**。于是池容量 = `16 × (R + S) × B`，而流式又要求
`S ≥ P − R`，两者合起来：

```
16(R + S) ≥ 16(R + P − R) = 16P = 整个 KV
```

**结论：在 per-layer plane 的池里，页粒度的流式在数学上不可能省内存。**
这也是参考实现能成立、而直接照搬其思路到 NInfer 会失败的原因：llama.cpp 的 attention 直接
吃 K/V 指针，所以它的 ring 是一块跨层共用的平缓冲区。

所以本设计的第一个决定是：为 16 个 full-attention 层新建一个**扁平槽池**（D1），
一个槽 = 一层的一页；表把（层，逻辑页）映射到全局槽号；注意力内核的地址是
`pool + slot × page_bytes`。

- 它不改变任何内核：`PagedKVLayerView` 的 `k_pages` / `v_pages` 改为指向扁平池的 K / V plane，
  `block_table` 改为指向**本层**那张表（`PagedKVLayerView.block_table` 本来就是 Tensor）。
  `paged_kv_address.cuh` 的 `plane + local_page × page_bytes` 公式不用动。
- 这是“新建 streamed pool”而非改 Main pool：局部、可回退；不开 `--kv-stream` 时不分配。

### 4.2 数据模型

```
Host（pinned，权威）                  设备：扁平槽池（KV 预算 3.07 GB）
┌──────────────────────┐             ┌───────────────────────────────────┐
│ 每层一个 host plane  │   H2D       │ K / V / Kscale / Vscale 四个平面   │
│ page 0 … max-1       │ ────────►   │ 槽区：                             │
│ (rk8v4: 104 448 B/页) │   D2H       │  [0, 16R)       每层 R 个常驻槽      │
└──────────────────────┘             │  [16R, 16R+S)   共享环（S 个槽）     │
                                     └───────────────────────────────────┘
每层一张表 [max_pages] I32（16 张，共 256 KB @ 262 K）
```

- 页大小沿用 **64 token**（`kPagedKVPageSize`），不引入新页尺寸。
- 常驻的是每层**最新的 R 页**（窗口，见 4.6），流式段是更早的部分，环跨层共享。
- 槽一律以“一层一页”为单位；`page_bytes` 不变，所以 `ops/kv_stream` 的拷贝内核形状可复用。

### 4.3 地址映射与不变量

层 `L` 的逻辑页 `p`（`0 ≤ p < active`）→ 全局槽：

```
p < active − R    →  slot = 16R + p          （流式）
p ≥ active − R    →  slot = L·R + (p mod R)  （常驻窗口）
```

- 同时占用的槽数 = `16R + (active − R) ≤ 16R + S` ✓。
- **两半都只与页号有关，与 `active` 无关**。这一点是必须的：如果写成
  `(p − (active − R)) mod R`，同一个常驻页的槽会随上下文增长而变，每个页边界都要重写整层表。
  写成 `p mod R` 时，离窗的页与进入的页**正好接同一个槽**（两者页号差 R）。
- 因此每条表项一生只写**两次**：页出现时（指向窗口槽）、离窗时（指向环槽）。每个页边界
  每层 2 条 → 32 条 4 字节 H2D + 32 次事件，可忽略。
- 不变量：
  1. 一份槽在同一时刻只服务一层；层 `l` 的 `consumed[l]` 之后才允许层 `l+1` 覆写。
  2. 表里指向槽的项在槽内容就绪（`ready[l]`）之前不被读。
  3. 常驻窗口与 host 副本对同一页的字节必须一致（写入路径保证，见 4.6）。
  4. 表项的槽号永远 `< 16R + S`（越界即 bug，用 assert 卡住）。

### 4.4 每轮执行序列（MVP，单流）

```
for l in full_attention_layers:            # 16 层
    wait(consumed[prev])                   # 环上一次性只有一层在用
    copy  layer l 的流式段  host → ring    # 每 (层,plane) 一条 2D 拷贝
    recordReady(l)
    wait(ready(l))
    attention(l)                           # 读常驻窗口 + 环
    recordConsumed(l)
```

- **每轮拷贝次数 = 层数 × plane 数**（rk8v4 是 4 个 plane 但表里只有 K/V/scales 三类 → 3 条；
  bf16 是 2 条），与页数无关：一条 `cudaMemcpy2DAsync`（宽 = 一页一层一个 plane 的字节，
  高 = 流式页数）即可，因为 host 的页与环的槽在各自 plane 内都是连续的。
  262 K 时约 48–64 条拷贝命令/轮。
- 这条序列**没有重叠**：拷完才能算。这是 MVP 的代价，也是 16 tok/s 的来源（§5）。
- **环的深度是策略，不是常数**（§4.10）：环必须至少能装一层的流式段，因为一层稠密 attention
  一次读完自己的流式段；但池子给的环往往比一层深——262 144 时 `S = P − R` 恰好一层，
  120 K 时同一个环能装四层。多出来的深度应当用来预取**后面几层**，让拷贝与当前层的
  attention 重叠（参考试：`stage_slots` + `cross_layer_prefetches` + `target_ring_working_set_ratio
  = 1.10`，即它主动把环留得略深于一层）。MVP 没用这部分，因为接入时只按“一层在用”写。

### 4.5 容量算术

设每层页数 `P`（262 144 → 4096），每页每层 `B = 104 448 B`，**流式池预算 `K = 2.64 GB`**
（总 KV 预算 3.07 GB 减去 MTP 池 0.428 GB，M0-4）：

```
16·R + S ≤ K / B = 25 300 页        （常驻 16 层，环跨层共享）
S ≥ P − R                           （整层流式段同时在场）
⇒ 15·R ≤ K/B − P      ⇒ R ≤ (25 300 − 4096) / 15 = 1413
```

| 上下文 | P | R（常驻/层） | f | S（环页） | 环显存 | 每轮搬运 | 每轮时长 | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 131 072 | 2048 | 1550 | 0.76 | 498 | 52 MB | 0.83 GB | 32 + 28 ms | 51 |
| 196 608 | 3072 | 1481 | 0.48 | 1591 | 166 MB | 2.66 GB | 104 + 28 ms | 23.6 |
| 262 144 | 4096 | 1413 | 0.34 | 2683 | 280 MB | 4.48 GB | 174 + 28 ms | **15.3** |

- 全流（`R = 0`）：`S = 4096` 页 = 428 MB，每轮 6.85 GB → 267 + 28 ms → 10.5 tok/s。
- `int8`（`B = 135 168 B`）：MTP 池 554 MB → `K = 2.52 GB` = 18 617 页 →
  `R = 968`（f = 0.24），`S = 3128`（423 MB），每轮 6.77 GB → **10.6 tok/s**。
- “每轮时长”按实测 25.7 GB/s 与 120 K 反推的 28 ms/轮计算（§5.2）。

### 4.6 写入路径（为什么常驻的是“最新”的页）

新 token 的 KV 由 `kv_cache_append` 产生，**必须落在设备上**（走 PCIe 写 host 会把 append 拖成
微秒级×页级的开销）。因此：

1. 新页落在常驻窗口内（`p ≥ active − R`）→ `kv_cache_append` 直接写设备，**不碰 host**。
2. 页滑出窗口（`p = active − R`）→ 在同一位置，先把该页 **D2H 到 host**（此时它已写满），
   再把进入的页 H2D 进同一个槽。每页边界每层 1 页 D2H + 1 页 H2D，共 16 × 2 × 102 KiB
   ≈ 3.3 MB / 64 token。
3. 因为页只在写满之后才离窗，host 副本永远不需要接收半页，也不需要每 token 写 PCIe。
4. **正在写的页（mutable tail）永远在窗口里**，因此它从不参与预取：环里那一格是给早先的页用的。

反过来，若按参考实现“最前面 R 页常驻”，新 token 全部落在流式段 → 每个 token 都要写 host 再
传回设备。这是 NInfer 不该照搬的第二个细节。

### 4.7 与 CUDA Graph 的关系（本设计与参考实现的关键差异）

参考实现流式时关图，因为它的 attention **直接吃 K/V 张量指针**：流式页要换缓冲区就只能换指针，
图就失效。

NInfer 的 attention 通过 **block table（显存数据）** 寻址。于是：

- 页 → 槽的映射是**表的内容**，不是图的参数。轮与轮之间表不变（4.2），图可以一直重放。
- 每轮的 H2D 拷贝由宿主机在**图外**投递（一条拷贝流 + 每层 `ready`/`consumed` 事件），图内每层
  attention 前放一个 `cudaEventWaitExternal` 节点——这正是 `structured_round.cpp:136` 已在用的
  机制，不改捕获拓扑。
- 只有 `active_pages`（每 64 token）或 `R`（重分区）变化时才需要重写表；前者是纯数据写入
  （一条小 kernel 或 `cudaMemcpyAsync`），**不需要重捕获**；后者改变 `S`，需要重捕获一次
  （实测捕获约 0.6–0.7 s；重分区有 64 次评估的冷却，代价被摊薄）。

> **M0 核实结论（已全部成立，可作为施工依据）**
>
> 1. **表地址稳定**：`KVExecutionTablePool` 的 `replicas_` 构造时一次性分配，`row()` 只是
>    它的切片视图 → 地址在整个进程生命周期内不变，发布只改内容。发布 = `pinned host_shadow_`
>    → `cudaMemcpyAsync` → 每 (row, replica) 一个 `RowFence` 事件；不相交条目不等，捕获中
>    直接跳过 fence。**每层表不需要新基础设施：`table_rows = 16` 即可**。
> 2. **事件重放语义**：`structured_round.cpp:131-146` 已是“每轮重录同一事件 + 图内等待”的
>    现成 idiom（捕获走 `cudaEventWaitExternal`，热身轮走 `cudaEventWaitDefault`）。
>    16 层 × 2 个事件照抄即可。
> 3. **拷贝可图外投递**：`copy_from_host` 是纯异步宿主函数，把目标页按连续 run 分组，每 plane
>    一条 `cudaMemcpy2DAsync`。环槽是连续区间 → 每层每 plane 正好一条。
> 4. **`physical_index(handle)` 只校 owner/generation** → 槽号可任意（只要属于同一个扁平池）。
>
> 采图时的注意事项：`publish_indices` 在捕获中会跳过 fence；所以**表必须在捕获前发布完**，
> 或者只在非捕获轮改表（每 64 token 的窗口转动正好可以放在轮与轮之间）。

### 4.8 自适应控制器

指标（沿用参考实现，因为它已在其硬件上验证过）：

| 指标 | 含义 | 来源 |
|---|---|---|
| `deadline_misses / samples` | 页没在需要前到达的比例 | 每层等待结束时刻 vs `ready` 时间 |
| `copy_busy_ratio` | 拷贝流忙碌比例 | 拷贝流的起止时间戳/轮时长 |
| `ring_peak_occupancy` | 环的峰值占用 | 分配器计数 |

规则：

1. `f` 够用（`S = 0`，全驻留）→ 完全不走流式路径，**与今天逐字节一致**。
2. 超出 → 从每层各释放一页，成为环的一页（**公平份额**，一次一页/层）。
3. 扩环（`R −= 1`）：`miss > 1%` 且 `copy_busy < 0.80`，连续 3 次评估。
4. 收环（`R += 1`）：`miss ≤ 1%` 且 `copy_busy < 0.50` 且 `ring_peak < 0.50`，连续 8 次。
5. 冷却 64 次评估；进入 decode 首轮直接跳到目标（避免开头几轮用错布局）。
6. **刹车**：`copy_busy ≥ 0.80` 时只收不放——这正是参考文章写的恶性循环防护。

`R` 的取值还要满足 4.5 的两个约束（`16R + S ≤ K/B`、`S ≥ P − R`），控制器只在可行区间内动。

### 4.9 与其它子系统的交互

| 子系统 | 交互 |
|---|---|
| MTP / DFlash2 | 它们有自己的小 KV 池（262 K 时 MTP 约 107 MB）。**不参与流式**，整层常驻。页数独立核算，预算里先扣掉。 |
| Vision | `--vision` 会再压 ~0.7 GB 预算；流式的 `R` 相应下降（公式不变，`K` 变小）。overlay 模式与流式互不干扰（不同池）。 |
| prefix cache（Legacy） | Legacy 的 checkpoint 会保存设备页；流式后“设备页”只代表常驻窗口，checkpoint 必须存 **host 权威副本**（否则恢复后缺页）。M1 先只支持 `--kv-stream` 与 Legacy 互斥（打开流式时禁用 checkpoint），M2 再做 host 侧快照。 |
| prefix cache（Hybrid） | `--use-alt-prefix-caching` 的 host tier 已有一套块页与往来引擎；流式与之叠加会把“页在 host”变成两套语义。M1 明确不支持，报错退出。 |
| 多序列 / 并发 | 表是每序列一行；流式的环是**每执行行**的资源，多序列同时 decode 会争环。M1 限制 `--max-concurrency 1`，M4 再按“序列数”分摊环。 |
| 多卡（TP/流水） | `PagedKVLayerView` 有 `page_stride/page_residue`（position-split 的 rank 只持有部分页）。流式与 position-split 的交互复杂，M1 明确只支持单设备；TP 下每 rank 独立流式留作后续。 |

### 4.10 第二阶段：重叠（M4a 跨层双缓冲 / M4b 分段流水）

MVP 的序列没有重叠：拷贝 4.56 GB 时 GPU 算不了别的（251 K 实测 177 ms 拷 + 23 ms 算）。
重叠有两条路，价格差很大，**必须按上下文分开算**。

**M4a 跨层双缓冲（不需要新 attention 入口）**：环里同时放层 `l` 和层 `l+1` 的流式段，
拷 `l+1` 期间层 `l` 做 attention。只需把“等 `consumed(prev)` 再拷”改成“按环深提前若干层”
（`ready/consumed` 事件已经是每层一组）。代价是环要更深，而环与常驻区同出一个固定池：

| 上下文 | R（页/层） | 流式页/层 | 双层环需要 | 现有环 2907 | 每轮拷贝 | 叠加前 | 叠加后 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 119 664 | 1189 | 681 | 1362 | 装得下（白拿） | 1.14 GB / 44 ms | 43.9 | **~62** |
| 176 000 | 1174 | 1576 | 3152 | 不够，R 少 15 页 | 2.63 GB / 102 ms | 22.6 | **~27** |
| 250 964 | 982 | 3114 | 6228 | 不够，R 少 207 页 | 5.20 GB / 202 ms | 14.1 | **14.0（洗白）** |

结论：**中长上下文白拿 +40%，到 251 K 退化成零**——环多深就要从常驻区切多少页，
而每切一页/层就多搬 `16 × 104 448 B`，正好把重叠省下的 23 ms 吃干。所以 M4a 应当做，
但不要把 262 144 的速度期待建在它上面。

**M4b 分段流水（改变环的深度需求）**：把每层的流式段按 `k` 页一段处理：

```
copy span 0 → attend span 0（偏 softmax）→ merge
              copy span 1 → attend span 1 → merge
                            ...
```

- 环只需 `k` 页，于是 `R` 可以更大：`R = (21 931 − k)/16`，262 K 时 `k = 16` →
  `R = 1369`，流式段 2727 页 → 每轮 4.56 GB（与 MVP 同量级）。
- 价值在重叠：每轮 ≈ max(177, 23) = 177 ms → **15.9 tok/s（+13%）**，且 `k` 越小、
  `R` 越大、拷贝越少，两者同向。
- 代价：需要一个新的 attention 入口（按页区间出偏结果 + 合并），参考实现在 `fattn.cu`
  里做了 2 141 行；NInfer 已有 TP prompt 的偏和合并基础设施可参考。
- 这不改变正确性：合并是精确的数值操作（按 `max_logit` 重新缩放），与参考实现相同。

两条路都**不改变“每轮必须把环外的 KV 全读一遗”这个下限**：251 K、`rk8v4` 时是 4.5–5.2 GB，
`25.77 GB/s` 下就是 175–200 ms。要想真正跳出这个下限，只能减少字节（更小的页 / 更大的常驻窗口），
而两者都与质量或显存直接相关（§5.3、§11）。

---

### 4.11 运行时接入：绕过哪一层（M1 的第二个结构性决定）

设计到这里都有一个未回答的问题：扁平池接进去后，今天那些“逻辑页 ↔ 物理页”的机制怎么办。
答案是把 text pool 的地址空间整个换掉，

**今天的两层**（`program/storage/kv_store.h`）：

| 层 | 职责 | 与流式的关系 |
|---|---|---|
| `LogicalKVPageStore` | 逻辑页按需 `materialize` 成一个 device lease；带 host replica、epoch、引用计数、`committed/protected columns` | **不适用**：它假设每个 occupied 逻辑页都有 device replica（`page.device_replica.emplace(...)`），而流式下非常驻页**故意没有** device replica；它的 device 页是需求分配，流式是确定映射 |
| `KVAddressSpaceStore` | 一条序列的逻辑页 → 物理页地址空间 | **不需要**：流式下地址空间就是 16 张每层表，内容由 `kv_stream_slot` 决定 |

**所以**：流式模式下 text pool 用 `KVStreamSpace` 取代前两层，它拥有：

1. 扁平 `DeviceKVPagePool` 的**全部槽 lease**（启动时一次性 `reserve` + `materialize`，
   于是 `available_pages() == 0`——流式池没有“空闲页”这个概念）。
2. 16 行每层表（`row_at(layer)`），启动时把流式段写一次（恒等映射），窗口每 64 token 转动。
3. host 权威副本（`HostKVArena` + `plan_host_kv_page_layout`），容量 = `P × page_bytes` 每层。
4. 每轮拷贝计划与 `ready`/`consumed` 事件。

**连带后果（M1 必须一起处理）**：

- `ProgramImpl` 的 `text_kv_pages` / `text_kv_addresses` 在流式下为空；所有使用它们的路径必须
  逐一处理：retention/checkpoint/prefix 路径**拒绝**（`--kv-stream` 已要求 context cache 关闭），
  其余（append / attention）只经 view，不经 store。
- `Vision` 的 KV tier 借用与 `KvExpertLoan` 的专家缓存都拿“空闲页”做抵押（
  `unpromised_free_pages()`）；流式池没有空闲页 → 这两个借款在流式下必须关闭。
- admission/pressure 读 `usable_pages()`（`pressure.cpp:2635`、`request_plan.cpp:246`）→
  流式下容量由 `max_context` 决定，不由池页数决定，这两处要改走 `max_context`。
- 写入路径（§4.6）：新 token 落常驻窗口，然后异步 D2H 到 host 权威副本；prefill 直接写 host 页
  （`qwen4_exp/executor.cpp:3457` 已有先例），再在 decode 首轮把常驻窗口拉回设备。
- 拷贝的投递点：解码轮的层循环（不是图内）——图内只放 `cudaEventWaitExternal` 节点。

这条决定让 M1 的范围变大（不是“加一个 allocator”，而是“多一套地址空间”），但它也是唯一正确的：
把流式塞进 `LogicalKVPageStore` 会在每次页搬动时继承需求分配的语义，而那正是我们要避开的。

---

### 4.12 原理图

**（1）扁平槽池与每层表**：一个槽就是一层一页；常驻槽按层分段，环槽跨层复用；
attention 的地址是 `pool + slot × page_bytes`，所以换表就等于换页。

```mermaid
flowchart TB
    subgraph HOST["Host（pinned，权威）"]
        HL["层 L 的 host plane<br/>page 0 … max−1"]
    end
    subgraph POOL["设备：扁平槽池（3.07 GB）"]
        RES["常驻槽区 [0, 16R)<br/>层 L → 槽 L·R … (L+1)·R"]
        RING["共享环 [16R, 16R+S)<br/>流式页 p → 槽 16R+p"]
    end
    subgraph TAB["每层一张表（16 × max_pages I32）"]
        TL["表 L：p &lt; active−R → 16R+p<br/>p ≥ active−R → L·R + (p−active+R) mod R"]
    end
    HL -- "每轮 H2D（每层 1–3 条 2D 拷贝）" --> RING
    TL -- "attention 解引用" --> POOL
```

**（2）一轮内的顺序**（MVP：环一次性只有一层在用）：

```mermaid
flowchart LR
    A["层 L：拷流式段 → 环"] --> B["record ready(L)"]
    B --> C["attention(L)<br/>读常驻窗口 + 环"]
    C --> D["record consumed(L)"]
    D --> E["层 L+1：覆写同一批环槽"]
```

**（3）自适应（带刹车）**：

```mermaid
flowchart TD
    A["整个 KV 能装进预算？"] -- "是" --> B["S = 0，完全不走流式"]
    A -- "否" --> C["从每层各释放一页 → 环多一页（R−1）"]
    C --> D{"miss > 1% 且 copy_busy < 0.80？"}
    D -- "是，连续 3 次" --> C
    D -- "否" --> E{"miss ≤ 1% 且 copy_busy < 0.5<br/>且 ring_peak < 0.5，连续 8 次？"}
    E -- "是" --> F["收环：R+1"]
    E -- "否" --> G["保持"]
    D -- "copy_busy ≥ 0.80" --> H["刹车：不再扩环"]
```

---

## 5 上界与基线

### 5.1 本机实测基线（2026-11-08，`rk8v4` + MTP3 + greedy，服务端同款 flags）

| 上下文 | prefill | decode | MTP 接受长度 | 峰值显存 |
|---|---:|---:|---:|---:|
| 8 192 | 2.37k tok/s | 114.3 tok/s | 2.76 | 12 672 MiB |
| 32 768 | 2.28k | 115.0 | 2.89 | 13 322 MiB |
| 65 536 | 2.11k | 117.6 | 3.10 | 14 188 MiB |
| 120 000 | 1.86k | 109.1 | 3.10 | 15 632 MiB |

32 K 同 prompt 的 KV 对照（decode）：`bf16` 121.8 / `int8` 116.6 / `rk4v4` 115.1 /
`rk2v4-e8` 122.2 tok/s；128 token 长生成 145.8 tok/s。

262 144 的启动探针（无草稿器）：`bf16` 需 17.62 GB、`int8` 9.90 GB、`rk8v4` 7.33 GB、
`rk4v4` 5.18 GB —— 全部失败；`rk2v4-e8` 3.38 GB 启动成功（但质量不可接受）。
带 MTP3 时可用运行时容量为 4 032 299 008 B。

### 5.2 上界

每轮时间的两部分（实测 25.7 GB/s；计算 28 ms/轮由 120 000 上下文 109.1 tok/s × 3.1 tok/轮 反推）：
| 方案 | 每轮搬运 | 拷贝 | 计算 | 每轮 | tok/s |
|---|---:|---:|---:|---:|---:|
| MVP `rk8v4` 262 144 | 4.48 GB | 174 ms | 28 | 202 ms | **15.3** |
| 分段流水 `rk8v4` | 4.20 GB | 164 ms | 28 | max=164 ms | **18.9** |
| MVP `int8` 262 144 | 6.77 GB | 263 ms | 28 | 291 ms | **10.6** |
| 分段流水 `int8` | 6.35 GB | 247 ms | 28 | 247 ms | **12.6** |
| 全流 `rk8v4`（R=0） | 6.85 GB | 267 ms | 28 | 295 ms | 10.5 |

- 带宽已实测：512 MiB × 30 次 = **25.77 GB/s H2D**、25.44 D2H；2D（4096 × 104 448 B）同速 25.80；
  并发 spin kernel 时 20.803 ms vs 串行 20.832 ms → **拷贝与计算完全并行**（M0-5）。
- 每轮接受长度取 3.1（120 K 实测）；262 K 时可能下降，届时按实测修正。
- 这些仍是**上界**：未计 attention 在 262 K 的额外开销、事件等待的抖动、页表写入。
- 分段流水只把环从 280 MB 降到 ~2 MB 级并多为 `R` 腾出 167 页/层（每轮少 6%），
  真正的收益是拷贝与计算重叠。

### 5.3 实测（M1，2026-11-08，RTX 4080 SUPER）

planner 实际给的切分（`--max-context 262144 --kv-capacity auto`，`auto` 按启动时的忙存定，
所以同一 dtype 在两次启动间会差几十页）：

| dtype | page_bytes | 槽数 | R（常驻/层） | S（环） | 窗口 token | 主机权威 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 262 144 | 12 760 | 577 | 3519 | 36 928 | 16.4 GiB |
| `rk8v4` | 104 448 | 20 002–21 945 | 1060–1189 | 2907–3036 | 67 840–76 096 | 6.4 GiB |
| `int8` | 135 168 | 24 318 | 1348 | 2748 | 86 272 | 8.3 GiB |
| `rk4v4` | 71 680 | 45 858 | 2784 | 1312 | 178 176 | 4.4 GiB |

三档与 §4.5 的差异是真实的：§4.5 用 2.64 GB 估的流式池，planner 实际给 `rk8v4` 1.96–2.09 GB
（`rk8v4` 的 MTP 池与主池同尺寸，438 MB），所以 R 比 1413 小。`S ≥ P − R` 恒为等号：环恰好装下
整段流式部分。

**首跑（`rk8v4` + MTP3 + `--prefill-chunk 256 --greedy --invariant-attention`）**：

| 项 | 值 |
|---|---|
| prompt | 250 964 token（`--max-context 262144`） |
| prefill | 3 min 11 s（1.31k tok/s） |
| decode | **14.1 tok/s**（32 token，2.2 s） |
| MTP 接受长度 | 2.82 |
| 峰值显存 | 15 028 MiB / 16 376 |
| 与 §5.2 上界 15.3 的差距 | 8%（未计 attention 在 262 K 的额外开销与等待抖动） |

**为什么只有 14 tok/s（与全驻留的 109 tok/s 不可比）**：流式每一轮都要把「窗口以外的 KV」
整段重传一遍：

```
每轮字节 = (active − R) 页 × 层数 × page_bytes
250 964 token：2732 × 16 × 104 448 = 4.56 GB → @25.77 GB/s = 177 ms
每轮总时长 = 177（拷） + ~23（算）= 200 ms；2.82 tok/轮 → 14.1 tok/s
```

GPU 在一轮的 200 ms 里约 **88% 在等 PCIe**。全驻留（无拷贝）时 120 K 是 109 tok/s，
这才是“平时那个数”，但它要 7.33 GB KV，262 144 在 16 GB 卡上不存在。

同一 prompt、`rk8v4`+MTP3、262 144 的实测曲线（窗口越大、每轮搬得越少）：

| prompt tokens | 环内页/层 | 每轮字节 | 每轮时长 | decode |
|---:|---:|---:|---:|---:|
| 80 404 | 68 | 0.11 GB | 33 ms | 86.7 |
| 119 664 | 686 | 1.15 GB | 67 ms | 43.9 |
| 250 964 | 2732 | 4.56 GB | 200 ms | **14.1** |

换 `rk4v4`（页 71 680 B、窗口约 145 K token）同一 250 964 token prompt：**25.7 tok/s**。
这说明可调的旋钮是「每轮字节 = (上下文 − 窗口) × 16 × page_bytes」：窗口越大、页越小就越快。
但 `rk4v4` 被质量下限排除，`int8` 的页更大（135 168 B）反而更慢（§5.2 的 10.6）。
M4 的分段流水只能把 14.1 抬到 ~19：重叠能藏住计算，藏不住 4.56 GB 的搬运时间。

**一致性（M2 的首个证据）**：80 404 token 的 prompt，`rk8v4` + MTP3，
`--invariant-attention`，全驻留（`--max-context 85000`）与流式环（262 144，68 页在环）**逐字节相同**
（48 token，MTP 接受长度同为 2.61）；decode 94.0 vs 86.7 tok/s。同格式的全驻留档无法在
262 144 上启动（`rk8v4` 需 7.33 GB），所以长上下文的正确性由环内字节与表项的直接校验 +
可启动的最大基线圈对齐证明（§9）。

**关图的代价实测**：8 192 上下文、`bf16`/`rk8v4` 全驻留（`S = 0`）时输出与关流式逐字相同，
decode 低 3.4%（正是关 CUDA Graph 的代价，§4.7）。

### 5.4 延迟推算（用实测切分与实测带宽，不做假设）

器件：`rk8v4` 每层每页 `104 448 B`、页 64 token、16 个全注意力层；实测 H2D `25.77 GB/s`；
planner 实际给 `R = 1189`（每层相同的常驻页，窗口 76 096 token）、环 `2907` 槽：

| 量 | 值 |
|---|---|
| 262 144 时全 KV | 4096 页/层 × 16 × 104 448 = **6.85 GB** |
| 常驻窗口（16×1189） | **1.99 GB** |
| 环（2907 槽，跨层共享） | **0.30 GB** |
| 一页一层的延迟 | 104 448 ÷ 25.77 GB/s = **4.05 µs** |
| 一层流式段（251 K） | 2733 页 = **11.1 ms** |
| 一轮 16 层 | 2733 × 16 = 43 728 页 = **4.57 GB = 177.2 ms** |
| 一轮计算（FFN+attention+MTP） | 200 − 177 = **22.8 ms** |

模型 `round = (active − R) × 16 × page_bytes ÷ 25.77 GB/s + 22.8 ms`，对实测：

| prompt | 活跃页/层 | 环内页/层 | 每轮字节 | 拷贝 | 推算 tok/s | 实测 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 80 404 | 1257 | 68 | 0.11 GB | 4.4 ms | 102.9 | 86.7（计算主导，常数 22.8 不适用） |
| 119 664 | 1870 | 681 | 1.14 GB | 44.2 | 41.8 | **43.9** |
| 176 000 | 2750 | 1561 | 2.61 GB | 101.2 | 22.6 | （未测） |
| 250 964 | 3922 | 2733 | 4.57 GB | 177.2 | 14.0 | **14.1** |

**反推 80 tok/s 需要什么**（这才是关键数字）：

| 重叠 | 每轮拷贝预算 | 可流式页/层 | 需要常驻窗口 | 需要设备 KV |
|---|---:|---:|---:|---:|
| 无（串行） | 12.4 ms | 192 | 250K token | **6.52 GB** |
| 完全重叠 | 35.2 ms | 544 | 227K token | **5.94 GB** |

本卡给 KV 的设备池只有 **2.29 GB**（1.99 常驻 + 0.30 环）。把 2.29 GB 全用在窗口上、环为零时，
窗口上限也只有 `2290 MB ÷ (16 × 104 448) = 1370` 页 = **87.7K token**。这就是为什么：

- ≤ ~88K token：可以全驻留，满速（实测 8K–64K 为 114–118 tok/s）；
- 88K–262K：必须流式，速度 = `2.8 ÷ ((active − R) × 16 × 104 448 ÷ 25.77 GB/s + 计算)`；
- 80 tok/s @ 262K 在这台卡上不存在，需要 ≥ 6 GB 的 KV 池（约 20 GB 的卡）。

### 5.5 窗口是速度的唯一旋钮（2026-11-08 实测，同一台卡）

同样的实现，只改「窗口覆盖多少上下文」和「每页多少字节」，速度差 5 倍。所有数字都是
`--prefill-chunk 256 --spec mtp --draft-tokens 3 --greedy --gdn-state-fp16`，
256 或 251 K 的 prompt 各跑 32 token：

| KV | `--max-context` | headroom | 契约页组 | R（页/层） | 窗口 token | 流式页/层 | 每轮拷贝 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rk4v4` | 262 144 | 1024 MiB | 45 858 | 2784 | 178 176 | 1138 | 50 ms | 25.7 |
| `rk4v4` | 262 144 | **128 MiB** | 46 958 | 2857 | 182 848 | 1065 | 47 ms | **45.8** |
| `rk8v4` | 262 144 | 1024 MiB | 21 945 | 1189 | 76 096 | 2733 | 177 ms | 14.1 |
| `rk8v4` | 262 144 | **256 MiB（现默认）** | 29 656 | 1704 | 109 056 | 2218 | 144 ms | **16.9** |
| `rk8v4` | 262 144 | **128 MiB** | 30 941 | 1789 | 114 496 | 2133 | 138 ms | **17.7** |

同一 189 864 token 的 prompt，换格式：

| KV | 窗口 token | 流式页/层 | 每轮拷贝 | decode |
|---|---:|---:|---:|---:|
| `rk4v4`（窗口 183 K） | 182 848 | 110 | 4.9 ms | **83.4** |
| `rk8v4`（窗口 114 K） | 114 496 | 1178 | 76 ms | **27.3** |

**结论一：`--kv-headroom-mib` 的默认 1024 对流式是浪费，已改为 256 MiB**（提交 `4077e33b`）。
流式关掉了 CUDA Graph，且 workspace 竞技场是**单独预留**的（实测峰值 100.3 / 容量 103.4 MiB，
与 headroom 无关），所以那 1 GB 只是空闲显存。实测：

| headroom | 池（页组） | R | 窗口 token | 空闲于启动后 | 251 K decode |
|---:|---:|---:|---:|---:|---:|
| 1024 MiB（旧默认） | 21 945 | 1189 | 76 096 | 1009.5 MiB | 14.1 |
| **256 MiB（现默认）** | 29 656 | 1704 | 109 056 | 241.5 MiB | **16.9** |
| 128 MiB（激进） | 30 941 | 1789 | 114 496 | 113.5 MiB | 17.7 |

选择 256 而不是 128：保留 2.5 倍于实测 workspace 峰值的裕量，代价只有 4.5%（16.9 vs 17.7）。
非流式路径的默认仍是 1024 MiB（它还包含图与捕获的 workspace），已验证 8192 全驻留不变
（页组 128/128）。实测通过的全部组合：
120 K 全驻留（起得来，之前 96 K 都起不来）、190 K、251 K、两种格式、长 prefill + MTP。
但这是**策略改动**，产品默认值要留裕量并做成 `--kv-stream` 下的独立默认（§9.12）。

**结论二：质量下限 `rk8v4` 是 251 K 速度的主要成本。** 同一个 251 K prompt，`rk4v4`
（每页 71 680 B vs 104 448 B）拿 45.8 tok/s，是 `rk8v4` 的 2.6 倍；190 K 时 83.4 vs 27.3，
是 3 倍。原因一半是字节少，一半是同一池子能装下更多页 → 窗口覆盖更多上下文。

**结论三：提速的旋钮按收益排序**（251 K、`rk8v4`）：

| # | 动作 | 代价 | 预期 251 K | 预期 120–190 K |
|---|---|---|---:|---:|
| 1 | headroom 256 MiB（**已是默认**，`4077e33b`） | 一条策略默认值 | **16.9**（实测） | 起得来 |
| 2 | M4a 跨层预取 | 环深 2 层，从常驻区切页 | ~20.5（+16%） | **+40%（白拿）** |
| 3 | M4b 分段流水 | 新 attention 入口（偏 softmax + 合并） | ~23（+30%） | +10% |
| 4 | 流式段用更小的页（`rk4v4` 级别） | **质量决定，需批准** | ~24（按字节算） | — |
| 5 | 2 + 3 同时 | 两者都要 | ~25（+41%） | +50% |

第 4 条若只对「流式段」生效、窗口仍保持 `rk8v4`，251 K 可到 ~24 tok/s（每轮
2133 × 16 × 71 680 = 2.45 GB → 95 ms）。它打破「同一格式、窗口内外精度一致」的合同，
属于产品决定，不擅自做。

---
### 5.6 拐点：装得下就不流式（和参考实现的曲线同形）

参考实现那张图（`benchmarks/README.md` 的 sweep）横轴是**容量**（`--ctx-size`），不是实际上下文。
容量装得下 KV 时它的 `pool_layout_make` 会把整层都算作常驻（`S = 0`），曲线与 stock 完全重合；
容量装不下才出现常驻/流式切分，才开始降速。本实现同形，实测（`rk8v4`+MTP3，120 K prompt）：

| 容量 | `--kv-stream` | 契约页组 | 切分 | decode | 与全驻留输出 |
|---:|---|---:|---|---:|---|
| 120 000 | 开 | **30 000 / 30 000** | `S = 0`，环为空 | 81.0 | **逐字节相同** |
| 120 000 | 关 | 30 000 / 30 000 | — | 83.6 | 基准 |
| 131 072 | 开 | 31 705 / 32 768 | 环 71 页，prompt 未触及 | 84.1 | — |
| 262 144 | 开 | 29 656 / 65 536 | `R = 1704`，166 页/层在环 | **81.7** | **逐字节相同** |

比值：容量 262 144、**实际 120 K** 上下文时 81.7 vs 全驻留 83.6 = **98%**（§5.5 的 64.1 是深度
环实验的产物，已随该实验撤回）。本卡的拐点（容量上限，`rk8v4`+MTP3，256 MiB headroom）在
**约 117 K**：131 072 容量起不来（运行期预留 4.12 GB + 0.13 GB > 4.03 GB 可用），120 000 能起。

**与参考曲线的差距只有一处：关图。** 本实现为流式关掉 CUDA Graph（−3.1%），参考实现**在流式下
仍捕获图**（`graph_decode_span_pages`、`ring->last_graph_decode`），所以它拐点左侧与 stock 完全重合。
这是已知可改进项，不影响拐点位置与降速形状。

### 5.7 深环：实测否决（勿重做）

环按最大容量分配，短上下文留出大量空闲深度，把环切成多份、让多层的流式段同时在环里、拷贝叠在
前一层 attention 上——听起来免费，实测**每个配置都更慢**（同一池、同一 prompt）：

| 配置 | depth 1 | 2 | 4 | 11 | 14 |
|---|---:|---:|---:|---:|---:|
| `rk4v4` 190 K prompt（池 46 958，深度 1 的 73.9 是旧二进制，同池对比以行为准） | **73.9** | 73.0 | 71.9 | 68.1 | — |
| `rk8v4` 120 K prompt（池 29 656） | **81.7** | 80.4 | — | — | 65.6 |

原因：重叠的拷贝不是躲在 attention 后面，而是与它争用（拷贝引擎写显存 + 驱动/事件开销），代价
大于省下的等待；单层深时一轮只剩 5–44 ms 的拷贝，而 interference 在 16 层×每轮的尺度上累积。
参考实现的 1.10 工作集比例与此一致：它也只比一层略深。

**教训**：短上下文的速度来自**更大的窗口**（§5.5 的 headroom 改动把 120 K 从 43.9 抬到 81.7），
不是来自重叠。想再快只有两条路：减少流式字节（更小的页/更大的窗口）或减少层的参与。
### 5.8 这套方法的理论上限（纯算术，与实测对照）

输入全部实测：`rk8v4` 每 token KV = 16 × 104 448 ÷ 64 = **26 112 B**；PCIe H2D 25.77 GB/s；
设备池 29 656 槽；MTP 每轮 ~2.8 token。**每轮计算用时取"关图 + 无拷贝"的实测值**：容量 120 000、
环空（S=0）时 81.0 tok/s → **33.7 ms/轮**，随上下文增长的只有 KV 从显存读的部分
（(12.5 GB 权重 + KV)/736 GB/s），262 144 时约 **40 ms**。

> 早先这里写的 23 ms 是 `167 − 144` 反推的，把已有的重叠算进了计算，是错的。那时的结论
> "动态窗口在 120 K 能 +39%"因此是假机会：**120 K 的拷贝已经完全免费**（有拷贝 81.7 vs
> 无拷贝 81.0 tok/s），重切窗口在那里收益为 0。

```text
S'(k) = (16P − pool) / (16 − k)      k 层环深时每层必须流式的页数（P = 容量页数 4096）
copy  = S'(k) × 16 × 104 448 ÷ 25.77 GB/s
tok/s = 2.8 / max(copy, 40 ms)                       理想重叠
tok/s = 2.8 / (copy + 40 ms)                         完全串行
```

| 环深 k | 每层流式页 | 窗口 | copy | 理想重叠 | 串行 | 实测 |
|---:|---:|---:|---:|---:|---:|---:|
| 1（现状） | 2 392 | 1 704 | 144 ms | 19.4（176 ms 未重叠） | 14.9 | **16.9** |
| 2 | 2 563 | 1 533 | 154 ms | **18.3** | 14.4 | 未测 |
| 4 | 2 990 | 1 246 | 180 ms | 15.7 | 13.2 | 未测 |

**环深和窗口抢同一个池，代价是 16P−pool ÷ (16−k)**：k=2 每轮多搬 7%，k=4 多搬 25%。
所以 251 K 的完美重叠只值 **+8%（16.9 → 18.3）**，不是早先估的 +16%。

三个硬常数不变：

- **池能全驻留的上下文 = 118.6 K token**；
- **计算能吸收的流式量 ≈ 40 ms × 25.77 GB/s = 1.03 GB = 39 K token**；低于它拷贝全免费
  （120 K 档流式 11 K → 实测免费），高于它每轮都在等 PCIe；
- **262 144 的天花板 = 18.3 tok/s（k=2、完美重叠）**，今天 16.9 已在 92% 处。

**还没定的一个数**：251 K 时拷贝流真实占用 144 ms（25.77 GB/s）还是 ~160 ms（2D 跨步的实效）？
前者说明还有 22 ms 计算露在外面（有 8% 可拿），后者说明我们已经在重叠上限的 97% 处（无事可做）。
这一条只能靠给拷贝流打计时事件测，不能靠算术。
### 5.9 MTP 的净加速与 rk4v4 基线

测试基准档改为 **`rk4v4`**（本卡不流式可容纳 190 K，对比最清楚）：prompt 186 452 token、
`--invariant-attention`、`--max-new 64`、容量 262 144 + `--kv-stream`（除第一行）。

| 配置 | decode | 每轮 | accept |
|---|---:|---:|---:|
| 190 K 全驻留，无 spec | 38.7 | 25.8 ms | — |
| 190 K 全驻留，MTP3 | **起不来**：需 4 103 599 616 B + 128 MiB，只有 4 032 299 008 B | | |
| 262 K 流式，无 spec | 38.2 | 26.2 ms | — |
| 262 K 流式，MTP1 | 62.8 | 33.4 ms | 1.88 |
| 262 K 流式，MTP2 | 64.3 | 37.3 ms | 2.38 |
| 262 K 流式，MTP3 | 63.5 | 39.1 ms | 2.48 |
| 262 K 流式，MTP4 | 65.0 | 37.4 ms | 2.70 |

- **MTP 净加速 1.66×**（63.5 vs 38.2），而不是 accept 的 2.48×：一次 4 token 验证轮是单 token
  轮的 **1.49×**（39.1 vs 26.2 ms）——drafts 每个都要在全上下文上做一次注意力。想吃到 2.4×
  只能削 draft 的注意力范围，那是数学变更，不在本线。
- **draft 数不是旋钮**：1/2/3/4 = 62.8/64.3/63.5/65.0（±2%）。
- **186 K 时流式与全驻留同速**（38.2 vs 38.7）：池 59 925 页把 186 K 整个容下，环空。
- 190 K 全驻留 + MTP 在这个模型上**永远**差 71 MB 装不下（MTP 池 428 MB），所以长上下文只有流式一条路。
- **MTP 在 262 K 上要付 1.06 GB 的池子**（rk4v4：无 spec 59 925 页 vs MTP3 45 085 页），
  等于窗口少 989 页/层 = **63 K token**。所以 262 K 用 MTP 的窗口只有 175 K token，
  而不用 MTP 有 238 K——MTP 仍值得（63.5 vs 38.2），但它把长上下文推进流式区的方式是这套
  池子里最贵的一项。
### 5.10 【已被 §5.11 推翻】rk4v4 下 251 K 的算术读数（当时以为已贴地板）

251 K prompt（3 922 页/层）、容量 262 144、MTP3、headroom 256：池 45 085 页组
→ 窗口 R = (45 085 − 4 096) ÷ 15 = **2 732 页 = 175 K token**，环 1 364 页 = 一层流式段。

```text
每轮流式 = (3 922 − 2 732) × 16 × 71 680 = 1.365 GB
拷贝地板 = 1.365 GB ÷ 25.77 GB/s = 52.96 ms
实测每轮 = 2.48 ÷ 45.8 tok/s = 54.15 ms      →  露在外面的只有 1.19 ms（2.2%）
PCIe 占空比 = 52.96 ÷ 54.15 = 97.8%
```

也就是说：**重叠已经做到 97.8%，再改调度最多 +2.2%。** 作者那份策略的第 4 条 guard
（"PCIe 饱和就停止扩环"）在这里就是全部结论——他的三个信号里 `copy_engine_busy ≈ 0.98`
出现在第一轮，重分区一进去就落在"不要再扩环"的分支上，所以**这套自适应用在我们身上不会
做任何事，而扩环反而会启动他描述的恶性循环**（每多一环，每轮多搬 7%，见 §5.8）。

唯一还能动的是**减少每轮字节**，而它只有三处：

| 手法 | 收益 | 代价 |
|---|---:|---|
| 池子按实测空闲显存（256 → 128 MiB 余量） | **+5%**（+117 页/层窗口 → 拷贝 52.96 → 47.8 ms） | 启动后只剩 ~113 MiB，需要失败回退兜底 |
| 流式段用更小的页（`rk4v4` → 半页） | 最多 +40% | **打破"窗口内外同一格式"，是产品决定，须批准** |
| 关 MTP 换 63 K 窗口 | 窗口 +36%，但 decode −40% | 不划算（1.66× 大于窗口收益） |

结论：调度侧（环深、页槽队列、span 回收）在 rk4v4 这个基准档上**收益上限 2.2%**，不做。
剩下的提速只有"更小的流式页"一条，且需要质量决定。
### 5.11 仪表实测：调度空间不是 2.2%，是 ~50%（§5.10 的算术被推翻）

`--kv-stream-report` 在 251 K（prompt 246 652）、容量 262 144、`rk4v4`+MTP3、池 47 894 页组上实测：

| 量 | 值 |
|---|---:|
| 窗口 / 环 | 2 919 页/层（186.8 K token）/ 936 页 × 16 层 = 1.02 GB/轮 |
| 每轮 | 2.33 ÷ 36.4 tok/s = **64.0 ms** |
| 拷贝流占用（copy busy） | **42.57 ms** |
| 计算流等拷贝的时间（exposed wait） | **22.65 ms** |
| 每层 | 拷贝 2.66 ms，其中 **1.42 ms 露在外面**，只有 1.24 ms 被本层计算盖住 |

**§5.10 错在哪里**：那里用"拷贝 52.96 ms > 计算 40 ms ⇒ 已在拷贝地板 97.8%"推断没有调度空间。
实测反驳了这个推断——§5.10 用的 45.8 tok/s 来自另一份配置（headroom 128、更小窗口），而且
"拷贝比计算长"并不等于"拷贝被隐藏"：**每层的上载都在等上一层的 `consumed`，注意力又在等
本层的 `ready`，所以每层露出 1.42 ms，16 层 22.65 ms**。完美重叠时一轮就是拷贝本身：

```text
完美重叠 = 42.57 ms/轮 → 2.33 ÷ 0.04257 = 54.7 tok/s
实测                                       36.4 tok/s        → +50%
```

只要让拷贝流不空转（参考实现的页槽队列 + span 级回收），251 K 这一档就有一半的余量，不是 2.2%。
`--kv-stream-report` 的事件对代价实测只有 0.3%（36.5 → 36.4 tok/s），可以安全地拿来判方向。
### 5.12 环深两层（已落地）：每轮 64.0 → 56.9 ms，暴露减半

按 §5.11 的数字把环分成 `ring_depth` 个区域（每区一层流式的量，层 `L` 用 `L mod depth` 区），
拷贝提前 `depth − 1` 层发出，并且只等**该区域上一任**的 `consumed`。深度 2 是"每层计算
2.58 ms ≈ 每层拷贝 2.90 ms"的那个点。

实测（2026-11-09T00:5x+08:00，提交 `a60647b1`，251 K / 容量 262 144 / `rk4v4` + MTP3）：

| | 环深 1（§5.11） | 环深 2 | 环深 3 |
|---|---:|---:|---:|
| 窗口 / 环 | 2 919 页/层 / 936×16 | 2 835 / 1 020×16 | 2 738 / 1 117×16 |
| 每轮上载 | 1 024 MiB | 1 090 MiB | 1 190 MiB |
| 拷贝占用 | 42.6 ms | 47.5 ms | 51.9 ms |
| 暴露 | 22.65 ms | **11.4 ms** | 11.7 ms |
| 每轮 | 64.0 ms | **56.9 ms** | 61.0–62.9 ms |
| 接受长度 | 2.33 | 2.62–3.00 | 2.86–2.95 |

结论三条：

1. **每轮 −11%，暴露减半**。tok/s 从 36.4 到 48–53 里有一部分来自接受长度，所以只看轮时间。
2. **深度 3 更差**（拷贝 +9%，而暴露一动不动）。暴露与环深无关 ⇒ 残余的 0.7 ms/层 不是"拷贝
   赶不上"，而是每轮固定的起停：轮首 `depth` 层没有可重叠的计算，轮末最后一层的拷贝无处可藏。
   再拿回这 11 ms 要**跨轮预取**，不是加深环。深环回到 2（`a60647b1`）。
3. 现在拷贝流占轮时间 **84%**，所以下一个两位数收益必须减少**拷贝量**（更小的流式页），而不是
   继续找重叠。

原始输出与命令存根：[`kv-streaming-measurements.md`](kv-streaming-measurements.md)（M-01…M-05）。

## 6 失败模式与回退

| 场景 | 行为 |
|---|---|
| 启动时 `K/B < P`（上下文装不进池） | 拒绝启动，报“需要 `--max-context ≤ N` 或更低 KV”，不静默降级 |
| 拷贝失败 / 事件超时 | 本轮作废，整个请求失败；设备池标记为 inactive，下个请求重新建立映射 |
| 自适应找不到可行 `R` | 固定到最大可行 `R`（流式段最小），记一条 warning，继续服务 |
| 与不支持的组合同时打开 | 启动时拒绝：Hybrid prefix cache、多并发 > 1、TP/流水、非 Qwen3.5 Dense |
| `--kv-stream` 关闭 | 代码路径与今天完全一致（不走新分配器、不建槽、不记指标） |

---

## 7 指标与观测

每轮一条结构化记录（沿用 `structured_round` 的日志通道）：

- `active_pages`、`resident_pages_per_layer`、`ring_slots`、`f`
- `streamed_bytes_this_round`、`h2d_commands`、`h2d_ms`、`wait_ms`
- `deadline_misses / samples`、`copy_busy_ratio`、`ring_peak`
- 调节事件：`{reason, from_R, to_R, at_evaluation}`

汇总（`--json` 输出、perplexity 报告）：
`tok/s`、峰值 VRAM、host RSS、累计 H2D 字节、平均等待、重分区次数。

---

## 8 里程碑与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0** ✅ | 核实表的地址与发布、`cudaEventWaitExternal` 重放语义、`copy_from_host` 图外/图内配对、MTP KV 页数、PCIe 带宽 | 全部成立；已写回 §4.5/§4.7/§5.2/§9。唯一修正：MTP 池 428 MB（非 107 MB）→ 流式池 2.64 GB，上界下调到 15.3 |
| **M1** ✅ | host 权威页 + 常驻窗口 + 共享环 + 每层表 + 单流序列（4.4） | 262 144 `rk8v4` + MTP3 上 250 964 token 的 prompt 跑完，decode **14.1 tok/s**、峰值 15 028 MiB（§5.3） |
| **M2** | 一致性 | 同格式 on/off 的 greedy token ID 与 logprobs 逐 token 一致（8 K、32 K、120 K 三档） |
| **M3** | 自适应控制器（4.8） | 全驻留阈值可复现；扩/收环在日志里可解释；`copy_busy ≥ 0.8` 时不再扩环 |
| **M4** | 分段流水（4.10） | 262 144 达到 17–18.9；环从 280 MB 降到 ~2 MB 级；`int8` 跑通 |
| **M5** | 与其它能力共存 | 多序列、Hybrid prefix cache、TP 各写清支持或拒绝 |

退出标准（整个交付）：§1 的五条目标全部满足，且 §5 的上界与实际差距有解释。

---

## 9 未核实与风险

1. ~~表在重放间的地址稳定性~~ → **已核实（M0-1）**：地址稳定，发布只改内容。剩余细节：表必须在
   捕获前发布（捕获中会跳过 fence），所以窗口转动放在轮与轮之间。
2. ~~`cudaEventWaitExternal` 的多次重放语义~~ → **已核实（M0-2）**：仓库内已有 idiom。
3. ~~25 GB/s 带宽~~ → **已实测（M0-5）**：25.77 GB/s H2D，且与计算完全并行。
4. **页粒度 64 token** 意味着 262 K 有 4096 页/层；每轮 48–64 条 2D 拷贝命令的投递开销约
   0.1 ms，可接受；但若改成逐页命令会炸（见 4.4）。
5. **Legacy checkpoint 与流式的语义冲突**（4.9）：M1 必须显式拒绝，不能“看起来能跑”。
6. attention 在 262 K 的耗时：实测 250 964 token 时每轮 ~71 ms（14.1 tok/s、2.82 tok/轮），
   高于 §5.2 估的 28 ms——这 43 ms 的差就是 15.3 → 14.1 的来源。若 M2 优化 attention 路线，
   上界可以拿回来。
7. ~~预算拆分待精确~~ → **已由 planner 实际数字取代（§5.3）**：`--kv-capacity auto` 给出的槽数
   比 §4.5 的估算小，R 相应小 7–25%；上界按实际 14.1 修正。
8. ~~环槽的显存粒度~~ → **已实测**：`rk8v4` 262 144 时 20 002–21 945 个槽的 free-list 开销
   未体现在显存差上（`gpu sequence used` 2.65 GiB 与预算一致）。
9. **主机权威副本必须按层分配**（M1 已修）：初版只分配一层的页，十六层共用同一批记录，
   上传后每层拿到同一层的字节；内存 6.4 GiB（`rk8v4`）/16.4 GiB（`bf16`）是这一决定的代价，
   也是 `bf16` 不可作为产品默认的原因。
10. **decode 轮的上下文必须携带流式空间**（M1 已修）：三个 decode 批上下文各自内联构造
    `ExecutionCore`，漏填两个字段时它们静默地不拷不等待，读到的环是上一层留下的字节。
11. **长上下文的逐 token 对比没有可利用的基线**：同格式全驻留装不下 262 144。M2 的等价性依靠
    (a) 可启动的最大上下文（如 `rk8v4` 85 000 / `rk4v4` 120 000）上的 on/off 逐字节对比，
    (b) 环内字节与表项的单次直接校验（已在 prefill 与 decode 轮各做过一次，全 OK）。
   若需更强证据，需加一个 host 权威与设备全量的逐页比对工具。
12. ~~`--kv-headroom-mib` 的默认 1024 在流式下是浪费~~ → **已改**：流式下默认 256 MiB
   （`kDefaultKvStreamHeadroomBytes`，实测 251 K 从 14.1 到 16.9，+20%）；显式传
   `--kv-headroom-mib 128` 可到 17.7，但启动后只剩 113.5 MiB，故不作默认。
13. **`rk4v4` 作对比基线的前提**：190 K 上下文下 `rk4v4` 全驻留仍装不下（需 4.23 GB），
   所以长上下文的“全驻留”参照系只能用 `rk4v4` **开流式但窗口覆盖 ~96%** 的那一档
   （190 K 时 83.4 tok/s，每轮只搬 110 页），它实际就是全驻留的速度。
## 10 施工顺序与触点（M1 落地地图）

按依赖顺序，每一步都能单独编译、单独验证：

| # | 改动 | 触点（已核实） | 验证 |
|---|---|---|---|
| 1 | 扁平槽池分配（K/V/Kscale/Vscale 四个平面，总槽数 `16R + S`） | `src/core/paged_kv_cache.{h,cpp}`、`src/core/paged_kv_storage.h` | 单测：槽数、页字节、对齐 |
| 2 | 每层表缓冲 `[16][max_pages] I32` + 表内核（流式段恒等，常驻窗口转动） | `src/models/qwen3_5/state/decoder_state.{h,cpp}` `layer_view()` | 单测：任意 `(active,R)` 下映射单射且不越界 |
| 3 | 视图切换到扁平池（`k_pages` 基址 + 本层 `block_table`） | `decoder_state.cpp:216-235`；`text.cpp:890` 调用点 | 全驻留时（`S=0`）与今天逐字节相同 |
| 4 | host 权威页（复用 `HostKVPageLayout` / `HostKVArena`），262 K `rk8v4` = 6.85 GB pinned | `src/core/host_kv_arena.{h,cpp}`、`program/storage/host_kv_store.h` | 启动预算分配正确、释放无泄漏 |
| 5 | 写入路径：新 token → 设备常驻窗口 + 异步 D2H；mutable tail 不预取 | `src/ops/kv_cache/append/*`、`execution/text.cpp` | 同 token 序列下 host 与设备字节一致 |
| 6 | 每轮序列：每层 1–3 条 2D H2D + `ready/consumed` 事件；图内等 | `program/structured_round.cpp:136` 的模式；decode 轮的构建处（M0 定位） | on/off 逐 token 一致；等待时长入日志 |
| 7 | 容量规划与选项：`--kv-stream`、host 预算、`R/S` 初值 | `src/runtime/engine/kv_capacity.cpp`、`include/ninfer/engine_options.h`、`apps/cli/options.cpp`、`src/serve/serve_options.cpp` | `--help` 与文档一致；关闭时零行为变化 |
| 8 | 拒绝组合：Hybrid prefix cache、`max-concurrency > 1`、TP/流水 | `src/runtime/engine/engine.cpp` 的校验段 | 启动报错文案明确，不静默降级 |

---


---

## 11 与参考实现的逐项对照（2026-11-08）

参考仓库 `RaymondHuang210129/llama.cpp-adaptive-kv-streaming`（本地只读副本
`D:\GithubRepo\llama.cpp-adaptive-kv-streaming`），其 README 自述的目标场景与本交付**完全相同**：
RTX 5070 Ti 16 GB、`Qwen3.8-27B` `UD-Q3_XL`、262 144 上下文、Q8_0 K / Q4_0 V、单槽。

### 11.1 机制：同一套设计，逐项对应

| 环节 | 参考实现 | 本实现 |
|---|---|---|
| 权威副本 | pinned host，`--kv-stream-stage-mib N` 定设备池 | 同（`HostKVArena`，6.4 GiB @`rk8v4` 262 144） |
| 切分 | 设备池 = 每层等量常驻页 + 跨层共享环（`llama_kv_stream_pool_layout_make`：`pages = (pool − scratch)/page_bytes/layers`，**每层相同**） | 同（`plan_kv_stream_split`：`R = (B − P)/(layers − 1)`，16 层同一 `R`） |
| 搬运单位 | 区域 =（层，token 区间，字节），页 256 token | 页 64 token，槽 = 一层的一页 |
| 搬运方式 | 每区域 K/V 各一条 `cudaMemcpyAsync` H2D 进 stage slot，`ready` 事件；compute 流等（`fattn.cu:1315-1335`） | 每层每 plane 一条 `cudaMemcpy2DAsync`，`ready/consumed/dropped` 事件 |
| 环深度 | `target_ring_working_set_ratio = 1.10` × 一层的流式页 | 恰一层（`S = P − R`） |
| 跨层预取 | 有：`stage_slots` + `cross_layer_prefetches`，32 页一批（`fattn.cu:145,1343`） | **无**（§4.10 M4a） |
| 自适应重切 | 有：`llama_kv_stream_partition_adapt`，三指标（deadline miss / copy busy / ring peak）+ 滞回 + cooldown，每次 ±1 页/层 | **无**（M3） |
| 内核选择 | `span_tuner` 在“整段”与“32 页有界流水”间按端到端延迟择优（≥0.5%） | 无（M4b） |
| 拒绝组合 | 仅 Qwen3.5、仅 target 上下文、`-np 1`、必须 FA（`llama-kv-stream-config.cpp`） | 同（+ 关图、要求 `--kv-capacity auto`） |

**关键点：参考实现也没有“每轮只拼一部分”的机制。**环被刻意留得只比一层略深（1.10）——
`partition_adapt` 求解 `ring >= 1.10 × (active − resident)` 的最大 `resident`——所以
**环外的每一页每轮都过 PCIe 一次**，与本实现一致。所谓“比例是竞争得来的”指的是
`resident/ring` 的边界在**固定池内**由三个反馈指标小幅推动（每次 1 页/层），不是每轮只搬一部分。

### 11.2 速度：同一台卡、同一个模型、同一上下文

| 上下文 | 参考实现（RTX 5070 Ti 16 GB，PCIe 5.0） | 本实现（RTX 4080 SUPER 16 GB，PCIe 4.0 25.77 GB/s） |
|---:|---:|---:|
| 8 K | ~50 tok/s | 86.7（80 K 点上） / 114（全驻留） |
| 49 K | ~40 | — |
| 114–120 K | ~30 | **43.9** |
| 176 K | ~20 | 22.6（推算，M4a 后 ~27） |
| 253 K | **~10** | **14.1** |

（参考数字来自作者 Medium 文与 Reddit 帖；ROCm 移植版在同一模型 262 144 上 9.5 tok/s。
本实现带 MTP，每轮 2.82 token，参考该配置无 MTP、每轮 1 token，这是差距的主要来源。）

### 11.3 问题所在：三个可兑现的差距，与一个不能兑现的期待

1. **M4a 跨层双缓冲（真差距）**：参考实现用共享环的额外深度把后几层的拷贝叠到当前层的
   attention 上，我们按“环上一次一层”串行。中长上下文白拿 +40%（120 K：43.9 → ~62），
   251 K 退化到零（环深度要从常驻区切）。**先做这个。**
2. **M3 自适应重切（真差距，但 M1 静态切分已足够）**：`auto` 池把常驻区顶到最大，重切只在
   工作集变化（多轮、prefill→decode）时有意义；参考的滞回参数可直接借用
   （grow 3 / shrink 8 / cooldown 64 evaluations，miss 阈值 1%，copy 饱和 80%）。
3. **M4b 分段流水（真差距，251 K 的唯一重叠来源）**：环只需 `k` 页 → `R` 变大 → 每轮 4.56 GB
   不变但可完全重叠 → **15.9 tok/s（+13%）**；代价是新的 attention 入口。
4. **不能兑现的期待**：`80–120 tok/s` 是**全驻留**的数（本卡 `rk8v4`+MTP 的上限约 99 K token）。
   262 144 上没有任何配置能让 KV 不再过 PCIe；参考实现用同一套机制在更快的总线上拿 ~10 tok/s。
   “每轮只拼一部分”在参考实现里不存在，也没有对应的算法——稠密 attention 一个 token 就要读全历史。

## 5.13 span 级流水：环按"一段"而非"一层"（设计，未实现，2026-11-09）

§5.12 的结论是"暴露与环深无关，再拿回要跨轮预取"。读完注意力的发射代码后，这个读法要改：
**暴露不是轮首轮末，是稳态饥饿**，而饥饿的根源是**环的粒度和计算的粒度差 26 倍**。

### 代码事实（2026-11-09 核对）

1. 稠密注意力**本来就是 split-KV**，段界是页的整数倍
   （`src/ops/softmax_attention/dense/causal_cache/small_t.cuh:180-192`）：
   `pages = div_up(div_up(window, kPagedKVPageSize), active); start = split * pages * 64`。
   251 K 窗口下 `CausalD256H24Kv4`（`SmallTSplitScale = 1`、`kCausalSmallTSplitKeyLimit = 3 968`、
   `SmallTMaximumSplits = 85`）得到 **85 段，每段 46 页 = 2 953 key**。
2. reduce 按**段号**处理 partial（`small_t_k8v4.cuh:707-717` 的 `reduce_output_kernel`：
   先对 85 个 partial 取最大 `m`，再算 `weights`，再 `causal_weighted_split_sum` 按 split 序号累加）。
   ⇒ 输出是"partial 集合 + 段号顺序"的纯函数，**与谁在什么时候算出来无关**。
3. 我们的流式页 = 64 token = `kPagedKVPageSize`，所以一段的流式页正好是整页的连续区间。

### 为什么这样拆是逐位相同的

同一批 partial、同一次 reduce ⇒ 同样的字节。要保证的是三件事不变：

- `window`、`visible_capacity`、`positions` 传原值。`causal_small_t_quantized_active_splits(window,
  …)` 按 window 算段数，传 span 的 key 区间会让段界变化，逐位相同立刻失效。
- partial 槽位仍按**总段数 85**步进（`causal_partial_*_index(..., split_count, ...)`），
  所以发射参数要新增 `split_begin`，而 `split_count` 仍是总数（今天它取自 `gridDim.y`）。
- 每层仍只做**一次** reduce，且顺序不变。

这与 `--invariant-attention`（`CausalKeyPartition::segment_keys`）是同一类保证：段是 key 空间的函数。
**不需要新金标，也不算质量变更。**

### 现状的浪费

| | 现状 |
|---|---:|
| 环 | `2 × 1 227 页 = 176 MB` |
| 计算每层实际消费粒度 | 46 页 |
| 过配倍数 | 26× |
| 每层拷贝 / 两层间可用计算 | 2.97 ms / 1.98 ms |
| 每层饥饿 | 0.99 ms × 16 = **15.8 ms/轮** |

那 176 MB 本可以是窗口：`2 454 槽 ÷ 16 层 = 153 页/层 = 9 800 token`。

### 容量算术（`16R + depth × span_pages ≤ 48 358`，P = 4 096，层 = 16）

| span | depth | 环槽 | 窗口 token | 尾部 | 每轮拷贝 | 每轮（含发射税） | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 整层（现状） | 2 | 2 454 | 183 616 | 67 384 | 47.5 ms | 56.7 ms | 69.2 |
| 4 段（184 页） | 3 | 552 | 191 232 | 59 768 | 42.8 | 43.5–44.2 | 88–90 |
| 2 段（92 页） | 3 | 276 | 192 320 | 58 680 | 42.1 | 44.9 | 87 |
| 1 段（46 页） | 3 | 138 | 192 832 | 58 168 | 41.7 | 47.3（税 5.4） | 83 |
| 1 段（无发射税） | 3 | 138 | 192 832 | 58 168 | 41.7 | **41.9** | **93.6（+35%）** |

发射税按每次 4 µs、85 段/层 × 16 层 = 1 360 次/轮估算；流式现在**不开 CUDA Graph**，
所以这一项必须靠"恢复流式图"消掉（handoff §4 第 4 项）。**保守口径 +25~35%，全部预测值，未实测。**

> 修订 §11.3 第 3 项：那里按"每轮 4.56 GB 不变"估了 +13%。字节其实会降（环变小、窗口变大），
> 所以预测上修到 +25~35%。

### 要改的三处

1. **`small_t_k8v4` 发射**：新增 `split_begin` 参数，`split` 由它推出；`split_count` 仍传总数。
   `causal_small_t_partition_range` 的 `window`/`active` 不变。
2. **`plan_kv_stream_split`**：环改为 `depth × span_pages`，窗口不再受 `depth·P` 约束；
   `advance`/`upload`/`kv_stream_ring_slot` 按段滚动（段内页连续，一次 DMA 约 8 MB，粒度健康）。
3. **Program 侧**（`src/models/qwen3_5/execution/text.cpp` 的流式分支）：每层按 span
   `upload → wait_ready(span) → 发射该 span → 记 consumed`；层末一次 reduce。事件数 16 → 16 × spans。

### 风险

| 项 | 说明 |
|---|---|
| 发射税 | 最大一项（−8~13%），靠恢复流式 CUDA Graph 吃掉 |
| 跨窗口边界的 span | 一段可能一半在窗口、一半在尾部；只等它的流式页 |
| 段数 85 未经日志确认 | 来自 `default_splits(251 000) → clamp 85` 的推导，跑一次读 `round_samples` 即可 |
| 图捕获 | 每 span 一次发射，图里节点数 ×spans；稳态每轮形状固定，可按页边界重捕 |

### 验收口径

1. **等价性**：`check --suite equiv` 仍 4/4 PASS（尤其 183 K 的 512 token 逐 id 相同）——
   这是这条设计"不算质量变更"的唯一凭据。
2. **速度**：`bench --case stream-251k`，比 56.7 ms 的基线。
3. 若逐 id 变了，说明三件不变量之一被破坏，先修不变量，不要调参。

## 5.14 主机侧注意力分担：算术上先出局（设计候选，不推荐）

想法：尾巴（窗口外 58–67 K token）每轮过总线，让 CPU 线程直接算这部分注意力，GPU 只算窗口，
按 online-softmax 合并（fp64 权重）。与 `--expert-dma-share`（CPU 算一部分专家）同构。

**它死在 FLOPs，不是死在精确性。** 每个（key, 层, query）要 `2 × 24 heads × 256 dim = 12 288` MAC，
对应 1 120 字节（rk4v4 每层每 token）⇒ 强度约 22 FLOP/字节；一轮 3.92 个 query 共享同一批字节，
有效强度约 86 FLOP/字节。

| | 数值 |
|---|---:|
| 一轮尾巴注意力 | `58 168 × 16 × 3.92 × 24 576 ≈ 90 GFLOP` |
| 一轮总时长（拷贝受限） | 41.7 ms |
| GPU 需要的持续算力 | ≈ 2.1 TFLOP/s（全窗口约 15 TFLOP/s） |
| 16 核 AVX-512 fp32 实际 | ≈ 0.1 TFLOP/s |
| **CPU 能接的比例** | **≈ 3%** |

即使用满 FP32 峰值也只有个位数百分比，收益可以忽略、工程量却是全表中最大的（要新增 CPU 端的
rk4v4 反量化 + QK^T/PV 核，还要和 DMA 抢内存带宽）。**结论：不做。**

顺带纠正一个说法：这条**不是**"降精度"。同样的反量化值、fp64 合并、分段更短反而更接近真值，
token 差异只会出现在近乎并列的位置——按"KLD / 智商不变"的判据它是可接受的；它出局的原因纯粹是
算力对不上带宽。真正被这条判据否决的是**尾部降精度**（改的是 KV 的表示），不是这条。
