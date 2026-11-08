---
id: kv-streaming-design
type: architecture-design
status: draft
title: 自适应 KV 流式：host 权威 KV + 设备 resident 区 + 环形 slot
parent: engine-architecture
depends-on:
  - paged-kv-cache
  - kv-stream-op
references:
  - phase-1-cutlist
  - resource-scheduling-context-cache
---

# 自适应 KV 流式（阶段 2 设计）

状态：**草案**。本文件是阶段 2 的设计权威；它记录已核实的机制、必须保持的不变量、
尚未决定的决策点与验收标准。实现不得绕过这里的合同。

## 1. 要解决的问题（实测数字）

RTX 4080 SUPER 16 376 MiB，IQ3_S 文本权重上卡 11.6 GiB 后，**可用于运行时容量的显存 =
4 757 913 600 B（4.43 GiB）**（`--vision` 关、`--spec` 关的启动探针，2026-11-08）。

单 token KV 字节（16 个 full-attention 层 × 4 个 KV 头）：

| `--kv-dtype` | 单 token 单头 | 每 token | 262 144 token | 262 144 启动探针 |
|---|---:|---:|---:|---|
| `bf16` | 1 024 B | 65 536 B | 16.41 GiB | 需要 17.62 GB → 失败 |
| `int8` | 528 B | 33 792 B | 8.26 GiB | 未探针（按 rk8v4 推算会失败） |
| `rk8v4` | 408 B | 26 112 B | 6.38 GiB | 需要 7.33 GB → 失败 |
| `rk4v4` | 280 B | 17 920 B | 4.28 GiB | 需要 5.18 GB → 失败（差 ~0.42 GB） |
| `rk2v4-e8` | 216 B | 13 824 B | 3.38 GiB | **启动成功** |

**质量下限（所有者 2026-11-08）**：`rk8v4` 是可用的最低档；`rk4v4` 及以下在真实任务上
幻觉与记忆明显变差、无法工作；流式做好之后倾向于用 **`int8`**。

因此阶段 2 的目标不是"让 262K 能启动"（`rk2v4-e8` 已经能），而是：

> **在 `rk8v4` / `int8` 的质量下跑 262 144 上下文**，并且 decode 速度可用、结果与全驻留逐 token 一致。

带宽上界（PCIe 4.0 x16，按有效 25 GB/s 估算，阶段 2 必须自测）：

| 配置 | 常驻显存 | 每 token 需搬 | 每轮上界 | 说明 |
|---|---:|---:|---:|---|
| `rk8v4` 262K | 4.2 GB | 2.18 GB | ~11.5 轮/s | 需把约 2/3 的 KV 留在显存里 |
| `int8` 262K | 4.2 GB | 4.06 GB | ~6.2 轮/s | 质量档更高，代价是每轮搬运几乎翻倍 |

**投机解码是这里的乘数，而不是"性能项"**：流式的代价按 **round** 计，不按 token 计。
一轮验证 K+1 个 token 只搬一次 KV，所以 `tok/s ≈ 每轮上界 × 每轮接受长度`。
本机实测 MTP3 的接受长度 3.33 tok/round（短上下文，77.8% 接受率），DFlash2 的窗口更大。
这意味着一开始就要把 MTP/DFlash2 纳入设计，而不是最后再调。

## 2. 数据流与所有权

```
权威副本：pinned host（每层 packed 页，身份页表）        ← 所有者：Program 的 host KV store
   │  H2D：按层、按 slot（图外拷贝流 + 外部事件）
   ▼
设备：resident 区（若干层的全部页） + ring（R 个 layer-sized slot）
   │  block table（显存数据，Engine 生命周期内地址稳定）
   ▼
paged attention（不变）
```

- 一个 **slot = 一层的一份完整 KV**（dense attention 在一层内要读整个可见前缀，
  这与 sparse 的"512 个 4-token 块"根本不同）。
- 层 `l` 的 slot = `l mod R`。ring 的顺序保证：层 `l` 算完后才能把层 `l+R` 拷进它的 slot。
- **不改 attention 数学**：流式只改变"页放在哪"。

## 3. 决策点（实现前必须定，写在实现 PR 里）

### 3.1 页命名空间（最大的一处结构性选择）

现状（已核实）：每个 layer 有自己的 plane（`state/decoder_state.cpp:plane_index()`），
但**同一个 pool 只有一张 execution table，一行 block table 被全部 layer 共用**，
即页组 `g` 在每层都是同一个物理页号。详见 `paged-kv-cache` §6.4 与
`state/decoder_state.cpp:layer_view()`。

两条路：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 新建 streamed pool | 流式层单独一个 pool（自己的 execution table、自己的容量曲线项） | 容量曲线多一维；`SequenceCapacityCurve` 要能表达"resident 页 + ring 页"两种寿命 |
| B. 给 Main pool 加 per-layer 表 | 保持一个 pool，块表按层索引 | 改 `PagedKVLayerView.block_table` 的消费者合同与发布路径（`paged-kv-cache` §11 的批量发布、pinned shadow fence） |

倾向：**先 A**（局部、可回退），B 留作第二阶段后期优化。

### 3.2 同步方式

已核实：`program/structured_round.cpp:136-141` 在捕获体内用
`cudaStreamWaitEvent(stream, uploaded_, cudaEventWaitExternal)` 产生**外部事件节点**，
每次 replay 等待图外先发布的 mask 上传。流式预取可以直接复用这条路径：

- 图外：拷贝流按层 H2D（`cudaMemcpy2DAsync` 或 `kv_stream` 的 copy kernel）+ `cudaEventRecord`；
- 图内：每层 attention 前 `cudaStreamWaitEvent(..., cudaEventWaitExternal)`。

未核实：捕获体内 fork 到第二条流的先例。**MVP 只做单流**（先拷后算），重叠作为第二步。

### 3.3 页内容与 host 写入

进入流式后，KV append 写 host 副本；设备侧只保留工作集。resident 层仍按今天的路径直接写设备。
对同一层，host 与 device 的 frontier 必须一致可验证（epoch + committed columns）。

## 4. 不变量

1. copy 期间源与目标都被 pin；目标在 copy 完成且 epoch/coverage 校验通过前不进入 address space
   或 block table。
2. 一个 GPU execution unit 内，选中行的 block table 不变、page payload 不搬迁。
3. 同一 `--kv-dtype` 下，streaming on/off 的 greedy token IDs 与 logprobs 一致。
4. 失败（OOM、slot 不足、拷贝失败）回到完整 inactive 终态，不留半改写映射。
5. 默认关闭；关闭时路径与今天逐字节一致。
6. 每轮的结构化日志记录：resident bytes、streamed bytes、H2D 次数与耗时、等待时间、
   调节原因（全驻留/释放/扩 ring/饱和）。

## 5. 自适应策略（按所有者给的参考文章，落到本仓库）

1. 全驻留：能装下就完全不流（R = 0，不开流式路径）。
2. 容量不足时：从每个 full-attention 层**公平**释放一小部分，汇成共享 ring。
3. ring 扩大的条件：预取跟不上 **且** PCIe 未饱和；饱和后停止扩大（否则形成
   "更大的 ring → 更少常驻 → 更多搬运 → 更慢预取" 的恶性循环）。
4. 带 hysteresis，避免来回调整；每次调整记一条日志。
5. 起始实现用**固定 bucket**（例如 ring 深度 2/4/8），稳定后再做连续调节。

## 6. 指标与验收

每档上下文（8K/32K/64K/128K/200K/262K）记录：

- prefill tok/s、decode tok/s（含 MTP/DFlash2 的每轮接受长度）、峰值 VRAM、host RSS；
- resident bytes、streamed bytes/token、H2D 次数与字节、等待时间、slot 命中/驱逐次数；
- 正确性：同格式 on/off 的 greedy IDs 与 logprobs 一致、planted-code recall、perplexity。

验收（阶段 2 出口）：

1. `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode；
2. 同格式 on/off 逐 token 一致；
3. 全驻留路径性能不回退（120K `rk8v4` 与基线一致）；
4. 失败可回退：`--kv-stream` 关闭即今天的行为；
5. 报告真实数字，包括"比上界差多少"的解释。

## 7. 未决与风险

- 页命名空间（§3.1）未定。
- 捕获体内多流 fork 未验证。
- host pinned 用量：262K `rk8v4` 的 host 副本约 6.9 GB，加上既有 8 GiB host cache 需要重新分配预算。
- `--vision` 会压缩可用显存（server 场景 120K `rk8v4` 只剩 145 MiB），流式必须与 Vision 共存。
- MTP 的额外 KV pool 会再占一份（每 token 1 632 B × 上下文）。
