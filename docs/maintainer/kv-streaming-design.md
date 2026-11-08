---
id: kv-streaming-design
type: architecture-design
title: 自适应 KV 流式：16 GB 上的 262 144 上下文
status: draft
parent: ninfer-kv-streaming
depends-on:
  - paged-kv-cache
  - kv-stream-op
references:
  - adaptive-kv-streaming-inputs
  - qwen3-5-model
---

# 自适应 KV 流式

目标、机制、预算、决策点与验收。外部资料（参考文章与 CUDA Graph 说明）在
[adaptive-kv-streaming-inputs](adaptive-kv-streaming-inputs.md)，它们是设计输入，不是本仓库的状态。

## 1 目标

在 RTX 4080 SUPER（16 376 MiB，sm_89，PCIe 4.0 x16）上，让 Qwen3.8-27B Dense 跑满
**262 144 原生上下文**，且：

- **质量下限 `rk8v4`**；`rk4v4` 及以下在真实任务上幻觉与记忆不可接受（所有者实测）。
  流式稳定后倾向 `int8`。
- decode 可用（见 §4 的真实上界），结果与全驻留**逐 token 一致**。
- 默认关闭；`--kv-stream` 打开。关闭时行为与今天逐字节一致。
- 只加流式，不删任何模型族。

## 2 现状（本机实测，2026-11-08）

* 权重上卡 11.6 GiB；IQ3_S 文本。
* 权重之后可用于运行时容量的显存：**4 757 913 600 B（4.43 GiB）**；带 `--spec mtp
  --draft-tokens 3` 时 **4 032 299 008 B（3.76 GiB）**。
* 固定开销实测：无草稿器约 0.44 GB，带 MTP3 约 0.96 GB；**带 MTP3 的 KV 预算约 3.07 GB**。

262 144 的启动探针（`--vision` 关）：

| `--kv-dtype` | 每 token | 262 144 需要 | 结果 |
|---|---:|---:|---|
| `bf16` | 65 536 B | 17.62 GB | 失败 |
| `int8` | 33 792 B | 9.90 GB | 失败 |
| `rk8v4` | 26 112 B | 7.33 GB | 失败 |
| `rk4v4` | 17 920 B | 5.18 GB | 失败（差 0.42 GB） |
| `rk2v4-e8` | 13 824 B | 3.38 GB | 成功 |

阶梯（`rk8v4` + MTP3 + greedy，128 token 生成）：

| 上下文 | prefill | decode | MTP 接受长度 | 峰值显存 |
|---|---:|---:|---:|---:|
| 8 192 | 2.37k tok/s | 114.3 tok/s | 2.76 | 12 672 MiB |
| 32 768 | 2.28k | 115.0 | 2.89 | 13 322 MiB |
| 65 536 | 2.11k | 117.6 | 3.10 | 14 188 MiB |
| 120 000 | 1.86k | 109.1 | 3.10 | 15 632 MiB |

32K 同 prompt 的 KV 对照：`bf16` 121.8 / `int8` 116.6 / `rk4v4` 115.1 / `rk2v4-e8` 122.2 tok/s。

模型几何（artifact 核实）：64 层，其中 **16 层 full attention**（3、7、…、63）、48 层线性注意力
（KV 不随上下文增长），4 个 KV 头、head_dim 256。所以流式只涉及 16 层。

**一层的 KV 大小**（262 144）：

| KV | 每层 | 16 层合计 |
|---|---:|---:|
| `rk8v4` | 427 MB | 6.85 GB |
| `int8` | 554 MB | 8.86 GB |
| `rk4v4` | 294 MB | 4.70 GB |

## 3 机制

### 3.1 已经具备的物理基础

- KV 以页存储，attention 通过 **block table（显存里的数据）** 寻址，页号不是 graph key
  （`paged-kv-cache`、`decode_graph`）。因此“换页”不需要重捕获图。
- 每层有独立 plane（`state/decoder_state.cpp` 的 `plane_index()`），但**全部层共用同一张
  block table 行**：页组 `g` 在每层是同一个物理页号。这是流式的第一个结构性约束。
- `ops/kv_stream` 已有可捕获的“host 块 → 设备 slot”拷贝：位置来自设备 I32、网格固定，
  `kv_stream_resolve` 在 GPU 内认领并搬运。
- 捕获体内等待外部事件的先例：`program/structured_round.cpp:136-141` 用
  `cudaStreamWaitEvent(stream, e, cudaEventWaitExternal)`。

### 3.2 流式形状

一块 **slot = 一层在当前上下文长度下的完整 KV**。层 `l` 用 `l mod R` 号 slot；层 `l` 算完，
才能把层 `l+R` 拷进同一个 slot。由此得到两条顺序保证：拷进 `l+R` 之前 `l` 已算完；算 `l+R`
之前拷贝已完成。MVP 用**单流**（先拷后算），第二步再把拷贝放到图外的第二条流、用外部事件让
图内每层等它。

resident 层（全驻留）与 streamed 层（走 ring）按上下文长度**自适应划分**，规则照参考文章
§自适应重分区：全驻留 → 公平释放一小部分成 ring → 只有“扩 ring 仍能隐藏延迟”时才扩 →
PCIe 饱和即停止（否则是“更大 ring → 更少驻留 → 更多搬运 → 更慢”的恶性循环）。

## 4 上界的真实算法（关键）

decode 的每轮代价**按轮计，不按 token 计**：一轮验证 K+1 个 token 只搬一次 KV。所以

    tok/s ≈ (每轮流过的字节 ÷ 有效带宽)⁻¹ × 每轮接受长度

且**预取能掩盖的部分可以忽略**：262 144 时一层拷贝 427 MB ÷ 25 GB/s ≈ 17 ms，
而一层计算 ≈ 109 tok/s × 3.1 tok/轮 下的一层份额 ≈ 0.5 ms。拷贝是计算的 30 倍，
重叠只能吃掉 3%。**这个阶段的 decode 就是带宽的函数**。

按 KV 预算 3.07 GB、ring 深度 2、有效带宽 25 GB/s（待实测，PCIe 4.0 x16 空闲 2.5 GT/s、
加载时 16.0 GT/s）：

| KV | ring | 常驻层 | 流式层 | 每轮搬运 | 每轮时间 | 轮/秒 | 带 MTP3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rk8v4` | 2×427 MB | 5 | 11 | 4.70 GB | 188 ms | 5.3 | **≈16 tok/s** |
| `int8` | 2×554 MB | 3 | 13 | 7.20 GB | 288 ms | 3.5 | **≈11 tok/s** |
| `rk8v4` | 1×427 MB | 6 | 10 | 4.28 GB | 171 ms | 5.8 | ≈18（但每次都等） |
| `rk8v4` 全流 | — | 0 | 16 | 6.85 GB | 274 ms | 3.6 | ≈11 |

（这些都是**上界**，未计 attention 计算与同步；实测只会更低。MTP 接受长度在 262K 可能低于
3.1。参考文章在 PCIe 5.0、无投机解码下 253K 约 13 tok/s，与这张表同量级。）

结论：**流式不是免费的**。它换来的是“能跑”，代价是 262K 时 16 GB + `rk8v4` 的
tok/s 从“不可能”变成“十几”。要提高这个数字，顺序是：减少 KV 字节 > 提高常驻比例 >
ring 深度 > 预取重叠 > 内核。

## 5 决策点（实现前定，写进本文件）

| 编号 | 问题 | 备选 | 倾向 |
|---|---|---|---|
| D1 | 页命名空间：流式层放哪 | A 新建 streamed pool（自己的 execution table 与容量曲线项）；B 给 Main pool 加 per-layer 表 | 先 A：局部、可回退 |
| D2 | host 权威副本归谁 | A 扩 `HostKVPageLayout`/`HostKVArena`；B Program 自建 pinned store | A，复用既有页几何 |
| D3 | slot 粒度 | A 整层（MVP）；B 页组（更细，ring 更省显存） | A 先行，B 视 262K 的 ring 开销再定 |
| D4 | ring 深度 | 固定 2（MVP）→ 自适应 | 固定 2 起步 |
| D5 | 与 prefix cache 的关系 | Legacy（默认）先支持；Hybrid 后接 | Legacy 先行 |

D3 之所以重要：262K 时一个 slot 是 427 MB，ring 深度 2 就吃掉 KV 预算的 28%。
页组粒度（例如 slot = 1/4 层）能把同样的预取深度压到 1/4 的显存，但排序与依赖更复杂。

## 6 不变量

1. 拷贝期间源与目标都被 pin；目标在拷贝完成且 epoch/coverage 校验通过前不进入 block table。
2. 一次 GPU 执行单元内，被选中行的 block table 不变、page payload 不搬迁。
3. 同一 `--kv-dtype` 下，streaming on/off 的 greedy token ID 与 logprobs 一致。
4. 失败（OOM、slot 不足、拷贝失败）回到完整 inactive 终态，不留半改写的映射。
5. 默认关闭；关闭路径与今天逐字节一致。
6. 每轮结构化日志：常驻字节、流式字节、H2D 次数与耗时、等待时间、调节原因。

## 7 验收

1. `rk8v4` 的 262 144 能启动并稳定 decode；`int8` 同（可能是更慢的一档）。
2. 同格式 on/off 逐 token 一致（8K、32K 两组指纹）。
3. 全驻留路径不回退：120K `rk8v4` 109.1 tok/s 为标尺。
4. 关掉 `--kv-stream` 即今天的行为。
5. 报告真实数字，并解释比 §4 上界差多少（同步、计算、接受的损失）。

## 8 里程碑

| 阶段 | 内容 | 完成条件 |
|---|---|---|
| M0 | 核实 §3.1 的四条前提（block table 稳定性、共用行、外部事件、page 发布） | 结论写进本文件 |
| M1 | host 权威 KV + 固定划分 + 固定 ring + 单流拷贝 | 262K `rk8v4` 跑通 |
| M2 | 一致性 | on/off 逐 token 一致 |
| M3 | 图外拷贝流 + 图内外部事件 | 每轮等待时间显著下降 |
| M4 | 自适应划分 | 全驻留阈值与停止扩 ring 的条件可复现 |
| M5 | `int8` | 262K `int8` 跑通并记录 |
