---
id: ninfer-kv-streaming
type: goal-and-requirements
title: 让 Qwen3.8-27B 在 16 GB 上跑满 262 144 上下文的 KV 流式
status: draft
---

# 目标：16 GB 上的 262 144 上下文

## 问题

RTX 4080 SUPER 有 16 376 MiB。Qwen3.8-27B 的 IQ3_S 文本权重上卡占 11.6 GiB，权重之后只剩
**4.43 GiB** 给运行时（带 MTP3 时 3.76 GiB）。而 262 144 的 KV 需要：

| `--kv-dtype` | 262 144 的 KV | 结果 |
|---|---:|---|
| `bf16` | 16.4 GB | 装机不可能 |
| `int8` | 8.3 GB | 装机不可能 |
| `rk8v4` | 6.4 GB | 装机不可能（差 3.3 GB） |
| `rk4v4` | 4.3 GB | 差 0.42 GB |
| `rk2v4-e8` | 3.4 GB | 能装，但**质量不可接受** |

所以今天的真实上限是“120 000 上下文 + `rk8v4`”（实测 decode 109.1 tok/s、峰值 15 632 MiB），
再往上只有两条路：降低 KV 质量（被所有者否决）或 **把每层 KV 的一部分留在 host，每轮按份额
拷回显存**（不是“某些层全驻留、某些层整层流式”）。

## 交付

在 16 GB 上跑 262 144 原生上下文，`--kv-stream` 选择启用，且：

1. **质量**：`rk8v4` 是最低档（`rk4v4` 及以下幻觉与记忆不可接受，所有者实测）；流式稳定后支持
   `int8`。
2. **数值**：同一 `--kv-dtype` 下，streaming 开/关的 greedy token ID 与 logprobs 完全一致。
3. **默认**：关闭。关闭时行为与今天逐字节一致；打开是 opt-in。
4. **诚实的速度**：262 144 时 decode 是 PCIe 带宽的函数，每层只能常驻约 34%（`rk8v4`，实测
   25.8 GB/s）：单流约 **15.3 tok/s**，拷贝与计算重叠后约 **18.9 tok/s**；`int8` 约 10.6 / 12.6。
   实际数字与解释一起报告。
5. **不牺牲现有能力**：全驻留路径（≤120 K）不回退；Vision、全部 KV 格式、MTP/DFlash2、
   多卡路径都不动。
6. **拷贝与计算重叠到拷贝流不再空转**：一层内的注意力按“一帧”（拷贝批次量级，上游取 32 页
   量级）边拷边发射，而不是等整层尾部搬完；页离开窗口后仍只搬一次、表项只改两次，
   但环占用的槽从“一层的尾部”降到“几帧”。
   判据是 `exposed_seconds / round`（M-10 口径）：251 K 状态 ≈ 15.8 ms。
   **落点：INT8 家族（`rk8v4` / `rk4v4` / `rk2v4` / `int8`）；`--kv-dtype k8v4` 是另一条 kernel，不在本项交付内。**
   **状态（2026-11-09，M-12）：机制已实现且逐位相同，但未达标。** 251 K 上帧路径的暴露
   从 15.8 降到 8.7 ms，每轮却从 56.7 涨到 61.1 ms（64.0 tok/s vs 69.2），因为链路本身占用
   每轮的 83%，层内切细只是在重排等待。因此**默认仍是整层拷贝**（实测最佳），帧路径由
   `NINFER_STREAM_FRAMES=1` 开启。本项未完成，下一步是减字节（帧级环 −6%，或更小的流式页），
   不是继续在层内切细。

## 不在本次范围

- 删除其它模型族或其它 GPU 架构（已明确放弃；能力保留）。
- UVM / `cudaMallocManaged` 路线（参考文章已证明页面抖动会毁掉 decode）。
- 降低 KV 质量来换容量。
- 预填充阶段的分块流式（本次只做 decode 的按层流式；预填充走既有 chunk 路径）。

## 关键依据

- 机制、预算与真实上界：[kv-streaming-design](../docs/maintainer/kv-streaming-design.md)
- 外部资料（参考文章全文 + CUDA Graph 说明）：
  [adaptive-kv-streaming-inputs](../docs/maintainer/adaptive-kv-streaming-inputs.md)
- 现有物理基础：[paged-kv-cache](../docs/maintainer/paged-kv-cache.md)、
  [engine-architecture](../docs/maintainer/engine-architecture.md)、
  [kv_stream 算子](../src/ops/kv_stream/SPEC.md)
- 模型几何：[qwen3-5-model](../docs/maintainer/qwen3_5-model.md)

## 完成条件

1. `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode；
2. on/off 逐 token 一致；
3. 120 K 全驻留的 decode 与峰值显存不差于基线（109.1 tok/s / 15 632 MiB）；
4. `--kv-stream` 关闭即今天的行为；
5. 报告真实 tok/s、峰值显存、实际搬运字节，并说明与设计上界的差距来源；
6. 251 K 的每轮拷贝占用与轮时间之比不再退化（帧级环落地前 47.5/56.7 ms），并按 M-11 的
   footprint 口径同时报告窗口长度；**同时证明 decode 真的搬了页**（`rounds_series` 的
   `upload_bytes` 非零，避免“少搬所以变快”的假加速）。
   **已验（2026-11-09，M-12）**：默认路径 251 K = 69.2 tok/s / 56.7 ms / 1 193.6 MiB/轮 /
   暴露 15.7 ms，与基线逐项相同（`20261109T045828Z-bench-m12-default-251k-e25152e0`）；
   等价性 4/4 PASS 且 decode 真搬页（`20261109T045055Z-equiv-m12-equiv-default-e25152e0`）。
   帧路径已在同一构建内验证逐位相同，但**更慢**，所以不作为交付路径（见交付项 6）。
