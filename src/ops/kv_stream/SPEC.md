---
id: kv-stream-op
type: module-design
status: draft
title: kv_stream：可捕获的 KV 驻留映射与 host→device 槽位拷贝
depends-on: [paged-kv-cache]
parent: paged-kv-cache
---

## Responsibility

一个 sparse-attention 层的**驻留映射**与**搬运**：把"host 里的权威 KV 块"变成"设备 pool 里的
slot"，并保证 attention 通过 block table 看到的块内容与 host 副本一致。四个入口
（`include/ninfer/ops/kv_stream.h`）：

- `kv_stream_map` / `kv_stream_map_bytes` —— 在固定显存里摆出映射（`table`、`slot_block`、
  `slot_stamp`、`slot_ref`、`copy_block`、`copy_slot`、`control`）。
- `kv_stream_resolve` —— 列出一个调用读取的块，缺失的块在 GPU 内被认领、按 CLOCK（second chance）
  取一个本调用不用的 slot、从 pinned host 拷进该 slot、改表。
- `kv_stream_evict` / `kv_stream_reset` —— 丢掉一段块的驻留 / 复位整张映射。
- `kv_stream_counters` —— 读回调用数、查表数、拷贝数、溢出标志。

**可捕获**是它的存在理由：位置来自一个设备 I32 字，拷贝数来自 `control[2]`，两次 launch 的网格
尺寸都固定，所以同一个 CUDA Graph 可以在任意位置、任意拷贝数下重放，不需要重新捕获。

## Contract

- host 副本布局 `[leading, 64, heads, pages]`（identity 页表，`kv_cache_append` 直接写）；
  device pool 布局 `[leading, 4, heads, slots]`；`table[b]` 是该块在 pool 里的 slot 或 `-1`。
- 一次调用最多 16 个 query（`kKvStreamMaxTokens`），每个 query 最多引用 512 个选中块加它自己
  所在的不完整块（`kKvStreamQueryBlocks = 513`）。
- pool 至少 `kv_stream_minimum_slots(tokens) = tokens × 513` 个 slot；不足时 `resolve` 返回错误
  （`control[3]` 溢出），不会静默丢块。
- 本调用写过、且本调用会读的块，拷完之后再刷新一次；写过但本调用不读的块从 pool 里移除，
  所以 slot 里永远不会留着比 host 副本更旧的字节。
- 一次调用使用的 slot 不会被本次调用驱逐（`slot_stamp == epoch` 的槽不参与 CLOCK 扫描）。
- 来源：Strata 的 `kv_stream`（MIT），归属见 `CONTRIBUTORS.md`。

## Boundary

- 拥有：映射的显存布局、驱逐策略、一次调用内的块→槽位解析、从 pinned host 的块拷贝。
- 不拥有：谁该被流式化（那是 Program/model 的决策）、KV 编码与页几何（`paged-kv-storage.h`）、
  attention 数学（`softmax_attention`）、host 副本的分配（`host_kv_arena` / qwen4_exp 的 cache）。
- 消费者只能通过 `PagedKVLayerView` 看到 pool 与 `table`；本模块不嵌入请求身份、allocator 或
  frontier 概念。
- 目前唯一的生产调用者是 Qwen3.8-Flash-Next（`src/models/qwen4_exp/executor.cpp` 的
  `--kv-resident`）；该模型族按 `SCOPE.md` 会被删除，**本算子保留**，作为 27B 自适应 KV 流式的
  机制来源（见 `kv-streaming-design`）。
