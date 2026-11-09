---
id: kv-streaming-measurements
type: architecture-design
title: KV 流式的测量参考表（条件 + 数值 + 复现方式）
status: active
parent: kv-streaming-design
references:
  - kv-streaming-strategy
  - kv-streaming-verification
---

# KV 流式的测量参考表

本文件是**参考表**：每一行是一个在当前条件下测到的量，每张表都带条件与复现方式。它不记录过程、
不记录日期、不记录尝试史——那些在 `bench/results/kv-stream/ledger.md`（记录区）与 git 历史里。
判据（哪个量必须落在什么范围才算通过）在
[kv-streaming-verification](kv-streaming-verification.md)；下一步做什么在
[kv-streaming-strategy](kv-streaming-strategy.md)。

## 口径

- **每轮时间**（`round_ms`）才是可比的量。`tok/s` 同时取决于 MTP 的接受长度（本轮 3.96 token/轮），
  接受长度随文本浮动，所以 tok/s 不单独作为判据。
- `stream.upload_bytes` 是**整个请求**的累计（含 prefill）。要求"每轮字节"时只能用报告里的
  `rounds_series`（逐轮），不能用总数除以轮数。
- 两臂比较必须同主机、同构建、同产物、同 prompt ids，且**按 rep 交错**运行
  （`CONTRIBUTING.md` 的 Comparable measurements；工具已内建）。
- `copy_busy_seconds` 含"等读者消费"的时间，不是纯传输时间；纯传输速率从字节 ÷ 链路峰值算。

## 条件

| 项 | 值 |
|---|---|
| 主机 | Linux，16 核，单 GPU 空闲 |
| GPU | RTX 4080 SUPER，16 376 MiB，sm_89 |
| 链路 | PCIe 4.0 x16，H2D 实测峰值 25.4 GB/s（标定值，用于所有带宽换算） |
| 构建 | `build-sm89`，Release，`cmake --build build-sm89 -j 16` |
| 产物 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（11.6 GiB 权重，含 `mtp`） |
| 页几何 | 64 token/页；`rk4v4` 71 680 B/页、`rk8v4` 104 448 B/页 |
| 通用旗标 | `--kv-capacity auto --gdn-state-fp16 --prefill-chunk 256 --greedy --invariant-attention --spec mtp --draft-tokens 3` |

## 表 1 decode 稳态（512 新 token）

| 上下文 | 格式 | 池/布局 | 每轮字节 | 每轮时间 | tok/s | 暴露 |
|---|---|---|---:|---:|---:|---:|
| 251 000 | `rk4v4` | 47 894 槽，窗口 2905 页/层，环 11 run | 1116.2 MiB | 54.4 ms | 72.8 | 记录于报告 |
| 251 000 | `rk8v4` | 31 583 槽，窗口 1813 页/层，环 20 run | 3366.9 MiB | 147.9 ms | 26.8 | 记录于报告 |
| 251 000 | `int8` | 4096 页/层池，窗口 1267 页/层，环 25 run | 5483.3 MiB | 236.1 ms | 16.8 | 203.1 ms |
| 183 000 | `rk4v4` 流式 | 780 MiB/轮 | 779.9 MiB | 46.6 ms | 85.4 | 6.7 ms |
| 183 000 | `rk4v4` 全驻留 | 0（全部常驻） | 0 | 37.1 ms | 106.8 | 0 |
| 180 000 | `rk4v4` 紧池 | 916.9 MiB | 916.9 MiB | 45.6 ms | 86.7 | 6.1 ms |
| 150 000 | `rk4v4` 紧池 | — | — | — | 99.4 | 记录于报告 |
| 150 000 / 180 000 | `rk4v4`（池装得下） | 0 | 0 | — | 107.1 / 106.0 | 0 |
| 100 000 | `rk4v4` 全驻留 | 0 | 0 | — | 118.9 | 0 |

## 表 2 prefill（同提示、同构建）

| 上下文 | 格式 | prefill tok/s | 上载 | 拷贝流忙 | 计算侧暴露 |
|---|---|---:|---:|---:|---:|
| 183 000 | `rk4v4` 流式 | 1476 | 45.7 GiB | 31.9 s | 0.0 s |
| 183 000 | `rk4v4` 全驻留 | 1481 | 0 | 0 | 0 |
| 251 000 | `rk4v4` 流式 | 1281 | 94.1 GiB | 55.5 s | 0.0 s |
| 251 000 | `rk8v4` 流式 | 1330 | 582.9 GiB | 99.4 s | 0.2 s |
| 251 000 | `int8` 流式 | 1331 | 1193 GiB | 120.0 s | 1.6 s |
| 150 000 / 180 000 | `rk4v4`（池装得下） | 1592.6 / 1593.6 | 0 | 0 | 0 |

同上下文（183 000）的 A/B 说明：**流式让 prefill 慢 0.3%**，搬运被注意力完全掩盖。
251 000 与 150 000 的 tok/s 差来自上下文变长的 O(P²) 计算成本，不是流式的代价。

## 表 3 等价性（同构建、同 prompt ids，`rk4v4`）

| 用例 | prompt token | ids | logprobs |
|---|---:|---|---|
| `equiv-8k` | 8 000 | 512/512 相同 | 逐位相同 |
| `equiv-100k` | 100 000 | 512/512 相同 | 逐位相同 |
| `equiv-183k` | 182 946 | 512/512 相同 | 逐位相同（0/512 有差） |
| `equiv-183k`，run 宽度 49 页 | 182 946 | 512/512 相同 | 逐位相同 |
| `equiv-100k-rk8v4`（交付格式） | 99 952 | 512/512 相同 | 逐位相同（全驻留 131.4 tok/s / 30.2 ms vs 流式 127.3 / 31.2，流式搬 57.8 MiB/轮、暴露 0.0 ms） | | 99 952 | 512/512 相同 | 逐位相同（`rk8v4`，全驻留 131.4 tok/s / 30.2 ms vs 流式 127.3 / 31.2，流式搬 57.8 MiB/轮、暴露 0.0 ms） |

## 表 4 灵敏度（已测过、可引用的单点）

| 变量 | 取值 | 每轮时间 |
|---|---|---:|
| run 宽度（180 000 紧池） | 49 / 96 / 128 / 192 页 | 45.5 / 45.4 / 45.5 / 46.0 ms |
| 环比起点（251 000） | 1.10（规划值） | 54.4 ms（控制器随后按实测移动） |

run 宽度四档的时间差在 ±0.7% 内、暴露恒 6.0 ms ⇒ 宽度不是这一档的杠杆；前视比例才是。

## 复现

```bash
# 一次稳态测量（两臂交错：变体在每个 rep 内交换顺序）
python3 tools/bench/run_kv_stream.py bench --case stream-251k-steady --reps 3 --tag <tag>

# 等价性（先录金标，再比对；需要原版二进制）
python3 tools/bench/run_kv_stream.py golden --suite equiv
python3 tools/bench/run_kv_stream.py check  --suite equiv

# 单独一例、指定格式（--kv-dtype 在子命令之前）
python3 tools/bench/run_kv_stream.py --kv-dtype rk8v4 bench --case stream-251k-steady --reps 1
```

每次运行在 `bench/results/kv-stream/runs/` 留下 argv、设备身份、逐轮样本与判定；运行记录的入库策略见
[bench/results/kv-stream/README](../../bench/results/kv-stream/README.md)。
