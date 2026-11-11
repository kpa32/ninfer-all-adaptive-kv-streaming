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
| 251 000 | `rk4v4` | 53 570 槽，窗口 3292 页/层，环 7 run | 693.0 MiB | 41.0 ms | **96.5** | 0.6 ms |
| 251 000 | `rk4v4`（默认前：embedding device、headroom 64） | 47 894 槽，窗口 2905，环 11 run | 1116.2 MiB | 54.4 ms | 72.8 | 13.5 ms |
| 251 000 | `rk8v4` | 31 583 槽，窗口 1813 页/层，环 20 run | 3366.9 MiB | 147.9 ms | 26.8 | 记录于报告 |
| 251 000 | `int8` | 4096 页/层池，窗口 1267 页/层，环 25 run | 5483.3 MiB | 236.1 ms | 16.8 | 203.1 ms |
| 183 000 | `rk4v4` 流式 | 780 MiB/轮 | 779.9 MiB | 46.6 ms | 85.4 | 6.7 ms |
| 190 000 | `rk8v4` 流式 | 窗口 1817 页/层，环 11 run | 1848.0 MiB | 83.2 ms | 47.2 | 44.6 ms |
| 190 000 | `rk8v4` 流式（新默认） | 窗口 +355 页/层（embedding host） | 1433.6 MiB | 66.7 ms | **58.8** | 28.4 ms |
| 183 000 | `rk4v4` 全驻留 | 0（全部常驻） | 0 | 37.1 ms | 106.8 | 0 |
| 150 000 | `rk4v4` 紧池 | — | — | — | 99.4 | 记录于报告 |
| 150 000 / 180 000 | `rk4v4`（池装得下） | 0 | 0 | — | 107.1 / 106.0 | 0 |
| 100 000 | `rk4v4` 全驻留 | 0 | 0 | — | 118.9 | 0 |
| 120 000 | `rk8v4` 全驻留 | 0 | 0 | 31.1 ms | 127.5 | 0 |

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
| `equiv-100k-rk8v4`（交付格式） | 99 952 | 512/512 相同 | 逐位相同（全驻留 131.4 tok/s / 30.2 ms vs 流式 127.3 / 31.2，流式搬 57.8 MiB/轮、暴露 0.0 ms） |
| `stream-251k`（回归金标，251 K prefill） | 250 946 | 64/64 相同 | 逐位相同（logprobs sha256 `8a06a1d4…`，`20261109T183207Z-equiv-e524df8a`） |
| `stream-150k-tight`（活体迁移 4 次，回归金标） | 149 966 | 512/512 相同 | 逐位相同（`20261109T184516Z-equiv-4c511b31`） |
| `stream-251k`，分帧 1 run/帧（C10） | 250 946 | 64/64 相同 | 逐位相同（`20261109T185335Z-equiv-c10-framed`）；每轮 166.7 ms vs 整层 54.7 |

## 表 4 灵敏度（已测过、可引用的单点）

| 变量 | 取值 | 每轮时间 |
|---|---|---:|
| run 宽度（180 000 紧池） | 49 / 96 / 128 / 192 页 | 45.5 / 45.4 / 45.5 / 46.0 ms |
| 环比起点（251 000） | 1.10（规划值） | 54.4 ms（控制器随后按实测移动） |

run 宽度四档的时间差在 ±0.7% 内、暴露恒 6.0 ms ⇒ 宽度不是这一档的杠杆；前视比例才是。

**`rk8v4` 的速度—上下文曲线**（同一池、同一 profile；窗口按**池的容量**规划，约 1 817 页/层，
不随请求的上下文变，所以每轮字节随上下文近似线性增长）：

| 上下文 | 每轮 | tok/s | 来源 |
|---:|---:|---:|---|
| 120 K | 31.1 ms | **127.5** | 实测（全驻留，零流式） |
| 141 K | — | ~127 | 新默认下的拐点：embedding host 把池从 31 583 推到 35 296 槽 |
| 190 K | 66.7 ms | **58.8** | 实测（新默认：embedding host + adaptive） |
| 205 K | ~80 ms | ~50 | 推算（190 K 实测校准） |
| 251 K | 130.9 ms | **30.1** | 实测（新默认） |
| 262 K | ~138 ms | ~29 | 推算 |

同 case 的默认前后（`--kv-stream` 不带其它旗标）：

| 格式 | 上下文 | tok/s | 每轮 | 每轮字节 |
|---|---:|---:|---:|---:|
| `rk4v4` | 251 K | 72.8 → **96.5** | 54.4 → 41.0 ms | 1116.2 → 693.0 MiB |
| `rk8v4` | 190 K | 47.2 → **58.8** | 83.2 → 66.7 ms | 1848.0 → 1433.6 MiB |
| `rk8v4` | 251 K | 26.8 → **30.1** | 147.9 → 130.9 ms | 3366.9 → 2952.5 MiB |

大收益在 `rk4v4`（+33%）而不在 `rk8v4`（+12%）：释放的 388 MiB 换成页数时，`rk8v4` 的页大 46%，
同样的字节只多 232 页/层（`rk4v4` 是 355）。

即：**50 tok/s 对应约 205 K 上下文**（`rk8v4` + MTP3、新默认）；默认前是 187 K。

## 表 5 CUDA Graph（183 000，`rk4v4`，512 新 token）

| 臂 | 每轮时间 | tok/s | 观测 |
|---|---:|---:|---|
| graph on（默认，全驻留） | 37.1 ms | 106.9 | 两次独立运行一致 |
| graph off（`--no-cuda-graph`，全驻留） | 37.7 / 37.8 ms | 105.1 / 105.0 | 两次独立运行 |
| 流式臂（两臂都关图，噪声基准） | 46.4 / 46.7 / 46.2 ms | 85.2 / 85.2 / 85.7 | 极差 0.5 ms |

`equiv-183k` 的 resident 臂，graph on（`20261109T192253Z`）vs off（`20261109T192831Z`、
`20261109T193412Z`）；两次 off 的 512 ids 与 on 逐位相同，即 C8。

- **收益 0.65 ms/轮（1.75%）**，与“每轮约 500–700 个 kernel × 1.3 µs 间隙”同量级：CPU 提交
  （约 1.5–2 ms）被 37 ms 的 GPU 工作完全掩盖，graph 只消掉 kernel 之间的 GPU 间隙。
  251 000 的每轮 54.4 ms 把同样的 0.65 ms 稀释成约 1.3%。
- **流式布局下不可捕获**：图路径为每个并发捕获行预留一页（`graphs.cpp reserve_capture_rows` 的
  `physical_page(alloc, 0)`），而流式地址提交时 `page_count = 0`（页由 `StreamedKVSpace` 发布），
  所以捕获在 380 µs 内以 `KV logical page is outside the address space` 失败
  （`20261109T191653Z`）。要让它成立，图内还必须包含每层的 copy issue 与其 `consumed` record：
  251 K 的环只有 11 run、一层要 46 run，`ahead = 0` 的串行档下把 issue 移到图外会等一个只有
  replay 才会 record 的事件而死锁。
## 表 6 布局：启动时规划一次（251 000，`rk4v4`）

池在启动时按 decode 形状规划一次（`plan_kv_stream_split`，环深按 `ring_ratio`），请求生命周期内
不变。这个决定有实测依据，两条都记在记录区（`bench/results/kv-stream/ledger.md`）：

| 臂 | prefill 上载 | prefill tok/s | decode 每轮 | 判读 |
|---|---:|---:|---:|---|
| 启动即 decode 形（现行） | 93.7 GiB | 1286 | **41.4 ms**（MTP 开，`20261111T001056Z`） | 交付形态 |
| prompt 形启动、首轮换形（已移除） | 93.7 GiB | 1286 | 54.5 ms（MTP 关，`20261109T202056Z`） | 换形收益 ≈0 |

- 换形（prompt 用环的地板 1.0、decode 首轮进入 1.10）的唯一实测收益是 prefill 上载 −1.6%
  （95.2 → 93.7 GiB），而 prefill 的计算侧暴露是 0.0 s（表 2）⇒ 时间收益为零；decode 反而 +0.2%。
- **Y3 被证伪**：strategy §6 预测 `rk8v4` 的 prefill 流量 582.9 → ≤450 GiB（−23%）。环从 1.10
  降到 1.0 只还出 `0.10 × 一层流式量`，251 K `rk4v4` 上 8 页/层（−0.8% 的流式页）。−23% 需要
  窗口 +12%，池里没有那些页。
- 环按**池的容量**规划（4 096 页），不是请求的前沿（3 922）：按前沿规划时环是 9 runs、
  look-ahead 1.13，稳态 60.5 ms（`20261109T201603Z`），比 11 runs 慢 10%。前沿只说明请求走到
  哪里，不说明环该多大。

## 表 7 ubatch（251 000，`rk4v4`，`stream-251k`）

| `--prefill-chunk` | decode 每轮 | 每轮字节 | 暴露 | 记录 |
|---:|---:|---:|---:|---|
| **256** | **54.1 ms** | 1112.4 MiB | 15.6 ms | `20261109T202605Z` |
| **512** | **54.3 ms** | 1112.4 MiB | 15.8 ms | `20261109T203005Z` |
| 768 | 58.9 ms | 1195.5 MiB | 20.3 ms | `20261109T203405Z` |
| 1024（产品默认） | 63.9 ms | 1314.8 MiB | 26.5 ms | `20261109T203802Z` |

- 四档全部 PASS（生成完整、无崩溃）。chunk 改变批形状，因此会移动 token 输出（M-08），这一轮
  只比时间与字节。
- **chunk 越大，窗口越小**：更大的 chunk 要更大的 workspace，从 KV 池里拿走页；251 K 上 1024 比 256
  少拿 202 MiB 的窗口，每轮多搬 202 MiB、慢 18%。
- 产品结论：`--kv-stream` 下 `--prefill-chunk` 应保持在 256/512；1024（非流式默认）对流式不利。

## 表 8 窗口的边界（251 000，`rk4v4`，`stream-251k`）

| 窗口/层 | 环 | 每轮字节 | 每轮时间 | 暴露 | ids | 记录 |
|---:|---:|---:|---:|---:|---|---|
| 0 页 | 32 runs | 4289.7 MiB | 193.2 ms | 144.9 ms | **第 39 个 token 分叉** | `20261109T214123Z` |
| **1 页** | 39 632 页 | 3290.6 MiB | 192.6 ms | 108.4 ms | **逐位 PASS** | `20261109T215412Z` |

- **零窗口在实现上不成立**：`kv_stream_window_slot` 在 `resident == 0` 时对每一页都返回槽 0，
  而新 token 必须落进本层的驻留页才能被 append 写。它规划合法、运行错数据。
  现在零窗口在**规划期**就不会产生：`plan_kv_stream_split` 给出的窗口总是可写的，池太小时回退
  uniform 1.0；`tests/test_kv_stream_layout.cpp` 断言规划出的窗口非零且槽位唯一。
- 最小窗口（1 页）是可跑的最坏情况：每轮搬整个上下文（3290.6 MiB）、193 ms/轮，逐位仍对。
  它只用于验证边界，不是产品配置。

## 复现

```bash
# 一次稳态测量（两臂交错：变体在每个 rep 内交换顺序）
python3 tools/bench/run_kv_stream.py bench --case stream-251k-steady --reps 3 --tag <tag>

# 等价性（先录金标，再比对；需要原版二进制）
python3 tools/bench/run_kv_stream.py golden --suite equiv
python3 tools/bench/run_kv_stream.py check  --suite equiv

# 单独一例、指定格式（--kv-dtype 在子命令之前）
python3 tools/bench/run_kv_stream.py --kv-dtype rk8v4 bench --case stream-251k-steady --reps 1

# 追加一条臂的旗标（例如关图）
python3 tools/bench/run_kv_stream.py bench --case equiv-183k --reps 1 --extra=--no-cuda-graph
```

每次运行在 `bench/results/kv-stream/runs/` 留下 argv、设备身份、逐轮样本与判定；运行记录的入库策略见
[bench/results/kv-stream/README](../../bench/results/kv-stream/README.md)。
