---
id: kv-streaming-worklog
type: architecture-design
title: KV 流式保真移植的工作日志与交接（复核 / 质量审查 / 收尾用）
status: active
parent: kv-streaming-design
depends-on:
  - kv-streaming-fidelity
  - adaptive-kv-streaming-inputs
---

# KV 流式保真移植：工作日志与交接

这份文件写给**没有参与本轮工作的人**：复核代码、做质量审查、或者接手收尾。它按"来源 → 为什么 →
改了什么 → 怎么实现 → 结果 → 已知缺陷 → 未完成工作 → PR 范围 → 怎么接手"组织。

分支：`kv-stream-faithful`（从 `kv-streaming` 的 `e3dd5bf3` 分出），39 个提交，27 个文件，
+3186 / −1021 行。

---

## 1. 权威来源（所有人必须先读这个）

| 来源 | 位置 | 用途 |
|---|---|---|
| **参考文章原文（唯一设计权威）** | `docs/maintainer/adaptive-kv-streaming-inputs.md` 附录 A（所有者提供的全文，逐字，只修机翻错字并逐处列明） | 决定"做什么"；所有要求都能在附录 A.1 的 R1–R13 表里找到出处与实现对应物 |
| **上游参考实现** | `D:\GithubRepo\llama.cpp-adaptive-kv-streaming`（分支含作者原始提交） | 决定"怎么做"；每一条都对应到一个具体 commit 与文件行号，见 `kv-streaming-fidelity.md` §10 |
| 上游作者的理论文档 | 上游 `README.md` 的 "Adaptive KV Streaming" 段、`benchmarks/README.md` | 池 = 钉住主机权威 + 共享 ring；池探针算法 |
| 本轮的逐条核对 | `docs/maintainer/kv-streaming-fidelity.md` | 上游结构（file:line）、逐概念对照、R1–R13 判定、三处结构性差距、收益上界、**§10 = 移植图纸** |
| 测量记录 | `docs/maintainer/kv-streaming-measurements.md` M-12 / M-12b / M-13 / M-14 / M-15 | 全部实测数字与实验设计 |

**关键点：上游的"设计图纸"不在 .md 里，而在代码注释与 `src/llama-kv-stream-plan.h` 的结构体注释里。**
`grep -rln "kv_stream" --include=*.md` 在上游只命中 README 与 benchmarks/README.md 两个文件。

---

## 2. 为什么改路线（本轮的决定）

前一轮的做法是"自己设计分帧"，结果是：机制逐位正确（ids 一致），但**更慢**（M-12：251K 上
64.0–67.7 tok/s vs 整层拷贝 69.2）。所有者的判断是：作者已经把这件事写过一遍，应当逐条迁移，
不要凭自己的形状再走一遍。于是本轮：

1. 把原文全文与 R1–R13 要求表落盘（取代原来的转述版），删除第二份权威；
2. 把上游 50+ 提交读成一张移植图纸（fidelity §10）；
3. 按图纸顺序移植 S1–S6（下节），参数一律沿用上游原值，不自创。

### 三处由对照得出的结构性结论（复核时最重要的判据）

1. **池切分原本在启动时按最大容量定死** ⇒ 违反 R7.1–R7.3（"装得下就不该流"、"只在必要时扩环"）。
   130K 的请求仍按 262144 容量预留环 ⇒ 大量槽位闲置。
2. **环按层绑区域**（`region = layer mod depth`）⇒ 结构上只能表达 `depth × 一层流式量`，
   永远得不到上游的 `1.10 ×`（窗口少约 9.5%，每轮字节多约 6.5%）。
3. **M-12 分帧失败的真正原因**：上游的 bounded span 是"为把环压到 1.10× 而付的代价"，
   我们只付了代价（更多发射、波次不整），没拿到收益（环没变）。

---

## 3. 改了什么（文件级，含上游对应提交）

| 文件 | 改动 | 上游对应 | 备注 |
|---|---|---|---|
| `src/core/kv_stream_layout.h` | **两次重写**：先改成平铺环 + `run_pages` + 不动点规划器；再改成**每层窗口** + 集中 decode 形状 + 二分求解 | `ab539d57d`、`d9dcd518c`、`4a84b5ccc`、`852094f51` | 纯函数，无 CUDA 依赖 |
| `src/core/kv_stream_slot_map.h` | 新增：环形游标分配对齐 run（run 永不跨界、FIFO 复用） | `4f2cb3e91` 的槽分配思想 | 纯函数 |
| `src/core/kv_stream_feedback.{h,cpp}` | 新增：deadline 差分、占空比外推、环峰值比 | `39704675e`、`158067c0a` | 纯函数 |
| `src/core/kv_stream_partition.{h,cpp}` | 新增：驻留/环分区控制器（目标窗口、迟滞、冷却、饱和刹车、过肥环修复、首轮跳目标） | `b29fe5859`、`158067c0a`、`92d72914f`、`93df8ac33` | 纯函数；参数全上游原值 |
| `src/core/kv_stream_signal.{h,cu}` | 新增：每层 pinned-mapped ready 标志 + 1 线程 deadline 内核 + 单 run 占空比采样 | `1ba8f4406`、`93df8ac33` | 唯一新增的 CUDA 文件 |
| `src/core/kv_stream_span_tuner.h` | 新增：bounded/unbounded 实测选择器（16 样本、≥0.5%、预热 1、布局变即 reset） | `50a7de723` | 纯函数；**交付默认关闭**，见 §7 |
| `src/models/qwen3_5/program/storage/kv_stream_space.{h,cpp}` | **重写寻址与调度**：每层窗口、平铺环、run 级拷贝与消费、每层一次块表发布、`repartition()`、`evaluate_partition()`、窗口式计量 | `4f2cb3e91`、`65d814435`、`b1d73aebe`、`c51644ec1`、`93df8ac33` | 本轮风险最集中的文件 |
| `src/models/qwen3_5/execution/text.{h,cpp}` | 删掉整条帧路径（`plan_stream_round` 的帧规划、`StreamFrameFeed` 帧游标、`NINFER_STREAM_FRAMES`/`NO_FRAMES`）；改由"环能提前几层"决定发射；接入 span 调优器（默认关） | — | 分层逻辑从"帧"回到"run" |
| `src/models/qwen3_5/program/planning/startup.cpp` | 规划器调用语义不变（budget/active/layers） | `2646f0aa2` | 现在能吃下"只装一层流式量"的探针候选 |
| `src/models/qwen3_5/state/decoder_state.cpp` | 池校验按新签名 | — | |
| `src/models/qwen3_5/program/program.cpp`、`include/ninfer/observability/resources.h`、`apps/cli/{main,run_report}.cpp` | `ring_depth`（区域数）→ `ring_runs`（run 数）；新增 `feedback{deadline_samples,deadline_misses,busy_windows,copy_busy_ratio}` 与每轮字段 | `158067c0a` | 观测面 |
| `tests/test_kv_stream_layout.cpp` | 重写为新布局模型的 10 组用例 | — | 纯逻辑 |
| `tests/test_kv_stream_feedback.cpp`、`test_kv_stream_partition.cpp`、`test_kv_stream_span_tuner.cpp` | 新增 | — | 纯逻辑，用例数字对齐上游 `tests/test-kv-stream-plan.cpp` |
| `tests/cmake/CoreTests.cmake`、`src/core/CMakeLists.txt` | 注册新目标/源文件 | — | |
| `docs/maintainer/adaptive-kv-streaming-inputs.md` | 附录 A 换成原文全文；新增 R1–R13 | — | 权威唯一化 |
| `docs/maintainer/kv-streaming-fidelity.md` | 新增 | — | 移植图纸 |
| `docs/maintainer/kv-streaming-measurements.md` | 新增 M-12…M-15 | — | 实测 |

---

## 4. 怎么实现的（关键不变量，复核时逐条对照代码）

1. **池恒等式**：`Σ layer_resident[i] + ring_pages ≤ slot_count`。
   规划器 `plan_kv_stream_uniform()` 用**二分**求最大可行窗口（左侧随 R 单调增），而不是定点迭代；
   环按 run 取整；`ratio` 装不下时退到 1.0（深度一）；`run_pages` 从请求值逐次减半直到能装下。
2. **窗口槽不变式**：`slot = layer_offset(layer) + page % resident(layer)`。
   纯函数 ⇒ 一个驻留页在窗口期内槽位不变，块表几乎不用改。超集情形（全部驻留）与今天的
   paged 缓存布局逐字节一致（单测 `all_resident_matches_today`）。
3. **流式前缀寻址**：页 `p` 落在环 run `(first_run + p / run_pages) % ring_runs` 的第 `p % run_pages`
   个偏移 ⇒ 一个 run 在设备上是**连续**的，一次 `copy_from_host_pages` 就能整批搬。
4. **两条事件链**：`ready_run_[r]`（该 run 的拷贝落地）与 `consumed_run_[r]`（该 run 已被读完）。
   写者在覆盖 run 前等待 `consumed_run_[r]`；消费者等待 `ready_run_[last]` **和** `ready_[layer]`
   （后者在**块表发布之后**记录 —— 缺这一条会有数值错误，见 §7）。
5. **块表发布**：每层每轮**一次** `publish_slots_at(layer, 0, [0, streamed))`（一个 strided H2D）。
   这是"每轮重新寻址还能负担得起"的关键。
6. **重分区**（`repartition`）：进出窗口的页各搬一次（出：D2H 到主机权威；进：从主机权威 H2D 回窗口槽），
   其余页只重发布表项；随后作废环记账、`run_cursor_ = 0`、`++layout_generation_`（让调优器重置）。
7. **控制器**（`evaluate_partition`）：每轮一次，输入是**上一窗口**的 deadline 比例、占空比、环峰值；
   没有新样本且非首轮 decode 就不评估；决策把槽位数交给 `resize_kv_stream_split()` 转成本布局能表达的
   (每层窗口, 环)。
8. **计量**：窗口开在 `begin_round`、关在下一次 `begin_round`；窗口没跑完就保持打开，绝不覆盖未读样本；
   每轮字节**无条件**记账（campaign 靠它判"是否真的在搬"）。

---

## 5. 结果（数字都在 measurements 里）

### 5.1 移植前的分帧实验（M-12，251K / `rk4v4` / MTP3）

| 变体 | tok/s | 每轮 ms | 暴露 ms | 上载 MiB/轮 |
|---|---:|---:|---:|---:|
| 整层拷贝（当时的交付） | 69.2 | 56.7 | 15.6 | 1193.6 |
| 帧 spf=20、64 页分块 | 67.5–67.7 | 57.9–58.2 | 11.1 | 1191.2 |
| 帧 spf=17（按波对齐） | 64.0–64.1 | 61.1–61.5 | 8.7 | 1191.2 |

### 5.2 交付格式首次实测（M-12b，`rk8v4` 251K）

**26.1 tok/s | 151.0 ms/轮 | 3603 MiB/轮 | 暴露 114.3 ms | 窗口 1670 页（106 880 token）**。

### 5.3 交付数字（M-16…M-21，交付提交 `af2a86b0`）

`check --suite equiv` 是在 `c4b02ae2` 上跑的（4/4 PASS）。之后只有文档与 bench 工具改动，
引擎源码没动（`git diff --stat c4b02ae2..af2a86b0` 只列 docs 与 `tools/bench/run_kv_stream.py`），
所以那次的结论对交付提交同样成立；要完全严格，重跑一次 `check --suite equiv` 即可。

对照基准是**移植前同一条 argv 的运行**（`cf90eaaf` / `73c3ae14`，静态 2× 环）。稳态口径用
`stream-251k-steady`（512 token），因为它才把一次性成本摊掉。

| 场景 | 移植前 | 现在 | 差 |
|---|---:|---:|---|
| 251K rk4v4 稳态每轮 | 56.6 ms | **54.4 ms** | **+3.9%** |
| 251K rk4v4 decode | 69.3 tok/s | **72.8 tok/s** | **+5.0%** |
| 251K rk4v4 每轮上载 | 1193.6 MiB | **1116.2 MiB** | **−6.5%** |
| 251K rk8v4 稳态每轮 | 151.0 ms | **147.9 ms** | **−2.1%** |
| 251K rk8v4 decode | 26.1 tok/s | **26.8 tok/s** | **+2.7%** |
| 251K rk8v4 每轮上载 | 3603.0 MiB | **3366.9 MiB** | **−6.6%** |
| 183K rk4v4 streamed decode | 89.9 tok/s | 85.2 tok/s | −5.2%（见下） |
| 183K streamed 暴露 | 15.6 ms/轮 | **6.5 ms/轮** | −58% |
| 183K 窗口 | 2111 页/层 | **2150 页/层** | +1.8% |
| 183K logprob 最大差 | 未测 | **0.000000（512/512）** | 逐位相同 |
| `check --suite equiv` | — | **4/4 PASS** | 金标 ids 全同 |

**183K 的 −5.2% 是唯一仍在的退步**，且原因与 251K 修掉的那个不同：183K 的环是 7 runs、一层只需
1 run，前视比例 7，门限不会动它；慢的是**每轮字节降到 779.9 MiB 后暴露仍有 6.5 ms**，即拷贝没有
完全藏在计算后面（bounded 流水能把它压到 0.1 ms，但发射代价 2.2×）。这条留给下一步（设备侧等待）。

### 5.4 定位 logprob 差异的八组实验（M-14/M-15，探针保留在树里）

| 编号 | 做什么 | 结果 | 结论 |
|---|---|---|---|
| E1 | `--kv-stream` + 池装得下全上下文（环 0 run） | 差 **0.000000** | 窗口寻址/块表/append/advance 全对 |
| E2 | 关 CUDA 图 | 差 0.606 | 图不是病因 |
| E3/E4 | 拷完立刻把环槽 D2H 回来比 + 打印"槽里实际是谁" | 1923/1949 不符，指纹指向**上一个使用者** | 探针时序假象（当时拷贝还在异步流上） |
| E5 | 探针先等 `ready_[layer]` 再比 | **3872/3872 全对** | 槽位映射与拷贝逐字节正确 |
| E6 | 拷贝之间主机同步（`NINFER_STREAM_SAFE=1`） | 仍差 0.087 | 不是拷贝与注意力的时序竞争 |
| E7 | — | — | 探针因一次原子编辑失败没编译进去 |
| E8 | 注意力**之后**再读同一槽 | **24/24 全错** | 槽内容在注意力期间被写坏 |
| E9 | 比对时打印"槽里是谁的页" | 槽里是**下一个轮到该 run 的层**的 page 0 | 环槽被后继层覆盖 |
| E10/E11 | `NINFER_STREAM_TRACE=1` 打消费序列 | `0,1,2,0,3,4,5,1,6,7,8,2,9,10,11,3` | **与 `compact_layer_indices` 的二次映射逐位吻合 = 根因** |

---

## 6. 已修复的缺陷（复核时可逐条验证）

1. **`ahead == 0` 时整层一次拷贝都没发**：旧逻辑只发"提前 ahead 层"的拷贝。修复后 183K 从
   `upload=0.0` 变成 779.9 MiB/轮。
2. **每轮字节只在 `--kv-stream-report` 时记账**：campaign 的 `stream_engaged` 因此误判 FAIL。
   现在每轮都记账，只有计时事件是可选的。
3. **规划器拒绝"只装一层流式量"的探针候选**：容量扫描第一个候选就失败 ⇒ 启动直接报错。
   现在零窗口是合法计划（全部流式），扫描得以继续。
4. **规划器的定点迭代在预算不足时下溢**：改为二分求解 + `ratio` 1.0 回退 + `run_pages` 逐次减半。
5. **池校验签名**：`kv_stream_pool_pages(split)` 不再需要 layers。
6. **绝对层号 vs full-attention 序号**：旧帧路径用绝对层号索引按 full-attention 序号编号的空间，
   后半模型的流式层会静默失配。现在的规则是：调用方映射一次 `compact_layer_indices[layer]`，
   空间内部一律用这个序数，**不得再映射**（§7 的根因就是多映了一次）。
7. **消费者没等块表发布**：`wait_ready`/`wait_run` 现在同时等 `ready_[layer]`（发布之后记录）。
   效果：logprob 差 0.118 → 0.079 nat（真实但不足）。
8. **层号被映射两次**（根因，见 §7）：`stream_layer` 直接用 `fidx`。
9. **`ahead` 没有上限**：夹到 `count − 1`，`issue()` 自带边界检查（`rk8v4`/251K 的崩溃）。
10. **span 流水按帧序而非页区间寻址**：改为用 hook 报告的 `(first_page, page_count)` 换算 run 区间
    （`wait_run_range`/`consume_run_range`），bounded 路径因此逐位正确（仍太贵，见 §8）。

---

## 7. 已修复的主缺陷（这轮的核心）

**症状**：`equiv-183k` 的 `streamed` 与 `resident` 比，ids 512/512 相同，**logprobs 最大差 0.118 nat**；
`check` 的 `logprobs_match_across_layouts` 判 FAIL。

**根因（静态的层号二次映射）**：`attn_mix` 的 `fidx` 参数**已经是**该层的 compact 序数（调用方先做过
`config_.compact_layer_indices[layer]`），函数里又映射了一次：

```cpp
const auto stream_layer = static_cast<std::uint32_t>(dimension(config_.compact_layer_indices[fidx]));
```

`compact_layer_indices[i]`（`src/models/qwen3_5/config.cpp:133`）是"第 i 层在**它自己那一类**里的序号"，
所以 16 个 full 层拿到的是 `0,1,2,0,3,4,5,1,6,7,8,2,9,10,11,3`：**0–3 各被消费两次、12–15 一次都没有**。
于是 `consume(错层)` 把别人的 run 标成已释放、`wait_ready(错层)` 等的是别人的事件，被遮住的 4 层从不
释放 run —— 拷贝就在活着的读者底下刷新环槽。读到的仍是同一页的上一轮字节，所以 **ids 全对、分布偏
0.1 nat**。

**为什么难定位**：它不崩、不报错、ids 一模一样；所有"看起来该怀疑"的地方经实验全部排除——
拷贝逐字节正确（E5）、关图仍错（E2）、完全串行化仍错（E6）、全驻留全对（E1）。前三个"修等待"的
尝试都不可能奏效，因为等待被记到了别人的 run 上。真正的证据是 `NINFER_STREAM_TRACE=1` 打出来的
消费序列，与 `config.cpp:133` 的构造规则逐位吻合。

**修复**：`stream_layer = static_cast<std::uint32_t>(fidx);`

**结果**（`check --case equiv-183k`，同 argv 同模型）：

| | 移植前 | 移植后 | +修 `wait_ready` 顺序 | **+修层号映射** |
|---|---:|---:|---:|---:|
| logprob 最大差 | 未测 | 0.118 | 0.079 | **0.000000** |
| 有差的 token | — | 508/512 | 510/512 | **0/512** |
| greedy ids | 512/512 | 512/512 | 512/512 | **512/512** |
| campaign | PASS（当时没有 logprob 判据） | FAIL | FAIL | **PASS** |

`check --suite equiv` 在交付提交 `9c9121a8` 上 **4/4 PASS**（`equiv-8k`、`equiv-100k` 金标 ids；
`equiv-183k` 的 resident 与 streamed 各 512 token，ids 与 logprobs 双重逐位相同）。

**这一轮另外两个真缺陷**（都已修）：
- `ahead`（允许前视的层数）没有上限，`ring_runs / runs_per_layer − 1` 在环较宽时可以超过层数 ⇒
  `issue()` 越界，`rk8v4`/251K 直接崩（`Streamed KV layer is out of range`）。现在夹到 `count − 1`，
  且 `issue()` 自带边界检查。
- `wait_ready`/`wait_run` 只等"最后一个 run 的拷贝"，**没等块表发布**。补上 `ready_[layer]` 之后
  logprob 差从 0.118 收到 0.079（真实但不足）。

---

## 8. 未完成的工作清单（接手从这里开始）

### 已结案（本轮收口）

| 项目 | 结论 | 证据 |
|---|---|---|
| logprob 逐位等价 | **0.000000（512/512）** | `check --suite equiv` 多轮 4/4 PASS，最后一次 `s17-suite`（`c4b02ae2`） |
| 251K 速度退步 | **已反转：72.8 tok/s vs 移植前 69.3（+5.0%），稳态每轮 54.4 vs 56.6** | `bench --case stream-251k-steady`（rk4v4，512 token，`s16-fair`），M-19 |
| 首轮 126 ms 尖峰 | 已消除 | 首轮 67.3 ms（移植前 63.8），逐轮样本 |
| 迁移实验把轮时算错 | 已修：迁移那一轮不计入试-run 均值、基线取稳态值、被否目标不再重试 | `[partition]` 打印与 M-19 |
| 环比判据（原 `1.10` 常量） | 改成"几何前视比例 + 实测字节节省"双判据，本机拒绝缩环 | `[partition] lookahead 1.375 -> 1.125 saving 18350080 of 1166376960 bytes` |
| span 流水选择 | 改成四级阶梯按实测选（整层/半层/2 runs/1 run），本机选"整层" | `kv_stream_span_tuner.h` + 单测 |

### 仍未做（完整清单，2026-11-09）

按"是不是交付条件"分成三档。**A 档是 SPEC 里写明的完成条件，目前没满足**，其余是扩展或技术债。

#### A. 交付条件缺口（SPEC「完成条件」逐条对账）

| # | 条件 | 状态 | 缺什么 |
|---|---|---|---|
| A1 | `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode | **一半** | `rk8v4` 已验（26.8 tok/s / 147.9 ms，M-20）；**`int8` 在 262 144 从未跑过**。要做到：`--kv-dtype int8 --max-context 262144 --kv-stream` 跑一次 bench（先看能不能启动），再跑一次 512 token 稳态 |
| A2 | on/off 逐 token 一致 | **只对 `rk4v4` 成立** | `check --suite equiv` 4/4、logprobs 0/512 有差，全部是 `rk4v4`（工具默认 dtype）。**交付格式 `rk8v4` 没有逐位证据**：16 GB 装不下它 183 K 的全驻留布局，原版金标录不了（M-12b 已记）。现有证据只有"能稳定 decode + 251 K 的 ids 与旧金标相同（64 token）"。要补：更大显存的卡上录 `rk8v4` 金标，或改用 `int8` 做等价性（同一 kernel 家族） |
| A3 | 120 K 全驻留 decode 与峰值显存不差于基线（109.1 tok/s / 15 632 MiB） | **未验** | decode 侧本轮只测了 `equiv-100k` 全驻留（118.9–119.1 tok/s，不同构建不同口径）；**峰值显存 15 632 MiB 这一项移植后没有重测**。要补：移植前的 footprint 口径重跑一次 |
| A4 | 报告真实 tok/s、**峰值显存**、实际搬运字节 | **缺峰值显存** | tok/s 与字节都齐（`timings` / `stream`）；峰值显存**不在 JSON 报告里**——只有各 arena 的 `peak_used_bytes`、启动后可用字节与 `workspace_logical_peak_bytes`。要补：进程级显存峰值采样（`cudaMemGetInfo` 轮询或 NVML），或把外部 `footprint` 子命令的结果并进报告 |

#### B. 功能与形态未覆盖

| # | 项目 | 状态 | 缺什么 |
|---|---|---|---|
| B1 | `ahead ≥ 1`（宽环）路径的**正面**验证 | 未验 | 当前门限**保留**宽环，所以它在实测路径里确实在跑且 logprobs 逐位相同，但不是单独验证。要补：临时恢复环比旋钮 → 2.0 → `check --case equiv-183k` |
| B2 | 数千 token 的长跑 | 只到 512 token（129 轮） | 冷却/实验路径在 150 K 紧池被真正走到过一次（M-21），但没有更长会话的数据；**重分区冷却的长期稳定性没有样本** |
| B3 | 130 K 以下的区间 | 只测 8 K / 100 K（equiv） | 100 K–150 K 之间、以及 8 K 以下的流式行为未测（理论上都应是"零搬"，但没有实测点） |
| B4 | 其它 KV 格式的流式 | 只测 `rk4v4`（等价+性能）、`rk8v4`（性能） | `int8`、`k8v4`、`rk2v4-e8`、`rk4v4-e8`、`nvfp4`、`bf16` 全部未测。其中 `int8` 属交付条件（A1）；`k8v4` 不在本项交付内（另一条 kernel） |
| B5 | 多卡：流式 × tensor-parallel / 流水段 | **未测，且没有显式守卫** | 代码里没有"`--kv-stream` 与 `--devices > 1` / `--tensor-parallel` 互斥"的检查；设计假设每段拥有整层的 KV 与自己的空间。要补：先加一条明确的拒绝或支持声明，再测 |
| B6 | prefill | 完全没动（所有者另行安排） | 具体缺口：无 chunk 级双缓冲；控制器/调优器**没有 prefill 样本**（`begin_round` 只在 decode 调用）；分帧流水被 `ph != Phase::Prefill` 显式排除；prefill 的暴露时间只记不归因；251 K 上一次 prompt 要搬约 100 GiB |

#### C. 技术债（不影响当前结论，但会影响后来人）

| # | 项目 | 状态 | 缺什么 |
|---|---|---|---|
| C1 | 反馈在非 report 运行下不完整 | 已知 | 拷贝占空比与暴露时间来自 CUDA 事件，只在 `--kv-stream-report` 时记录 ⇒ 生产默认路径下控制器的输入是**残缺的**（deadline miss 仍从设备侧采样得到，不受影响）。要补：把计时与报告解耦（计时常开、事件按需关闭）或说明只依赖设备侧信号 |
| C2 | 测量旋钮都是环境变量 | 已知 | `NINFER_STREAM_RUN_PAGES` / `_BOUNDED` / `_VERIFY` / `_DEBUG` 都靠环境变量，没有 CLI 旗标与运行时告警；`stream.run_pages` 进报告算部分补偿。要补：要么收进 CLI（可见可审计），要么在文档里明确它们是诊断口 |
| C3 | 没有 GPU 级自动化测试 | 已知 | 四个单测都是纯逻辑；逐字节探针（`NINFER_STREAM_VERIFY`）只在 eager 下手动开。要补：`BUILD_TESTING=ON` 的构建树 + 一个小的真实 cache 测试目标 |
| C4 | 上游后续能力未移植 | 有意 | phase arena、图切分、多卡、图像 prefill、MTP/DFlash/Gemma 的 handoff，以及"**驻留旧前缀、最新页在环里**"的布局（与本实现的"最新页驻留"相反） |
| C5 | 集中 decode 形状（S6）的实际收益 | 未测 | 池恒等式下它不改变每轮字节，只改拷贝层数与 run 长度；`rk8v4`/251 K 是唯一真正触发它的场景，没有做 A/B |
| C6 | 运行记录不入库 | 有意 | `bench/results/kv-stream/runs/` 累计 126 MiB 留在主机；仓库只保留被结论引用的少量。复现靠"按决策索引重跑用例"，不是 checkout 目录 |


### prefill 的现状（为后续的 prefill 专项留证据）

本轮**没有**为 prefill 做任何专门优化，只把现状核清：

- **路径**：prefill 与 decode 共用同一套 `StreamedKVSpace`（按 run、按层、`wait_ready` 挡在注意力之前）。页的发布是**按 chunk**的：`prefill.cpp` 每个 chunk 前调 `advance(1 + (chunk_end-1)/64)`，只发布"这个 chunk 即将写入"的页；`prefill.cpp` 调 `ensure_sequence_kv_mapped(..., publish_stream_pages=false)`，整段 prompt **不提前发布**（否则前几页会在写之前就被窗口挤出去）。
- **成本形态**：窗口随 chunk 滑动，每 chunk 要把流式前缀**整段重搬一遍**，所以 prefill 上载 ≈ 窗口字节 × chunk 数。
- **实测**（报告里的 `stream.prefill`）：

  | 运行 | prefill tok/s | prefill 上载 |
  |---|---:|---:|
  | 251K 移植前 | 1282.3 | 107 GiB |
  | 251K 现在 | 1281.3 | 94 GiB |
  | 180K（池装得下） | 1593.6 | 12.6 GiB |
  | 150K（池装得下） | 1592.6 | 0 |

- **prefill 期间自适应不工作**：`begin_round` 只在 `!prefill && stage==0` 调用，所以分区控制器、span 调优器、deadline 采样**只有 decode 样本**；prefill 用启动时那套 split，分帧流水也被 `ph != Phase::Prefill` 显式排除。
- **缺口**：没有 chunk 级双缓冲；prefill 的暴露时间单独记在 `prefill_exposed_seconds_`，本轮没有归因它。

### 一条要记住的判据

速度问题**必须看逐轮样本**，不能只看 `round_ms_median`：本轮四个 ~170 ms 的尖峰在 129 轮里只把
中位数从 54.4 拉到 55.1，却把 tok/s 从 72.8 拉到 67.2。`--report-json` 的
`decode_rounds.samples` 才是归因依据。

---

## 9. PR 范围

**包含**
- 逐字原文与 R1–R13 要求表（替换旧转述版）
- 移植图纸与逐条核对（`kv-streaming-fidelity.md`）
- 反馈计量、分区控制器、设备侧 sampling、运行时重分区、平铺环 + run 级拷贝/消费、
  span 流水（按页区间寻址，由调优器实测选择，本机实测选 unbounded）、每层窗口 + 集中 decode 形状
- 观测面：`ring_runs`、`feedback{...}`、每轮 deadline/busy 字段
- 四个纯逻辑单测（layout / feedback / partition / span tuner）
- 测量记录 M-12…M-15

**不包含（明确不改）**
- 上游后续的生命周期提交（phase arena、图切分、多卡、图像 prefill、MTP/DFlash/Gemma handoff）
- 生产代码里的 `NINFER_STREAM_VERIFY` 探针（诊断用，默认不启用，保留作为逐字节 oracle；
  `NINFER_STREAM_SAFE` 与 `NINFER_STREAM_TRACE` 已随缺陷定位完成而删除）
- 计算精度相关的任何改动（本轮**没有**放宽任何数值判据；`rk8v4` 仍是质量下限）
- 250K 以上的上下文、非 `rk*` 家族的 KV 格式

**验收标准（建议写进 PR 描述）**

1. `check --suite equiv` 4/4，ids 与 logprobs 均逐位相同 —— **已满足**（`c4b02ae2`，`s17-suite`）；
2. `bench --case stream-251k-steady --kv-dtype rk4v4`（512 token）：不再比移植前慢 ——
   **已满足**：54.4 ms/轮、72.8 tok/s（移植前 56.6 / 69.3），每轮字节 −6.5%；
   暴露 ≤ 2 ms/轮这一条**未满足**（13.5 ms），它需要设备侧等待，见 §8 #2；
3. `bench --case stream-251k-steady --kv-dtype rk8v4`：记录窗口/环/每轮字节与 tok/s ——
   **已满足**：147.9 ms、26.8 tok/s、3366.9 MiB/轮（移植前 151.0 / 26.1 / 3603.0）；
4. 池装得下时 `--kv-stream` 零代价 —— **已满足**：`stream-150k` / `stream-180k` 每轮 0 字节、
   107.1 / 106.0 tok/s（M-21）；
5. 四个纯逻辑单测全绿；增量构建无 error —— **已满足**。


---

## 10. 与上游/原文的差异（复核时不要当成缺陷）

| 项 | 上游/原文 | 本实现 | 为什么 |
|---|---|---|---|
| 页大小 | 256 token（Q8_0/Q4_0） | 64 token（`rk*` 家族三平面） | 本引擎的 paged KV 页 |
| 每轮 token | 1 | 3–4（MTP verify） | 所有"每轮"与"每 token"换算要乘接受长度 |
| 流式的那一半 | 最新页在环里（旧前缀驻留） | 最新页驻留（旧前缀流式） | 追加只写设备；代价是环要容下整个旧前缀 |
| UVM | 作者日常开 | 不使用 | 我们自己做驱逐 |
| 卡 | 5070 Ti / PCIe 5.0 / 实测 50 GB/s | 4080 SUPER / PCIe 4.0 / 实测 25.4 GB/s | 数字不可直接搬；本轮所有默认值都由本机实测决定 |
| 参数化 | 固定池 + 任意 ring_slots | 每层窗口 + 环（整数倍 run） | 布局能力不同；`1.10×` 现在可表达 |
| 自适应 | 控制器按模型决策 | 控制器 + **迁移实验**（试 4 轮、比中位数、慢则回退、被否不重试） | 本机实测控制器的字节模型会提出灾难性缩环（M-21：-46%），必须有实测裁判 |

---

## 11. 怎么接手（命令级）

```bash
# 同步（Windows 端 push，主机只 fetch/reset）
git fetch origin kv-stream-faithful && git reset --hard origin/kv-stream-faithful

# 增量构建（禁止全量）
CCACHE_DISABLE=1 cmake --build build-sm89 -j 16

# 纯逻辑单测（不需要 GPU）
g++ -std=c++20 -Isrc tests/test_kv_stream_layout.cpp -o /tmp/t1 && /tmp/t1
g++ -std=c++20 -Isrc src/core/kv_stream_feedback.cpp  tests/test_kv_stream_feedback.cpp  -o /tmp/t2 && /tmp/t2
g++ -std=c++20 -Isrc src/core/kv_stream_partition.cpp tests/test_kv_stream_partition.cpp -o /tmp/t3 && /tmp/t3
g++ -std=c++20 -Isrc tests/test_kv_stream_span_tuner.cpp -o /tmp/t4 && /tmp/t4

# 等价性（走 bench，不要手跑二进制）
PYTHONPATH=. python3 tools/bench/run_kv_stream.py check --suite equiv
PYTHONPATH=. python3 tools/bench/run_kv_stream.py check --case equiv-183k

# 性能（稳态口径用 512 token 的用例；64 token 的用例只看中位数，会放大一次性成本）
PYTHONPATH=. python3 tools/bench/run_kv_stream.py --kv-dtype rk4v4 bench --case stream-251k-steady
PYTHONPATH=. python3 tools/bench/run_kv_stream.py --kv-dtype rk8v4 bench --case stream-251k-steady
PYTHONPATH=. python3 tools/bench/run_kv_stream.py --kv-dtype rk4v4 bench --case stream-150k stream-180k stream-150k-tight

# 逐轮归因（**速度问题必须看这个**，中位数会藏住尖峰）
python3 - <<'PY'   # 在主机仓库根目录执行
import json,glob,os,statistics
d=glob.glob("bench/results/kv-stream/runs/*<tag>*")[0]
c=glob.glob(os.path.join(d,"*","streamed","rep-01","report.json"))[0]
s=[x["seconds"]*1000 for x in json.load(open(c))["decode_rounds"]["samples"]]
print(len(s), statistics.median(s), statistics.mean(s), max(s))
print([(i,round(v,1)) for i,v in enumerate(s) if v>80])
PY

# 诊断开关（全部默认关闭）
NINFER_STREAM_DEBUG=1    # 布局/门限/试-run 的打印（bench 会继承环境变量，落在 stderr.log）
NINFER_STREAM_VERIFY=1   # 每层拷完把环槽 D2H 回来与主机权威逐字节比（需 --no-cuda-graph）
NINFER_STREAM_BOUNDED=0|1 # 钉住流水形状：0=整层一次发射，1=每 run 一次
```

**先看哪些文件**：`src/core/kv_stream_layout.h`（不变量与 `kv_stream_lookahead_ratio`）→
`kv_stream_space.cpp` 的 `upload` / `wait_ready` / `consume` / `advance` / `repartition` /
`evaluate_partition`（顺序、事件与实验）→ `text.cpp` 的层循环（发射策略与帧阶梯）→
`kv_store.h` 的 `ensure_mapped_to_tokens`（前沿发布）。

**记录纪律**：任何结论都落 `kv-streaming-measurements.md`，带上运行目录名
（`bench/results/kv-stream/runs/<UTC>-<tag>-<commit>`）、argv、目的与判据；不要用临时探针，
不要手跑二进制做结论（自动容量会抖）。
