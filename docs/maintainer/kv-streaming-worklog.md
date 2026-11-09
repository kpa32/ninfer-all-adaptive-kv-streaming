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

### 5.3 新框架首次上卡（M-13，`equiv-183k`）

| 项 | resident 基线 | streamed（新框架） | streamed（移植前） |
|---|---:|---:|---:|
| greedy ids | 512/512 | **512/512** | 512/512 |
| 每轮上载 | — | 779.9 MiB | 822.9 MiB |
| 暴露 | — | **0.9 ms/轮** | 15.6 ms/轮 |
| decode | 106.0 | 94.2 tok/s | 89.9 |
| 窗口 / 环 | — | 2150 页/层 / 7 runs（≈1.23×） | 2111 / 1512 槽（2×） |

### 5.4 定位 logprob 差异的九组实验（M-14/M-15，全部保留在树里）

| 编号 | 做什么 | 结果 | 结论 |
|---|---|---|---|
| E1 | `--kv-stream` + 池装得下全上下文（环 0 run） | 差 **0.000000** | 窗口寻址/块表/append/advance 全对 |
| E2 | 关 CUDA 图 | 差 0.606 | 图不是病因 |
| E3/E4 | 拷完立刻把环槽 D2H 回来比 + 打印"槽里实际是谁" | 1923/1949 不符，指纹指向**上一个使用者** | 探针时序假象（当时拷贝还在异步流上） |
| E5 | 探针先等 `ready_[layer]` 再比 | **3872/3872 全对** | 槽位映射与拷贝逐字节正确 |
| E6 | 拷贝之间主机同步（`NINFER_STREAM_SAFE=1`） | 仍差 0.087 | 不是拷贝与注意力的时序竞争 |
| E7 | — | — | 探针因一次原子编辑失败没编译进去 |
| E8 | 注意力**之后**再读同一槽 | **24/24 全错** | 槽内容在注意力期间被写坏 |

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
6. **绝对层号 vs full-attention 序号**：旧帧路径用 `fidx` 索引按 full-attention 序号编号的空间，
   后半模型的流式层会静默失配；现在统一 `compact_layer_indices`。
7. **消费者没等块表发布**：`wait_ready`/`wait_run` 现在同时等 `ready_[layer]`（发布之后记录）。
   效果：logprob 差 0.118 → 0.079 nat（真实但不充分，见下）。

---

## 7. 未修复的缺陷（接手的第一件事）

**症状**：`equiv-183k` 的 `streamed` 与 `resident` 比，ids 512/512 相同，**logprobs 最大差 0.079 nat
（510/512 有差）**；`check` 的 `logprobs_match_across_layouts` 判 FAIL。

**证据链（已排除）**：
- 拷完就比是逐字节正确的（E5，3872/3872）；
- 完全串行化拷贝仍然错（E6）⇒ 不是拷贝/注意力的时序竞争；
- 关图仍然错（E2）⇒ 不是 CUDA 图；
- 全驻留（环 0 run）逐位正确（E1）⇒ 窗口、块表、append、advance 本身对；
- **注意力读完后再读同一个槽，内容变了**（E8，24/24）⇒ 有别的东西在注意力期间**写**了环槽。

**首要假设**：本轮新增 token 的 KV append 通过**块表**解析页号，而该页的表项还是旧的/未发布，
于是 append 写进了环槽（同时新页的键也没落到窗口槽）。这与"ids 对、分布偏"完全吻合。
最可能的位置：`src/models/qwen3_5/program/storage/kv_store.h:1693`（`ensure_mapped_to_tokens` →
`stream_space_->advance(...)`）与**本轮 KV 写入**的相对顺序 —— prefill 路径已在写入之前
（`prefill.cpp:1369` 的注释即此意），decode 这条要核实。

**下一步（按顺序）**：
1. 在 `advance()` 与 append 都加一次性打印（页面、表项、流），确认 decode 轮里谁先谁后；
2. 若 append 在前：把 `advance` 提到 append 之前，或在 append 前强制发布本轮新页；
3. 重跑 `check --case equiv-183k`（期望 `logprobs_match_across_layouts=true`）；
4. 通过后跑 `check --suite equiv`（4/4）与 `bench --case stream-251k`（`rk4v4` 与 `rk8v4` 各一次）。

**另一个已知限制**：`NINFER_STREAM_BOUNDED`（bounded span 派发）默认关闭。原因是 span 的下标
（按 split 计数）与 run 的下标（按页计数）只在"run 恰好是整数个 span"时才对齐；对不齐时最后几个
span 等待的是空事件，会读到未落地的槽 —— 曾测到 0.13 nat 的偏差。要开启必须先让派发单位与拷贝单位
同源（让 `splits_per_run` 由 `run_pages × 64 / segment_keys` 精确整除，或让 hook 直接用 run 下标）。

---

## 8. 未完成的工作清单

| # | 项目 | 状态 | 备注 |
|---|---|---|---|
| 1 | §7 的 logprob 缺陷 | **未修** | 阻塞 `check --suite equiv` 4/4 |
| 2 | bounded span 派发（`NINFER_STREAM_BOUNDED`） | 默认关 | 需先把 span 与 run 对齐 |
| 3 | `check --suite equiv` 4/4 | 未跑 | 需要 #1 |
| 4 | 251K bench（`rk4v4` / `rk8v4`） | 未跑 | 需要 #1 |
| 5 | 130K–180K 的收益实测（R7.1 效应） | 未测 | 规划上这是本轮最大的期望收益处 |
| 6 | `1.05/1.10/1.25/1.50` 四档 ring ratio 扫描 | 未做 | 上游 1.10 是它自己硬件的值，不宜照抄结论 |
| 7 | 重分区代价实测（一次重分区多少毫秒） | 未测 | 上游用 64 轮冷却摊掉 |
| 8 | `--kv-stream-report` 之外的计时样本清理 | 可选 | 目前 `report_` 关闭时仍会记每轮字节 |
| 9 | 上游的"驻留旧前缀、最新页在环里"布局 | 未移植 | 与本实现的"最新页驻留"相反；对本轮收益无影响，但影响 append 走向（见 §7） |

---

## 9. PR 范围

**包含**
- 逐字原文与 R1–R13 要求表（替换旧转述版）
- 移植图纸与逐条核对（`kv-streaming-fidelity.md`）
- 反馈计量、分区控制器、设备侧 sampling、运行时重分区、平铺环 + run 级拷贝/消费、span 调优器（默认关）、
  每层窗口 + 集中 decode 形状
- 观测面：`ring_runs`、`feedback{...}`、每轮 deadline/busy 字段
- 四个纯逻辑单测（layout / feedback / partition / span tuner）
- 测量记录 M-12…M-15

**不包含（明确不改）**
- 上游后续的生命周期提交（phase arena、图切分、多卡、图像 prefill、MTP/DFlash/Gemma handoff）
- 生产代码里的 `NINFER_STREAM_VERIFY` / `NINFER_STREAM_SAFE` 探针（诊断用，默认不启用；收尾时可删或保留）
- 计算精度相关的任何改动（本轮**没有**放宽任何数值判据；`rk8v4` 仍是质量下限）
- 250K 以上的上下文、非 `rk*` 家族的 KV 格式

**验收标准（建议写进 PR 描述）**
1. `check --suite equiv` 4/4，ids 与 logprobs 均逐位相同；
2. `bench --case stream-251k --kv-dtype rk4v4`：`stream_upload_mib_per_round > 0`，暴露 ≤ 2 ms/轮，
   decode 不低于移植前（69.2 tok/s）；
3. `bench --case stream-251k --kv-dtype rk8v4`：记录窗口/环/每轮字节与 tok/s（基线见 M-12b）；
4. 四个纯逻辑单测全绿；增量构建无 error。

---

## 10. 与上游/原文的差异（复核时不要当成缺陷）

| 项 | 上游/原文 | 本实现 | 为什么 |
|---|---|---|---|
| 页大小 | 256 token（Q8_0/Q4_0） | 64 token（`rk*` 家族三平面） | 本引擎的 paged KV 页 |
| 每轮 token | 1 | 3–4（MTP verify） | 所有"每轮"与"每 token"换算要乘接受长度 |
| 流式的那一半 | 最新页在环里（旧前缀驻留） | 最新页驻留（旧前缀流式） | 追加只写设备；代价是环必须容下整个旧前缀（见 §8 #9） |
| UVM | 作者日常开 | 不使用 | 我们自己做驱逐 |
| 卡 | 5070 Ti / PCIe 5.0 / 实测 50 GB/s | 4080 SUPER / PCIe 4.0 / 实测 25.4 GB/s | 数字不可直接搬 |
| 参数化 | 固定池 + 任意 ring_slots | 每层窗口 + 环（整数倍 run） | 布局能力不同；`1.10×` 现在可表达 |

---

## 11. 怎么接手（命令级）

```bash
# 同步（Windows 端 push，主机只 fetch/reset）
git fetch origin kv-stream-faithful && git reset --hard origin/kv-stream-faithful

# 增量构建（禁止全量）
CCACHE_DISABLE=1 cmake --build build-sm89 -j 16

# 纯逻辑单测（不需要 GPU）
g++ -std=c++20 -Isrc tests/test_kv_stream_layout.cpp     -o /tmp/a && /tmp/a
g++ -std=c++20 -Isrc src/core/kv_stream_feedback.cpp  tests/test_kv_stream_feedback.cpp  -o /tmp/b && /tmp/b
g++ -std=c++20 -Isrc src/core/kv_stream_partition.cpp tests/test_kv_stream_partition.cpp -o /tmp/c && /tmp/c
g++ -std=c++20 -Isrc tests/test_kv_stream_span_tuner.cpp -o /tmp/d && /tmp/d

# 等价性（走 bench，不要手跑二进制）
PYTHONPATH=. python3 tools/bench/run_kv_stream.py check --suite equiv
PYTHONPATH=. python3 tools/bench/run_kv_stream.py check --case equiv-183k

# 性能
PYTHONPATH=. python3 tools/bench/run_kv_stream.py bench --case stream-251k
PYTHONPATH=. python3 tools/bench/run_kv_stream.py --kv-dtype rk8v4 bench --case stream-251k

# 诊断探针（只用于 eager 运行；会同步，慢）
NINFER_STREAM_VERIFY=1 build-sm89/apps/ninfer <argv>… --no-cuda-graph
NINFER_STREAM_SAFE=1   build-sm89/apps/ninfer <argv>…
```

**先看哪些文件**：`src/core/kv_stream_layout.h`（不变量）→ `kv_stream_space.cpp` 的 `upload` /
`wait_ready` / `consume` / `repartition` / `advance`（顺序与事件）→ `text.cpp` 的层循环（发射策略）→
`kv_store.h:1693`（decode 侧前沿发布，§7 的嫌疑点）。

**记录纪律**：任何结论都落 `kv-streaming-measurements.md`，带上运行目录名
（`bench/results/kv-stream/runs/<UTC>-<tag>-<commit>`）、argv、目的与判据；不要用临时探针，
不要手跑二进制做结论（自动容量会抖）。
