# KV 流式：工作交接

> **读这一份就能接着干，不需要重跑任何东西。** 最后更新：2026-11-09T10:30Z，
> HEAD `2ecfcb6e`（分支 `kv-streaming`，已推送）。
> 设计权威：`kv-streaming-design.md`（§5.13 之前是历史，**§5.13.1 是最新事实**）；
> 测量存根：`kv-streaming-measurements.md`（M-01…M-11，每条都有时间/提交/命令/输出）；
> 运行记录：`bench/results/kv-stream/`（原版金标、逐次 argv、逐轮指标与判定）；
> 外部资料：`adaptive-kv-streaming-inputs.md`；参考实现源码：`D:\GithubRepo\llama.cpp-adaptive-kv-streaming`（只读）。

---

## 0 目标与当前状态

### 0.0 目标格式（所有者 2026-11-09 再次明确）

- **交付目标是 `rk8v4`**（质量下限，SPEC 交付项 1）；稳定后 `int8`。
- **`rk4v4` 只是测试时的替身**（窗口 181 440 token、拷贝便宜 1.7 倍，便于快速对比）。
  把在 `rk4v4` 上量到的数字**直接当交付结论**是错的，必须在 `rk8v4` 上补测。
- 两者同属 **INT8 家族**（`small_t_i8_*`）⇒ 内核改动一次覆盖；**`k8v4`（`Fp8KeyNvfp4Value`）
  是另一条 kernel，不是交付路径**，别再搞错（这就是上一轮返工的根因，见 §0.2）。

### 0.1 当前实际状态（2026-11-09 完，M-12）

**帧（span）流水：实现完成、逐位相同、但实测更慢，故不作为默认。默认仍是整层拷贝。**

| 251 K（`rk4v4`+MTP3，同一构建） | tok/s | 每轮 ms | 暴露 ms/轮 | 上载 MiB/轮 |
|---|---:|---:|---:|---:|
| **默认（整层拷贝，交付）** | **69.2** | **56.7** | 15.7 | 1 193.6 |
| 帧路径（`NINFER_STREAM_FRAMES=1`）| 64.0–67.7 | 57.9–61.5 | 8.7–11.3 | 1 191.2 |

- 为何不赚钱：链路每轮要 1 191 MiB ÷ 25.4 GB/s ≈ 47 ms，占每轮 56.7 ms 的 83%；层内切细
  只能重排等待（暴露 15.7 → 8.7 ms），额外的发射与流事件把每轮抬回去（M-12 完整表）。
- 机制在 INT8 家族里是通的：`CausalSmallTInvocation::split_begin/split_end` → `launch_tc_partial_i8`
  → 两个 kernel（`split = split_begin + blockIdx.y`、`split_count = split_capacity`）；
  帧宽 = `splits_per_frame × split_keys ÷ 64`，`split_keys` 与内核 `causal_small_t_partition_range`
  同规则；开头 `plan_stream_round()` 定本轮是否分帧，`layer_copy = space && (prefill || !framed)`
  管拷贝与 consumed 两者，拿不到帧就退回整层拷贝。
- 已验证：默认路径 `check --suite equiv` **4/4 PASS**（`20261109T045055Z-…-e25152e0`）；
  分帧开启时 183 K 流式行也 PASS（ids + logprobs 逐位、`upload=824.7 MiB/轮`，
  `20261109T035348Z-equiv-dbg8-cb327a1a`）；251 K 默认 = 69.2 tok/s / 56.7 ms，与基线逐项相同。
- 工具：`run_kv_stream.py` 加了全局 `--kv-dtype`（**要放在子命令之前**），可测 `rk8v4`。
- 下一步（未做）：要收益就得减字节——帧级环（环只装几帧，窗口 +≈6%）或更小的流式页；
  继续在层内切细已被 M-12 证伪。

### 0.1.0 上一轮的诊断（保留，解释为何返工）

### 0.1.0 上轮结论（保留，解释为什举振了）

已落的 8 个提交（`98973519`…`224d057b`）只改了 `small_t_k8v4.cu`，而那是 `--kv-dtype k8v4`
（`Fp8KeyNvfp4Value`）的 kernel；交付目标 `rk8v4`、`int8` 与测试替身 `rk4v4` 走 INT8 家族。
由此产生一个**已修**的缺陷：`20830005` 把 decode 的按层拷贝改成 `prefill` 专属，而 INT8 家族
拿不到帧 hook ⇒ 该格式 decode **一次都不搬**。实测记录
`runs/20261109T003445Z-bench-frames-224d057b` 的 `rounds_series` 逐轮 `upload_bytes=0`，
它的 102.6 tok/s 就来自少搬的那 ~1.1 GB/轮，**该数字作废**。

正确口径与路由事实写在设计 §5.13.1（含上游出处 `llama.cpp-adaptive-kv-streaming` 的
`kv-stream-span-tuner.h` / `fattn.cu:145-146,1983-2058,378-391`：span = 32 页量级、
按端到端计时自适应选择）。

### 0.1.1 开工清单（压缩后从这里接，一步不要跳）

| # | 步骤 | 文件 | 完成判定 |
|---|---|---|---|
| 1 | 恢复 decode 按层拷贝兜底（**correctness，先做**） | `src/models/qwen3_5/execution/text.cpp`，`run_stage_layers` 里两处 `kv_stream_space_ != nullptr && prefill` | 帧 hook 拿不到时仍按层 `upload`/`wait_ready`/`consumed_event`；`rk4v4` 的 `rounds_series.upload_bytes > 0` |
| 2 | `small_t_i8` 能接受 split 区间 | `CausalSmallTInvocation`（`launch.h`）放 `CausalSplitRange` + `frames`；`small_t_i8_launch.cuh::launch_tc_partial_i8` 与两个 tier kernel（`small_t_i8.cuh`、`small_t_i8_pipelined.cuh`）用 `split = split_begin + blockIdx.y`、`split_count` 仍传总容量 | 主机单 TU 能编；默认区间逐字节不变 |
| 3 | 帧发射循环 + **层末一次归约** | 同上，仿 `small_t_k8v4.cu` 的 `dispatch(begin,end)` 结构 | 归约仍是 `causal_attention_small_t_reduce_output_kernel`（按 split 序号累加） |
| 4 | 帧大小改成可调参数 | `causal_attention_frame_plan`（`small_t.cu`）按 INT8 几何算；`frame_pages` 从启动参数来 | 删掉硬编码 782–931 页；起点取上游量级（32 页） |
| 5 | 增量构建 + CPU 单测 | 主机 `CCACHE_DISABLE=1 cmake --build build-sm89 -j 16`；`g++ -std=c++20 -O1 -Isrc -Iinclude tests/test_kv_stream_layout.cpp` | 链接通过；`kv stream layout: all checks passed` |
| 6 | 上卡（**需批准**） | `python3 tools/bench/run_kv_stream.py bench --case stream-251k --reps 1` | `check --suite equiv` 4/4 逐 id 相同；`rounds_series.upload_bytes` 非零；**`rk8v4` 必测一次** |

### 0.1.2 关键数字（引用前先核口径）

| 项 | 值 | 出典 |
|---|---:|---|
| 251 K 基线（`rk4v4`+MTP3） | 69.2 tok/s，每轮 56.7 ms = 47.5 拷贝 + 9.2 空转 | `runs/20261108T201204Z-bench-baseline-73c3ae14`，3 次 |
| 暴露（新基准口径） | 15.8 ms/轮（工具 per-round 口径 9.2 ms） | 同上 |
| 每轮上载 | 1 193.6 MiB；接受长度 3.94 | 同上 |
| `rk4v4` 流式窗口（容量 262 144 + `--kv-headroom-mib 16`） | 2 835 页 = **181 440 token**（⇒ 该配置下流式仅在 181 K 之后触发，100 K 永远不流式） | M-10 / handoff §5 |
| `equiv-183k` 用例的窗口 | **135 104 token**（容量 184 064 + headroom 1008）⇒ 183 K 提示尾部 ≈ 748 页，基线时每轮真实搬 823 MiB | M-10 |
| `rk8v4` 流式窗口 | 约 1 789 页 = **114 496 token**（每页 104 448 B） | 设计 §5.5 表 |
| 本卡 SM | **80**（运行报告 `compute capability 8.9, 80 SMs`） | `runs/…T003445Z` 日志 |
| `--invariant-attention` 的段 | `segment = 3 136` key（容量 262 144）；251 K 窗口 ⇒ `splits ≈ 81` | 实测打印 `[stream] dispatch … segment=3136` |
| 页 | 64 token；`rk4v4` 71 680 B/页/层，`rk8v4` 104 448 B/页/层，全注意力层 16 | 设计 §5.3 |

### 0.1.3 已作废的东西（不要重新引入）

- **102.6 tok/s / 38.4 ms 每轮**（`runs/…T003445Z`）——解码零 I/O 的假加速。
- **+25~35%** 旧预测；“一帧 = 一个 wave = 782–931 页”的硬编码；**76 SM**。
- `--embedding-residency host` 的 +38%（所有者 22:35 已否决作为本特性下一步，也非我方工作）。
- `equiv-183k` 在 183 K 下本来就有真实尾部（窗口 135 104），所以它的 `stream_upload_mib_per_round`
  应大于 0；若出现 0，说明 decode 根本没搬页（就是 §0.1 那个缺陷），不是“没触发流式”。

### 0.2 需求与批准留根（原始对话，2026-11-08）

| 时间（UTC） | 所有者原话（节选） |
|---|---|
| 22:19:38 | “`llama.cpp-adaptive-kv-streaming` 的文档和优化策略再看看，目前我们的 256k 解码还是太慢了……先找资料找方向，找到和我讨论，不要任何动工” |
| 22:29:35 | “我不接受质量降低……如果你有计划就跟我说明，然后可以动工，但是要说清楚，必须让我知晓” |
| 22:35:45 | “`--embedding-residency host` 我能不知道吗，你测试了又如何，都是现有的东西，跟你无关”（⇒ 该方案不作为本特性的下一步） |
| 22:45:28 | “先写文档，待会我压缩后你再动工；主机侧注意力也说明下” |
| 22:47:48 | “`small_t_k8v4` 这个是不是只暂时优化 k8v4 的？其他的 kv 还没优化是吗？简单快速回答问题”（⇒ 当时回答错成“k8v4 = 我们用的 rk4v4”，是后续返工的根因） |
| 22:50:43 / 23:00:31 / 23:13:45 | “读取你刚才写的文档，然后开始吧。最好 SPEC TODO 都更新下” / “你继续，直到工作完成” / “一次写完……这轮一次性把功能迭代完毕” |

需求本体：就是 `llama.cpp` 那个 fork 的 **bounded-span pipeline**（一层内边拷边算），
落点是交付格式（INT8 家族），不是 `k8v4`。

**等价性已验（2026-11-08）**：183 K 上流式与**原版二进制**逐 id 相同（512 个 token，跨 8 页/层），
且同一构建里全驻留与流式的 logprob 数组逐位相同。记录：
`bench/results/kv-stream/runs/20261108T195936Z-equiv-73c3ae14`（8k / 100k / 183k 四行全 PASS）。
复现只需 `python3 tools/bench/run_kv_stream.py --model <artifact> check --suite equiv`。

| 配置（251 K 提示 / 容量 262 144 / `rk4v4` + MTP3） | 每轮 | decode | 拷贝占用 | 计算流等拷贝 |
|---|---:|---:|---:|---:|
| M1 环深 1 | 64.0 ms | 36.4 | 42.6 ms | 22.65 ms |
| 环深 2 | 56.9 ms | 48–53 | 47.5 ms | 11.4 ms |
| 环深 3（已否决） | 61.0–62.9 ms | 46.9 | 51.9 ms | 11.7 ms |
| 环深 2 + 跨轮预取 | 54.1 ms（单次） | 41.4（接受长度偏低那次） | — | — |

**关键判断（2026-11-09 修正）**：拷贝流占每轮 84%，但每轮还有 9.2–15.8 ms 是计算流空等拷贝，
而这部分能靠“一层内边拷边算”取回（上游同一手段，见设计 §5.13.1）；**这是本阶段的主攻**。
另一个方向是“更小的流式页”（最多 +40%），它改的是“窗口内外同一格式”的质量合同，
需要所有者签字，不在本次范围。

**投机后端（M-11）**：DFlash2 每轮比 MTP-7 便宜 9.3–11.2 ms，但它多占 1.80 GiB ⇒ 窗口从
182 912 降到 77 888 token。**窗口固定时那份搬运惩罚对所有后端一样，所以 DFlash2 全程领先约 29%，
换后端不赚钱**；收益只在“把权重还给窗口”。自适应后端切换已延后为独立特性 PR（§4.7）。
日常使用：`--spec mtp --draft-tokens 3`；再加 `--embedding-residency host`（嵌入表 388.4 MiB
换 355 页 = 22 720 token 窗口，预测 251 K 每轮 −15.4 ms；未实测）—— 那只是**使用配方**，
不是本特性的工作项（所有者 22:35 已明确）。

上表用的是**重复提示词**（M-02…M-06）。新基准改用"长文本 + 续写数字列表到 800"的提示
（251 K、接受长度 3.94、`--no-thinking`），不可与上表逐项相减：

| 新基准（251 K、容量 262 144、`rk4v4` + MTP3、3 次） | 值 |
|---|---:|
| decode | 69.2 tok/s |
| 每轮 | 56.7 ms |
| 每轮上载 | 1 194 MiB |
| 暴露（计算等拷贝） | 15.8 ms |
| 接受长度 | 3.94 |

每轮时间与上表环深 2 的 56.9 ms 基本相同；tok/s 高是接受长度 3.94 对 2.24–3.0 的结果，不是调度变快。
记录：`bench/results/kv-stream/runs/20261108T201204Z-bench-baseline-73c3ae14`。

---

## 1 环境、构建与成本（先读这一节，能省 20 分钟）

| 项 | 值 |
|---|---|
| 构建机 | `kpa@192.168.2.13`，Linux 16 核 |
| 仓库 | `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`（分支 `kv-streaming`） |
| 构建树 | `build-sm89`（Release，`BUILD_TESTING=OFF`，`CMAKE_CUDA_ARCHITECTURES=89`） |
| 模型 | `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（11.6 GiB） |
| GPU | RTX 4080 SUPER 16 376 MiB，驱动 615.71.09，空闲 9 MiB |
| 本地编辑 | `D:\GithubRepo\ninfer-all-adaptive-kv-streaming`（Windows） |

### 1.1 构建规矩（违反会被回滚）

- **只允许增量构建**：改 `.cpp` 后
  `git pull --ff-only && cmake --build build-sm89 --target ninfer -j 14`（2.6 s）。
- **永远不要新建构建树、不要全量编译**。曾为"新旧二进制对比"在 `/tmp` 和
  `~/ninfer-fork/ninfer-base` 各建一棵树 → 每次 20 分钟，用户已明确禁止。**需要对比就用原版二进制或
  运行期开关，不要重建。**
- 被全树包含的头（`paged_kv_cache.h`、`kv_store.h`、`context.h`、`text.h`）改动 = 20+ 分钟，
  **必须攒成一次批量提交**。
- 不要 `git switch`；不要对已推送的提交 amend / force-push（本会话踩过，已用
  `git rebase --onto` 修复）。

### 1.2 原版二进制（等价性金标的唯一来源）

```
/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer      # 用户自己编译的原版，2026-11-05
```

- 那个目录里有 `用户自己部署的,AI看到后别动和你无关.md`：**只准执行它做测试**，不写、不编、不
  `git` 操作、不改它的模型或配置。
- 版本串：`ninfer 2026-11-04.3+2026-11-04.3-15-gf6928e00`。
- 它**没有** `--kv-stream`（原版无流式），所以它只能给出**全驻留**的答案——这正是我们要的：
  我们的流式实现必须和这个原版的答案逐 token 相同。

---

## 2 已落地（代码地图）

> 流式主体（扁平池/表/环/逐层调度/观测）已落地且已验（M-10）；帧（span）流水的状态见 §0.1——
> 它目前只在 `small_t_k8v4.cu`（`--kv-dtype k8v4`）里，不在交付路径上。

| 组件 | 文件 | 说明 |
|---|---|---|
| 容量算式与槽映射 | `src/core/kv_stream_layout.h` | `plan_kv_stream_split`（**环深参数**，默认 2；池子 ≥ 6 个上下文才取深环）、`KVStreamSplit{resident_pages, ring_pages, ring_depth}`、`kv_stream_ring_extent`、`kv_stream_slot`、`kv_stream_window_slot`、`kv_stream_ring_slot(layer, page)`（层 `L` 固定用 `L % depth` 区） |
| 扁平池几何 | `src/models/qwen3_5/state/decoder_state.{h,cpp}` | 平面按 kind（4 个）、每层一行表、`plan_cache(..., KVStreamSplit*)` |
| 表与页搬运 | `src/core/paged_kv_cache.{h,cpp}` | `row_at`、`publish_slots_at`、`copy_to_host_pages`、`copy_from_host_pages(source, layout, first_page, count)` |
| 运行时空间 | `program/storage/kv_stream_space.{h,cpp}` | 主机权威每层一段；`advance()`（新页→窗口槽；离窗页先 D2H 再指环槽）；`upload()`（**只补该区缺的页**，`region_layer_`/`region_pages_` 记账）；`ready/consumed/dropped` 事件；`reset()` 清区记账；`TransferStats`（窗口/环/每轮字节/copy busy/exposed wait）；`begin_round`/`wait_ready`/`transfer_stats` |
| 逐层调度 | `execution/text.cpp` `run_stage_layers` | full 层：等 `consumed(目标层 − depth)` → `upload(目标层)` → `wait_ready(本层)` → attention → `record consumed(本层)`。目标层 = `(compact + depth − 1) % layers`，`prefill` 时不回绕；**回绕的那一半就是跨轮预取** |
| 计划接入 | `planning/startup.cpp`、`planning/request_plan.cpp`、`runtime/engine/kv_capacity.cpp`、`materialization.cpp` | 槽轴 `layers × logical_pages`；流式请求 text 额度 0；`plan_kv_stream_split` 的零窗口结果是规划器的最小锚点（**不要判为非法**，缓存本身才拒绝它） |
| 池尺寸与回退 | `runtime/engine/kv_capacity.{h,cpp}`、`model_instance.cpp`、`include/ninfer/base.h` | 池 = 实测空闲 − `kDefaultKvStreamHeadroomBytes`（**64 MiB 地板**）；装不下就按 curve 重算并回退 64 MiB/次、最多 6 次；显式容量不缩；`cudaGetLastError()` 清掉失败 `cudaMalloc` 的latch |
| 观测 | `include/ninfer/observability/{resources,snapshot}.h`、`program.h`、`engine_core.h`、`apps/cli/main.cpp` | `StreamedKvStats` → `RuntimeStats.streamed_kv` → summary 打印；`--kv-stream-report` 才记计时事件（默认关，开关代价在噪声内） |
| 选项 | `--kv-stream`、`--kv-stream-report` | 要求 `--kv-capacity auto`、`--max-concurrency 1`、单设备、context cache 关 |
| 单测 | `tests/test_kv_stream_layout.cpp` | 全 `constexpr` 规划/映射断言（含环深、区归属、零窗口锚点）。**未在 CMake/ctest 下跑过**：远端用 `g++ -std=gnu++20 -I src -I include tests/test_kv_stream_layout.cpp && ./a.out` 直接编译运行，全绿 |

---

## 3 测量存根（不要再重跑，先看这里）

全部在 `kv-streaming-measurements.md`，每条含时间、提交、命令、原始输出、以及它决定了什么：

| 编号 | 内容 | 结论 |
|---|---|---|
| M-01 | 池按实测空闲 + 强制触发回退 | 47 894 页组、启动后余 53.5 MiB；`--kv-headroom-mib 1` → 48 816 失败→47 880 起来 |
| M-02 | 环深 1 的仪表 | 每轮 64.0 ms、拷贝 42.6、**暴露 22.65** |
| M-03 | 环深 2 | 每轮 **56.9 ms**、暴露 11.4（3 次重复 56.6/56.8/56.9） |
| M-04 | 环深 3 | 每轮 61.0–62.9、暴露 11.7 → **否决**；暴露与环深无关 ⇒ 是轮首/轮末固定开销 |
| M-05 | 旧深环（120K/190K） | 作废理由：那里环是空的 |
| M-06/M-07 | 跨轮预取 | 单次每轮 54.1 ms（−5%）；等价性已在 M-10 里一并验证（该提交之后的构建） |
| M-08 | 数值脆弱性探针 | 同二进制、同 `--seed 42`，只改 `--prefill-chunk` 256→128，**token 就变了** ⇒ 这个 246K 的 α 提示词不能用作等价性 oracle；比较必须固定 `--prefill-chunk` |
| M-09 | 全驻留 190K + MTP3 装不下 | 差 71 MB；两种布局的重叠区间只有 ~183–186 K |
| M-10 | 原版金标与等价性（8k/100k/183k）+ 251K 基线 | 512 token 逐 id 与逐位 logprob 全部相同；251 K 每轮 56.7 ms、69.2 tok/s、1 194 MiB/轮、暴露 15.8 ms |
| M-11 | 投机后端占用与草稿成本 | DFlash2 权重 +1.80 GiB ⇒ 窗口 −105 024 token（77 888）；但每轮比 MTP-7 便宜 9.3–11.2 ms ⇒ **窗口固定时它全程赢 29%，换后端无收益**；收益在“把权重还给窗口” |
| 设计 | span（帧）流水（§5.13，事实以 §5.13.1 为准） | 环从“整层尾部”改成“几帧”轮转，一层内边拷边算 ⇒ 取回每轮 9.2–15.8 ms 空转。**落点：INT8 家族；`k8v4` 不算交付**。量级预测 +18~24%，**未实测、待重算** |

**确定性事实**：同一二进制、同一命令行、`--seed 42 --greedy` 连跑两次，token ids 完全相同（已验）。

---

## 4 未完成 / 待审的硬缺口（按优先级）

1. **等价性金标：已完成（M-10）**。`scripts/kv-stream-golden.sh` 已被
   `tools/bench/run_kv_stream.py` 取代（后者记录 argv、逐轮指标、设备身份与判定）并被删除。
   重新录金标：`golden --suite equiv --force`（原版，约 4 分钟）→ `check --suite equiv`（我们的，约 12 分钟）。
2. **跨轮预取的等价性：已随第 1 项关闭。** 若以后再改调度，先跑 `check --suite equiv`。
3. **长上下文回归基线：已录**（`golden --suite perf`，`oracle: self`，标注为回归基准而非等价性 oracle）：
   `prompt-shape`（8 K 下必须生成满 512 token 的哨兵）、`stream-251k`、`resident-8k`。
4. **流式下恢复 CUDA Graph**：−3%（≤130 K），当前为流式关图。
5. **更小的流式页**（打破窗口内外同一格式）：最多 +40%，**等用户批准**。
6. 低优先：M4b 分段流水、M2 `int8` 一档一致性、M5 共存（多序列/Hybrid prefix/TP）、
   `tests/cmake/CoreTests.cmake` 里注册 `ninfer_kv_stream_layout_test`。
7. **自适应草稿器后端（DFlash2 ↔ MTP）：延后，作为独立特性 PR**（数据见 M-11）。
   用户决定：日常先用 `--spec mtp --draft-tokens 3` 跑流式，继续维护与优化；等常规手段用尽再做。
   做之前必须先查证三件事（均不需 GPU）：
   (a) `kv_stream_space` 的设备页能否运行时增加（`kv_capacity_increment` 是 1 页，但需确认
       释放 DFlash2 权重后新页真的拿得到）——它决定“1.80 GiB ⇒ +105 024 token 窗口”是否成立；
   (b) 换出 DFlash2 权重是否只需释放，还是必须重建 CUDA Graph（graph 绑权重指针）；
   (c) 同一请求中途换后端是否改变 token（`--invariant-attention` 只消掉 attention 那部分）。
   若三条都成立，设计是：**单向切换**（上下文单调 ⇒ 拐点至多穿一次）、
   判据 `R × Δt > S`（剩余 token × 速度差 > 切换成本）、里程碑 16 K 评估 + 迟滞、
   内层宽度继续交给 `--adaptive-mtp`、DFlash2 侧钉 7（每轮成本对 k 平坦）。
   MTP → DFlash2 方向不可逆（drafter 上下文 K/V 来自目标第 5/19/33/47/61 层隐状态，未保存），
   除非加“仅投影保温路径”（每 token 一次 `[25600×5120]` 线性，估 0.3–0.5 ms）。
8. **MTP 宽度扫描（`--draft-tokens` 3..K、`--adaptive-mtp`）由用户自行测量，不在本项目范围。**
   用户以前用过 `--adaptive-mtp`；宽度与流式的关系（拷贝受限时判据应是“每轮提交 token 数”
   而不是“每轮时间”）已写在 M-11，但**没有本卡实测**，不要把外推当结果引用。
9. **`--embedding-residency host`：白拿的显存，已写进推荐配方，但加速数字是外推。**
   产物 `text/token_embedding` 是 `gguf_iq2_s` 388.4 MiB，`tie_word_embeddings: false` ⇒ 可搬。
   要证实只差一个 footprint 变体（`--embedding-residency host`），需用户批准再跑。
10. **帧（span）流水：已做完一环，实测更慢，已退出默认路径（M-12）**。
    上游同名字段：`kv-stream-span-tuner.h`、`fattn.cu:145-146,1983-2058`（span = 32 页量级、
    按端到端计时自适应）。我们这一轮的形态：INT8 家族一次调用内分帧发射、层末一次 reduce、
    帧宽与内核同规则、默认仍整层拷贝（`NINFER_STREAM_FRAMES=1` 开帧）。
    251 K 同一构建 A/B：整层 69.2 tok/s / 56.7 ms / 暴露 15.6 ms；帧 64.0–67.7 / 57.9–61.5 /
    8.7–11.3。**结论：链路已占每轮 83%（1 191 MiB ÷ 25.4 GB/s ≈ 47 ms），层内切细只是重排
    等待，额外的发射与流事件反而收费。**
    下一步要动每轮字节数，不是切得更细：
    a) 帧级环（`kv_stream_layout.h` 的 `frame_pages`/`kv_stream_region` 已就绪且单测通过；
       待接进 `StreamedKVSpace` 的拷贝寻址——现在还是 `ring_region + page` 的整层寻址），
       估算窗口 +≈6%、每轮 −3 ms；
    b) 更小的流式页（改质量合同，需所有者签字）；
    c) `rk8v4` 上的帧路径至今未测（窗口 114 496 token，尾部更大，量级应当更不利，不得外推）。
    已验：等效 4/4（默认路径）+ 分帧开启时 183 K 行也逐位相同；251 K 默认与基线逐项相同。
11. **主机侧注意力分担：已按算术排除，不要做**（§5.14）。尾巴注意力一轮约 90 GFLOP，
    16 核 CPU 只到 0.1 TFLOP/s ⇒ 只能接 ~3%。它不降质量，纯粹是算力对不上带宽。

**已被实测否决，不要重做**：环深 3（M-04）；不加回退机制就把 headroom 压到 16–32 MiB；
用 120K/190K 那种"环是空的"配置去论证调度收益（这正是 §5.8 和 M4a 两次错误测量的根因）。

---

## 5 用户硬性约束（违反会被回滚）

- **不删任何东西**（模型族、架构、KV 格式都保留）；任何删减逐条审核。A 组删除批次留在旧分支 `stash@{0}`。
- **绝不碰 `.thinkrail/`**（TODO 存储，已 gitignore；连目录都不许删改）。
- 测试**自清理**：不留进程、临时文件、显存（"那边是部署的真实电脑，不是垃圾场"）。
- 禁止：sudo、装/升级包、改 `start.sh` 核心参数、`git clean -fdx`、`reset --hard`、force-push、
  重写历史、杀非本次启动的进程、下载模型。
- **先提交、再改 TODO 状态**；结论必须来自真实命令输出，失败即停并报告，禁止编造。
- **禁止全量编译**（≈30 分钟）：只允许增量 `cmake --build build-sm89`；ccache 会悬挂 ⇒ `CCACHE_DISABLE=1`。
- **动工/测 GPU 先说明计划并获批准**；能手算就手算；“停”即停。
- **主机不能 push**：Windows 端 push，主机只 `git fetch && git reset --hard origin/kv-streaming`；
  **构建期间不得动源码**（曾经因此报废一轮 70 分钟构建）。
- 不要为常量改动跑验证测试（`--kv-headroom-mib` 外部已可控）。
- 测试**必须有留痕**：结果进 `kv-streaming-measurements.md`（时间/提交/命令/输出），不要只留在会话里。
- **测试一律 `--kv-dtype rk4v4`**（便宜、窗口 181 440），但**交付结论必须在 `rk8v4` 上复核一次**
  （所有者 2026-11-09 明确：“我的目标是 rk8v4，目前先用 rk4v4 测试”）。
- **KV 质量下限 `rk8v4`**（SPEC 交付项 1）；`rk4v4` 只作测试基线。

---

## 6 基准参数的既有认知（避免重复推导）

- `--invariant-attention` 是 on/off 逐字节对比的**必要条件**（否则注意力按宽度选不同 split-KV 路线）。
- 基准档：`--prefill-chunk 256 --gdn-state-fp16 --greedy`（`--seed 42` 一起用时输出可复现）。
- `rk4v4`：每页 71 680 B、每 token 17 920 B、页 = 64 token；`rk8v4`：每页 104 448 B。
- PCIe 4.0 x16 实测 25.77 GB/s（H2D），与计算独立；有效拷贝速率 24–25 GB/s。
- 容量的自动分配把池定在 `layers × logical_pages + resident` 的上限与实测空闲之间，
  所以同一 dtype 两次启动的槽数可差几十页。
- `plan_kv_stream_split`：`R = (pool − depth·P) / (layers − depth)`；`depth ≥ 2` 只在
  `pool ≥ 6·P` 时采用（否则窗口的比重叠值钱）。
- 轮时间 = `acceptance_length / decode_speed`；接受长度随采样文本浮动，**tok/s 不能跨配置直接比**。

---

## 7 命令速查（压缩后的新会话直接照抄）

```bash
# 0) 同步代码（Windows 端已 push 之后；构建期间不得再改源码）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  git fetch origin && git reset --hard origin/kv-streaming'

# 1) 增量构建（只允许增量；ccache 会挂）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  CCACHE_DISABLE=1 cmake --build build-sm89 -j 16 2>&1 | tail -20'

# 2) 主机单 TU 单测（不需 GPU）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  g++ -std=c++20 -O1 -Isrc -Iinclude tests/test_kv_stream_layout.cpp -o /tmp/kvl && /tmp/kvl'

# 3) 等价性（必须 4/4、逐 id 相同；约 12 分钟）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  python3 tools/bench/run_kv_stream.py check --suite equiv'

# 4) 251K 基准（PASS 后再看数字；注意看 stream_upload_mib_per_round 必须 > 0）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  python3 tools/bench/run_kv_stream.py bench --case stream-251k --reps 1 --tag <标签>'

# 5) 单次运行的结构化报告（查 decode 是否真的搬了页）
python3 - <<'EOF'
import json,glob
f=sorted(glob.glob("bench/results/kv-stream/runs/*/stream-251k/streamed/rep-01/report.json"))[-1]
d=json.load(open(f)); s=d["stream"]
print(f); print({k:s[k] for k in ("upload_bytes","download_bytes","resident_pages","streamed_pages","rounds")})
print(s["rounds_series"][:3])
EOF
```

**性能读法**：`round_ms_median` 与 `stream_exposed_ms_per_round` 一起看；
`stream_upload_mib_per_round == 0` 意味着 decode 没搬页，数字全部作废（§0.1 的教训）。

**`rk8v4` 复测**（交付目标，必须做一次）：同一条 `bench` 命令，把 case 的 dtype 覆盖成 `rk8v4`
（工具默认 `rk4v4`）。窗口会从 181 440 降到约 114 496 token，尾部更大、拷贝更多，
所以 `rk8v4` 的收益应该比 `rk4v4` 更明显 —— 但必须实测，不得外推。
