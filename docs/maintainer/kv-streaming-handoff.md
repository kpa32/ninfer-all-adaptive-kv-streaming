# KV 流式：工作交接

> **读这一份就能接着干，不需要重跑任何东西。** 最后更新：2026-11-08T20:20Z，
> HEAD `73c3ae14`（分支 `kv-streaming`，已推送）。
> 设计权威：`kv-streaming-design.md`（§5.11、§5.12 是最新实测）；
> 测量存根：`kv-streaming-measurements.md`（M-01…M-10，每条都有时间/提交/命令/输出）；
> 运行记录：`bench/results/kv-stream/`（原版金标、逐次 argv、逐轮指标与判定）；
> 外部资料：`adaptive-kv-streaming-inputs.md`；参考实现源码：`D:\GithubRepo\llama.cpp-adaptive-kv-streaming`（只读）。

---

## 0 一句话状态

262 144 原生上下文在 16 GB 卡上跑通，流式 decode 从 M1 的 **36.4** 提到 **48–53 tok/s**
（`rk4v4` + MTP3 + 251 K 提示，提交 `47f14a7c`）。

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

**关键判断**：拷贝流已经占每轮 84%，所以下一个两位数收益只能来自**更小的流式页**（最多 +40%，
需要用户批准改变“窗口内外同一格式”的质量合同），不是继续找重叠。

**投机后端（M-11）**：DFlash2 每轮比 MTP-7 便宜 9.3–11.2 ms，但它多占 1.80 GiB ⇒ 窗口从
182 912 降到 77 888 token。**窗口固定时那份搬运惩罚对所有后端一样，所以 DFlash2 全程领先约 29%，
换后端不赚钱**；收益只在“把权重还给窗口”。自适应后端切换已延后为独立特性 PR（§4.7）。
日常使用：`--spec mtp --draft-tokens 3`。

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

## 2 已落地（代码地图，截至 `a525f693`）

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
- **所有测试一律 `--kv-dtype rk4v4`**（除非正在对比格式本身）。
- **不要为常量改动跑验证测试**（`--kv-headroom-mib` 外部已可控）。
- 测试**必须有留痕**：结果进 `kv-streaming-measurements.md`（时间/提交/命令/输出），不要只留在会话里。
- KV 质量下限 `rk8v4`；`rk4v4` 只作测试基线。

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
