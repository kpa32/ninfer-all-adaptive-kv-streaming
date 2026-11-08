# KV 流式：工作交接

> **读这一份就能接着干，不需要重跑任何东西。** 最后更新：2026-11-09T03:0x+08:00，
> HEAD `a525f693`（分支 `kv-streaming`，已推送）。
> 设计权威：`kv-streaming-design.md`（§5.11、§5.12 是最新实测）；
> 测量存根：`kv-streaming-measurements.md`（M-01…M-08，每条都有时间/提交/命令/输出）；
> 外部资料：`adaptive-kv-streaming-inputs.md`；参考实现源码：`D:\GithubRepo\llama.cpp-adaptive-kv-streaming`（只读）。

---

## 0 一句话状态

262 144 原生上下文在 16 GB 卡上跑通，流式 decode 从 M1 的 **36.4** 提到 **48–53 tok/s**
（`rk4v4` + MTP3 + 251 K 提示，提交 `47f14a7c`）。其中调度那一半（环深 2 + 跨轮预取）
**性能已测、等价性未验**——这是当前唯一的硬缺口，见 §4。

| 配置（251 K 提示 / 容量 262 144 / `rk4v4` + MTP3） | 每轮 | decode | 拷贝占用 | 计算流等拷贝 |
|---|---:|---:|---:|---:|
| M1 环深 1 | 64.0 ms | 36.4 | 42.6 ms | 22.65 ms |
| 环深 2 | 56.9 ms | 48–53 | 47.5 ms | 11.4 ms |
| 环深 3（已否决） | 61.0–62.9 ms | 46.9 | 51.9 ms | 11.7 ms |
| 环深 2 + 跨轮预取 | 54.1 ms（单次） | 41.4（接受长度偏低那次） | — | — |

**关键判断**：拷贝流已经占每轮 84%，所以下一个两位数收益只能来自**更小的流式页**（最多 +40%，
需要用户批准改变"窗口内外同一格式"的质量合同），不是继续找重叠。

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
| M-06/M-07 | 跨轮预取 | 单次每轮 54.1 ms（−5%）；**等价性未验** |
| M-08 | 数值脆弱性探针 | 同二进制、同 `--seed 42`，只改 `--prefill-chunk` 256→128，**token 就变了** ⇒ 这个 246K 的 α 提示词不能用作等价性 oracle；比较必须固定 `--prefill-chunk` |

**确定性事实**：同一二进制、同一命令行、`--seed 42 --greedy` 连跑两次，token ids 完全相同（已验）。

---

## 4 未完成 / 待审的硬缺口（按优先级）

1. **等价性金标还没录**。`scripts/kv-stream-golden.sh` 已改成正确方向
   （`--record` 用**原版**二进制录 ids 到 `docs/maintainer/kv-stream-golden/`，比对模式跑我们
   自己的构建，并额外跑一次 `--kv-stream`），但**还没跑过、目录还是空的**。
   一条命令即可（3 个 case，约 10 分钟，GPU 空闲时）：
   ```bash
   bash scripts/kv-stream-golden.sh --record      # 原版：8k / 100k / 184k
   bash scripts/kv-stream-golden.sh               # 我们的：全驻留 + 184k 的流式，逐 id 比对
   ```
   为什么是这三个 case：>186 K 时全驻留 + MTP 在这张卡上起不来（差 71 MB），<181 440 token 时
   流式根本不参与（窗口装得下）——**184 K 是两种布局都能回答、且流式确有其事的唯一区间**。
2. **跨轮预取（`47f14a7c`）缺等价性证据**。缺陷方向一旦存在就是"读到没搬完的页"，只能靠逐 token
   比对发现。上面第 1 项做完即关闭。
3. **`ours-*` 的流式 ≥184 K 回归金标**：原版答不了 >186 K，所以长上下文的金标只能由我们自己的构建
   录（明确标注为"回归基准，不是等价性 oracle"），尚未录。
4. **流式下恢复 CUDA Graph**：−3%（≤130 K），当前为流式关图。
5. **更小的流式页**（打破窗口内外同一格式）：最多 +40%，**等用户批准**。
6. 低优先：M4b 分段流水、M2 `int8` 一档一致性、M5 共存（多序列/Hybrid prefix/TP）、
   `tests/cmake/CoreTests.cmake` 里注册 `ninfer_kv_stream_layout_test`。

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
