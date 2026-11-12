# 接手文档：流式上下文缓存（`context-cache-tier`）

**这是接手第一份要读的东西。** 目标、现状、证据、下一步、坑，全在这一页；
设计推导过程在 `docs/maintainer/kv-stream-context-cache.md`（§1–§21，逐节带行号与实测）。

---

## 1. 背景（为什么会走到这里）

`--kv-stream` 让 27B 模型在 16 GB 卡上跑 192 K 上下文：**每页 KV 的权威副本在 pinned host，
设备只持一段窗口**。这套机制本身已经成熟（`src/ops/kv_stream/*`、几何、环、事件、量化内核，
长期实测 100 tok/s @183 K、95 @251 K），**本分支不动它**。

问题出在**缓存语义**：流式空间自己拥有全部槽与整块宿主 arena，于是 ninfer 原有的
"上下文缓存体验"——续聊命中、取消后续聊、多会话保留、会话文件——在流式下**要么缺失、要么用了别人的字节**。
缺陷 #11–#15（`[stale reservation]`、`[context cache exhausted]`、`[frontier beyond space]`、
`[cancel 后同一句]`、`[静默假命中]`）是同一族：**声明的上下文比字节活得更久**。

所有者的要求（原话记录在 `docs/maintainer/kv-stream-context-cache.md` §12）：
**「流式只是扩上下文，不破坏 ninfer 原有使用体验和流程」**——开着流式，缓存行为要与驻留模式一致。

## 2. 任务

让 `--kv-stream` 下的缓存行为与驻留模式一致：

| 能力 | 定义 |
|---|---|
| 续聊命中 | 同一会话下一轮只 prefill 增量，ids 与冷跑逐 token 相同 |
| 同一 prompt 重发 | 命中 prompt 末尾的 checkpoint（TurnClosure / 端点锚点） |
| 取消后续聊 | 取消不清空上下文；能发布锚点/salvage 时必须能续 |
| 多会话 A→B→A | 第二个会话不覆盖第一个；A 回来能命中（**当前未达标**） |
| 会话文件 | `--slot-save-path` + `POST /slots/{id}?action=save\|restore\|erase`（**当前未达标**） |
| 不崩、不错答 | 任何缓存状态都不得导致崩溃或静默错答 |

**硬约束**（所有者指定，见 §12）：`src/ops` 零改动；`text.cpp` 的拷贝编排形状与事件顺序不变；
拒绝"有疑问就重算"（冷启动只用于 `space_pages == 0`）；先量后改；未测就写"未测"。

## 3. 当前现状（一句话）

**单会话的缓存体验已达标并有逐 token 证据；多会话被显式拒绝（等待"每 region 一窗口"落地）；
会话文件已放行入口但仍红在覆盖判据。**

### 3.1 已达标（实测，可复跑）

| 能力 | 配置 | 证据 |
|---|---|---|
| 续聊命中（endpoint） | 默认 | `reuse=1 base=23424`，ids 与冷跑逐 token 相同 |
| 取消后续聊（prefill 中途取消） | 默认 | 同一测试 `reused 23424`，进程存活 |
| 同一 prompt 重发 | `prepare_tokens(..., anchor_prompt_end=true)` | `reused 900`，ids 相同 |
| 换会话被接纳、不出错 | 默认 | 全部用例 ids 与冷跑逐 token 相同 |
| 流式核心不受影响 | — | `src/ops` diff = 0 |

复跑：`ctest -R "kv_stream_(reuse|eviction)_real"` → **2/2 通过**（约 130 s + 76 s，RTX 4080 SUPER）。

### 3.2 未达标

| 项 | 现状 | 卡在哪 |
|---|---|---|
| 多会话 A→B→A | 启动期**拒绝** `max-private-continuations > 1` | 单窗口无法服务两个上下文；两版交换机制都失败（§20.19–§20.24）。正解见 §5 |
| 会话文件 | serve 已接受 `--slot-save-path`；`save` 返回 500 `retained session Text KV coverage is inconsistent` | `session_snapshot` 用 `mapped_pages`（流式恒 0）判覆盖；导出形状未定（§21.11–21.13） |
| 全量回归 | **未跑**（自引擎改动以来） | 需 `ctest -j 1`（385 例，约 1.5 h） |
| 性能复核 | **未测** | 需 251 K `round_ms_median ≤ 41.4 ms × 1.01` |

## 4. 代码地图（改哪儿、别碰哪儿）

| 位置 | 作用 | 状态 |
|---|---|---|
| `src/models/qwen3_5/program/storage/kv_stream_space.{h,cpp}` | 宿主 arena、窗口/环、块表发布、region 状态 | 本分支主战场 |
| `src/models/qwen3_5/program/storage/kv_store.h` | `KVAddressSpaceStore`：`stream_can_serve`、`enter_request` 调用、流式截断 | 有多处流式特例 |
| `src/models/qwen3_5/program/prefill.cpp` | 请求进入（`enter_request` 在 reuse 分支**之前**）、reuse 分支 | 顺序敏感 |
| `src/models/qwen3_5/program/planning/startup.cpp` | 流式形状的守卫（锚点、宿主状态槽、private 上限） | 现在拒绝多 owner |
| `src/models/qwen3_5/program/storage/session_snapshot.cpp` | 会话写/读 | 流式导出未实现 |
| `src/runtime/engine/context_cache/resource_manager.h` | 目录、候选、驱逐、发布后 demote | 有大量 `[candgen]`/`[evict]` 调试输出 |
| **不要动** `src/ops/kv_stream/*`、`text.cpp` 的拷贝编排 | 核心流式能力 | 所有者硬约束 |

## 5. 下一步（按顺序，已设计，不再猜）

### 5.1 多会话：每 region 各自的窗口与环（**优先**）

**为什么**：所有者的 serve 参数里没有 `--slot-save-path`，会话文件不会被触发；真正会触发的是多会话。

**做法**（§21.14）：空间里的单份 `split_` → `region_splits_`，`split()` 返回当前 region 的那份；
region r 的设备槽基址 = `r * (pool_pages / regions_)`。R=1 时沿用 `cache.stream_split()`（逐字节不变）；
R>1 时各 region 用 `plan_kv_stream_split(pool_pages / R, pages_per_layer, layers, runs, ratio)` 规划。

**收益**：不存在"换窗口"，`flush_window`/`upload_window`/`window_source_` 一概不需要——那类 bug 从结构上消失。

**代价（先算）**：每 region 的环至少 1 个 run（128 页），R=2 时每层窗口小 128 页，251 K 每轮字节上升，需实测。

**验收**：`private=2` 时 A→B→A `reused > 0` 且 ids 与冷跑逐 token 相同；随后跑两个 fast 用例 + 251 K 判据。

### 5.2 会话文件（若所有者需要跨重启）

1. 加独立访问器 `covered_tokens(handle)`（**不要**改 `mapped_pages` 语义，§21.5 已因它回滚一次）；
2. 导出/导入前先定案"一条页记录是一层还是全部层"（§21.13 的实验：`--no-cuda-graph` 下打印
   `text_payload / state_bytes / images`，或用驻留模式存一次会话看真实文件大小）；
3. 写入 `session_snapshot` 的流式分支，跑 `parity.py --only P5`。

## 6. 验证与工具

```bash
# 构建（测试机 debian, RTX 4080 SUPER 16 GB, sm_89）
cmake -S . -B build-verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CCACHE_DISABLE=1 cmake --build build-verify -j 16

# 两个 fast 用例（必须绿）
NINFER_TEST_ARTIFACT=$HOME/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  ctest --test-dir build-verify -R "kv_stream_(reuse|eviction)_real" --output-on-failure

# 一致性验收（serve 级，P1..P6）
python3 tools/bench/kv_stream/parity.py --model <artifact> --bin build-verify/apps/ninfer-serve

# 等价套件（改动后必须 4/4）
python3 tools/bench/run_kv_stream.py check --suite equiv
```

调试开关（全部 stderr，生产默认关）：

| 开关 | 输出 |
|---|---|
| `NINFER_STREAM_DEBUG=1` | 请求进入（owner/region/depth/window/target）、窗口上传/回写、锚点恢复、截断原因、首步 hidden/logits 与槽号 |
| `NINFER_STREAM_VERIFY=1` | 字节 oracle：环/窗口与宿主记录逐页比较，计数进 `stream.verify` |
| `NINFER_MAT_DEBUG=1` | 目录候选（`[candgen] CAT/IDX`）、物理可行性、`peak/held/limits` |
| `NINFER_ADMISSION_DEBUG=1` | 准入阻塞原因、`[evict]` 释放路径 |

## 7. 坑（这一轮真的踩过，别重犯）

1. **不要用 `sed` 改 C++**：三次把 `\n` 写进字符串或吞掉函数签名，每次 ~5 分钟。用编辑工具。
2. **调试打印必须排序**：`cudaMemcpy` 在默认流上读到的是旧值，曾导致"两个不同请求打印相同 logits"的假象。
   用 `cudaMemcpyAsync(..., device.stream)` + `cudaStreamSynchronize`。
3. **同一进程只能有一个 27B 引擎**：第二个会 `cudaMalloc` OOM（测试里用同一个 engine 顺序跑）。
4. **测试时序**：取消必须发生在 prefill **进行中**（`prompt ≈ 3000 句 ×12 token`，10 s 取消）；
   在 prefill 完成后取消会发布"越过 prompt 末尾"的端点——那是正确行为，但同一 prompt 永远用不上。
5. **槽接口零基且要 filename**：`POST /slots/0?action=save`，body `{"filename": "..."}`。
6. **`private=1` 是当前唯一被证明的配置**：`private=2` 会被启动守卫拒绝（这是有意的）。
7. **`--max-context` 很小 + CUDA Graph** 会让启动失败（`cudaErrorStreamCaptureInvalidated`）；
   测量用 `--no-cuda-graph`。
8. **owner stamp 的单位是"目录条目"**（slot + generation），不是"当时在跑的请求"——按请求算会把
   取消后留下的 checkpoint 冤枉成"别人的字节"（§20.10）。

## 8. Git 与文档

| 项 | 值 |
|---|---|
| 工作分支 | `context-cache-tier`（HEAD `eefaeb37`，140 个提交自 `ff23cbe0`） |
| 基线 | `kv-stream-27b-16g` @ `ff23cbe0`（= 本分支起点） |
| 剪裁版 | `pruned-qwen38-27b-16gb` @ `bcbd38f5`（未动） |
| 回退点 | tag `kv-stream/layout-switch-before-removal` → `c8bdb380` |
| 设计/推导 | `docs/maintainer/kv-stream-context-cache.md`（§1 语义与代价、§2 基线契约、§3 偏差、§10–§13 计划重审、§16 基线代码清单、§17 融入形态、§20 region 编年、§21 ub-mode 日志） |
| 历史台账 | `bench/results/kv-stream/ledger.md`、`layout-rename-defect.md`、本目录 `handoff.md`（M1 时期） |
| 上游参考 | `docs/maintainer/kv-streaming-strategy.md`（唯一计划权威）、`kv-streaming-design.md`、`kv-streaming-fidelity.md`、`kv-streaming-measurements.md`、`kv-streaming-verification.md`（C1–C16） |

## 9. 进度（诚实口径）

| 维度 | 完成度 | 依据 |
|---|---|---|
| 单会话命中/取消/不崩 | **100%** | 逐 token 对比 + 2/2 用例 |
| 多会话 | 20% | region 分区与窗口机制已就位，交换机制已删除，正解已设计未实现 |
| 会话文件 | 20% | 入口放行，导出/导入未实现，阻塞在载荷层结构定案 |
| 回归/性能 | 0% | 未跑 |
| **整体** | **约 50%** | 按上面四项等权，且"单会话"是最常走的一条 |

## 10. 接手第一小时建议

1. 读 `docs/maintainer/kv-stream-context-cache.md` 的 §3（偏差）、§16（基线代码清单）、§21（日志）；
2. 在测试机构建并跑 §6 的两个 fast 用例，确认基线是绿的；
3. `git log --oneline ff23cbe0..HEAD | head -40` 看这轮改过什么；
4. 从 §5.1 的第一片开始（`region_splits_`），先证 R=1 逐字节不变，再提 `regions_`。
