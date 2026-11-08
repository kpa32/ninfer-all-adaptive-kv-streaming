# KV 流式：工作交接（临时文档，M1 完成后删除）

> 读这一份就能接着干。最后更新：HEAD `842fb5eb`。
> 设计权威：`docs/maintainer/kv-streaming-design.md`；外部资料：
> `docs/maintainer/adaptive-kv-streaming-inputs.md`。

## 0 一句话状态

262 144 的**启动路径已经全通**（扁平槽池、主机权威副本 6.85 GB、容量按槽解析、请求被接纳、
页面归属干净），现在卡在**请求执行第一步**：事务/提交路径仍假设序列有逻辑页，抛
`logical KV page handle is stale`。修掉它就能第一次真正 decode。

## 1 分支、提交、环境

- 分支 `kv-streaming`，已推 `origin/kv-streaming`。最近提交（本阶段 12 个）：
  `cf56e416`…`842fb5eb`；更早：`9f9c66e6` 容量算式与槽映射、`b8251f77` 扁平池几何、
  `6541c366`/`88b42318` 运行时接入设计、`be6d5a79` M0 收尾。
- Linux 构建机：`kpa@192.168.2.13`，仓库 `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`
  （已 checkout `kv-streaming`），构建树 `build-sm89`（`BUILD_TESTING=OFF`，Release）。
- 模型：`~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（15 GB，27B Dense）。
- 本地：`D:\GithubRepo\ninfer-all-adaptive-kv-streaming`；参考实现只读：
  `D:\GithubRepo\llama.cpp-adaptive-kv-streaming`。

### 构建成本（重要，别再踩）

| 改什么 | 耗时 |
|---|---|
| 一个 `.cpp`，`--target ninfer` | **2.6 s** |
| `git pull` 后全量（四个程序） | 30 s |
| **改被全树包含的头**（`paged_kv_cache.h`、`kv_store.h`、`text.h`） | **20+ 分钟** |

规则：**头文件改动攒成一次批量提交**；日常只改 `.cpp` 并用
`cmake --build build-sm89 --target ninfer -j 14`。不要再用 `git switch`（整树重检出）。

### 运行模板

```
./build-sm89/apps/ninfer <model> --prompt "..." --max-context 262144 \
  --kv-capacity auto --kv-stream --kv-dtype rk8v4 --gdn-state-fp16 \
  --prefill-chunk 256 --spec mtp --draft-tokens 3 --greedy --max-new 8
```

`--kv-capacity auto` 是 `--kv-stream` 的硬要求；CLI 默认已关 context cache（流式也要求关）。

## 2 当前卡点（精确）

顺序：`engine ready` → 请求被接纳 → `error: worker crash: logical KV page handle is stale`
→ cleanup 报告 `left 0 main / 0 backend Device KV pages ... without an owner`（干净）。

即：`LogicalKVPageStore::require()` 收到一个**默认构造的逻辑页句柄**。流式序列
`address.page_count == 0`，所以任何"遍历页 0..page_count-1"的循环都不会跑；出问题的是
**不经过 page_count 就取页**的地方。候选（按可能性）：

1. `ProgramImpl::bind_sequence_kv` / `refresh_state_views`（`storage/context.cpp`）
2. `transactions/commit.cpp` 的提交路径
3. `prefill.cpp:475` 附近 `commit_activation` 之后的 KV restore 循环
4. `materialization.cpp:834` 之后 `prepare_kv_restores`（`mapped_pages(address)==0` 时可能取 `[0]`）

定位手段（最省时）：在 `LogicalKVPageStore::require` 里临时
`std::fprintf(stderr, "%s\n", __PRETTY_FUNCTION__)` 不行（拿不到调用者），改用
`gdb --batch -ex run -ex bt` 跑一次即可拿到调用栈；或临时 `__builtin_return_address(0)`。

## 3 下一步（有序）

1. **修卡点**：让绑定/提交路径在 `text_stream_space != nullptr` 时跳过逻辑页遍历，只保留
   行与前沿。多半只需在两三个 `for (page…)` 外层加分支。
2. **跑 8192 全驻留档**（`--max-context 8192 --kv-capacity auto --kv-stream`）：此时
   `S = 0`、无任何搬运，输出必须与不开流式逐字相同。这是扁平池本身的验收。
3. **跑 262 144**：第一次真环形搬运。对照 §5 上界 15.3 tok/s，记录
   `summary` 的 decode 速度、峰值显存、`summary planned slack`。
4. **M2 一致性**：同格式 on/off greedy token ID 与 logprobs 逐 token 一致（8K/32K/120K）。
5. **M3 控制器**（扩环/收环/刹车）、**M4 分段流水**（偏 softmax 合并）、**M5 共存**。

## 4 已落地的东西（在哪里、怎么用）

| 组件 | 文件 | 说明 |
|---|---|---|
| 容量算式与槽映射 | `src/core/kv_stream_layout.h` | `plan_kv_stream_split`（取最大可行窗口）、`kv_stream_slot`、`kv_stream_window_slot`、`kv_stream_ring_slot`。全 `constexpr`，有单测 `tests/test_kv_stream_layout.cpp` |
| 扁平池几何 | `src/models/qwen3_5/state/decoder_state.{h,cpp}` | `plan_cache(..., KVStreamSplit*)`：平面变成"每类 KV 一个"、保留容量解析给的页组数、表行 = 每层一行 + 序列行；`PagedKVCache::flat_slots()/stream_split()` |
| 每层表按行取/按槽发布 | `src/core/paged_kv_cache.{h,cpp}` | `row_at`、`publish_at`、`publish_slots_at`（按页号，不走 lease） |
| 按页号搬运 | 同上 | `copy_to_host_pages` / `copy_from_host_pages` |
| 运行时空间 | `src/models/qwen3_5/program/storage/kv_stream_space.{h,cpp}` | 自带 `HostKVArena`（`P × page_bytes` 主机权威）；`advance(active_pages)` 发布新页与其离窗页（离窗先 D2H 再改表）；`upload(layer, copy)` 把该层流式段一次拷进环；`ready/consumed` 事件；`reset()` |
| 地址空间接入 | `storage/kv_store.h`（`KVAddressSpaceStore`） | `enable_streaming(space)`；`prepare_activation`/`commit_activation`/`activate`/`deactivate`/`ensure_mapped_to_tokens`/`resize_entitlement`/`release_growth_entitlement`/`commit_frontier`/`destructive_truncate` 全部有流式分支 |
| 计划接入 | `planning/startup.cpp`、`planning/request_plan.cpp`、`runtime/engine/kv_capacity.cpp`、`runtime/contract/resources.h` | 槽轴：`maximum_pages64 = layers × logical_pages`；`logical_main_pages` 让 `resolved_tokens` 报逻辑上下文；MTP 池保持 `logical_pages`；流式请求 KV 额度 = 1 页组 |
| 逐层搬运与等待 | `execution/text.{h,cpp}` | `set_kv_stream(space, copy)`；`run_layers` 开头 `issue_kv_stream_uploads()`（每层先等上一层 `consumed`）；`run_stage_layers` 在 full 层 attention 前等 `ready`、后记 `consumed` |
| 关图 | `planning/startup.cpp` | `impl->use_cuda_graph = inputs.use_cuda_graph && !kv_stream` |
| 选项链 | `include/ninfer/engine_options.h`、`models/load_options.h`、`apps/cli/*` | `--kv-stream`，help 已写 |

## 5 已核实的事实（不要重猜）

- 页 = **64 token**；每 (layer, rank, kind) 一个 plane（旧布局）；16 个 full-attention 层；
  4 个 KV 头；head_dim 256。`rk8v4` 每层每页 = **104 448 B**。
- 块表：`replicas_` 构造时分配 → 地址恒定；发布 = pinned shadow + `cudaMemcpyAsync` +
  每 (row, replica) 一个 `RowFence`；**捕获中跳过 fence** → 表要在捕获前或轮间发布。
- 图内等外部事件：`structure_round.cpp:131-146` 的 `cudaEventWaitExternal` idiom 现成。
- `copy_from_host` 是纯异步宿主函数，按连续 run 分组，每 plane 一条 `cudaMemcpy2DAsync`。
- MTP 池 = 主池尺寸 + `ceil((draft−1)/64)` 页，`mtp_layers = 1` →
  262 144 `rk8v4` 下 **428 MB**（旧笔记的 107 MB 是错的），且已含在 3.07 GB 预算内。
- PCIe 4.0 x16 实测 **25.77 GB/s H2D / 25.44 D2H**；2D 同速；**并发 kernel 不减速**
  （20.803 vs 20.832 ms）→ 拷贝引擎与计算独立。
- 关 CUDA Graph 实测：6452 prompt tokens、96 new → 116.9（开图）vs **113.0 tok/s**（关图），
  −3.3%，每轮 +0.9 ms；262K 每轮 ~190 ms 时约 0.5%。
- 容量的 affine 校验：`finalize_sequence_plan_impl` 比 `device_reservation_bytes`、
  `kv_capacity`、`extra_rank_reservation_bytes` 三者，流式下 `kv_capacity` 必须是逻辑上下文
  （`page_count(capacity) × 64`），否则报 `resolved KV capacity does not match`。
- 池的租约会**饿死 admission**（`reserve` 拿不到页）；`available_pages()`/`usable_pages()` 是
  admission 的输入。

## 6 我犯过并已写入设计的错（别再退回去）

1. **per-layer plane 下流式数学上不可行**：`16(R+S) ≥ 16P`。必须扁平槽池 + 每层表。
2. **常驻槽写成 `(p − (active−R)) mod R` 是错的**：它会随 `active` 变，每个页边界都要重写整层表。
   正确的是 **`p mod R`**（离窗页与进入页正好同槽）。单测已钉住"一页一生最多换一次槽"。
3. **prefill 不能整段预发布**：会把首屏页提前推出窗口，写入落环、下一 chunk 用空主机记录覆盖。
   必须**逐 chunk** `advance(cursor + remaining)`。
4. **写入路径**改为：新 token 只写设备窗口；页离窗时**一次 D2H**（它此时已写满）。
5. 设计文档 §4 已把这些改正，§4.11 记录了运行时绕过哪一层。

## 7 实测基线（可引用）

| 上下文 | prefill | decode | MTP | 峰值 |
|---|---:|---:|---:|---:|
| 8 192 | 2.37k | 114.3 | 2.76 | 12 672 MiB |
| 32 768 | 2.28k | 115.0 | 2.89 | 13 322 MiB |
| 65 536 | 2.11k | 117.6 | 3.10 | 14 188 MiB |
| 120 000 | 1.86k | 109.1 | 3.10 | 15 632 MiB |

带 MTP3 的运行时容量 **4 032 299 008 B**。262 144 上界（`rk8v4`、25.77 GB/s）：
流式池 2.64 GB = 25 300 槽 → R = 1413/层（f = 0.34）、环 2683 页（280 MB）、
每轮 4.48 GB → **15.3 tok/s**；分段流水 **18.9**；`int8` **10.6 / 12.6**。

## 8 用户硬性约束（违反就会被回滚）

- **不删任何东西**（模型族、架构、KV 格式都保留）；任何删减要逐条审核。
- **绝不碰 `.thinkrail/`**（TODO 存储在那；已被 `.gitignore`，但仍不许删改目录）。
- 测试**自清理**：不留进程、临时文件、显存（"那边是部署的真实电脑，不是垃圾场"）。
- 禁止：sudo、装/升级包、改 `start.sh`、`git clean -fdx`、`reset --hard`、force-push、
  重写历史、杀非本次启动的进程、下载模型、改系统配置。
- **先提交、再改 todo 状态**（宿主会在标记完成时自动提交）。
- 一切结论必须来自**真实命令输出**；失败就停下报告，禁止编造。
- 基准必须用服务端等价 flags；`rk8v4` 是最低质量档，`rk4v4` 及以下不可接受。

## 9 工作流状态

- 路由：**brainstorming**（功能变更），设计已获批准，正在执行第 8 步 Build。
- task-spec：`.thinkrail/context/TASK-kv-streaming.md`（Product/Design/执行顺序）。
- TODO 计划：M0 全部完成；M1 部分完成（容量算式、扁平池几何已完成；池/表/写入/序列/跑通/
  一致性/实测未完成）；M3–M5 未开始。
- 第 9 步（amend specs）**还没做**：`docs/maintainer/paged-kv-cache.md` 的"plane 按层"
  在流式下已变成"按 kind"，等流式跑通后一起改。

## 10 未验证项

- `tests/cmake/CoreTests.cmake` 里新注册的 `ninfer_kv_stream_layout_test` **没被 CMake 执行过**
  （`build-sm89` 是 `BUILD_TESTING=OFF`）。单测本身在 GPU 机上用
  `g++ -std=c++20 -Wall -Wextra -Werror` 独立跑过并通过。
- `row_at` / `publish_at` / `publish_slots_at` / 两个 `*_pages` 拷贝：只编译通过，尚未被运行覆盖。
- MTP 池在 262 144 下的实际占用与 `mtp_base_pages` 的拆分需用 planner 数字复核（误差 >5% 要重算 R）。
