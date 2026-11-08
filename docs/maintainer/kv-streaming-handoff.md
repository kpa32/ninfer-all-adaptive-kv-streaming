# KV 流式：工作交接（临时文档，全部里程碑完成后删除）

> 读这一份就能接着干。最后更新：HEAD `a0633d69`（M1 完成）。
> 设计权威：`docs/maintainer/kv-streaming-design.md`；外部资料：
> `docs/maintainer/adaptive-kv-streaming-inputs.md`。

## 0 一句话状态

**M1 完成**：262 144 原生上下文在 16 GB 卡上跑通 —— 250 964 token 的 prompt，
`rk8v4` + MTP3，decode **14.1 tok/s**（设计上界 15.3 的 92%），峰值显存 15 028 MiB / 16 376。
一致性首个证据：80 404 token 的 prompt 在全驻留与「68 页在环」下**逐字节相同**。
下一步是 M2（把一致性扩到多档）、M3（自适应控制器）、M4（分段流水）、M5（共存）。

## 1 环境与命令

- Linux 构建机：`kpa@192.168.2.13`，仓库 `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`
  （分支 `kv-streaming`），构建树 `build-sm89`（Release，`BUILD_TESTING=OFF`）。
- 模型：`~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`。
- 本地：`D:\GithubRepo\ninfer-all-adaptive-kv-streaming`。

### 构建成本（别再踩）

| 改什么 | 耗时 |
|---|---|
| 一个 `.cpp`，`cmake --build build-sm89 --target ninfer -j 14` | **2.6 s** |
| 被全树包含的头（`paged_kv_cache.h`、`kv_store.h`、`context.h`、`text.h`） | **20+ 分钟** |

**头文件改动必须攒成一次批量提交。** 不要 `git switch`（整树重检出）。

### 本机验证过的命令

```bash
# 全驻留基线（同 prompt、同 dtype、同 flags）
./build-sm89/apps/ninfer <model> --messages /tmp/p430.json --max-context 85000 \
  --kv-capacity auto --kv-dtype rk8v4 --gdn-state-fp16 --prefill-chunk 256 \
  --spec mtp --draft-tokens 3 --greedy --invariant-attention --max-new 48

# 流式（262 144，环 68 页）
... --max-context 262144 --kv-capacity auto --kv-stream ...   # 其余同上
```

`--invariant-attention` 是 **on/off 逐字节对比的必要条件**（见 `qwen3_5-model.md` 的
“Invariant attention”：不加它，注意力按调用的窗口/宽度选不同 split-KV 路线，logits 会有可观测差异）。

## 2 已落地（代码地图）

| 组件 | 文件 | 说明 |
|---|---|---|
| 容量算式与槽映射 | `src/core/kv_stream_layout.h` | `plan_kv_stream_split`、`kv_stream_slot`、`kv_stream_window_slot`（`p mod R`）、`kv_stream_ring_slot`（`16R + p`）。全 `constexpr`，单测 `tests/test_kv_stream_layout.cpp` |
| 扁平池几何 | `src/models/qwen3_5/state/decoder_state.{h,cpp}` | `plan_cache(..., KVStreamSplit*)`：平面按 **kind**（4 个）、表每层一行、`layer_view` 用 `row_at(layer)`、`batch_layer_view` 走 `single_row_paged_kv_batch_view` |
| 表与页搬运 | `src/core/paged_kv_cache.{h,cpp}` | `row_at`、`publish_slots_at`（按页号）、`copy_to_host_pages`、`copy_from_host_pages`、`page_index` |
| 运行时空间 | `program/storage/kv_stream_space.{h,cpp}` | 主机权威 **每层一段**（`host_pages(layer)`）、16 张层表、`advance()`（新页→窗口槽；离窗页先 D2H 再指环槽 `16R+p`）、`upload(layer, copy)`、`ready/consumed/dropped` 三组事件、`reset()` |
| 地址空间接入 | `program/storage/kv_store.h`（`KVAddressSpaceStore`） | `enable_streaming`/`streaming()`；`prepare_activation`/`commit_activation`/`activate`/`deactivate`/`ensure_mapped_to_tokens`/`resize_entitlement`/`release_growth_entitlement`/`commit_frontier`/`destructive_truncate` 全有流式分支；**`page_count` 恒 0**（含义仍是“已物化逻辑页数”） |
| 计划接入 | `planning/startup.cpp`、`planning/request_plan.cpp`、`runtime/engine/kv_capacity.cpp`、`materialization.cpp` | 槽轴 `maximum_pages64 = layers × logical_pages`；容量报逻辑上下文；流式请求 **text 额度 0**、MTP 额度走原公式 |
| 逐层搬运 | `execution/text.cpp` `run_stage_layers` | full 层 attention 前：`wait consumed(prev)` → `upload(layer)` → `wait ready(layer)`；attention 后 `record consumed(layer)`。环只有一份，**必须逐层交错** |
| 空间传递 | `program/context.h`（`ExecutionCore` 两字段）→ `graphs.cpp`、`prefill.cpp`、`program_impl.cpp`、`commit.cpp`、`decode.cpp`（三处内联） | 每个跑文本层的上下文都带 `text_stream_space`/`text_stream_copy` |
| 关图 | `planning/startup.cpp` | `use_cuda_graph = inputs.use_cuda_graph && !kv_stream`（实测代价 3.4%） |
| 选项 | `--kv-stream`，要求 `--kv-capacity auto`、`--max-concurrency 1`、单设备、context cache 关 | `apps/cli/*`、`engine_options.h`、`load_options.h` |

## 3 M1 修掉的六个缺陷（都是真实报错/错解，别再退回）

1. **`page_count` 语义**：`ensure_mapped_to_tokens` 曾把“已到达页数”写进 `address.page_count`，
   而这个字段全仓含义是“已物化的逻辑页数” → 读者遍历到未物化页，报 `logical KV page handle is stale`。
   现在流式下恒为 0，前沿只存在于 `StreamedKVSpace::active_pages()`。
2. **额度与 owner 校验**：流式请求的 text 额度必须是 **0**（它不占任何池页），否则
   `materialized sequence does not match its active entitlement`。MTP 池是普通池，额度保持原公式
   （自己算会差 2 token，报 `KV coverage exceeds active entitlement`）。
3. **prefill 的整段预发布**：起手把 `prompt_tokens` 整段发布，第一个 chunk 立刻回退 → 报
   `A streamed KV frontier only grows`。起始调用必须 `publish_stream_pages=false`。
4. **环是共享的**：曾经一次性把 16 层全上传再逐层算，后面的层覆盖前面的页。必须逐层交错。
5. **主机权威只分配了一层**（最隐蔽）：十六层共用同一批主机记录 → 每层上传的都是最后一层的字节。
   我的自检因此是**同义反复**（拿环和同一批被覆盖的记录比）而"通过"。修法：`page_bytes × P × layers`，
   `host_pages(layer)` 取该层的段。
6. **decode 上下文没带空间**：`decode.cpp` 里三个批上下文（ordinary/MTP/DFlash）内联构造
   `ExecutionCore`，漏了两个字段 → decode 轮不拷不等，读到 prefill 遗留的环。全部补齐。

**排查手法（值得复用）**：拦截 `__cxa_throw` / `std::set_terminate` 打印 `backtrace_symbols_fd`，
再 `addr2line -f -C -e build-sm89/apps/ninfer <offset>` 定位抛点 —— 只需重编 `.cpp`，不必为头文件里的
inline 抛出点等 20 分钟。

## 4 关键实测（2026-11-08，RTX 4080 SUPER 16 376 MiB）

planner 实际切分（`--max-context 262144 --kv-capacity auto`；`auto` 随启动忙存浮动，同 dtype 两次
启动可差几十页）：

| dtype | page_bytes | 槽数 | R/层 | 环 S | 窗口 token | 主机权威 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 262 144 | 12 760 | 577 | 3519 | 36 928 | 16.4 GiB |
| `rk8v4` | 104 448 | 20 002–21 945 | 1060–1189 | 2907–3036 | 67 840–76 096 | 6.4 GiB |
| `int8` | 135 168 | 24 318 | 1348 | 2748 | 86 272 | 8.3 GiB |
| `rk4v4` | 71 680 | 45 858 | 2784 | 1312 | 178 176 | 4.4 GiB |

- **262 144 + `rk8v4` + MTP3，prompt 250 964 token**：prefill 1.31k tok/s（3 min 11 s）、
  decode **14.1 tok/s**、MTP 接受 2.82、峰值 15 028 MiB。每轮 ≈71 ms（其中 attention ≈43 ms，
  比设计估的 28 ms 贵，是 15.3→14.1 的全部来源）。
- **一致性**：80 404 token prompt，`rk8v4`+MTP3，`--invariant-attention`：
  全驻留（85 000）与流式（262 144，68 页在环）**逐字节相同**（48 token，MTP 接受同为 2.61）；
  decode 94.0 vs 86.7 tok/s。
- **全驻留时（S=0）**：8 192 上 `bf16`/`rk8v4` 输出与关流式逐字相同，decode −3.4%（关图代价）。
- 环内字节与表项：曾经在 prefill 与 decode 轮各做过一次直接校验（D2H 环槽 ↔ 主机段，并核
  表项 `entry[p] = 16R+p` / `L·R + p mod R`），修完第 5 项后**未再复测**（诊断代码已删）。

## 5 下一步（M2 → M5）

1. **M2 一致性**：至少两档（建议 80K 与 120K token）× 两种 dtype（`rk8v4`、`int8`），
   全部加 `--invariant-attention`，比较 stdout 逐字节；120K 那档基线用 `rk4v4`（基线装得下）。
   若要更强证据，写一个 host 权威 ↔ 设备全量的逐页比对工具（现在的诊断已删，需要重建）。
2. **M3 控制器**（设计 §4.8）：三指标（deadline miss、copy busy、ring peak）+ hysteresis + 刹车；
   同时把 §7 的指标接进 summary（`resident`/`ring`/`uploaded_bytes`/`wait_ms`）。
3. **M4 分段流水**（§4.10）：把 14.1 抬到 ~19（拷贝与计算重叠）。需要按页区间出偏结果 + 合并。
4. **M5 共存**：多序列、Hybrid prefix cache、TP 各自「支持或明确拒绝」。
5. **收尾**：`docs/maintainer/paged-kv-cache.md` 的「plane 按层」要改成「按 kind（流式下）」；
   `tests/cmake/CoreTests.cmake` 里 `ninfer_kv_stream_layout_test` 仍**未**在 CMake 下跑过。

## 6 用户硬性约束（违反会被回滚）

- **不删任何东西**（模型族、架构、KV 格式都保留）；任何删减要逐条审核。
- **绝不碰 `.thinkrail/`**（TODO 存储；已 gitignore，但目录也不许删改）。
- 测试**自清理**：不留进程、临时文件、显存。
- 禁止：sudo、装/升级包、改 `start.sh`、`git clean -fdx`、`reset --hard`、force-push、
  重写历史、杀非本次启动的进程、下载模型。
- **先提交、再改 todo 状态**；结论必须来自真实命令输出，失败即停下报告。
- KV 质量下限 `rk8v4`；`rk4v4` 只可用作**测试基线**，不可作为交付默认。

## 7 已知限制与代价

- 主机权威副本 6.4 GiB（`rk8v4`）/16.4 GiB（`bf16`）是 pinned 内存；`bf16` 流式因此不适合作为
  产品默认（卡能跑，但吃 16 GB 主机内存）。
- 关 CUDA Graph（M1 选择）：8K 全驻留实测 −3.4%；262K 时被 71 ms/轮稀释。
- `--kv-capacity auto` 是硬要求（池大小要按可用显存算），因此 R 在不同启动间会有几十页的浮动。
- 长上下文的逐 token 对比没有可负担的全驻留基线（这正是功能存在的理由），M2 的等价性只能在
  能启动的最大上下文上做。
