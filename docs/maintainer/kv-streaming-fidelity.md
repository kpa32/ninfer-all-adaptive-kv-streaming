---
id: kv-streaming-fidelity
type: architecture-design
title: KV 流式保真核对：作者理论 / 上游实现 / 我们的现状
status: active
parent: kv-streaming-design
depends-on:
  - adaptive-kv-streaming-inputs
---

# KV 流式保真核对

**这份文件的用途**：把 [参考文章原文](adaptive-kv-streaming-inputs.md)（附录 A，唯一设计权威）和作者
自己那份实现（`D:\GithubRepo\llama.cpp-adaptive-kv-streaming`）逐条摊开，对照 NInfer 的现状，给出
差距、结论和施工顺序。任何实现决定都要能追到这里的某一行；追不到的就是我们自己加的，必须单独标注。

核对方法：先读原文逐字抽 R1–R13（`adaptive-kv-streaming-inputs.md` 附录 A.1），再读上游实现
（`src/llama-kv-stream-*.{h,cpp}`、`ggml/src/ggml-cuda/fattn.cu`、`ggml/include/ggml-cuda.h`、
`src/llama-kv-cache.cpp`、`src/llama-context.cpp`、`README.md`、`benchmarks/README.md`），
最后读我们自己的实现。

---

## 1. 上游的实现结构（事实，带出处）

一个**固定大小的 CUDA 池**，按页切成两类用途，边界可以在运行时移动：

```
pool_pages = layers * resident_pages_per_layer + ring_slots          （恒等式，见下）
```

- 池里前半是**驻留页**：每层 `resident_pages_per_layer` 页，连续摆放
  （`fattn.cu:437-511` `kv_stream_resident_cache_layout`）。
- 池里后半是**环形缓冲**（代码里叫 scratch / stage）：`ring_slots` 个页槽，所有流式层共享
  （`ggml-cuda.h:40-49` 的 `stage_bytes`/`stage_slots`）。
- 权威副本在**钉住的主机内存**里，token-major 紧凑排列，所以连续 N 页是一次连续 H2D
  （`fattn.cu:1285-1300` 的注释与实现；`README.md` 第一段）。
- 一个**平铺的槽池 + 动态分配**：每页的请求（layer, page）在每轮开头登记，环里哪些槽给它由调度器
  决定（`fattn.cu:1590-1685` `graph_add_attention`、`1464-1496` `schedule_batch`），槽位复用以
  `consumed` 事件为界（`fattn.cu:1546-1560`）。
- **拷贝批处理**：同一层、主机地址连续的请求合成一批，最多 32 页，一次 memcpy K + 一次 memcpy V
  （`fattn.cu:1359-1440`，常量 `KV_STREAM_COPY_BATCH_PAGES = 32`）。
- **消费侧 span 合并**：把"槽位连续、token 连续"的就绪页合成一个更宽的 K/V span 交给同一个内核，
  所以 decode 每层的发射次数是 `ceil(流式页 / 32)`，而不是每页一次；`bounded span`（32 页）与
  "整段一次发射"由**实测调优器**二选一（`fattn.cu:1983-2040`、`kv-stream-span-tuner.h` 全文）。
- **反馈计量**：只采样**一个批次**的拷贝时间，按本轮上载数外推成占空比
  （`fattn.cu:1258-1283`），再加一个设备内核在"计算真正需要这一页"的时刻读 ready 标志，统计
  deadline 命中/失约（`fattn.cu:1246-1256`、`1993-2005`）。
- **自适应重分区**：`llama_kv_stream_partition_adapt`（`llama-kv-stream-plan.cpp:355-490`）。
  目标划分 = 找最大的 `resident` 使 `ring >= target_ring_working_set_ratio * streamed`，
  **默认比例 1.10**（`llama-kv-stream-plan.h:171`）。扩环一次退一页/层、收环一次还一页/层；迟滞
  3/8 次评估、冷却 64 次；判据是 `deadline_miss_ratio`、`copy_engine_busy_ratio`、
  `ring_peak_occupancy_ratio`，**饱和刹车 `COPY_SATURATED = 0.80`**（同文件 403-441 行）。
- **decode 布局**：prompt 阶段用均匀布局（可增长、不需重分区）；decode 类微批把同一份页预算
  **集中到少数被 split 的层**，其余层整层驻留，集中度受环工作集约束；目标为 0 时回到 prefill 布局
  （`fattn.cu:477-505`、`llama-kv-cache.cpp:1382-1390`）。
- **在图上运行**：每轮 graph 开始时登记请求、末尾填槽，层开始时放行本层可变尾页并回填空槽
  （`fattn.cu:1564-1720`）；全部在 CUDA Graph 内，跨流等待是图内 event 节点。

### 1.1 一句话概括上游的三段论

1. 池固定 → 2. 驻留/环的边界**只在运行时按反馈移动** → 3. 消费侧用 span 合并把"环只装 1.1 倍
   一层流式量"变成可行，从而把窗口做到最大。

---

## 2. 我们的实现结构（事实，带出处）

- 同样是固定池：`layers * resident_pages + ring_pages`（`src/core/kv_stream_layout.h:87-90`），
  `plan_kv_stream_split` 在**启动时**按**最大容量**算好，之后不变
  （`src/models/qwen3_5/program/planning/startup.cpp:146-152`）。
- 环被**静态切成 `ring_depth` 个区域**，`region = layer mod ring_depth`，槽位 =
  `ring_base + region*extent + offset`（`kv_stream_layout.h:56-80`、`kv_stream_ring_slot`）。
- 权威副本在钉住主机内存，逐层一段（`StreamedKVSpace::host_pages`），拷贝走
  `copy_from_host_pages` → 每平面一次 strided 传输（`src/core/paged_kv_cache.cpp:1004`）。
- 消费侧**只通过块表间接寻址**：内核按 split 读 `block_table[first_page + page]`
  （`src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh:251-260`）。
- 每轮开头决定是否分帧（`plan_stream_round`，`text.cpp:1518`），分帧时逐帧
  `wait → dispatch(区间) → consumed`（`text.cpp:1693-1720`），**默认整层拷贝**
  （`NINFER_STREAM_FRAMES` 未设置时 `layer_copy = true`，`text.cpp:2235`）。
- 反馈：只有每层事件对的 copy/exposed 计时（`kv_stream_space.h:236-250`），没有 deadline 采样、
  没有占空比的可靠估计（分帧时 `copy_busy` 报 409 ms/轮，超过轮时间，结论作废）。
- 没有重分区控制器、没有 decode 布局、没有 span 调优器。

---

## 3. 逐概念对照

| 概念 | 上游 | 我们 | 判定 |
|---|---|---|---|
| 主机为权威 | 是，token-major 紧凑 | 是，逐层一段 | ✅ 同 |
| 块表间接寻址 + 图内换页 | 是 | 是（图内捕获） | ✅ 同 |
| 跨流事件（拷贝/消费） | 每页/每批 | 每层/每帧 | ⚠️ 粒度不同 |
| 预取领先于消费者 | 请求表 + 层开始回填 | `depth-1` 层提前 + 轮末预取下一轮 | ⚠️ 功能同、调度不同 |
| 精确注意力 | 全上下文精确 | 全上下文精确 | ✅ 同 |
| 环的形状 | **平铺槽池 + 动态分配** | **静态 `depth` 区域 + `layer mod depth`** | ❌ 结构差 |
| 环的大小 | **`1.10 x` 一层流式量**，目标 | **`2 x` 一层流式量**，写死 | ❌ 多占窗口 |
| 驻留/环边界 | **运行时移动**（迟滞 + 冷却 + 刹车） | **启动时定死，按最大容量算** | ❌ 缺控制器 |
| 短上下文的代价 | ring 从最小开始，够驻留就不流 | 按最大容量切分 ⇒ **短上下文也预留大环** | ❌ 真实浪费 |
| 层间分布 | prefill 均匀 / decode 集中到少数层 | 全部层一律同窗口 | ❌ 缺 decode 布局 |
| 拷贝命令数 | 每 32 页 2 条 | 每层每平面 1 条 | ✅ 同量级 |
| 消费侧发射 | span 合并（≤32 页/次）+ 调优器二选一 | 整层一次，或逐帧（更粗） | ❌ 缺 span 合并与调优 |
| 反馈信号 | deadline 命中率 + 占空比外推 + 环峰值 | 只有事件对计时（busy 失真） | ❌ 缺两个信号 |
| 自检 | `--kv-stream-stage-mib` 池探针 + 回退 | headroom 自动容量 | ⚠️ 手段不同 |

结论：**骨架（主机权威、块表换页、图内跨流事件、精确注意力）我们和上游一致；缺的是三块"动态性"
——环的大小由反馈决定、层的分布按阶段变、消费粒度按实测选。** 这三块恰好是原文 R7 的全部内容。

---

## 4. R1–R13 对照

| # | 原文要求 | 现状 | 差距 |
|---|---|---|---|
| R1 | 只流式全注意力层 | ✅ `full_attention_layers` 作为 `layers` | 无 |
| R2 | 一层算完其 KV 即闲置，可移除 | ✅ `consumed` 事件后区域可复用 | 无 |
| R3 | 必须提前预取，不能用时才拷 | ✅ `depth-1` 层提前 + 轮末预取 | 无 |
| R4 | 环的槽位在各层之间循环复用 | ⚠️ 复用，但**按层固定绑区域**，不是"任何页可进任何槽" | 结构差（见 §5） |
| R5 | `tok/s ≤ 带宽 ÷ 每轮流式字节` | ✅ 已实测吻合：`rk8v4` 3 603 MiB→142 ms、151 ms/轮 | 无（这是物理，不是缺陷） |
| R6 | 按上下文决定流多少、留多少 | ⚠️ 有窗口/尾部，但**比例在启动时定死** | 缺动态 |
| R7.1 | 能全驻留则 ring 禁用 | ❌ 池按最大容量切分，短上下文仍预留 `2×(P−R)` 环 | **真实浪费** |
| R7.2 | 需要流式时从小规模开始，公平分到各层 | ❌ 一步到位按最大容量 | 缺 |
| R7.3 | 只在必要时扩环 | ❌ 无扩环（无控制器） | 缺 |
| R7.4 | 上下文越长每 token 流得越多，最终 PCIe 成瓶颈 | ✅ 已实测（`rk8v4` 94% 占空比） | 无 |
| R7.5 | 恶性循环链条 | ⚠️ 不会发生（无扩环），但也无刹车可依赖 | 缺 |
| R7.6 | **PCIe 饱和就停止扩环** | ❌ 无（没有占空比信号，也没有扩环动作） | 缺 |
| R8 | 目标：暴露趋 0，轮时间不因调度变长 | ⚠️ 暴露 15.6 ms/56.7（rk4v4）；帧路径暴露降但轮变长 → 违反后半句 | 缺正确判据 |
| R9 | 自己控制驱逐，KV 不降精度 | ✅ 显式驱逐、`rk8v4` 为质量下限 | 无 |
| R10 | 测量口径（N+256、256 token 平均、同环境对照） | ✅ `tools/bench/run_kv_stream.py` | 无 |
| R11 | 能全驻留区间不得劣化 | ❌ **未验证**：`equiv-183k` 只是等价，没有"同上下文全驻留 vs 流式"的速度对照 | 缺实验 |
| R12 | 数量级预期按带宽/字节换算 | ⚠️ 已换算，但此前用作者数字外推过（已作废） | 口径已修正 |
| R13 | 中端卡的 agent 负载；15 tok/s 为下限 | ✅ `rk8v4` 251K 26.1 tok/s 高于该下限 | 无 |

---

## 5. 三个结构性差距（必须做，不能绕过）

### 5.1 池的切分是启动时定死的（违反 R7.1–R7.3）

`plan_kv_stream_split` 用**最大容量**（`logical_pages = capacity/64`）算 `R`，于是：

- 一个 130K 的请求、配置成 262 144 容量，仍然按 4 096 页/层切池 ⇒ 环占 `2×(4096−1670)=4852` 槽，
  而它实际只需要 `2×(2032−1670)=724` 槽 ⇒ **4 128 个槽闲置在环里**（≈ 13% 的池），这些槽本可以是
  驻留页。直接后果：130K 上下文每轮还要搬 362 页/层（≈605 MiB），而按 R7.1 该是零。
- 反过来，上下文接近容量时 `R` 又无法再涨（环永远是 2 倍），窗口被结构性地压小。

这就是原文第 1、2、3 条要解决的事，也是我们最大的一处不符。

### 5.2 环是"按层划分区域"，不是"平铺槽池"

我们的槽位是 `f(层, 页)` 的纯函数（这是块表"一页只改两次"的前提），代价是：

- 环只能是 `depth × 一层流式量`，因为"一个区域 = 一层"是硬绑定；
- 无法表达"环 = 1.1 倍一层流式量"，因为区域必须整数倍；
- 无法让不同层的页共享同一个槽（上游可以：任何空闲槽给任何页）。

上游用**动态槽分配**换来了 1.1 倍环；它付出的代价是块表项要重写（"repartition invalidates compact
resident addresses"，`llama-kv-stream-plan.cpp:405-407`），并用冷却时间摊掉。

### 5.3 消费侧粒度只有"整层"和"帧"两档，且没有调优器

上游把"就绪且槽位连续"的页合并成更宽的 span（≤32 页）交给同一个内核，再用**实测**在
"bounded span"和"整段一次"之间二选一（≥0.5% 才切）。我们只有整层一次（默认）和逐帧（更粗、
实测更慢），**没有中间档，也没有实测选择**。

**这一条解释了帧式流水在本机的失败**：上游的 bounded span 是"为了把环压到 1.1 倍而付的代价"
（换来更少字节），我们只付了代价（更多发射、波次不整齐），没拿到收益（环没变）。
——这也是之前"层内分帧更慢"结论的真正解释：**不是分帧错，是分帧没有配上缩环。**

---

## 6. 预期收益（诚实上界，按 R5 换算）

以 `rk8v4`、251K、池 31 572 槽、`P = 4 096` 页/层为算例：

| 方案 | 环 | 窗口 | 每轮尾部页 | 每轮字节 | 相对现在 |
|---|---|---:|---:|---:|---:|
| 现在（`depth=2`） | 4 852 | 1 670 | 2 252/层 | 3 603 MiB | 1.00 |
| 保真（`1.10×` + 动态） | 2 760 | 1 816 | 2 106/层 | 3 370 MiB | **−6.5%** |

⇒ 251K 大约是 **26.1 → 27.8 tok/s（+6~7%）**，不是数量级变化：R5/R7.4 说这一档已经是 PCIe 受限。
保真实现真正的价值在两处，都不在 251K 这个点上：

1. **中短上下文**（130K~180K）：按 R7.1 该"能不流就不流"，我们现在却按最大容量预留环 ⇒ 每轮白搬
   几百 MiB。140K 附近可能是**成倍**的差别（待测）。
2. **全曲线自洽**：8K→251K 一条曲线不再需要人工选 `depth`；饱和刹车保证不会滑进 R7.5 的恶性循环。

---

## 7. 施工顺序（每期独立验收，不合并批准）

**P0 计量地基**（没有它，控制器就是猜）
- 加 deadline 采样（设备内核读 ready 标志，统计命中/失约）与占空比估计（采样一个拷贝批、按本轮
  上载数外推），落进运行报告；修掉现在分帧时 `copy_busy` 失真的问题。
- 验收：251K 下报告出的占空比与 `每轮字节 ÷ 25.4 GB/s ÷ 每轮时间` 相差 <5%。

**P1 动态重分区（R7.1–R7.6）**：同一布局，`R`/`depth` 可在轮间改变
- 启动按**当前容量**、从最小环开始；能全驻留就完全不流式（R7.1）。
- 控制器照抄上游规则：miss>1% 且占空比<0.80 连续 3 次 ⇒ 扩环（R−1）；miss≤1% 且占空比<0.50 且
  环峰值<0.50 连续 8 次 ⇒ 收环（R+1）；冷却 64 轮；**占空比 ≥0.80 只收不放**。
- 重分区时重建块表并搬动受影响页；上限：每 64 轮最多一次。
- 验收：130K 上下文每轮字节 ≤ 现在的 1/4；251K 不劣化；`check --suite equiv` 4/4 逐位相同。

**P2 decode 布局（R7 + 上游 `kv_stream_resident_cache_layout`）**：decode 时把页预算集中到少数
被 split 的层，其余层整层驻留；prompt 回到均匀布局。
- 验收：同一上下文下每轮事件/等待数下降，轮时间不劣化。

**P3 环降到 1.1 倍 + 消费侧 span 合并 + 调优器（R4 + 上游 span tuner）**
- 环改为"平铺槽池 + 动态分配"，块表项在页进/出环时改写（一页仍只改两次）。
- 消费侧按"槽位连续"合并 span，默认 32 页一档；调优器用端到端轮时间在
  bounded / unbounded 之间二选一（16 样本、≥0.5%）。
- 验收：251K `rk8v4` 每轮字节 −6.5% 且 decode 提升 ≥5%；调优器在 `rk4v4` 上独立复现同一选择。

---

## 8. 与原文不同、必须单独标注的地方

| 项 | 为什么不同 | 处理 |
|---|---|---|
| **MTP 验证轮**（每轮 4 token） | 原文是每轮 1 token 的普通 decode；我们每轮验证 3–4 个草稿 token，计算/拷贝比更高 | 所有"每轮"数字按 `token = 轮 × 接受长度` 折算；控制器按**轮**评估，与原文按 token 等价 |
| **`rk8v4` 页 = 64 token** | 原文 / 上游是 256 token 页（Q8_0/Q4_0） | 页更小 ⇒ 同样字节的命令数×4；span 合并按页计（32 页 = 2 048 token） |
| **页内含分组量化尺度** | `rk8v4`/`rk4v4` 的 K 是 rotated int8 + int4 V，每页 3 个平面 | 拷贝批处理的"主机连续"判断要按平面分别成立 |
| **多 GPU 流水线** | 上游单卡 | 流式只在本机 stage 内，跨 stage 与本特性无关 |
| **无 UVM 依赖** | 原文作者日常开 UVM | 我们不使用 UVM；R9 的"自己控制驱逐"我们本来就满足 |

---

## 9. 尚未核实 / 风险

- 上游 `1.10` 这个比例是在 Q8_0/Q4_0、256 token 页、50 GB/s 上得到的，我们的 64 token 页和
  25.4 GB/s 未必同值；**P3 必须实测扫 1.05/1.10/1.25/1.50 四档**，不能照抄数字（R12 的口径）。
- 重分区会重写块表并搬页，代价未测；上游用 64 轮冷却摊掉，我们需要实测一次重分区的毫秒数。
- 130K 的预期收益是按字节算的，未实测；P1 验收必须先测"现在 130K 每轮字节"作为基线。
- 我们的 decode 是 CUDA Graph（拓扑 + frontier 分档），重分区/布局切换会改变图参数；上游用
  `reconfigure` 显式重建，我们需要确认图缓存键要把"布局代数"纳入，否则会用到旧布局的图。

---

## 10. 移植图纸：上游提交序列（一条已经走过的路）

作者把整套机制拆成了 50 多个小提交（2026-08-20 → 10-19），每个 ≤ 1000 行、多数带单元测试。
**我们不再自己设计形状，按这条序列逐条移植**，只做三处必要的本地化：页 = 64 token 且 K/V/尺度三平面、
decode 是拓扑+frontier 分档的 CUDA Graph、每轮验 3–4 个草稿 token。

核心序列（08-20 → 08-29）与落点：

| 上游提交 | 它建立什么 | 我们的落点 | 现状 |
|---|---|---|---|
| `2646f0aa2` | 块级驻留规划器 | `plan_kv_stream_split` | ✅ 已有（静态） |
| `411af01db` | extent 预留 + 收缩迟滞 | 容量在启动时定死 | ⬜ 不需要 |
| `693040765` | 块级区域构造 | 槽位映射 | ✅ 已有 |
| `f04350785` | 钉住主机运行时（权威在主机） | `HostKVArena` + `StreamedKVSpace` | ✅ 已有 |
| `50ae50c0f` | 受限 stage 传输（槽=一页，K/V 分平面） | `copy_from_host_pages` | ✅ 已有 |
| `a4378d218` | 多块合并为一次精确注意力 | split + 层末单次 reduce | ✅ 已有 |
| `f0957c082` | 稳定驻留页（块表少改） | `kv_stream_window_slot` | ✅ 已有 |
| `39704675e` | **deadline 驱动预取调度器**（请求表 + 状态 + 空闲槽分配） | 新增 `src/core/kv_stream_prefetch.h` | ❌ **缺** |
| `1ba8f4406` | 异步流水（先发拷贝、消费等事件） | `depth-1` 提前 + 轮末预取 | ✅ 已有 |
| `b29fe5859` | **分区控制器初版**（miss/busy/peak + 迟滞 3/8） | 新增 `src/core/kv_stream_partition.h` | ❌ **缺** |
| `c51644ec1` | 驻留边界移进固定池（运行时改 R） | `StreamedKVSpace` 重分区 | ❌ **缺** |
| `82a9a538c` | 跨层预取（层边界回填槽） | 轮内请求表驱动 | ❌ **缺** |
| `93df8ac33` | 环按反馈自适应（CUDA 侧 feedback 接线） | `feedback_fn` 等价物 | ❌ **缺** |
| `158067c0a` | 批量上传 + **约束反馈**（busy 外推、feedback floor） | 运行报告 + 控制器输入 | ❌ **缺** |
| `92d72914f` | 重分区限速（冷却 64） | 控制器参数 | ❌ **缺** |
| `4f2cb3e91` | 合并相邻流式页（槽连续 ⇒ 宽 span） | `small_t_i8` 的 span 合并 | ❌ **缺** |
| `540cbe990` | 合并相邻页上传 | 已是每层每平面一次 | ✅ 等价 |
| `1eb2faaa0` | 精确脏页跟踪 | 整页搬；decode 只追加尾页 | ⬜ 复核 |
| `725bac3c5` | token-major staging 局部性 | 每层一段连续 | ✅ 已有 |
| `65d814435` | bounded span 与拷贝重叠 | span ≤32 页 | ❌ **缺** |
| `ab539d57d` | **环按层工作集定尺寸**（`1.10×`） | 布局规划器 | ❌ **缺** |
| `d9dcd518c` | decode 布局受环约束（集中到少数层） | 每层驻留页分布 | ❌ **缺** |
| `b1d73aebe` | 布局切换迁移驻留页 | 重分区搬页 | ❌ **缺** |
| `4a84b5ccc`/`852094f51` | 布局上限 / 多波小环边界 | 边界条件 | ❌ **缺** |
| `50a7de723` | 类型泛化 + **span 调优器** | `kv_stream_span_tuner` 等价物 | ❌ **缺** |

后续提交（08-31 起：phase arena、graph splits、多卡、opt-in 接线、图像 prefill、MTP/DFlash/Gemma
handoff）**不移植**：它们对应上游特有的生命周期，我们的对应物已经存在或没有该特性。

### 10.1 移植顺序（= 上游顺序，砍掉我们已有的）

1. **S1** `39704675e` + `158067c0a` 的反馈部分：deadline 采样与占空比外推，先做成纯函数 + 单测
   （上游就是 `llama-kv-stream-plan.cpp` + `tests/test-kv-stream-plan.cpp` 的写法）。
2. **S2** `b29fe5859` + `92d72914f`：分区控制器（纯函数 + 单测）。
3. **S3** `c51644ec1` + `82a9a538c` + `93df8ac33` + `158067c0a` 的接线部分：控制器接进
   `StreamedKVSpace` 与每轮循环，块表重发布，跨层预取回填。
4. **S4** `d9dcd518c` + `b1d73aebe` + `4a84b5ccc` + `852094f51`：decode 布局与迁移。
5. **S5** `ab539d57d` + `4f2cb3e91` + `65d814435` + `50a7de723`：环 1.10× + span 合并 + 调优器。

每一阶段的上卡验收口径都写在第 7 节；数值参数（1.10、0.80/0.50、3/8、64）先按上游原值落地，
再用实测的扫档确认（第 9 节的第 1 条风险）。

---

## 11. 全表面复核（2026-11-10）：这张表的框错了，一条结论也错了

### 11.1 错在哪

**（1）一条结论是假的。** §3 那张对照表里写着「块表间接寻址 + 图内换页 | 是 | 是（图内捕获）| ✅ 同」。
事实是 `src/models/qwen3_5/program/planning/startup.cpp:1510`：

```cpp
impl->use_cuda_graph = inputs.use_cuda_graph && !inputs.features.kv_stream;
```

**流式一开，CUDA Graph 全部关掉。** 上游把整轮换页放在图内（`fattn.cu:1564-1720`，跨流等待是图内
event 节点），我们则是"关掉图、主机逐层发拷贝"。这一条被写成"✅ 同"，是本轮最大的失实。
**归属要说清楚：这一行不是本分支引入的。** `git show e3dd5bf3:src/models/qwen3_5/program/planning/startup.cpp`
（本分支的起点）第 1509 行就是同一个写法；旧 `kv-streaming-handoff.md` 的第 4 条还记着
「流式下恢复 CUDA Graph：−3%（≤130 K）」——即当时实测**开图更慢**，但那条记录没有 argv、没有逐轮
样本，本分支从未复测。所以它不是"漏做"，是**未复测的旧结论**被我在 §3 写成了"✅ 同"（见 [strategy](kv-streaming-strategy.md) 的 C7 条目）。
prefill 侧同理：`begin_round` 的 `!prefill` 与 prefill 被排除在分帧之外，也都在 `e3dd5bf3` 上已存在。


**（2）对照表的组织轴是"decode 一轮怎么跑"。** 于是四类机制从头到尾没有成为表里的一行：

1. 归一化布局：prompt 用 uniform、decode 用 concentrated、下一个 prompt chunk 回退
   （`llama-kv-cache.cpp:1400-1412` 的 `decode_layout_pages = 0` 注释写着 "A zero target restores prefill"）；
2. 每 ubatch 适配：`llama-context.cpp:1381-1387` 在 `process_ubatch` 里调 `kv_stream_adapt`，**不分阶段**；
3. 写路径脏行与镜像：`llama-kv-cache.cpp:1758-1764` 的 `mark_dirty_rows_fn`，每个 ubatch 都报；
4. 图内调度：`kv_stream_graph_schedule_batch` 认领槽位、跨层预取、`ready_flags_device` 设备侧放行。

这四类不在表里 ⇒ 移植时自然也没被当成工作项 ⇒ "遗漏"不可能被自己发现。

**（3）我把已记录的工作项删掉了。** `kv-streaming-handoff.md` 在 `1d3eae83` 被"归档"删除，而它
第 4、5 条与第 2 节里记着两件本分支**从未替代**的东西：**流式下恢复 CUDA Graph（旧实测 −3%，≤130 K）**
与**跨轮预取＝`depth` 回绕的那一半**。我重写的待办清单两项都没收，等于把前任记下的工作项从我方
文档里抹掉（现已作为 C7/C8 补回，带出处）。这是本次最该被追责的动作：不是没做到，是**把待办删了**。


### 11.2 上游机制 × 我方现状（逐条，带位置）

| # | 机制 | 上游位置 | 我方 | 判定 |
|---|---|---|---|---|
| 1 | 主机权威副本 + 块表换页 | `llama-kv-cache.cpp` | `kv_stream_space.cpp`（host arena + 每层表） | ✅ 有 |
| 2 | 平铺槽池 + run 宽度 | `fattn.cu` transfer ring | `kv_stream_layout.h` 环/run | ✅ 有 |
| 3 | 设备侧 ready 标志 + deadline 采样 | `fattn.cu:1285-1345`（`ready_flags_device`） | `core/kv_stream_signal.cu` | ✅ 有 |
| 4 | 分区控制器（迟滞 3/8、冷却 64、1.10） | `llama-kv-stream-plan.cpp:355` | `core/kv_stream_partition.*` | ✅ 有 |
| 5 | 跨度调优器（16 样本 / 0.5%） | `ggml-cuda/kv-stream-span-tuner.h` | `core/kv_stream_span_tuner.h` | ✅ 有 |
| 6 | **prompt/decode 双布局 + 回退** | `llama-kv-cache.cpp:1400-1412` | **无**：`kv_stream_space.cpp:407` 把 `entering_decode_layout` 硬编码成 `false`；也没有"回到 uniform"的路径 | ❌ 缺 |
| 7 | **每 ubatch 适配（含 prompt）** | `llama-context.cpp:1381-1387` | **无**：`begin_round` 只在 decode 调（`text.cpp:2168` 带 `!prefill`） | ❌ 缺 |
| 8 | **写路径脏行 / 镜像标记** | `llama-kv-cache.cpp:1758` | **无**：`advance()` 无条件回写跌出窗口的页（`kv_stream_space.cpp:790`） | ❌ 缺 |
| 9 | **图内换页 + 图内跨流事件** | `fattn.cu:1564-1720` | **无**：`startup.cpp:1510` 关图 | ❌ 缺（最贵） |
| 10 | **槽位认领 + 跨层/跨轮预取调度** | `kv_stream_graph_schedule_batch`（工作守恒、可跨轮） | **无**：跨轮预取被我删掉（`0c6927c7`），只发 `current + ahead` | ❌ 缺 |
| 11 | prompt 时跨度=整层、decode 才用调优跨度 | `fattn.cu:1761`（`Q->ne[1] == 1 ? … : UINT32_MAX`） | 行为近似（prefill 被排除在分帧之外），但**没有"分档"这个机制** | ⚠️ 近似 |
| 12 | MTP/DFlash KV 作为独立驻留角色 | `llama-kv-stream-plan.h:11-13` `role::mtp` | 无 | ⏸ 上游也没接线 |
| 13 | region 规划器（驻留优先级、`pin_tail`） | `llama-kv-stream-plan.h:13-95` | 无（我们用几何公式） | ⏸ 上游也没接线 |
| 14 | 主机侧与图并行的拷贝编排 | `kv_stream_graph_upload` | `upload()` 每层顺次发 run | ⚠️ 部分 |

第 12、13 条的判定依据：`llama_kv_stream_plan_make` / `regions_make` / `extent_make` /
`prefetch_dispatch` 在上游**除单测外没有任何调用者**（`grep` 于 `src/` 与 `ggml/` 为 0），
运行时真正用的是 `resident_cache_layout`（`fattn.cu`）+ `partition_adapt`/`feedback_delta`
（`llama-kv-cache.cpp`）。所以它们不构成"我们欠的移植"，但**它们的机制思想（producer 依赖、
工作守恒的槽位填充）在 CUDA 侧是活的**，见第 10 条。

### 11.3 后果（可量化）

| 症状 | 直接原因 | 已有实测 |
|---|---|---|
| 251K decode 每轮 54.4 ms（rk4v4）、147.9 ms（rk8v4） | 无图（第 9 条）+ 无跨轮预取（第 10 条） | [measurements](kv-streaming-measurements.md) 表 1 |
| 251K prefill 1281 tok/s、搬 ~94 GiB | 不换布局（第 6 条）+ 不适配（第 7 条）+ 每 chunk 重传尾部 | [measurements](kv-streaming-measurements.md) 表 2 |
| prefill 形状对 prefill 不利 | decode 的集中布局被用于 prompt | 第 6 条 |
| 控制器 `entering_decode_layout` 分支是死代码 | `kv_stream_space.cpp:407` 硬编码 false | 第 6 条 |
| 集中布局可能从未真正被选中 | 同上 + 无测量记录形状 | 第 6 条 |
| MTP KV 仍整层驻留（占容量） | 第 12 条（上游也未接线） | — |

### 11.4 计划：把"病猫"做成"老虎"（T1–T7，各自独立验收）

按"收益 ÷ 风险"排序。每期都要求：改前先有测量、改后逐位不破、报告里能看见它的数字。

| 期 | 内容 | 验收判据 |
|---|---|---|
| **T0** | **先量后改**：在 183K/251K 上量 `--no-cuda-graph` ↔ graph 的差（旧记录说开图 −3%，未复测；没有这条数字，T1 就是拍脑袋） | 两个容量、两个 KV 格式各一组逐轮样本；结论写进 measurements |
| **T1** | **CUDA Graph 与流式的关系**：按 T0 的实测决定是"补图内换页"还是"维持关图并写清理由"；若补，则块表/control 更新做成图内合法形状，布局代数进图缓存键 | 251K rk4v4 不慢于 72.8 tok/s；graph vs eager 的 logprobs 512/512 逐位 |
| **T2** | **双布局**：prompt 用 uniform、decode 用 concentrated、下一个 prompt chunk 回退 uniform | 251K prefill 字节与 tok/s 优于 1281/94 GiB；decode 不退；日志与报告能读出当前布局 |
| **T3** | **每 chunk 适配 + 脏页回写** | prefill 期出现 `[partition]` 决策；prefill 回写字节 ≤ 实际写过页数 |
| **T4** | **跨轮预取 + 槽位认领**（照上游 `producer_attention_index` 的依赖思想，而不是删掉预取） | `--kv-stream-ring-ratio 2.0` 下 251K `check --case equiv-183k` PASS，且每轮不慢于 1.10 |
| **T5** | **跨度分档**：prompt 整层 / decode 阶梯，一套代码两个档 | prefill 与 decode 各自用对档；`[stream] ladder` 只在 decode 出现 |
| **T6** | **草稿 KV（MTP）流式**（本机 rk8v4 251K 的容量/字节收益） | 同一容量下池更大或每轮字节更少，且逐位不破 |
| **T7** | **全矩阵验证**：262 144 的 prefill+decode × {rk4v4, rk8v4, int8} × resident(≤110K)/streamed；作者口径曲线 | 每格有 ids 与 logprobs 判据；曲线上能看见流式生效点 |

### 11.5 流程错误（必须记下来，防止再犯）

1. 我把"只做 decode 的按层流式"写进 SPEC 的"不在本次范围"，**没有把它作为一次交付范围缩减提交批准**，
   后来又把 prefill 归到"所有者另行安排"，等于把缩掉的东西改写成别人的任务。
2. 对照表以 decode 流水为轴，**没有按"上游机制的完整清单"来写**，所以 11.2 里第 6–11 条从一开始就
   不在比较对象里。
3. 第 92 行那条"✅ 同"是在没有核对 `startup.cpp:1510` 的情况下写的**推断**，却被写成事实。

### 11.6 本分支到底做了什么、没做什么（git 证据，防止误判为"推倒重来"）

`kv-stream-faithful` 是 `kv-streaming`（`e3dd5bf3`）之上的 **87 个提交**，`e3dd5bf3` 是它的祖先，
`git log HEAD..origin/kv-streaming` 为空——**没有丢掉父分支的任何提交**。

**加进来的**（`git diff --stat e3dd5bf3..HEAD`：20 文件 +2174/−715）：
`kv_stream_feedback` / `kv_stream_partition` / `kv_stream_signal` / `kv_stream_slot_map` /
`kv_stream_span_tuner` 五个新模块（上游机制），`kv_stream_space` +843/−…的重构（平铺环 + run 级
拷贝/消费 + 逐层窗口），`text.cpp` 的发射与消费重写，以及四个真缺陷的修复（层号二次映射、块表发布
未等待、`ahead` 无上限越界、span 按帧序而非页区间寻址）。

**减出去的**：`NINFER_STREAM_FRAMES` 那条固定帧路径（改为调优器驱动）与 `depth` 回绕的跨轮预取
（因 ids 漂移，见 C8）。

**没碰的**：`startup.cpp:1510` 的关图、`text.cpp` 里 prefill 被排除在分帧与适配之外、写路径脏行、
prompt/decode 双布局——这四件都在父分支上就是如此（C7/C8 与 §11.2 第 6–9 条）。

结论：需要的是**补这六件 + 补齐验证**，不是废掉分支重建。
