# SPEC：单目标 NInfer + Qwen3.8-27B 自适应 KV 流式

状态：draft（阶段 1 审计产出，等所有者确认）
权威关系：范围由 [`SCOPE.md`](SCOPE.md) 定义；本文件定义目标、架构现状与改动面、阶段验收。
删减执行细节见 [`CUTLIST.md`](CUTLIST.md)，基线测量见 [`BASELINE.md`](BASELINE.md)，
删减风险见 [`RISKS.md`](RISKS.md)，仓库既有架构权威仍是 `docs/maintainer/` 下的文档。

## 1. 目标

把本仓库的产品范围收成**一条路线**（Qwen3.5 Dense 27B，RTX 30/40/50 平台），然后在这条路线上
实现**自适应 KV 流式**，让 Qwen3.8-27B Dense 的 262 144 原生上下文在单张 16 GiB 的
RTX 4080 SUPER 上稳定、正确、可测量地运行。

三个目标按顺序成立：

1. **可维护**：删掉**其它模型的支持**（Gemma 4、Flint、DeepSeek-V4.1-Flash，以及待定的
   Flash-Next / MoE），保留 27B 的全部功能、全部 KV 与权重格式、Vision、TP/pipeline；
   平台的保留规则见 `SCOPE.md` §1（sm_86/89/120a 保留）。
2. **能装下**：KV 不再受显存硬墙限制；host 权威 KV + 设备 resident 区 + 有界 ring + 按层预取 +
   自适应划分。
3. **快**：在 16 GiB 上把长上下文 decode 做到 PCIe 4.0 x16 带宽允许的上界附近，并用真实测量报告。

## 2. 非目标

见 `SCOPE.md` §0–§3。要点：**删减不等于删功能**——27B 上可用的一切保持可用；
不做 lossy 默认；不引入第二套 artifact/推理路线；不在本次分叉里为其它模型新增工作。
把“其它模型的支持”去掉是唯一被授权的删减方向，逐条见 `CUTLIST.md`。

## 3. 现状架构（审计结论，按代码核实）

### 3.1 执行边界与所有权

四层边界（`docs/maintainer/engine-architecture.md`）：Gateway → Frontend → Engine → Program。
与本次改动相关的唯一权威：

| 事实 | 所有者 | 代码 |
|---|---|---|
| 请求顺序、round 成员 | Scheduler | `src/runtime/engine/scheduler.h`、`admission_policy.*` |
| 逻辑缓存策略、retention | ResourceManager | `src/runtime/engine/context_cache/`（8.4k 行） |
| 物理 State/KV、reservation、placement、CUDA Graph | Program | `src/models/qwen3_5/program/` |
| 物理 KV 页池、block table、execution row | KV Store（Program 内） | `src/core/paged_kv_cache.*`、`src/models/qwen3_5/program/storage/kv_store.h` |
| host packed 页与 arena | Host KV Store | `src/core/host_kv_arena.*`、`storage/host_kv_store.h` |

关键现状（决定了阶段 2 的改动面）：

1. **active request 的 KV 必须全部在 Device**。admission 为 active request 预留
   prompt + 最大输出增长的完整 Device 页（`kv_lease_growth` 只是把它变成有界窗口）。
   host replica 目前服务的是 **inactive checkpoint**（device→host 降级、session snapshot），
   而不是 active sequence 的按层流式工作集。
2. KV 池是 `P=64` 的 typed plane pools，closed plane order 为 page-major
   `[X,P,H,N_physical]`。**每个 layer 的 K/V/(scale) 是独立 plane**
   （`state/decoder_state.cpp:plane_index()` = `layer × shares × kinds + rank × kinds + kind`），
   但**同一个 pool 只有一张 execution table，也就是一张 block table 行被全部 layer 共用**
   （`layer_view()` 里 `block_table = execution_tables_.row(row, holder)`）。
   即：**页组 g 在全部 layer 的 plane 中是同一个物理页号**。
   qwen4_exp 的流式为什么能用“每层一个小池”，正是因为它有自己的一套 program/storage；
   qwen3_5 要做到“每层独立 ring”必须新增一个 pool（带自己的 execution table）或引入 per-layer 表。
3. 消费端只看 `PagedKVLayerView`（每层 plane tensors + 一行 block table）与
   `PagedKVBatchLayerView`（`block_tables[Nlogical,C]` + `table_rows[B]`）。
   page ID 是显存数据，不是 graph key；plane base 与 table base 在 Engine 生命周期内稳定。
4. CUDA Graph 捕获**单流**：`core/decode_graph.cpp` 的 `capture()` 只对一条 stream 做
   `cudaStreamBeginCapture(..., ThreadLocal)`，`DecodeGraphExecutable` 提供 instantiate/update/upload/launch。
   但仓库**已经在捕获体内使用外部事件节点**：
   `program/structured_round.cpp:136-141` 在 `cudaStreamIsCapturing()==Active` 时用
   `cudaStreamWaitEvent(stream, uploaded_, cudaEventWaitExternal)`，让每次 replay 等待图外
   先发布的 mask 上传。这正是流式预取需要的同步原语（图外拷贝流 + 图内等待），
   **不需要**重新捕获。仓库里**没有**“捕获体内 fork 到第二条流并汇合”的先例
   （`cudaStreamWaitEvent` 只出现在 TP/stage/overlay 的非捕获路径与上述外部事件处）。
5. 已存在的流式基础设施（**阶段 2 的主要复用对象**）：
   - `ops/kv_stream/kv_stream.cu`（410 行）+ `include/ninfer/ops/kv_stream.h`：
     block→slot residency map、GPU 内 CLOCK 驱逐、从 pinned host 直接拷贝到 pool slot 的两个
     kernel（resolve + copy），**全部可捕获**（位置与拷贝数从显存字读取）。
     几何固定为 `kKvStreamPageTokens=4`、`kKvStreamMaxTokens=16`、每 query 513 block；
     host 布局 `[leading,64,heads,pages]`（identity 页表），pool 布局 `[leading,4,heads,slots]`。
   - `src/models/qwen4_exp/executor.cpp` 是它的生产用户（`--kv-resident N|all|auto`，
     12 个 sparse-attention 层，默认 32 768 驻留）。它证明了：per-layer pool 张量
     （`state.pool[i]`）、host cache view、map 显存、以及 capture 后的 replay 都能工作。
   - `core/host_kv_arena.*`：固定容量 pinned arena、per-plane packed layout、
     recipe/apply 的无分配 adopt 语义、`split()`/subview。
   - `core/evictable_kv_pool.*`、`core/vmm.*`：VMM 借还（整层/整窗口粒度）。
6. KV 页面的 attention 路径是 **paged 直接寻址**：不在 kernel 外做 gather，
   也不建立 request-contiguous staging copy（`paged-kv-cache.md` §10.4 的约束）。
   这对 dense 层流式意味着：**一层 attention 需要该层在整个可见前缀上的 KV**
   （不是 sparse 的 512 block），因此 ring 的槽位粒度必须是"一层的一份完整 KV"，
   而不是 sparse 的 4-token block。

### 3.2 目标模型的数字（用于容量预算）

按 `SCOPE.md` 的几何，单 token 单 KV 头的物理字节（`docs/maintainer/paged-kv-cache.md` §4.3 的表）：

| KV 格式 | 单 token 单头 | 16 层 × 4 头 = 每 token | 262 144 token |
|---|---:|---:|---:|
| `bf16` | 1 024 B | 64 KiB | 16.0 GiB |
| `rk8v4` | 408 B | 25.5 KiB | 6.4 GiB |
| `rk4v4` | 280 B | 17.5 KiB | 4.4 GiB |
| `rk2v4-e8` | 216 B | 13.5 KiB | 3.4 GiB |

权重 10.9 GiB，卡 16.0 GiB。即使 `rk2v4-e8`，262 144 token 也只剩约 1 GiB 余量给
workspace / State / Graph / 分配器；`bf16` 与 `rk8v4` 必须依赖流式才能达到原生窗口。
4080 SUPER 的 PCIe 4.0 x16 有效带宽估计约 25 GB/s（阶段 2 必须自测），因此
**全流式的上界 ≈ 25 GB/s ÷ 每 token 流过字节**：`rk4v4` 全流 4.7 GB/token 只有约 5.3 tok/s；
把 12/16 层留在显存（每 token 只流 4 层 ≈ 1.18 GB）上界约 21 tok/s。
阶段 3 的优先级因此确定为：**减少 KV 字节 > 提高 resident 比例 > 预取/重叠 > kernel**。

## 4. 目标架构变化（阶段 2 的设计约束）

### 4.1 数据流

```
          权威副本 (pinned host, HostKVArena extent, per-layer packed layout)
                                   │  H2D (按层、按 slot)
                                   ▼
        Device: resident 区 (完整 16 层的部分页) + ring (R 个 layer-sized slot)
                                   │  block table (显存数据)
                                   ▼
                        paged attention (不变)
```

- 权威 KV 一旦进入流式模式就写在 host 副本上（`kv_cache_append` 写 host view），
  device 只持有工作集；resident 层仍按今天的路径直接写 device。
- 每个流式层在 device 上拥有 `R` 个 slot 中的固定槽位 `l mod R`；一个 slot = 该层一份完整
  （或当前 frontier 范围内的）KV。ring 的消费者只有"下一个流式层"，因此槽位映射可以是静态表，
  每 token 只需搬字节，不需要改 page ID（这是与 sparse `kv_stream` 的 map 最大的不同）。
- 预取：层 `l` 的 attention 结束后即可预取层 `l+R`；MVP 允许串行（copy 后 compute），
  优化阶段再引入第二条拷贝流 + 事件等待。
- 自适应：全驻留 → 容量不足时按 full-attention 层**公平**释放一小部分汇成 ring →
  预取跟不上且 PCIe 未饱和时才扩大 ring → 饱和后停止扩大；带 hysteresis，避免 thrash，
  每次调整写入结构化日志。

### 4.2 必须保持的不变量

1. 页生命周期：copy 期间源与目标都被 pin；目标在 copy 完成并通过 epoch/coverage 校验前不进入
   任何 address space 或 block table。
2. 一个 GPU execution unit 内，选中行的 block table 不变、page payload 不搬迁。
3. `--kv-dtype` 与 streaming 正交；同一格式 streaming on/off 的 greedy token IDs 与 logprobs 一致。
4. 失败必须回到完整 inactive 终态：OOM、缺 slot、拷贝失败都不得留下半改写的映射。
5. 默认关闭；关闭时路径与今天逐字节一致。

### 4.3 已识别但**尚未在代码中核实**的机制（阶段 2 前置）

已核实的有利事实：外部事件节点在 qwen3_5 的捕获图里已有生产用法
（`program/structured_round.cpp`，`cudaEventWaitExternal`），因此“图外拷贝流 + 图内等待”这条
重叠路线不需要新机制，只需要把事件从 mask 上传换成 per-layer 预取完成。

仍未核实：

- **页命名空间**：流式层要“每层独立 ring + 静态表”时，是新建一个 streamed pool（自己一张
  execution table，代价是容量曲线多一维）还是给 Main pool 加 per-layer 表（代价是改
  `PagedKVLayerView.block_table` 的消费者合同与发布路径）。两种都要先算清 `page_group_count`
  与 `SequenceCapacityCurve` 的影响（见 `paged-kv-cache.md` §3.3）——这是阶段 2 设计文档的第一个决策点。
- 捕获体内 fork 到第二条流（`cudaEventRecord` 在捕获流 → `cudaStreamWaitEvent` 到拷贝流 → 拷贝
  → 汇合）是否有先例；没有就先走外部事件，不改捕获拓扑。
- 单层能否在“独立命名的页命名空间”下与 resident 层共用一套 append/attention 调用
  （plane 已经按层存在，障碍只在共享的 block table 行），以及 `SequenceCapacityCurve`
  如何表达“resident 页 + ring 页”两种不同寿命的容量。
- block table publication 的 pinned shadow H2D（`paged-kv-cache.md` §12 不变量 18）在每 token
  改写同一行时的 fence 语义。
- 262 144 上下文下 attention 的 split/workspace 是否随"KV 在 host"变化（不应变化，但需确认
  kernel 只看到 device 指针）。

## 5. 阶段与验收

### 阶段 1：删减（本文件交付的范围）

执行条件：`CUTLIST.md` 的 A 组（三个别的模型族 + 登记点）被所有者逐项确认；B 组默认不动。
每步：小步 commit → Linux `git pull --ff-only` → `./build.sh` → 8K smoke →
固定 prompt greedy token IDs 与基线一致。回归就回退该 commit。

验收：
- 删除清单内每一项可独立回退（一次提交一个族）；
- 目标路径的 8K/32K/120K smoke 与基线 token 序列一致；
- `AGENTS.md`/`CONTRIBUTING.md` 的族清单按 `SCOPE.md` 改写（平台节保留 sm_86/89/120a）；
- 仓库内不再有指向已删族的构建目标、测试、文档链接（`git diff --check` + 文档链接检查）。

### 阶段 2：KV 流式（MVP → 正确性 → 自适应 → 性能）

1. MVP：固定 resident/streaming 划分、固定 R、按层预取 N 层、单流 copy+compute、
   `--kv-stream`（或同义开关）默认 off。
2. 正确性：8K/32K/64K/128K 逐级加长，同格式 on/off greedy IDs 与 logprobs 一致；
   长文 planted-code recall；`ninfer-perplexity` 一致。
3. 自适应：全驻留起步、公平释放、按 PCIe 饱和度停止扩大、hysteresis、每次调整有日志。
4. 性能：ring 深度、预取距离、copy 批量、pinned 内存、MTP/DFlash2 交互；先 C=1，再 2–4。
5. 验收：262 144 不 OOM；decode 稳定；正确性达标；报告真实 tok/s；resident 路径不回退。

### 阶段 3：4080 SUPER 优化

先测 8K/32K/64K/128K/200K/262K 的 prefill/decode、VRAM、H2D 字节、wait 时间；
按"KV 压缩 > resident 比例 > 预取/重叠 > kernel/route"的顺序每次只改一个变量；
保留基线对照；报告均值/范围与最差变化；不引入 lossy 默认。

## 6. 度量与报告合同

每次测量记录：`git rev-parse HEAD`、分支、build 参数、CUDA/driver、GPU 名/显存、
完整启动参数、每档上下文的 prefill/decode tok/s、峰值 VRAM、host RAM、
streaming 指标（resident bytes、streamed bytes/token、H2D 次数/字节、wait 时间、hit rate）、
失败日志原文。口径与 `CONTRIBUTING.md` 的"comparable measurements"一致：
同一主机、同一 build、每个 prompt 带唯一 nonce（避免 prefix cache 复用）、交错跑。

## 7. 交付物

| 文件 | 性质 | 生命周期 |
|---|---|---|
| `SCOPE.md` | 产品范围契约 | 长期权威 |
| `SPEC.md` | 目标/架构/验收 | 长期权威 |
| `CUTLIST.md` | 阶段 1 执行清单 | 阶段 1 结束后并入 `SCOPE.md`/`README.md` 并删除 |
| `BASELINE.md` | 分叉时的真实测量 | 数字迁入 `docs/performance/` 后删除 |
| `RISKS.md` | 删减风险登记 | 阶段 1 结束后删除 |
| `docs/maintainer/kv-streaming-design.md` | 阶段 2 设计 | 长期权威（阶段 2 产出） |
