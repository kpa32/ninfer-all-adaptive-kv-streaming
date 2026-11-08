# RISKS：删减与流式的风险登记

状态：阶段 1 审计产出。每条都给"触发条件 / 影响 / 处置"。诚实标注哪些是已核实、
哪些只是看到迹象、哪些还没测。

---

## 1. 环境与操作风险（已发生/已核实）

### R0 越界删除（流程风险，最高优先级）— **所有者明确警告**

- 所有者 2026-11-08：删减的只是**产品范围（其它模型）**，不是功能；27B 特性与 RTX 30/40/50 支持必须保留；
  **未经审核的删除可被整体回滚，相关工作作废**。
- 对本 agent 的约束：① 动手前先提交 `CUTLIST.md` 条目与理由；② 一条一个 commit；
  ③ 不把“顺手清理”当成删减；④ 任何“看起来没人用”的代码先归 B 组。
- 到本文件为止的审计只新增了 5 份根目录文档，未删除/修改任何源码、测试、脚本。

### R1 Linux 机器上的 git 根目录与脚本不一致 — **已核实**

- 机器上 `/home/kpa/ninfer-fork` **不是** git 仓库；真正的仓库在
  `/home/kpa/ninfer-fork/ninfer-all-adaptive-kv-streaming`（`master` @ `6eb311fd`，干净）。
- `start-mtp.sh` 里 `SERVER="$SCRIPT_DIR/ninfer-all/build-sm89/apps/ninfer-serve"`，
  而实际产物路径是 `$SCRIPT_DIR/ninfer-all-adaptive-kv-streaming/build-sm89/apps/ninfer-serve`
  —— **该脚本今天启动不了**。
- `start-mtp.sh` 还包含 `sudo nvidia-smi -pm 1` 与 `sudo nvidia-smi -i 0 -pl 320`，
  与你"不要 sudo"的要求冲突。
- 处置：不改脚本（你的规则）；阶段 2 起用显式命令启动（`LD_LIBRARY_PATH` + `--model` 位置参数），
  并在报告里给出完整命令行。若你希望修脚本，那是单独的、需要你同意的改动。

### R2 构建时间与机器共享 — **已核实**

- 16 核、64 GiB；审计期间机器上有一个 `./build.sh` 正在跑（PID 16123，`-j16`），
  负载 12–16。按你的指示我没有并发构建、没有杀进程。
- 影响：任何"全量构建耗时"的量测都必须在那台机器空闲时做，否则数字不可比。
- 处置：构建耗时只做相对对照（同一时段、同一 `-j`），并记录当时的 `loadavg`。

### R3 PCIe 链路：空闲 2.5 GT/s，加载时 16.0 GT/s — **已核实（两侧都测到）**

- 空闲：`current_link_speed = 2.5 GT/s`；`Max=16.0 GT/s ×16`。
- 加载中（服务端 15.9 GiB 占用、decode 运行时）：`current_link_speed = 16.0 GT/s`。
- 影响：链路会升速，因此阶段 2 的带宽假设（PCIe 4.0 x16，有效约 25 GB/s）成立；
  但**有效带宽仍需自测**（协议开销、小传输、双向竞争都会拉低）。
- 处置：阶段 2 第一次跑流式时做一次 pinned→device 大块连续拷贝的微基准，写进 `BASELINE.md` §5.5。

### R5 用户自己的 device profile 标定被丢弃 — **已核实（日志）**

- 启动日志：`the entry for nvidia-geforce-rtx-4080-super-sm89 was calibrated against route catalog
  a0d4d93c979588e1 (build 2026-11-04.3-15-gf6928e00), this build's is e614262150e40eaf; it is ignored`。
- 影响：当前跑的是**内建** profile（78 routed keys，2026-10-16），不是最近的实测标定。
  阶段 3 比较 kernel/route 改动时，基线必须固定用同一个 profile（内建或重标定后的）。
- 处置：在阶段 3 开始前由你决定是否花 GPU 时间 `ninfer-calibrate`；那之前所有 A/B 都注明
  "内建 profile"，不把 profile 变化混进 kernel 改动的对比里。

### R4 Windows 侧不能编译 CUDA — 已核实

- 本机（Windows）只做编辑与只读审计；所有构建/运行在 Linux 机器上。
- 影响：任何"我编译过了"的说法都必须来自 Linux 的真实日志。

---

## 2. 不能删的东西（按代码证据）

| 项 | 证据 | 如果删了会怎样 |
|---|---|---|
| `NINFER_SM8X_COMPAT` 路径 | 实际构建命令带 `-DNINFER_SM8X_COMPAT=1 -DNINFER_TIER_AMPERE=1` | sm_89 会失去路由与回退，几乎全部 kernel 失效 |
| `src/ops/kv_stream/*` + Strata 归属 | `qwen4_exp` 在用；阶段 2 的参考实现 | 阶段 2 从零开始，且违反 `CONTRIBUTORS.md` 的归属 |
| `HostKVArena` / `transfer_work` / `evictable_kv_pool` / `vmm` | paged KV、context cache、Vision overlay、suspend 都在用 | KV 流式没有物理基础 |
| `qwen3_5/program/storage/*`（kv_store、host_kv_store、state_store、session_snapshot） | 上下文缓存与 checkpoint 的唯一实现 | prefix reuse / slot 持久化全部失效 |
| `softmax_attention/dense/**` 的 `context`/`packed`/`prompt` 三组路由 | prefill、decode、verify 分别用它们 | 长上下文 prefill 或 decode 直接不可用 |
| `linear/gguf/**` + `linear/q4,q5,q6,q8/**` | artifact 实际格式（见 `SCOPE.md` §2） | 目标模型加载失败 |
| `linear_attention`、`gdn_*`、`dynamic_grouped_conv` | 48 个 GDN 层 | 模型数学不成立 |
| `device_profiles.json` 的 4080 SUPER 条目 + `ninfer-calibrate` | `--device-profile auto` 依赖它 | 路由退回通用系数，性能不可解释 |
| sm_86 / sm_89 / 120a 的架构分支与各自回退路径 | 所有者要求保留 RTX 30/40/50 支持 | 删了就是删功能（你的规则） |
| 九种 `--kv-dtype` 格式与各自的 attention/codec 变体 | 27B 的可选特性，`docs/performance.md` 有实测 | 删了就是删功能 |
| FP8 / NVFP4 / T2 / STQ 权重单元 | 同一个 27B 的其它 artifact + 120a 构建需要 | 无法加载那些 artifact |
| Vision（`execution/vision*`、`load/vision*`、`program/vision_control*`、`src/media/decode`）| 所有者已确认永久保留；`start-mtp.sh` 在用 | 删了就是删功能 |
| `core/tensor_parallel.*`、`stage_link`、`stage_plan`、pipeline CLI | 多卡是功能，不是“别的模型” | 删了就是删功能 |
| `runtime/engine/context_cache/**` + `runtime/prefix_cache/**` | Legacy + Hybrid 两种 prefix cache | serving 的缓存模式失效 |

---

## 3. 上游 cherry-pick 冲突风险

本仓库保留 `.git` 历史与 `origin`（`kpa32/ninfer-all-adaptive-kv-streaming`）。
删除整族/整卡/整格式会与未来上游改动产生结构性冲突：

| 删除动作 | 冲突面 | 缓解 |
|---|---|---|
| A1–A4（Gemma 4 / Flint / DeepSeek） | 上游对 `engine.cpp`、`registry.*`、`models/CMakeLists.txt`、`tests/CMakeLists.txt` 的任何改动都会冲突 | 登记点集中在一个 commit（A4），冲突时整块取上游再重放删族 |
| 未来若要删 Flash-Next / MoE（B1/B2） | `engine.cpp`、`program/` 与 `ops/` 的共享层，冲突面最大 | 阶段 2 之后再评估；按“收敛到单一路径”而不是“删 if 分支”重写 |
| 保留项（Vision、KV 格式、FP8/NVFP4、sm_70/75、TP/pipeline） | 不产生冲突：它们保持与上游同形 | — |

非冲突面（安全）：`SCOPE.md` / `SPEC.md` / `CUTLIST.md` / `RISKS.md` 是本仓库新增文件，
上游没有对应物。

---

## 4. 需要数值验证的改动

| 改动 | 为什么不能只靠编译通过 | 验证 |
|---|---|---|
| A1–A4 删族 | 注册点/前端方言删错会让另一个族静默退回 Qwen 语义 | 8K smoke 的 greedy token IDs 与基线比对 + `ninfer_tests` 的 qwen3_5 子集 |
| 阶段 2 流式 | 页驻留改变的是“数据在哪”，必须是逐字节等价 | 同格式 on/off 的 greedy IDs 与 logprobs 一致；planted-code recall；perplexity |
| （保留）改任何共享 kernel | 九种 KV 格式与 FP8/NVFP4 权重仍要可加载 | 对应 Op 测试 + 真实 artifact 启动 |
| 阶段 2 流式 | 页驻留改变的是"数据在哪"，必须是逐字节等价 | 同格式 on/off 的 greedy IDs 与 logprobs 一致；planted-code recall；perplexity |

注意：`CONTRIBUTING.md` 的 "lossless by default" 要求"同格式 streaming on/off 结果一致"是
**可验证的合同**，不是口号。阶段 2 必须把它做成测试，而不是一次性人工对比。

---

## 5. 阶段 2 的技术风险（尚未在代码中核实）

| 编号 | 风险 | 现状证据 | 处置 |
|---|---|---|---|
| S1 | `decode_graph` 只捕获单流，跨流同步怎么做 | `core/decode_graph.cpp` 仅 `cudaStreamBeginCapture(ThreadLocal)`；但 `program/structured_round.cpp:136-141` 已在捕获体内用 `cudaEventWaitExternal`（外部事件节点），且仓库无 `cudaGraphAddEventWaitNode` 的直接调用 | 用“图外拷贝流 + 每层/每 token 一个外部事件 + 图内 wait”实现重叠；捕获体内 fork 二流属未验证项，MVP 不用 |
| S2 | 每层独立 ring 与现有页命名空间冲突 | `state/decoder_state.cpp` 的 `plane_index()` 说明每个 layer 有自己的 plane，但 `layer_view()` 让全部 layer 共用同一行 block table；即页组 g 在每层都是同一物理页号 | 阶段 2 先在一个新 streamed pool（自己的 execution table）上做，不动 Main pool 的发布路径 |
| S3 | block table 的 pinned shadow H2D 与"每 token 改写同一行"的 fence | `paged-kv-cache.md` §12 不变量 18 | 流式层用**静态表**（槽位固定为 `l mod R`），把发布频率降到 0 |
| S4 | host pinned 内存上限 | 预计 host KV < 8 GiB，64 GiB 主机足够；但 `RLIMIT_MEMLOCK`/`vm.max_map_count` 未查 | 首次分配失败就停下报告，不改系统配置 |
| S5 | 262K 上下文下 attention 的 split/workspace 假定 KV 在 device | 未核实 | 阶段 2 第一步就读 `softmax_attention` 的 launch 分裂逻辑，确认只吃指针 |
| S6 | 物理上界（带宽）远低于算力，容易把"流式没生效"误读成"kernel 慢" | 已算：`rk4v4` 全流 4.4 GiB/token ÷ 25 GB/s ≈ 5.7 tok/s | 每次测量同时记录 streamed bytes/token 与 H2D 时间，先算上界再谈 kernel |
| S7 | 自适应调参会 thrash | 文章的经验（ring 只在预取跟不上且 PCIe 未饱和时扩大）| 加 hysteresis、每次调整记日志、用固定 bucket 起步 |

---

## 6. 报告纪律（对本次工作自身的要求）

- 只报告真实命令输出；SSH 失败/构建失败/超时就写"未完成 + 原因"，不写推测数字。
- 每个性能数字带条件：commit、build 参数、CUDA/driver、卡、上下文长度、并发、KV 格式、
  是否 MTP、是否命中 prefix cache。
- 每档上下文至少一次冷前缀（避免 prefix cache 把 prefill 变成 0）与一次重复前缀（验证复用）。
- 报最差变化，不只报最好一次；同一主机的 A/B 才可比。
