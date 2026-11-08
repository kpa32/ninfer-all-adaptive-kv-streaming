# RISKS：删减与流式的风险登记

状态：阶段 1 审计产出。每条都给"触发条件 / 影响 / 处置"。诚实标注哪些是已核实、
哪些只是看到迹象、哪些还没测。

---

## 1. 环境与操作风险（已发生/已核实）

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

### R3 PCIe 链路在空闲时回落到 2.5 GT/s — **已核实，但需加载时复核**

- `/sys/bus/pci/devices/0000:08:00.0`：`current_link_speed = 2.5 GT/s`，
  `max_link_speed = 16.0 GT/s`，`width = 16`。
- 影响：**如果加载时链路不升到 16 GT/s，KV 流式的带宽上界会差 6 倍以上**，
  阶段 3 的所有 tok/s 结论都会失效。
- 处置：阶段 2 第一次跑流式时，在推理进行中读一次 `current_link_speed`，并用一次
  H2D 微基准（pinned→device，大块连续拷贝）测有效带宽，把两个数字写进 `BASELINE.md`。

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

---

## 3. 上游 cherry-pick 冲突风险

本仓库保留 `.git` 历史与 `origin`（`kpa32/ninfer-all-adaptive-kv-streaming`）。
删除整族/整卡/整格式会与未来上游改动产生结构性冲突：

| 删除动作 | 冲突面 | 缓解 |
|---|---|---|
| A1–A3（三族）| 上游对 `engine.cpp`、`registry.*`、`models/CMakeLists.txt`、`tests/CMakeLists.txt` 的任何改动都会冲突 | 把这三处改动集中在一个 commit（A4），冲突时整块取上游再重放删族 |
| A5（权重格式）| `linear.cpp` dispatcher、`ops/CMakeLists.txt`、`tensor-formats.md` 是上游高频改动区 | 删格式的提交只动"格式枚举 + 源列表 + 文档表"，不动共享的 tile/launch 代码 |
| A6/A7（架构）| `CMakeLists.txt` 顶部的架构门控被上游频繁调整（CUDA 版本、120a 限制） | 保留顶层结构，只删具体分支；冲突时以上游为准重放 |
| `qwen3_5` 内部（B 组）| 最大冲突面 | 阶段 2 之后再动，且按"单一路径收敛"而不是"删分支" |

非冲突面（安全）：`SCOPE.md` / `SPEC.md` / `CUTLIST.md` / `RISKS.md` 是本仓库新增文件，
上游没有对应物。

---

## 4. 需要数值验证的改动

| 改动 | 为什么不能只靠编译通过 | 验证 |
|---|---|---|
| A5 删格式 | dispatcher 删错分支会让另一个格式静默走错 kernel | 固定 prompt greedy token IDs（8K/32K）+ `ninfer_tests` 的 linear/attention 目标子集 |
| A6/A7 删架构 | 陷阱单元（trap）与回退路径共享宏；删错会让 89 编译出空实现 | 构建后跑 8K smoke + 一个长 prefill（32K）+ `--help` 启动 |
| B2 删 KV 格式 | 每个格式有独立的 Op oracle 与 golden 行；`rk8v4` 与 `int8` 共享 key 编码 | `tests/ops/test_kv_cache_append.cpp`、`causal_cache.cpp` 的保留格式用例 + perplexity |
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
