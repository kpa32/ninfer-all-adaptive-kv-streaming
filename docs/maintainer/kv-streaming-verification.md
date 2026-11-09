---
id: kv-streaming-verification
type: module-design
title: KV 流式的验证机制与评分规则（每条得分都必须有具名检查）
status: active
parent: kv-streaming-strategy
references:
  - kv-streaming-strategy
  - kv-streaming-measurements
---

# KV 流式的验证机制与评分

本文件定义**怎么证明它对、怎么打分**。规则只有一条：**分数只能来自具名检查**——每个检查有
ID、命令、判据、状态与证据位置；没跑过的检查不得分，缺证据的结论不写进任何文档。

对齐三处权威：
- 本仓库：`CONTRIBUTING.md:200`（新 kernel 要有 oracle + `compute-sanitizer` racecheck/synccheck）、
  `tests/README.md`（注册与跑法）、`tests/cmake/CoreTests.cmake`（CTest 注册）、`.github/workflows/ci.yml:411`（`ctest` 是门禁执行点）；
- 作者套件：`tests/test-kv-stream-*.cpp`（plan / config / softmax / cuda-buffer / cuda-set-rows /
  cuda-attn / bench-config，共 71 个具名用例）、`benchmarks/test_*.py`、`tools/kv-stream-bench/`；
- 作者测量：`benchmarks/benchmark_kv_stream.py`（8 K 步进容量扫描 + 自动探池 + 可续跑 CSV + PNG/SVG）。

## 1 六个检查层（L1–L6）

| 层 | 跑在哪 | 依赖 | 上游对应 | 我方现状 |
|---|---|---|---|---|
| **L1 纯逻辑** | CTest（无 GPU） | 无 | `test-kv-stream-plan`（30）、`config`（5）、`softmax`（3）、`bench-config`（3） | layout/feedback/partition 已注册；**span_tuner 未注册**；config 守卫无测试 |
| **L2 内核 oracle + sanitizer** | CTest（GPU，缺卡时退出 77） | 构建树 `BUILD_TESTING=ON` | `cuda-set-rows`（9） | **无**（`kv_stream_signal.cu`、`small_t_i8` 区间发射都没有） |
| **L3 空间/几何 GPU** | CTest（GPU） | 同上 | `cuda-buffer`（7） | **无** |
| **L4 注意力逐位** | CTest（GPU） | 同上 | `cuda-attn`（24） | **无**（我们改成端到端等价，轴不同） |
| **L5 端到端等价与回归** | `tools/bench/run_kv_stream.py`（真机 + 产物） | GPU 空闲、产物、金标 | 无（作者只有基准） | ✅ 这是我们的强项：`check --suite equiv` |
| **L6 门禁与文档** | 脚本/CI | 无 | 无 | 4 项未过（见 G 系列） |

L1–L4 是本仓库规则要求的部分（`CONTRIBUTING.md:200`、`tests/README.md`）；L5 是我们对作者的补充；
L6 是合并门槛。

## 2 检查清单（ID 即契约）

### 2.1 正确性（C）

| ID | 检查 | 命令 / 判据 | 状态 | 证据 |
|---|---|---|---|---|
| C1 | 等价套件（`rk4v4`，8 K / 100 K / 183 K） | `run_kv_stream.py check --suite equiv` ⇒ 4/4，ids 与 logprobs 逐位 | **PASS** | `20261109T113826Z-equiv-s17-suite-c4b02ae2` |
| C2 | `rk8v4` 逐位（110 K，全驻留装得下） | 同上，新增 `equiv-rk8v4` 用例 | 未跑 | — |
| C3 | `int8` 262 144 启动 + 512 token 稳态 | `bench --kv-dtype int8 --case stream-251k-steady` | 未跑 | — |
| C4 | 262 144 prefill 逐位 | 251 K 上 resident(不可行)/流式 vs 金标 ids + 尾 token logprob | 未跑（183 K 已过） | C1 覆盖 183 K |
| C5 | 活体重分区（迁移）逐位 | 训练期强制迁移一次，前后 512 token 逐位 | 未跑 | — |
| C6 | 零窗口 / `ratio=1.0` 可规划且可跑 | 单测 ∈ L1；GPU 需一次 `--kv-stream-ring-ratio 1.0` 跑通 | 半（仅 L1） | `test_kv_stream_layout` |
| C7 | 一进程两请求（`reset()` 路径） | `7dc5fccd5` 的串行提示场景，逐位 | 未跑 | — |
| C8 | graph vs eager 逐位 | `--no-cuda-graph` 对照 | 未跑（依赖 P4） | — |
| C9 | 环→表→attention 字节一致 | `NINFER_STREAM_VERIFY=1` 逐字节 oracle | 手动可用，**未入 CTest** | worklog §5.4 |
| C10 | span 区间发射逐位 | `small_t_i8` 的 run 区间与整层结果比对 | 未跑 | — |
| C11 | 无守卫组合被拒 | 纯函数 `kv_stream_rejection`（`src/core/kv_stream_guards.h`）的单测覆盖 9 组组合与报错顺序；CLI 可达路径 `--kv-capacity 8192 --kv-stream` 实跑 | **PASS** | `test_kv_stream_guards` 全过；实跑报 `error: --kv-stream sizes its KV pool from free memory: pass --kv-capacity auto`。注：单请求 CLI **不暴露** `--max-concurrency`，序列守卫只经 Engine 生效，由单测覆盖 |
| C12 | MTP/DFlash2 KV 不受影响 | 流式下 MTP 接受率与 resident 相同 | 未跑 | — |

### 2.2 性能（P）（判据都是"不可回归"，不是"更快"）

| ID | 检查 | 判据 | 状态 |
|---|---|---|---|
| P1 | 251 K `rk4v4` decode | ≥ 72.8 tok/s（54.4 ms/轮），逐轮 max/median ≤ 1.25 | **PASS**（[measurements](kv-streaming-measurements.md) 表 1） |
| P2 | 251 K `rk8v4` decode | ≥ 26.8 tok/s（147.9 ms/轮） | **PASS**（[measurements](kv-streaming-measurements.md) 表 1） |
| P3 | 183 K streamed decode | ≥ 85.2 tok/s（46.4 ms/轮） | **PASS**（[measurements](kv-streaming-measurements.md) 表 1） |
| P4 | 251 K prefill | ≥ 1281 tok/s（`rk4v4`） | **PASS**（[measurements](kv-streaming-measurements.md) 表 2） |
| P5 | 每轮字节 | ≤ 基线 +2%（1116.2 / 3366.9 MiB） | **PASS** |
| P6 | 暴露时间 | 183 K ≤ 7.9 ms/轮；251 K 记录不设门限（带宽受限） | **PASS**（183 K） |
| P7 | 全驻留不回退 | `equiv-100k` resident ≥ 118 tok/s，且 ≤120 K 不启用流式 | **PASS** |
| P8 | 交错 A/B（`CONTRIBUTING:171`） | 两变体按 rep 交错；提示词不加 nonce（新进程、且会毁金标） | **PASS**：`equiv-183k --reps 2` 交错跑完，两臂 PASS（[measurements](kv-streaming-measurements.md) 表 1） |

### 2.3 内核与工具（K / T）

| ID | 检查 | 判据 | 状态 |
|---|---|---|---|
| K1 | `kv_stream_signal` 的 oracle | 测试自己的记账即规格：ready=样本、未 ready=miss、四 run 互不串、重新 ready 仍是样本、未采样窗口报"无" | **PASS**（`tests/test_kv_stream_signal.cu`） |
| K2 | `compute-sanitizer --tool racecheck` | 0 error（`tests/README.md:96-110` 的跑法） | **PASS**：0 hazards |
| K3 | `compute-sanitizer --tool synccheck` | 0 error | **PASS**：0 errors；另跑 initcheck 也是 0 |
| K4 | 跨架构可编 | `sm_70/75/80/86/89/120a` 全部构建通过 | 未做（本地仅 `sm_89`） |
| T1 | bench 工具单测 | `tests/test_kv_stream_tool.py`（11 个用例，`python -m pytest` 在仓库根跑）纳入 CI | **PASS**（`ci.yml` 新增一步 + `change-scopes.py` 把该文件计入 python 范围） |
| T2 | span tuner + guards 注册 | `CoreTests.cmake` 有 `ninfer_kv_stream_span_tuner_test` 与 `ninfer_kv_stream_guards_test` | **PASS**（文件已注册） |

### 2.4 门禁与文档（G，来自 `CONTRIBUTING.md`）

| ID | 要求 | 现状 |
|---|---|---|
| G1 | `docs/cli.md` 写 `--kv-stream` | **PASS**：新增 "Streaming KV past the pool" 一节 + Common options 两行 |
| G2 | README What's New 一行 | **PASS**：已有一行 |
| G3 | `docs/serving.md` 写明边界 | **PASS**：新增 "Streaming KV is not a server mode" |
| G4 | 新 kernel 的 racecheck/synccheck 记录 | **PASS**（= K2/K3） |
| G5 | 测试注册进 CTest | 缺 span_tuner ⇒ FAIL（= T2） |
| G6 | 测量交错运行 | **PASS**：`run_case` 改为 rep 在外、变体在内（[measurements](kv-streaming-measurements.md) 表 1） |
| G7 | 文档只写当前状态（无日期日志/尝试过程） | worklog/measurements 违反 ⇒ FAIL |
| G8 | Conventional Commits | 遵守 ⇒ PASS |

## 3 评分规则

**得分 = Σ(权重 × 通过) / Σ(权重)**，通过只认 §2 的 PASS；"半"按 0.5 计；未跑 = 0。

| 维度 | 权重 | 检查集合 |
|---|---:|---|
| 正确性 | 40% | C1–C12（每项等权，C1/C11 各算 1，C9 未入 CTest 计 0.5） |
| 性能回归 | 20% | P1–P8 |
| 内核与工具 | 15% | K1–K4、T1–T2 |
| 门禁与文档 | 15% | G1–G8 |
| 上游机制覆盖 | 10% | §11 的 M 表（有对应实现且被检查覆盖才算） |

等级：
- **A 可交付**：总分 ≥ 90 且正确性 ≥ 95 且门禁 = 100%
- **B 可合并**：总分 ≥ 80 且正确性 ≥ 90 且门禁 = 100%
- **C 可用预览**：正确性 ≥ 85（主流格式逐位）+ 有可用边界说明
- **D 不可用**：其余

## 4 当前得分（按 §2 逐项算出来，不是估计）

| 维度 | 通过/总 | 得分 | 依据 |
|---|---|---:|---|
| 正确性 | 6/12（C1、C11 全过；C6、C9 半） | **20/40** | C1 等价套件 + C11 守卫表与新 CLI 实跑 |
| 性能回归 | 8/8（P1–P8 全过，P8 为交错） | **20/20** | 5 条实测基线 + 交错口径 |
| 内核与工具 | 5/6（T1、T2、K1–K3 全过；K4 跨架构未做） | **12.5/15** | oracle + racecheck + synccheck 落地 |
| 门禁与文档 | 6/8（G1、G2、G3、G4、G6、G8 过；G5 等 CTest 树、G7 文档结构） | **11.25/15** | 只剩注册与文档结构 |
| 机制覆盖 | ~10/40 工作包 | **2.5/10** | §11 |
| **总分** | | **≈66/100 ⇒ D** | 正确性 6/12 是唯一挡 C 线的项 |

**这就是为什么不能说"方向对了"**：方向对不对要由 M 表与 P 表证明（P1–P7 过、机制 25%），
而"结果对、可交付"要由 C 表与 G 表证明（C 只过 1 项、G 只过 1 项）。当前结论只能是
**D 级：可跑、主流场景正确性有一项硬证据，但不达合并与交付线**。

## 5 与作者套件的差距（为什么我们的 L2–L4 是空的）

| 作者用例组 | 用例数 | 我们 | 处理决定 |
|---|---:|---|---|
| `test-kv-stream-plan`（planner/extent/prefetch/partition/feedback） | 30 | layout 14 + feedback 9 + partition 14 | 等价替代；**prefetch dispatch 与 extent 迟滞我们没有实现，因此无测试**（要补实现再补测试） |
| `test-kv-stream-config`（opt-in / 支持配置 / 逐条拒绝 / 池划分） | 5 | 无 | **要补**（C11 的来源） |
| `test-kv-stream-softmax`（分段合并） | 3 | 不适用 | 块表间接寻址不需要 |
| `test-kv-stream-cuda-buffer`（staging 几何、pinned 可达、池移动） | 7 | 无 | **要补**（L3） |
| `test-kv-stream-cuda-set-rows`（量化写路径、在线写入、几何） | 9 | 无 | **要补**（L2，覆盖环↔host 字节） |
| `test-kv-stream-cuda-attn`（tuner、逐位、驻留页、多波、布局发布） | 24 | 无（用 L5 替代） | **要补最小子集**：单页/多页/宽 batch 的逐位，等价于 C4/C10 |
| `test-kv-stream-bench-config`（基准配置校验） | 3 | `test_kv_stream_tool.py` 部分覆盖 | 纳入 CI 即可（T1） |

## 6 执行与落盘

- L1–L4：`ctest --test-dir <build> -j`；`BUILD_TESTING=ON` 的构建树；缺 GPU 时用例退出 77（`tests/README.md`）。
- L5：`python3 tools/bench/run_kv_stream.py …`；每条检查在 `bench/results/kv-stream/` 留运行记录，
  在 [measurements](kv-streaming-measurements.md) 追加一行"结论 → 检查 ID → 运行记录"。
- L6：合并前逐条跑 §2.4；未过的项在 PR 描述里点名，不许写"基本完成"。

**本轮的直接动作**（并入 P0）：T2（注册）、K2/K3（sanitizer）、C11（守卫测试）、T1（工具测试进 CI）、
G1/G2/G3（文档）、G7（文档结构整改）、P8（交错测量）。这八项都不碰算法，先把"能不能证明"这条命脉接上。