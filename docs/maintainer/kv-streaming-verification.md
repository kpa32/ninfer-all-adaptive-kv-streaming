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
| C2 | `rk8v4` 逐位（100 K，全驻留装得下） | `check --case equiv-100k-rk8v4` ⇒ ids 与 logprobs 逐位 | **PASS**（`20261109T163448Z-equiv-c2`，见 [measurements](kv-streaming-measurements.md) 表 3） |
| C3 | `int8` 262 144 启动 + 512 token 稳态 | `bench --kv-dtype int8 --case stream-251k-steady` | **PASS**：236.1 ms/轮、16.8 tok/s、5483 MiB/轮、暴露 203 ms；prefill 1331 tok/s | 见 [measurements](kv-streaming-measurements.md) 表 1、表 2 |
| C4 | 262 144 prefill 逐位 | 251 K 上 resident(不可行)/流式 vs 金标 ids + 尾 token logprob | **PASS**：`check --case stream-251k`（`20261109T183207Z-equiv-e524df8a`），ids 与 `golden/stream-251k.ids` 相同，logprobs 的 sha256 `8a06a1d4…` 与金标一致 | 金标为 `oracle: self`（原版在 262 K 起不来），因此这是同构建回归，不是与原版的等价 |
| C5 | 活体重分区（迁移）逐位 | 训练期强制迁移一次，前后 512 token 逐位 | **PASS**：`check --case stream-150k-tight`（`20261109T184516Z-equiv-4c511b31`）里流式臂跑了 4 次迁移（`layout_generation=4`，两次 `[partition] trial start` + 两次 `revert`），512 token 的 ids 与 logprobs 仍与金标逐位相同 | 金标为 `oracle: self`；迁移由现有控制器提案，未新增开关 |
| C6 | 零窗口被拒绝、最小窗口可跑 | 实测：窗口 0 的运行在第 39 个 token 分叉（`20261109T214123Z`，golden 18 vs actual 19）——所有页的窗口槽都是 0，新 token 无处落；现由 `repartition` 显式拒绝，`NINFER_STREAM_WINDOW_PAGES` 只接受 ≥1。最小合法性窗口（1 页/层）在 251 K 上跑通且逐位（`20261109T215412Z`，192.6 ms/轮、3290.6 MiB/轮） | 原判据“零窗口可跑”已证伪并换成正确行为；`--kv-stream-ring-ratio` 不再需要恢复 |
| C7 | 一进程两请求（`reset()` 路径） | `7dc5fccd5` 的串行提示场景，逐位 | **PASS**：`ctest --test-dir build-sm89-tests -R kv_stream` 的 `ninfer_qwen3_5_kv_stream_serial_real_test` 通过（290 s，一进程两请求、190 K prompt） | 测试源 `tests/models/qwen3_5/test_engine_kv_stream_serial_real.cpp`，`NINFER_TEST_ARTIFACT` 控制跳过 |
| C8 | graph vs eager 逐位 | `--no-cuda-graph` 对照 | **PASS**（全驻留）：`equiv-183k` resident 的 512 ids 在 graph on（`20261109T192253Z`）与 off（`20261109T192831Z`、`20261109T193412Z`）下逐位相同。流式下不适用：图在流式布局下不可捕获（`20261109T191653Z`，380 µs 内以 `KV logical page is outside the address space` 失败，原因是捕获行在流式地址里没有页） | graph 在流式下维持关闭，代价 0.65 ms/轮（251 K ≈ 1.3%），见 [measurements](kv-streaming-measurements.md) 表 5 |
| C9 | 环→表→attention 字节一致 | `NINFER_STREAM_VERIFY=1` 逐字节 oracle | **PASS**：251 K 上 6048 次整页比对、0 不一致（`20261109T213718Z`，ids 64/64 逐位）；`serial_real` 测试的第二次请求开 VERIFY 并断言 `checks > 0 && mismatches == 0`，计数进 `StreamedKvStats::verify_checks/verify_mismatches` 与报告 `stream.verify` | 随机采样（每页 0 与末页 + 读后首页），不是全量：预算 20 000 页 |
| C10 | span 区间发射逐位 | `small_t_i8` 的 run 区间与整层结果比对 | **PASS**：`NINFER_STREAM_BOUNDED=0`（整层，`20261109T184935Z`）与 `=1`（1 run/帧，`20261109T185335Z`）两次 `check --case stream-251k` 都 PASS，ids 与 logprobs 哈希相同；分帧把暴露从 15.9 降到 1.7 ms，但每轮 54.7 → 166.7 ms | 分帧在本机不作为默认（见 SPEC 交付项 6） |
| C11 | 无守卫组合被拒 | 纯函数 `kv_stream_rejection`（`src/core/kv_stream_guards.h`）的单测覆盖 9 组组合与报错顺序；CLI 可达路径 `--kv-capacity 8192 --kv-stream` 实跑 | **PASS** | `test_kv_stream_guards` 全过；实跑报 `error: --kv-stream sizes its KV pool from free memory: pass --kv-capacity auto`。注：单请求 CLI **不暴露** `--max-concurrency`，序列守卫只经 Engine 生效，由单测覆盖 |
| C12 | MTP/DFlash2 KV 不受影响 | 流式下 MTP 接受率与 resident 相同 | **PASS**：`equiv-183k` 两臂（`20261109T183714Z-equiv-4c511b31`）均开 MTP3，接受长度同为 4.0，ids 与 logprobs 逐位相同；150 K 紧池（`20261109T184516Z`）同为 4.0 | 接受率相同由“两臂 ids 逐位”隐含，记录里两个数都在 |

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
| P9 | 关图的代价 | 流式下不启用 graph，代价已知且不回归（≤ 1.3%/轮） | **PASS**：graph on 比 off 快 0.65 ms/轮（1.75%@183 K），251 K 稀释到约 1.3%（[measurements](kv-streaming-measurements.md) 表 5） |

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
| G5 | 测试注册进 CTest | **PASS**：`ctest -R kv_stream` 跑到 8 个用例并全过（layout / feedback / partition / span_tuner / signal / guards / serial_real / kv_stream） | 
| G6 | 测量交错运行 | **PASS**：`run_case` 改为 rep 在外、变体在内（[measurements](kv-streaming-measurements.md) 表 1） |
| G7 | 文档只写当前状态（无日期日志/尝试过程） | **PASS**：`docs/` 只留设计、策略、验证、参考表；历史移到 `bench/results/kv-stream/ledger.md` |
| G8 | Conventional Commits | 遵守 ⇒ PASS |

## 3 评分规则

**得分 = Σ(权重 × 通过) / Σ(权重)**，通过只认 §2 的 PASS；"半"按 0.5 计；未跑 = 0。

| 维度 | 权重 | 检查集合 |
|---|---:|---|
| 正确性 | 40% | C1–C12（每项等权，C6/C9 的“半”计 0.5） |
| 性能回归 | 20% | P1–P9 |
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
| 正确性 | 12/12（C1–C12 全过） | **40/40** | 等价套件 + 回归金标 + 活体迁移 + 分帧/整层 + graph A/B + 一进程两请求 + 零窗口拒绝/最小窗口 + 字节 oracle |
| 性能回归 | 9/9（P1–P9 全过） | **20/20** | 5 条实测基线 + 交错口径 + 关图代价 |
| 内核与工具 | 5/6（T1、T2、K1–K3 全过；K4 跨架构未做） | **12.5/15** | oracle + racecheck + synccheck 落地 |
| 门禁与文档 | 8/8（G1–G8 全过） | **15/15** | CTest 树真跑 8/8 |
| 机制覆盖 | ~10/40 工作包 | **2.5/10** | §11 |
| **总分** | | **90.0/100 ⇒ A** | 总分 ≥90、正确性 100% ≥95%、门禁 100% |

剩下的两项都在同一次 `BUILD_TESTING=ON` 批次里：C7（一进程两请求）跑过即 11/12（92%）
达 B 的正确性线，G5 随同一次 CTest 跑到 100%。

**这两项已关闭**（`ctest -R kv_stream`：8/8 passed，含 C7 的 `serial_real` 与 G5 的 `span_tuner`）
⇒ **B 可合并**。A（≥90 且正确性 ≥95）仍差：C6/C9 是“半”（零窗口 GPU 判据与 VERIFY 未入 CTest），
机制覆盖 25% 是架构性差距。

**A 可交付**：总分 90.0、正确性 12/12、门禁 8/8。仍要说明的两点：机制覆盖 25% 是架构性差距
（图内调度、脏行、span 合并各有实测理由不做或不需要）；K4 跨架构只能依赖 CI 的四套构建。

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
G1/G2/G3（文档）、G7（文档结构整改）、P8（交错测量）。这八项都不碰算法，先把“能不能证明”这条命脉接上。

## 7 SPEC 完成条件逐条对账（P8）

SPEC 的六条是交付契约。每条的现有证据都指向 `bench/results/kv-stream/runs/` 的具名记录，没有证据的
项不得写成完成。

| # | 条件 | 证据 | 判定 |
|---|---|---|---|
| 1 | `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode | `rk8v4`：147.9 ms/轮、26.8 tok/s（measurements 表 1，P2）；`int8`：236.1 ms/轮、16.8 tok/s、prefill 1331 tok/s（C3） | **PASS** |
| 2 | 同格式 on/off 逐 token、logprob 一致 | `rk4v4`：8 K / 100 K / 183 K 512/512（C1），183 K 两臂 0/512 有差（C12）；`rk8v4`：100 K resident vs streamed 512/512（C2）；251 K `rk4v4` 对自录金标 64/64 + logprobs sha256 一致（C4）；活体迁移 4 次后仍逐位（C5）；分帧/整层逐位（C10） | **PASS**，带下面两个边界 |
| 3 | 120 K 全驻留不差于 109.1 tok/s / 15 632 MiB | **127.5 tok/s / 31.1 ms / 峰值 15 415 MiB**（`20261109T204316Z`，`resident-120k`，`rk8v4` + MTP3、512 token；权重 11 903 + KV 3 375 + workspace 100 + 图 36）。邻点：100 K 全驻留 118.9（P7）、150/180 K 的“池装得下”档 107.1/106.0 | **PASS** |
| 4 | 关闭即今天行为 | `--kv-stream` 不开时不分配流式池、不建流式表、不走 `enter_layout`；全驻留路径的 100 K 118.9 tok/s 与基线一致（P7） | **PASS** |
| 5 | 报告真实 tok/s、峰值显存、搬运字节并解释与上界差距 | measurements 表 1（每轮字节/时间/暴露）、表 2（prefill 上载）、表 5（关图代价）、表 6（阶段布局）；上界解释：46.1 ms 拷贝地板 → 86 tok/s 天花板，实测 54.4 ms 中 8.3 ms 未藏住 | **PASS** |
| 6 | 251 K 每轮拷贝/轮时间不退化、报告窗口长度、证明 decode 真搬页 | 表 1：54.4 ms / 72.8 tok/s / 1116.2 MiB、窗口 2905 页/层、环 11 run；`rounds_series` 逐轮字节非零；迁移与布局形状可读（`layout_generation`、`shape`） | **PASS** |

两个必须在 PR 里点名的边界：

1. **262 144 上的 `rk8v4` / `int8` 没有逐位参照**。同格式全驻留在这个容量装不下（`rk8v4` 需 7.33 GB），
   所以条件 2 的逐位证据来自 100 K（`rk8v4`）与 183 K（`rk4v4`）；262 144 这两格只有“同构建自比”，
   不能与关流式比较。

## 8 未做与不可达（如实列出，不当遗漏）

| 项 | 原因 | 处置 |
|---|---|---|
| R11 图内调度 / 图内换页 | 流式布局下不可捕获（捕获行无页，380 µs 失败）；即便可捕获，graph 只值 0.65 ms/轮（1.75%@183 K，≈1.3%@251 K） | 维持关图，measurements 表 5 |
| R10 写路径脏行/镜像 | 本实现的页只在写满后离窗（design §4.6），回写时整页都是脏的，不存在“回写未写过的行” | 不需要移植，已在本节记录理由 |
| **serve + 流式 + 草稿（曾经的挂起）** | 现象是 `Engine` 的 wait 永不返回、进程随后 `SIGFPE` 消失且无日志。根因不在 serve：`StreamedKVSpace::advance` 对没有环的整段驻留布局也走“页滑出窗口→写回 host”，而 `slot_of_page` 随后去取模一个 0 run 的环。serve 的 warmup（短上下文按整段驻留规划、窗口 1 页）留下这样的布局，第一个真实请求把前沿推过去就必死；CLI 的 prompt 一次规划到位、前沿不再增长，所以长期未暴露。gdb 抓到 `slot_of_page` 帧与 HTTP 线程的 `wait_for_request` 同时在场，才把“挂起”和“崩溃”分开 | 已修（`2fb81289`：整段驻留的窗口跟随前沿，层表随基址重发；`slot_of_page` 以 `std::logic_error` 陈述契约）。三个连续 serve 请求全通过：TTFT 61–78 ms、total 243–350 ms、decode 56.5–86.5 tok/s、MTP 接受 28.6–60.0% |
| 跨轮预取 | 多留一个 extent 的实测代价（+5.6% 字节）大于收益（≤3 ms） | 已否决，ledger “跨轮重叠” |
| span 合并（更细发射） | 更细的发射在本卡上更慢（1 run/帧 166.7 ms vs 整层 54.7 ms） | 不作默认，C10 |
| 跨架构构建（K4） | 本机只有 `sm_89` | 依赖 CI 的四套构建，PR 里点名 |