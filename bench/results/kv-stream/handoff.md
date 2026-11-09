# KV 流式：交接（压缩后从这里接）

> 临时交接文档，不是产品文档。任务收口后删除；历史与证据仍在 `bench/results/kv-stream/` 与 git。

## 0 一句话状态

`--kv-stream` 已能在 16 GB 卡上跑 262 144 上下文（`rk4v4` 72.8 tok/s、`rk8v4` 26.8、`int8` 16.8），
解码逐位正确、prefill 不动数值、迁移不破数值；**缺的是**：流式下的 CUDA Graph（结构性）、
prompt/decode 双布局与 prefill 参与（流量）、写路径脏页（写回），以及 C7/C9/G5 三项门禁。

**下一步是 P0 剩下的 graph 基线**（用户已定：压缩后开始 P0）。

---

## 0.1 目标与完成条件（`SPEC.md`）

在 16 GB 上跑 262 144 原生上下文，`--kv-stream` 选择启用，且：

| # | 条件 | 现状 |
|---|---|---|
| 1 | `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode | **达成**：251 K 记录 147.9 ms/26.8、236.1 ms/16.8 |
| 2 | on/off 逐 token 一致（ids + logprobs） | 8 K/100 K/183 K 逐位（C1/C2）；262 K 无全驻留对照，物理不可行；C4 用回归金标覆盖 251 K prefill |
| 3 | 120 K 全驻留不差于 109.1 tok/s / 15 632 MiB | **未直接测**；P7 用 100 K 全驻留 118.9 作代理 |
| 4 | 关闭即今天行为 | 守卫表（C11）与关闭路径；无单独具名检查 |
| 5 | 报告真实 tok/s、峰值显存、搬运字节并解释上界差距 | **达成**（表 1/2 + M3/M4） |
| 6 | 251 K 拷贝/轮不退化、报告窗口长度、证明真搬页 | **达成**：54.4 ms/轮、1116.2 MiB/轮、暴露 13.5 ms；窗口长度原先只有最小值，本轮已补每层窗口 |

不在本次范围：其它模型族/架构、UVM、降 KV 质量、prefill 分块流式。

---

## 0.2 工作方式（用户硬性要求，违反会被叫停）

1. **每步先说明要做什么，等用户回话，再动手**；不要连续自作主张。
2. **动手前先搜**：项目里已有 bench、用例、金标、开关、测试；不要造轮子。
3. **动作要审批**；清理/回滚也要先说明。
4. **不租 pod**：测试机 `kpa@192.168.2.13` 一直可用。
5. **不做全量编译**；不编 `ninfer_tests`（约 30 分钟）。只增量单目标 `ninfer`。
   测试批次单独排期（后台跑）并先获批。
6. **不装依赖**（Python 用机器上的 3.13.5，工具纯标准库；CUDA 13.4 已在）。
7. 记录要真：没跑就写“未跑”，不利结果写在最终回复里。
8. 提交后推送并同步测试机；提交用 Conventional Commits。

---

## 0.3 禁地与环境

| 项 | 值 |
|---|---|
| 测试机 | `kpa@192.168.2.13`（debian，16 核/62 GB），RTX 4080 SUPER 16 376 MiB，驱动 615.71.09 |
| 工具链 | CUDA 13.4.92，cmake 3.31.6，`python3` 3.13.5 |
| 我们的树 | `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`，分支 `kv-stream-27b-16g` |
| 构建 | `build-sm89`（Release、sm_89、`BUILD_TESTING=OFF`）；二进制 `untagged+e524df8a` |
| 模型 | `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` |
| oracle | `~/ninfer/ninfer-all/build-sm89/apps/ninfer`（原版 master `f6928e00`，只执行） |
| **禁地** | `~/ninfer/` 原文：“这不是你的目录，禁止任何事项.”——不读不写不建不编；oracle 只由工具执行 |
| GPU | 单卡共用：一次一个作业，启动前查空闲；bench 工具忙卡会拒绝 |
| 时钟 | Windows 与 Linux 差 10–25 秒，`tar` 会报“未来时间戳”；不要依赖 mtime 判断新旧 |

---

## 0.4 分支与最近提交

本地与测试机都在 `kv-stream-27b-16g`，HEAD `28c758d5`。本轮新增：

| 提交 | 内容 |
|---|---|
| `ef00948d` | 报告每层窗口、规划容量、分区次数、形状（形状用标量，**不要放 `std::string`**） |
| `e524df8a` | 删掉参考表里重复的一行 |
| `4c511b31` | C4：251 K prefill 逐位 |
| `24be7202` | C5：4 次活体迁移逐位 |
| `f3561664` | C7 测试（引擎级两请求），已注册，**未跑** |
| `211183d1` | C10/C12 |
| `28c758d5` | C6 的 GPU 半不可复现，写明原因 |

---

## 1 文档权威与要求

| 文档 | 作用 |
|---|---|
| `SPEC.md` | 交付与六条完成条件 |
| `docs/maintainer/kv-streaming-strategy.md` | **唯一计划**：P0–P8、依据、验算、验收 |
| `docs/maintainer/kv-streaming-verification.md` | C/P/K/T/G 具名检查与评分（只有具名检查能得分） |
| `docs/maintainer/kv-streaming-measurements.md` | 参考表（条件 + 数值 + 复现） |
| `docs/maintainer/kv-streaming-fidelity.md` | 上游机制逐条对照（12 条活机制：5 有、2 半、5 缺） |
| `docs/maintainer/kv-streaming-design.md` | 施工前设计与算术（部分已被实现取代） |
| `bench/results/kv-stream/ledger.md` | 历史记录区：决策、作废数字、被否路径 |
| `CONTRIBUTING.md` | 无损、每族、每平台、默认、可比测量、文档当前状态、测试/sanitizer、提交风格 |

**P0–P8 摘要**（验收是契约）：

- **P0**：报告形状/每层窗口/分区次数（**已做**）；graph on/off 8 组（**未做**，需先让流式可建图）。
- **P1**：prompt=uniform、decode=concentrated、下一 chunk 回退；ubatch 256/512/768/1024 四档。
- **P2**：prefill 每 chunk 跑控制器与跨度阶梯；写路径脏行/稳定页镜像。
- **P3**：跨轮重叠——**已实测否决**（+6% 字节 > 可回收 3 ms）。
- **P4**：按 P0 数字决定补图内换页或维持关图并写清代价。
- **P5**：MTP KV 进池（净 −5% 字节），按实测留或否。
- **P6**：一进程两请求（C7，测试已写未跑）；`docs/serving.md` 边界（已写）；`int8` 262 144（已跑）。
- **P7**：span 融合、相邻页合并，按实测决定。
- **P8**：全矩阵、曲线、SPEC 对账、分数表。

**C 表现状**：C1/C2/C3/C4/C5/C10/C11/C12 **PASS**；C6/C9 **半**；C7 **测试就绪待跑**；C8 **未跑**（依赖 P4）。
按 verification 公式：正确性 9/12（含两个半）≈ 30/40，总分约 78 → 仍 C 线（B 需 ≥80 且正确性 ≥90%）。

---

## 2 已落地（代码地图）

| 模块 | 位置 | 作用 |
|---|---|---|
| 布局规划 | `src/core/kv_stream_layout.h` | uniform/concentrated 规划、环/run、每层窗口、`kv_stream_shape_of`、`kv_stream_shape_name` |
| 分区控制器 | `src/core/kv_stream_partition.{h,cpp}` | 迟滞 3/8、冷却 64、1.10、饱和刹车 |
| 设备反馈 | `src/core/kv_stream_signal.{h,cu}` | ready 标志、deadline 采样（`record_deadline<<<1,1>>>`） |
| 跨度调优 | `src/core/kv_stream_span_tuner.h` | 16 样本、0.5%，阶梯在 `text.cpp:2168-2199` 驱动 |
| 流式空间 | `src/models/qwen3_5/program/storage/kv_stream_space.{h,cpp}` | 主机权威、平铺环、run 级拷贝/消费、迁移实验、报告统计 |
| 发射/消费 | `src/models/qwen3_5/execution/text.cpp` | 每层 `upload → wait_ready → attention → consume`；`!prefill` 才 `begin_round`（2168） |
| prefill 接入 | `src/models/qwen3_5/program/prefill.cpp:1369` | 每 chunk `advance()` 发布页 |
| 报告 | `include/ninfer/observability/resources.h`、`apps/cli/run_report.cpp`、`apps/cli/main.cpp` | `stream` 段 + CLI 打印 |
| 守卫 | `src/core/kv_stream_guards.h` | 9 组拒绝组合（C11） |
| 测试 | `tests/test_kv_stream_{layout,feedback,partition,span_tuner,guards}.cpp`、`test_kv_stream_signal.cu` | L1/L2 |
| 引擎级测试 | `tests/models/qwen3_5/test_engine_kv_stream_serial_real.cpp` | C7，`NINFER_TEST_ARTIFACT`，跳过 77 |
| 基准工具 | `tools/bench/run_kv_stream.py` | cases/golden/check/bench/report/compare |

**已实现的机制**（对照 fidelity §11.2 的 12 条活机制）：主机权威 + 块表换页、平铺槽池 + run、
设备侧 ready/deadline、分区控制器（另加 trial/revert）、跨度调优器 = **5 条**；
prompt/decode 分档与主机编排 = **2 条半**；
图内换页 + 跨轮/跨层调度、prefill 参与、脏页/镜像、双布局（运行期）= **5 条缺**。

---

## 3 本轮测量与记录（都在测试机 `bench/results/kv-stream/runs/`）

| 用途 | 记录 | 结果 |
|---|---|---|
| 报告字段 251 K 稳态 | `20261109T181407Z-bench-p0-fields-8176a982` | 72.5 tok/s、54.6 ms、1116.2 MiB/轮；`shape=uniform`、16×2905 页、环 11 run、`layout_generation=0` |
| 报告字段复跑 | `20261109T181857Z-bench-p0-fields-b-8176a982` | PASS |
| C4 251 K prefill | `20261109T183207Z-equiv-e524df8a` | ids 64/64、logprobs sha256 `8a06a1d4…` 与金标相同 |
| C12 183 K 两臂 | `20261109T183714Z-equiv-4c511b31` | resident 106.3 / streamed 85.0；接受长度均 4.0；逐位 |
| C5 150 K 紧池 | `20261109T184516Z-equiv-4c511b31` | `layout_generation=4`；两次 trial + 两次 revert；512 token 逐位 |
| C10 整层 | `20261109T184935Z-equiv-c10-whole-f3561664` | 71.3 tok/s、54.7 ms、暴露 15.9 ms |
| C10 分帧 1 run/帧 | `20261109T185335Z-equiv-c10-framed-f3561664` | 23.6 tok/s、166.7 ms、暴露 1.7 ms；ids/logprobs 与整层相同 |
| C6 极限池 | `/tmp/zero-window.json`（未入库） | `resident_pages=761`、`streamed_pages=0`、upload=0；零窗口只在 L1 |
| 诊断（待删） | `…T180205Z-bench-diag8k`、`…T180255Z-bench-diag150k`、`…T180528Z-bench-diag-clean`、`…T180935Z-bench-diag150k-b` | 崩溃归因用，结论已入文档 |

---

## 4 关键数字（引用前核口径）

- 251 K `rk4v4`：**54.4 ms/轮、72.8 tok/s、1116.2 MiB/轮、暴露 13.5 ms**；窗口 2905 页/层、环 11 run（1408 页）、`shape=uniform`、容量 4096 页。
- 251 K `rk8v4`：147.9 ms、26.8 tok/s、3366.9 MiB/轮；`int8`：236.1 ms、16.8 tok/s、5483.3 MiB/轮、暴露 203 ms。
- 183 K：流式 85.4 tok/s / 46.6 ms / 779.9 MiB；全驻留 106.8 / 37.1；prefill 1476 vs 1481（−0.3%）。
- 150 K 紧池：96.5 tok/s、38.8 ms；150 K 全装下：107.0、0 字节/轮。
- 带宽标定 **25.4 GB/s**；1116.2 MiB → **46.1 ms 拷贝地板**；54.4 − 46.1 = 8.3 ms 未藏住，唯一兑现方式（多留一个 extent）实测 +1.9 ms 更慢。
- 池启动后只剩 53.5 MiB；`kDefaultKvStreamHeadroomBytes` 64 MiB。

---

## 5 已作废 / 不要再试

- 跨轮预取（多留一个 extent）：+5.6% 字节、净 +1.9 ms 更慢，已结案。
- 深环 2.0×：56.5 vs 54.6 ms，已从出货路径移除（`7734165e`），`plan_kv_stream_lookahead` 仅保留给未来。
- 分帧作为默认：逐位正确但 3 倍慢（本次 C10 再次确认）。
- `--kv-stream-ring-ratio` 开关：已在 `6a44fc25` 移除；C6 的 GPU 半因此不可复现。
- 环从 2× 改 1.10× 的“字节模型”决策：只看字节会掉前视（M-13/M-17）。
- 用 UVM、降 KV 精度换容量：所有者已否决。
- 作者三个 prefill 提交（`b2ec300a2`/`3552d1b7d`/`46c1778f7`）：strategy §9.4 判定不移植（本机 prefill 只慢 0.3%、暴露 0、staging 已 per-run）；**用户看板曾写“移植”，这是未决分歧点**。

---

## 6 未完成的硬缺口（按用户可感知价值排序）

| # | 缺口 | 收益 | 入口 |
|---|---|---|---|
| 1 | `--embedding-residency host` 在流式下的效果 | 释放 388.4 MiB → +355 页/层 → 每轮约 −407 MB（1116→709 MiB，拷贝 46→28 ms），**从未在流式下量过** | 先 A/B 量，再决定是否做成流式默认 |
| 2 | P5：MTP KV 进池 | 每轮约 −5% 字节 | `startup.cpp:162-164`、`mtp_layers=1`；需独立驻留角色 |
| 3 | P0 graph（**用户指定的下一步**） | 251 K 上界 0.7%/轮；≤130 K 旧记录 −3%（未复测） | `startup.cpp:1507` 的 `&& !inputs.features.kv_stream` |
| 4 | P1 双布局 + ubatch | prefill 流量（Y3），时间不变差 | `entering_decode_layout` 硬编码 false（`kv_stream_space.cpp:446`）、`text.cpp:2168` |
| 5 | P2 prefill 适配 + 脏页 | `[partition]` 进 prefill；回写 ≤ 脏页量 | `prefill.cpp:1369`、`advance()` 无条件 D2H（`kv_stream_space.cpp:838-870`） |
| 6 | C7 / C9 / G5 | 门禁，不是产品能力 | 测试树批次（30 分钟，后台、需批准） |
| 7 | P8 全矩阵、曲线、SPEC 对账、分数表 | 交付证据 | 依赖上面几项 |

---

## 7 下一步：P0 的 graph 基线（压缩后从这里接）

### 7.1 现状

`src/models/qwen3_5/program/planning/startup.cpp:1507`：

```cpp
impl->use_cuda_graph = inputs.use_cuda_graph && !inputs.features.kv_stream;
```

流式一开就关图，所以 P0 的“graph on”在流式下**根本不存在**。P0 与 P4 共用同一处改动。

### 7.2 要做的事

1. 让流式下可建图（先做成诊断开关，避免改默认）：把 `&& !kv_stream` 换成“流式时也允许捕获”，
   并确认每层的 `wait_ready/consume` 在捕获区内的语义（`cudaStreamWaitEvent` 在捕获中合法，
   `structured_round.cpp:136` 已有图内等外部事件的先例）。风险点：块表发布与 `ready_` 事件的
   顺序必须仍在图内正确；若捕获导致数值或时序不对，P4 就结案为“维持关图”，并写清代价。
2. 工具侧：给 `run_kv_stream.py` 的 case 加 graph on/off 变体（`--no-cuda-graph` 已有 CLI 旗标；
   工具需要能把旗标分别给两臂）。
3. 跑 8 组：183 K/251 K × rk4v4/rk8v4，各 on/off，落记录。
4. 判据：P4 的“251 K 不慢于 72.8 tok/s”；C8 的“graph vs eager 512/512 逐位”；Y2 的“251 K 不超过 +1%（很可能 −1…−3%）”。

### 7.3 不要做的事

- 不要先改默认；先诊断开关 + 8 组数据。
- 不要为 graph 去全量编译测试包。
- 不要碰 `~/ninfer`。

---

## 8 命令速查（照抄改路径即可）

```bash
# 同步（本地 push 之后；构建期间不要再改源码）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && git fetch -q origin && git reset --hard -q origin/kv-stream-27b-16g'

# 增量构建（只单目标；ccache 不可靠，别加）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && cmake --build build-sm89 --target ninfer -j 16'

# 251 K 稳态（约 4 分钟）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && python3 tools/bench/run_kv_stream.py \
  --binary build-sm89/apps/ninfer \
  --oracle ~/ninfer/ninfer-all/build-sm89/apps/ninfer \
  --model ~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  bench --case stream-251k-steady --reps 1 --tag <tag>'

# 等价/回归
… check --case equiv-183k | stream-251k | stream-150k-tight

# 读报告字段
python3 - <<'EOF'
import json,glob
p=sorted(glob.glob("bench/results/kv-stream/runs/*<tag>*/stream-251k-steady/streamed/rep-01/report.json"))[-1]
s=json.load(open(p))["stream"]
print(s.get("shape"), s.get("layout_generation"), s.get("resident_pages"), s.get("ring_runs"))
print(min(s["layer_resident"]), max(s["layer_resident"]))
EOF

# 测试树（不要顺手跑；一次批次、后台、先获批）
cmake -S . -B build-sm89-tests -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_TESTING=ON
```

**单文件语法检查**（比测试树便宜，但不能替代运行）：`g++ -std=c++20 -fsyntax-only -Iinclude
-Isrc -Itests -I/usr/local/cuda/include <file>.cpp`。

---

## 9 踩过的坑（别再犯）

1. **跨库结构体加 `std::string` → 流式启动即 SIGSEGV，无日志**。`StreamedKvStats` 由模型库按值
   返回给 runtime/CLI；改成标量 `shape`（0/1/2/3）后正常。定位方法是“干净源码同用例 PASS vs 改动版崩”。
2. **`active_pages` 在请求结束时被 reset 清 0**，报告里会读成 0。形状改为按**规划容量**判定，
   字段在 0 时省略，不打印“0 页”。
3. **`pgrep -f` 自匹配**：清理残留时把执行命令的 shell 自己杀掉了。用 `[r]un_kv_stream[.]py`。
4. **`tar` 跨 Windows/Linux 的时间戳在未来**，构建依赖判断会受影响；改完公共头先 `touch` 再编。
5. **测试包全量构建约 30 分钟**（`ninfer_tests` 一个二进制含所有用例）；不要为单个测试触发。
6. **bench 工具会拒绝忙卡**；中断 SSH 客户端不会终止远端进程，要用 pgrep 确认。
7. 结论必须落到记录：`runs/` 未入库，文档写“结论 → 记录名”；不要引用不存在的目录。

---

## 10 文件清单（本轮工作树）

已提交：`ef00948d`、`e524df8a`、`4c511b31`、`24be7202`、`f3561664`、`211183d1`、`28c758d5`。
未提交（测试机/本地都可能存在）：`bench/results/kv-stream/runs/` 下的诊断记录（待删）。
待办：`/tmp/zero-window.json` 未入库；C7 测试未运行；G5 未跑。
