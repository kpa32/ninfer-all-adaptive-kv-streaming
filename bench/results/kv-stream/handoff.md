## Goal

在 16 GB 单卡上把 Qwen3.8-27B 的 262 144 上下文做成可交付的 `--kv-stream` 能力：`rk8v4` 与 `int8`
都能启动并稳定 decode；同一格式下流式开/关 greedy ids 与 logprobs 逐位一致；关闭即今天行为；
报告真实速度、显存与搬运字节。当前阶段：**压缩上下文后从 P0 的 CUDA Graph 基线继续**。

## Constraints & Preferences

**用户的工作方式（违反会被叫停）**
- 每步先说明要做什么，等用户回话，再动手；不要连续自作主张。
- 动手前先搜：项目里已有 bench、用例、金标、开关、测试，不要造轮子。
- 动作要审批；清理/回滚也先说明。
- 不租 pod（测试机一直可用）；不装依赖；不做全量编译，不编 `ninfer_tests`（约 30 分钟）。
- 只增量构建单目标；测试批次单独排期、后台跑、先获批。
- 提交后推送并同步测试机；Conventional Commits。
- 结论要真：没跑写“未跑”，不利结果写在最终回复里，不藏。

**项目硬规则（`CONTRIBUTING.md`）**
- 无损优先：优化不得改输出；逐位是判据（ids 不够，logprobs 也要逐位）。
- 每个模型族、每个平台（`sm_70/75/80/86/89/120a`，CUDA 12.x/13.x）。
- 默认即最快无损配置；产品模式走 CLI/服务选项，env 只能做诊断/A-B。
- 可比测量：同主机/同构建/同产物/同 prompt、交错运行、无其它负载。
- 文档只描述当前状态（无日期日志/尝试过程）；历史放 `bench/results/kv-stream/ledger.md`。
- 新 kernel：独立 oracle + 一次 `compute-sanitizer` racecheck 与 synccheck；测试注册进 CTest。

**SPEC 六条完成条件**
1. `rk8v4` 与 `int8` 的 262 144 都能启动并稳定 decode；
2. 同格式 on/off 逐 token、logprob 一致；
3. 120 K 全驻留不差于 109.1 tok/s / 15 632 MiB；
4. 关闭即今天行为；
5. 报告真实 tok/s、峰值显存、搬运字节并解释与上界差距；
6. 251 K 每轮拷贝/轮时间不退化、报告窗口长度、证明 decode 真搬页。

不在范围：其它模型族/架构、UVM、降 KV 质量、prefill 分块流式。

**禁地**：测试机 `~/ninfer/` 原文“这不是你的目录，禁止任何事项.”——不读不写不建不编；
其中的原版二进制只由 bench 工具**执行**。

## Progress

### Done

**P0 报告字段（已落地并提交）**
- `StreamedKvStats` 增加每层窗口 `layer_resident`、规划容量、分区次数 `layout_generation`、形状 `shape`；
  `window_bytes` 改为**按层求和**（原来 `最窄窗口 × 层数`，集中布局下偏小）；CLI 打印布局行。
- 形状用**标量** 0/1/2/3（未跑/全驻留/均匀/集中），理由见 Key Decisions。
- 实测 251 K：`shape=uniform`、16 层窗口全为 2905 页（容量 4096）、环 11 run、`layout_generation=0`；
  54.6 ms/轮、72.5 tok/s、1116.2 MiB/轮（基线 54.4/72.8，±0.5%）。

**C 表（verification 的具名检查）**
- C1/C2/C3/C11：此前已 PASS。
- **C4 PASS**：`check --case stream-251k`（`20261109T183207Z`），ids 64/64 与金标相同，
  logprobs 的 sha256 `8a06a1d4…` 与金标一致；金标 `oracle: self`（原版在 262 K 起不来），属回归口径。
- **C5 PASS**：`check --case stream-150k-tight`（`20261109T184516Z`），流式臂发生 4 次迁移
  （`layout_generation=4`，两次 `[partition] trial start` + 两次 `revert`），512 token 逐位相同。
- **C10 PASS**：`NINFER_STREAM_BOUNDED=0`（`184935Z`：71.3 tok/s、54.7 ms、暴露 15.9 ms）与
  `=1`（`185335Z`：23.6 tok/s、166.7 ms、暴露 1.7 ms）的 ids 与 logprobs 完全相同。
- **C12 PASS**：`equiv-183k`（`20261109T183714Z`）两臂接受长度均 4.0、ids 逐位。
- **C6 半**：`--kv-stream-ring-ratio` 已在 `6a44fc25` 移除，GPU 判据无法复现；用
  `--kv-headroom-mib 3072` 验证窗口收缩到 761 页/层可跑（`streamed_pages=0`、upload=0）。
- **C7 测试已写**：`tests/models/qwen3_5/test_engine_kv_stream_serial_real.cpp`（一进程两请求，
  190 K prompt，`NINFER_TEST_ARTIFACT` 控制跳过），注册进 `tests/models/qwen3_5/tests.cmake`；
  语法检查通过，**未运行**。

**P0 graph 结案（已落地并提交）**
- 诊断：`startup.cpp` 曾加 `NINFER_STREAM_CAPTURE=1` 允许流式捕获，跑一次 `stream-251k`
  （`20261109T191653Z`）→ **380 µs 内失败**：`KV logical page is outside the address space`。
- 根因：`graphs.cpp reserve_capture_rows` 为每个捕获行 `create_active(1,…)` 后取第 0 页，
  而流式地址 `commit_activation` 把 `page_count = 0`（页由 `StreamedKVSpace` 发布）。
- 收益实测（183 K 全驻留，`equiv-183k` resident）：graph on 37.1 ms vs off 37.7 / 37.8 ms
  = **0.65 ms/轮（1.75%）**；251 K 稀释到约 1.3%；两次 off 的 512 ids 与 on 逐位相同（C8 PASS）。
- 上限解释：CPU 提交 1.5–2 ms 被 37 ms 的 GPU 工作掩盖，graph 只消 kernel 间隙（≈500–700 × 1.3 µs）。
- 结论：维持关图，开关已删（`eae816e9`）；measurements 表 5、verification C8/P9。

**提交（本地与测试机一致，HEAD `eae816e9`）**
`ef00948d` 报告字段 · `e524df8a` 表格修复 · `4c511b31` C4 · `24be7202` C5 ·
`f3561664` C7 测试 · `211183d1` C10/C12 · `28c758d5` C6 · `f408528b` 交接文档 ·
`7a49bde1` 格式 · `823c361a` 捕获诊断 · `25173bf0` bench `--extra` · `eae816e9` graph 结案。

### In Progress

- 无。下一步是 `--embedding-residency host` 在 251 K 流式下的 A/B（见 Next Steps），尚未开始。

### Blocked

- **C7/C9/G5**：需要一次 `BUILD_TESTING=ON` 的测试树构建（约 30 分钟），须单独获批并后台跑。
- **K4 跨架构**：本地只有 `sm_89`，未做。

## Key Decisions

- **跨轮重叠否决**：多留一个 extent 换 7 ms 暴露，实测 +5.6% 字节、净 +1.9 ms 更慢（ledger 两条决策记录）。
- **深环 2.0× 否决**：56.5 vs 54.6 ms，已从出货路径移除（`7734165e`）；`plan_kv_stream_lookahead` 只留给未来。
- **分帧不作默认**：逐位正确（C10），但每轮 54.7 → 166.7 ms，本机三倍慢；阶梯选整层。
- **`--kv-stream-ring-ratio` 移除**（`6a44fc25`）：扫描可能规划出错误切分；C6 的 GPU 判据因此写为“半”。
- **作者三个 prefill 提交不移植**（strategy §9.4）：本机 prefill 只慢 0.3%、暴露 0、staging 已 per-run。
  **注意：用户看板曾写“移植”，这是唯一未决分歧点**，动手前必须确认。
- **形状用标量而非 `std::string`**：`StreamedKvStats` 跨模型库/runtime 按值返回，加 `std::string`
  导致流式启动即 SIGSEGV 且无日志；定位方式是“干净源码同用例 PASS vs 改动版崩”。
- **形状按规划容量判定**：请求结束会 reset 前沿（`active_pages` 归 0），按瞬时前沿判形状会把
  uniform 读成 all-resident；字段为 0 时不输出。
- **C4/C5 用 `oracle: self` 金标**：原版无 `--kv-stream`、在 262 K 起不来；文档已注明是回归口径，
  不是与原版的等价。
- **不做图内换页**：graph 在这条路径上只值 0.65 ms/轮（1.75%@183 K，1.3%@251 K），上限就是
  kernel 间隙；要让它覆盖流式，图内必须含每层 copy issue + `consumed` record + 每次 replay
  更新 memcpy 源（251 K 的环 11 run < 一层 46 run ⇒ `ahead=0`，图外 issue 全部层会死锁）。
- **用户可感知价值排序**（据实）：减字节（embedding-residency 388 MiB → 每轮约 −407 MB，未量）
  > P5 MTP 池（−5% 字节）> P0 graph（251 K 上界 0.7%/轮）> P1 双布局/P2 脏页（今天被计算掩盖）。
- **不租 pod**：`kpa@192.168.2.13` 一直可用，租机是错误建议（已撤回）。

## Next Steps

1. **`--embedding-residency host` 在 251 K 流式下的 A/B（可感知收益最大项）**
   - 现状：embedding 常驻占 388.4 MiB；流式池每轮 1116.2 MiB、46.1 ms 拷贝地板、暴露 13.5 ms。
   - 预期：释放 388.4 MiB → +355 页/层 → 每轮约 −407 MB（1116 → 709 MiB），拷贝 46 → 28 ms。
   - 做法：先确认该选项在流式下的守卫与交互（`kv_stream_guards.h`、CLI 帮助、`serve_options`），
     再用 bench 两臂 A/B（默认 vs `--embedding-residency host`），`--extra` 透传；
     判据：ids 逐位（C4 口径）+ 每轮字节/时间/暴露 + 池容量变化。
2. **C 表收尾**（发布前，非产品能力）：C9（VERIFY 逐页计数进 `StreamedKvStats`）、G5/C7 的
   测试树批次（`BUILD_TESTING=ON`，约 30 分钟，需单独获批）。
3. 之后按可感知收益：P5 MTP KV 进池（约 −5% 字节）→ P1 双布局 / ubatch 适配。

## Critical Context

**环境**
- 测试机 `kpa@192.168.2.13`（debian，16 核/62 GB），RTX 4080 SUPER 16 376 MiB，驱动 615.71.09。
- CUDA 13.4.92（`nvcc` 不在非登录 PATH，用 `/usr/local/cuda/bin/nvcc`），cmake 3.31.6，`python3` 3.13.5。
- 仓库 `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`，分支 `kv-stream-27b-16g`。
- 构建 `build-sm89`（Release、sm_89、`BUILD_TESTING=OFF`）；二进制 `untagged+eae816e9`（与源码一致）。
- 模型 `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`；
  oracle `~/ninfer/ninfer-all/build-sm89/apps/ninfer`（原版 master `f6928e00`，只执行）。

**关键数字（引用前核口径）**
- 251 K `rk4v4`：54.4 ms/轮、72.8 tok/s、1116.2 MiB/轮、暴露 13.5 ms；窗口 2905 页/层、环 11 run。
- 251 K `rk8v4`：147.9 ms、26.8 tok/s、3366.9 MiB/轮；`int8`：236.1 ms、16.8、5483.3、暴露 203 ms。
- 183 K：流式 85.4 tok/s / 46.6 ms / 779.9 MiB；全驻留 106.8 / 37.1；prefill 1476 vs 1481（−0.3%）。
- 150 K 紧池 96.5 tok/s / 38.8 ms；150 K 全装下 107.0、0 字节/轮；100 K 全驻留 118.9。
- 带宽标定 25.4 GB/s；1116.2 MiB → 46.1 ms 拷贝地板；54.4 − 46.1 = 8.3 ms 未藏住。
- 池启动后剩 53.5 MiB；流式默认 headroom 64 MiB。
- graph（183 K 全驻留）：on 37.1 ms / off 37.75 ms = 0.65 ms/轮（1.75%）；251 K 推算 1.3%。

**命令**
```bash
# 同步（本地 push 后；构建期间不改源码）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && git fetch -q origin && git reset --hard -q origin/kv-stream-27b-16g'
# 增量单目标
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && cmake --build build-sm89 --target ninfer -j 16'
# 251 K 稳态（约 4 分钟）
ssh kpa@192.168.2.13 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && python3 tools/bench/run_kv_stream.py \
  --binary build-sm89/apps/ninfer --oracle ~/ninfer/ninfer-all/build-sm89/apps/ninfer \
  --model ~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  bench --case stream-251k-steady --reps 1 --tag <tag>'
# 单文件语法检查（不能替代运行）
g++ -std=c++20 -fsyntax-only -Iinclude -Isrc -Itests -I/usr/local/cuda/include <file>.cpp
```

**坑**
- 跨库结构体不要加 `std::string`（SIGSEGV、无日志）。
- 请求结束 `active_pages` 归 0；形状按容量判。
- `pgrep -f` 会自匹配；用 `[r]un_kv_stream[.]py`。
- Windows/Linux 时钟差 10–25 秒，`tar` 会报“未来时间戳”；改公共头先 `touch` 再编。
- 中断 SSH 客户端不会终止远端进程；用 pgrep 确认。bench 工具忙卡会拒绝。
- 全量测试包 30 分钟；不要为单个测试触发。

**C 表现状**：C1/C2/C3/C4/C5/C8/C10/C11/C12 PASS；C6/C9 半；C7 待跑。
P 表 P1–P9 全 PASS。按 verification 公式：正确性 10/12 ≈ 33.3/40，总分约 81；但正确性 83%
未达 B 的 90%、G5 未过，所以仍是 **C 线**。跑完 C7 + G5（同一次测试树批次）即 11/12（92%）
且门禁 100%，那时才是 B。

<read-files>
SPEC.md
docs/maintainer/kv-streaming-strategy.md
docs/maintainer/kv-streaming-verification.md
docs/maintainer/kv-streaming-measurements.md
docs/maintainer/kv-streaming-fidelity.md
bench/results/kv-stream/ledger.md
bench/results/kv-stream/README.md
src/models/qwen3_5/program/planning/startup.cpp
src/models/qwen3_5/program/storage/kv_stream_space.cpp
src/models/qwen3_5/program/storage/kv_stream_space.h
src/core/kv_stream_layout.h
src/models/qwen3_5/execution/text.cpp
src/models/qwen3_5/program/prefill.cpp
tools/bench/run_kv_stream.py
tests/models/qwen3_5/test_engine_kv_stream_serial_real.cpp
</read-files>

<modified-files>
bench/results/kv-stream/handoff.md
docs/maintainer/kv-streaming-verification.md
docs/maintainer/kv-streaming-measurements.md
include/ninfer/observability/resources.h
src/core/kv_stream_layout.h
src/models/qwen3_5/program/storage/kv_stream_space.h
src/models/qwen3_5/program/storage/kv_stream_space.cpp
src/models/qwen3_5/program/program.cpp
apps/cli/run_report.cpp
apps/cli/main.cpp
tests/test_kv_stream_layout.cpp
tests/models/qwen3_5/test_engine_kv_stream_serial_real.cpp
tests/models/qwen3_5/tests.cmake
</modified-files>
