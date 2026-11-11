# 交接文档：KV 流式 + 前缀复用（当前状态）

本文件是**接手第一份要读的东西**：现在到哪一步、哪些已验证、哪些没做完、怎么复跑、怎么回退。
历史决策与作废数字在 `ledger.md`；单个缺陷的证据档案在 `layout-rename-defect.md`。

## 0. 当前状态（一句话）

前缀复用在流式下**已实现并逐位验证**；一个"布局改名不搬字节"的**正确性缺陷已修复**，并据此**移除了
请求中途换布局与自适应控制器**（唯一收益 0.2%，实测为零）；**几何不变性（C14）已成自动化用例**。
特性层验证全绿，见 §1.3。

| 项 | 值 |
|---|---|
| 分支 | `kv-stream-27b-16g` |
| HEAD | `bf0554da`（末次提交见 `git log`） |
| 回退点（换形在、已修好） | tag `kv-stream/layout-switch-before-removal`、分支 `kv-stream/preserved-layout-switch`（都指向 `c8bdb380`） |
| 测试机 | `kpa@192.168.2.13`（debian），仓库 `~/ninfer-fork/ninfer-all-adaptive-kv-streaming` |
| 模型 | `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` |
| 构建 | `build-verify`（Release、`BUILD_TESTING=ON`、sm_89，**全新目录**，见 §4） |

**测试机状态**：源码树与 Windows 端**同一提交、零未提交改动**（`git status --porcelain` 只剩未跟踪文件）。
上一轮查 bug 时留在树里的 8 处诊断补丁（`NINFER_ZERO_ARENA`、`NINFER_STATE_DEBUG`、
临时放开 device 槽守卫、`printed<4096` 等）**已全部清除**：`git reset --hard` 后 grep 这些标记为 0，
shell 与 `~/.bashrc` 里也没有任何 `NINFER_*` 变量。旧的 `build-sm89`、`build-sm89-tests`
（含补丁时代的对象文件）**已删除**，不要在它们上面取证据。

## 1. 提交与实测（每条都能复验）

### 1.1 保留/复用（C13）

| 提交 | 内容 | 证据 |
|---|---|---|
| `03879140` | 流式请求结束保留上下文（frontier + 一张 **host** StateImage，KV 不复制），下一请求按前缀复用 | 第二请求 `reuse=1 base=4614`、`cache 4,614 (100.0%, private endpoint)`、TTFT 2 ms |
| `8561809e` | `--kv-stream` **默认保留并复用**（去掉开发旗标）；端点固定 host shelf；guard/启动拒绝 device 槽 | `--no-prefix-reuse` 仍可关；两请求 ids 与冷跑一致 |
| `884cfc66` | serve 不再静默覆盖 `--device-state-slots`，改为带修复 flag 的拒绝 | `--device-state-slots 1` 实测报错退出 |

### 1.2 布局缺陷与移除（C14）

| 提交 | 内容 | 证据 |
|---|---|---|
| `c8bdb380` | **修正确性**：`repartition` 改名必须搬字节 | 几何 1413 vs 1372 两臂逐位一致（修复前 `-0.1798` vs `-0.8109`） |
| `4844f2a6` | 缺陷档案 `layout-rename-defect.md`（含定性：架构选择错误，非笔误） | 8 组证据、7 条排除、复现配方、回退命令 |
| `68941676` | 池在**启动时**按 decode 形一次规划；`enter_layout`/`evaluate_partition` 恒 false | 六组几何（`resident` 1413→40、`ring` 24→35）答案逐位相同 |
| `0b508bad` | **删除换形与控制器**：`kv_stream_partition.{h,cpp}`、`repartition`/`enter_layout`/`evaluate_partition`、`advance` 加宽分支（改为显式报错）、试探/冷却状态、`kv_stream_ring_peak_ratio`、报告字段 `stream.layout_generation` | 见 §1.3 |
| `c28abd33`…`bf0554da` | 两个真机测试改成自说明：`serial_real` 的前沿按请求算术（`prompt+generated−1`）而非常量；几何用例用**两个 headroom**（1/2 GiB）造两种池 | §1.3 |

### 1.3 特性层验证（全新树 `build-verify`，全部通过）

| 检查 | 结果 |
|---|---|
| `ctest -R kv_stream`：`layout` / `feedback` / `span_tuner` / `signal` / `guards` / `ninfer_kv_stream_test` | 全部 PASS |
| `kv_stream_serial_real`（190 K，冷跑 + 保留两臂） | **PASS 749 s**（此前从未在测试树跑过；冷跑臂 `ids identical`、`rounds 7→14`、字节 oracle `736 checks / 0 mismatches`） |
| `kv_stream_geometry_real`（C14，新增） | **PASS 86 s**：`2889 vs 1866 pages/layer, 11 vs 20 ring runs` → `8 ids and 6 logprobs identical` |
| `geometry_driver`：预留 0 / 64 / 512 MiB | `resident` 1413 / 1372 / 1075、`ring` 24 / 24 / 26 → 三组 `ids equal=1 logprobs equal=1` |
| `reuse_driver` | `retained=1 frontier=4614` → `reused=4614`、`equal_ids=1`、`OK` |
| `check --suite equiv` | **4/4 PASS**（8 K 149.6、100 K 119.2、183 K resident 106.3 / streamed 100.2 tok/s；streamed `upload 356.6 MiB/轮`、`exposed 0.1 ms`），campaign `20261111T073620Z-equiv-verify-clean-bf0554da` |
| 251 K `bench --case stream-251k --reps 2` | **PASS**：`95.0 tok/s`、`41.4 ms/轮`、`upload 689.1 MiB/轮`、`exposed 1.0 ms`；与删除前同配方测得的 95.0 / 41.4 / 689.1 **完全一致**（性能中性）。campaign `20261111T074344Z-bench-verify-clean-bf0554da` |
| 8 K CLI 冒烟（`--report-json`） | `shape=all-resident`、`ring_runs=0`、报告 `stream` 里**已无** `layout_generation`，其余 25 键齐全，exit 0 |

## 2. 测试层次与界限（别把两层混用）

1. **特性判定（流式开）**——唯一能判断"功能对不对"的一层：上面 §1.3 的 8 个 ctest 用例、
   `check --suite equiv`、两个手工驱动、251 K bench、CLI 冒烟。
2. **共享面回归（流式关，老参数）**——全量 ctest（286 个）的其余部分：算子/内核 131、qwen3_5 模型 34、
   runtime/artifact/Python 工具。**只有 2 个测试文件打开 `kv_stream`**（`serial_real`、`geometry_real`），
   所以全量 ctest **不是**"流式下的全量"，它证明的是：删掉 `TransferStats::layout_generation`、
   `text.cpp` 的调优器复位、`prefill.cpp` 的调用、`kv_stream_partition` 之后**共享代码没被踩坏**。
3. **不存在的第三层**："老测试 × 流式开"的矩阵。现有测试把参数写死，要做必须先把测试参数化（新工程）。
4. **已知无覆盖**：`grep -rn "layout_generation" tests/` = 0 ⇒ 报告字段的删除**没有测试保护**，
   只能由 §1.3 的 CLI 冒烟兜住。不要声称它被测试覆盖。

## 3. 未完成事项

### U1 251 K 的归因（可选，需批准）
- 现状：`95.0 tok/s`；同配方历史记录 ~96.5 tok/s（配置已不可比），差 ~1.5%，记录区无相邻基线。
- 做法：与 tag `kv-stream/layout-switch-before-removal` 做**交错** A/B（第二套 build = 一次全量编译，
  约 35–50 分钟）。判据：轮时间中位差 ≤ +1%、prefill tok/s ≥ −1%；否则考虑分配式槽位方案。
- 本轮已证：删除前后同构测得的 251 K 数字完全一致（§1.3），所以 U1 只影响"与更早记录的比较"。

### U2 跨格式/长会话（产品面，未跑）
- `rk8v4` 251 K 稳态（历史 `147.9 ms`）、`int8` 262 144（历史 `236.1 ms`、16.8 tok/s，C3）**本轮未复跑**。
- 192 K/251 K 多轮对话、serve 两请求 ids 对照（C13 的 serve 证据来自上一轮，未在本树复跑）。
- serve 与四个模型族（Qwen3.5-core / Flash-Next / Gemma 4 / Flint / DeepSeek-V4.1）交叉冒烟。

### U3 合规项（与本次删除无关，仍未做）
- `compute-sanitizer` racecheck/synccheck（`kv_stream_signal.cu`）；
- 跨架构构建确认（`sm_70/75/80/86/89/120a`）；
- 测量口径的 `--interleave`（bench 目前逐变体顺序跑）。

## 4. 环境与常用命令

```bash
# 同步（本地 push 后）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && git fetch -q origin kv-stream-27b-16g && \
  git merge --ff-only -q FETCH_HEAD && git log --oneline -1 && git status --porcelain | grep -v "^??"'

# 全新构建（Release + 测试树 + sm_89；-j 16 = 16 核；非登录 shell 必须显式给 nvcc）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  cmake -S . -B build-verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && \
  CCACHE_DISABLE=1 cmake --build build-verify -j 16'

# 特性层
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  NINFER_TEST_ARTIFACT=$HOME/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  ctest --test-dir build-verify -R "kv_stream" --output-on-failure'
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  python3 tools/bench/run_kv_stream.py --binary build-verify/apps/ninfer \
    --model $HOME/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
    check --suite equiv --tag <tag>'

# 共享面回归（串行，真实模型用例同时跑会抢卡）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  NINFER_TEST_ARTIFACT=... ctest --test-dir build-verify -j 1 --output-on-failure'

# 手工驱动（必须与库同一次构建重新链接；见 tools/bench/kv_stream/README.md）
cd build-verify && LIBS=$(ninja -t commands apps/ninfer | tail -1 | sed 's/^.*-o apps\/ninfer //')
/usr/bin/g++ -std=c++20 -O2 -I../include -I../src -I/usr/local/cuda/include \
  -c ../tools/bench/kv_stream/geometry_driver.cpp -o /tmp/geometry_driver.o
eval "/usr/bin/g++ -O3 -DNDEBUG /tmp/geometry_driver.o -o /tmp/geometry_driver $LIBS"
LP_RESERVE2=512 /tmp/geometry_driver <artifact>      # 期望 ids equal=1 logprobs equal=1
/tmp/reuse_driver <artifact>                          # 期望 OK
```

**日志只写 `/tmp/kvlogs/`，不要落在 `~`。** 注意 `/tmp` 是 tmpfs：`mv` 到别的文件系统是复制+删除，
后台进程的 fd 会写到已删除的 inode 上（本轮因此丢过一次 ctest 输出，正确做法是直接重定向到目标路径）。

## 5. 决策与回退

- **换形已移除**：收益实测为零（decode +0.2%、prefill 上载 −1.6% 且该时间本被掩盖），代价是"换形即改名"。
  详见 `layout-rename-defect.md` §8。
- **回退**（如需恢复换形通道）：
  ```bash
  git checkout kv-stream/layout-switch-before-removal -- \
    src/core/kv_stream_layout.h src/core/kv_stream_partition.h src/core/kv_stream_partition.cpp \
    src/models/qwen3_5/program/storage
  ```
- 若将来测出 decode 退化 > 2%，备选是**分配式槽位**（run 粒度 ID + 发布表）：换形只搬跨边界页，
  比"整窗回写 host"快；这是唯一预留方案。

## 6. 硬性工作约定（沿用）

- 动手前先说明；先搜项目已有基建；不租 pod；增量构建优先，全量构建须明确批准。
- 源码只经 `commit → push → git pull` 同步到测试机；**不要手工改测试机源码**。
- 提交/推送后同步测试机；Conventional Commits；结论要真（没跑写"未跑"）。
- 任何"逐位"结论必须带几何指纹（`resident_pages`/`ring_runs`/`layer_resident`）。
- 同一张卡上**一次只跑一个 GPU 任务**；全量 ctest 用 `-j 1`。
