# CUTLIST：阶段 1 删减清单 v2（待所有者逐项批准）

状态：**提案，未执行**。分支 `target/4080s-27b` 到这份文件为止只新增文档，未删除/修改任何源码、
测试、脚本、构建文件。

## 0. 你这次给的方向（已并入 `SCOPE.md`）

1. **只保留 RTX 4080 SUPER = sm_89** 的支持（确认：4080 SUPER 是 compute capability 8.9 / Ada）；
   sm_70 / sm_75 / sm_80 / sm_86 / 120a 全部删掉以减少编译时间；**以后项目成功了再对比加回来**。
2. **Flash-Next 不要**，和 Qwen 27B 无关的模型都不要。
3. **多卡不要**，但"如果代码耦合极深、牵涉大就别动"。
4. **从模块相对独立、简单的开始**。
5. 删减仍需你**逐项批准**；一条一个 commit；测试要自清理（跑完不留进程、不留临时文件）。

## 1. 先回答两个问题

### 1.1 "是 89 吗？"

是。RTX 4080 SUPER = `sm_89`（Ada，80 SM）。`build.sh` 现在就是
`-DCMAKE_CUDA_ARCHITECTURES=89`；机器上的二进制只含 89 镜像。

但要分清两件事：

- **构建只编 89** —— 已经成立，不需要改。
- **源码里仍带着别的卡的实现** —— 这些文件在当前 89 构建里**照样被编译**（只是永远不会被调用）。
  例：`src/ops/linear/sm70/sources.cmake` 被无条件 `include`，5 个 `.cu` 一起编；
  `linear/nvfp4`(19 个 `.cu`)、`linear/fp8`(15)、`linear/t2`(6)、`linear/stq`(1) 同样无条件编。
  所以"删别的卡/别的格式"是真的能省编译时间，不是只改一个变量。

### 1.2 "Flash-Next 和流式支持代码是什么关系？"

- **Flash-Next**（`Qwen3.8-Flash-Next`，代码 `src/models/qwen4_exp`）是**另一个模型**：
  15 277 行模型代码 + 3 680 行自己的 Engine core + 7 151 行测试。你不需要它，可以删。
- 但它内部有一条 **`--kv-resident`** 的实现：**KV 整体放在 pinned host 内存里，显存里只放一个小的
  slot 池，按层从 host 往显存搬**，而且搬运是在 CUDA Graph 里执行的。
  这是本仓库**唯一已经在跑**的"host 权威 KV + 显存小池 + 按层搬运"的例子，是 27B 流式的参考实现。
- 真正可复用的内核是 `src/ops/kv_stream/kv_stream.cu`（**410 行**）。它是**算子**，不是模型代码，
  和 Flash-Next 模型本身分开。27B 的第二阶段会直接调用它（或按它的设计写）。
- 结论：**删模型，留算子**。`ops/kv_stream` 410 行、目前没有调用者（等第二阶段的 27B 用），
  删了它等于把最难的那部分（GPU 内 slot 表 + CLOCK 驱逐 + 可捕获的 host→device 拷贝）从头再写一遍。
  如果你坚持连它也删，我就按"第二阶段自己写"来做，但那会多出几周工作量。

## 2. 分组：从简单/独立 → 复杂/耦合

| 组 | 内容 | 独立性 | 体量 | 编译单元 |
|---|---|---|---|---|
| **A1** | sm_70 / sm_75 / 120a 的实现（卡死了也不会被 89 调用） | **高**（纯删除 + 少量 CMake/宏） | ~5 000 行 | ~25 `.cu` |
| **A2** | Flash-Next（`qwen4_exp`）+ 它独占的 4 个算子 | **高**（qwen3_5 不引用它） | 15 277 + 3 680 + 2 498(ops) | ~50 |
| **A3** | Gemma 4 | **高**（单向依赖 qwen3_5 共享代码） | 8 014 + 2 953 | ~35 |
| **A4** | Flint（含 `qwen3_5/frontend/flint_output` 方言） | **高** | 4 085 + 2 668 | ~25 |
| **A5** | DeepSeek-V4.1-Flash | **高** | 8 201 + 3 147 | ~40 |
| **B1** | 权重格式 FP8 / NVFP4 / T2 / STQ | 中（要改 dispatcher/loader/文档） | 19 194 | **~41** |
| **B2** | KV 格式 fp8 / k8v4 / nvfp4（以及 int8 / rk4v4-e8 的选项） | 中 | ~5 600 | ~16 |
| **B3** | Qwen3.5 MoE 35B-A3B | **低**（与 Dense 共用整个 program） | 见 §4 | ~30 |
| **B4** | 多卡：tensor-parallel / pipeline stage | **低**（15 个 qwen3_5 文件带 rank 分支） | 3 536 + 模型内分支 | ~15 |
| **B5** | Docker/多架构打包、Windows 专用路径 | 中 | `tests/release` 2 555 | — |

收益口径：A 组每条省的是"整族的编译单元 + 测试 + 文档面"；B1/B2 省的是**编译时间大头**，
但代价是"少支持一种权重/KV 编码"。

---

## 3. A 组（建议执行，按此顺序，一条一个 commit）

### A1 卡支持：只留 sm_89

| 删什么 | 文件 | 依据 |
|---|---|---|
| sm_70（Volta） | `src/ops/linear/sm70/`（14 文件 3 074 行，含无条件被 include 的 `sources.cmake`）、`src/ops/common/sm70_emulation.cuh`、`pre_sm80.cuh`、`softmax_attention/.../prompt_sm70.{cu,h}`、`small_t_sm70.cuh`、`mma.cuh` 的 Volta 分支 | 只有 V100 会走；89 构建里是死代码但**照样编译** |
| sm_75（Turing） | `src/ops/common/sm75_mma.cuh`、`softmax_attention/.../prompt_bf16_sm75.cuh`、`small_t_bf16_sm75.cuh`、`q4/q5/q8` 里的 `NINFER_SM75` 分支 | 同上 |
| sm_80 / sm_86 | `CMakeLists.txt` 的架构白名单与 `NINFER_ARCH_AMPERE` 判定（保留 `NINFER_SM8X_COMPAT`，它是 **89 自己的路径**）、`docs/gpus.md` 的 80/86 行 | 89 与 80/86 共用同一条兼容路径，删的只是"允许别的架构号" |
| 120a（Blackwell） | `src/core/pdl.cuh`、`src/ops/common/blackwell_routes.h`、`src/ops/linear/nvfp4/*_tma*`、`src/ops/linear/fp8/*tma*`、`src/ops/sparse_moe/prefill` 的 TMA 分支、`fp8_sm86_stubs.cpp`/`nvfp4_sm86_stubs.cpp`、`CMakeLists.txt` 的 120a/CUDA-13 版本约束、`NINFER_ARCH_BLACKWELL`/`NINFER_SM120_*` 定义 | 89 上编译成陷阱、永不执行 |
| 文档/打包 | `docs/build-linux.md` 的 V100/Turing 节、`docs/gpus.md` 的 70/75/80/86/120a 行、`docs/maintainer/build-system.md`、Docker 的多架构矩阵 | 描述与现实一致 |

**保留不动**：`NINFER_SM8X_COMPAT`（89 实际走的兼容路径）、`NINFER_TIER_AMPERE`、4080 SUPER 的
device profile 与 `ninfer-calibrate`。

风险：低—中。唯一的真风险是把"89 也在用的宏"当成"别的卡的宏"删掉。验证方式：每删一类都构建 +
8K smoke，并且 `grep -rn "NINFER_SM8X_COMPAT" src` 的数量不减少。

### A2 删 Flash-Next（`qwen4_exp`），**保留 `ops/kv_stream`**

- 删：`src/models/qwen4_exp/`（30 文件 15 277）、`src/runtime/engine/qwen4_exp_core.{h,cpp}`（3 680）、
  `tests/models/qwen4_exp/`（7 151）、`tests/convert/test_qwen4_exp*.py`、`tests/fixtures/qwen4_exp/`、
  `tools/convert/qwen4_exp*.py`、`tools/reference/qwen4_exp.py`、`docs/qwen3-8-flash-next.md`、
  `docs/maintainer/qwen4-exp-model.md`、`src/models/qwen4_exp/loading_sources.cmake` 的 include、
  `engine.cpp` 的判定与 core 分支、`registry.h` 的 `Qwen4Exp`。
- **连带删**（`grep` 确认 qwen3_5 零引用）：`src/ops/{qsa_indexer, sparse_attention, ple_inject,
  hyper_connection}`（合计 2 498 行），以及它们的 `sources.cmake` include 与测试。
- **保留**：`src/ops/kv_stream/*`（410 行，第二阶段用）、`src/ops/candidate_selector`（27B 的 DFlash2
  在用：`execution/draft.cpp`、`load/dflash2.cpp`、`planning/startup.cpp`）。
- 风险：低。qwen4_exp 只单向依赖 qwen3_5 共享代码，删它不改 27B 数学。

### A3 Gemma 4 / A4 Flint / A5 DeepSeek-V4.1-Flash

与前一份清单相同（文件、测试、文档、tools、依赖方向、风险都已核实），此处只补两点：

- A4 必须同时处理 `src/models/qwen3_5/frontend/flint_output.{h,cpp}` 与 `frontend.cpp` 的
  `ToolCallSyntax::Flint` 分支，否则留下无人调用的方言代码。
- 三个族各自的一个 commit，互不依赖；**每步等你点头再下一步**（或你一次批准 A1–A5，我按序执行并
  每步贴真实构建/运行输出）。

---

## 4. B 组（需要你单独决定）

### B1 权重格式 FP8 / NVFP4 / T2 / STQ —— **编译时间最大的一块**

- 体量 19 194 行、**约 41 个 `.cu` 单元**，且当前 89 构建**无条件编译**它们。
- 代价：删掉后**无法加载**同一 27B 的其它 artifact（官方 `groupwise-int`、MXFP8、NVFP4、
  Ternary Bonsai 等），只剩 GGUF + 原生 q4/q5/q6/q8（IQ3_S 就是这套）。
- 你的原话："后期项目成功了再通过对比把别的加回来" —— 按这个原则，**可以删**。
- 我的建议：**删**（编译时间收益最大），但这是"少支持一种权重编码"，请你明确点头。
  操作上分 4 个 commit（NVFP4 → FP8 → T2 → STQ），每个都构建 + 8K smoke。

### B2 KV 格式 fp8 / k8v4 / nvfp4（可选再删 int8 / rk4v4-e8 选项）

- 约 5 600 行、16 个 `.cu`（attention 变体 + codec）。
- 代价：`--kv-dtype` 少几种选择。你要保留的 `bf16 / rk8v4 / rk4v4 / rk2v4-e8` 不受影响
  （`rk8v4` 的 key plane 就是 int8 的 rotated G64 编码，所以 `int8` 的 codec 不能删，只能删"选项"）。
- 建议：删 `fp8`/`k8v4`/`nvfp4` 三种（独立文件），保留 `int8`/`rk4v4-e8`（删了也省不了代码）。

### B3 Qwen3.5 MoE（35B-A3B）

- 与 Dense **共用整个 program**：`program/planning`、`execution`、`load` 里到处是 MoE 分支，
  `ops/{sparse_moe,moe_experts,moe_route}` 9 511 行、`core/{expert_delivery,expert_tier}` 约 3 000 行、
  CLI `--expert-*` 一整族。
- 按你的"耦合极深就别动"规则：**建议不动**（至少不在阶段 1）。

### B4 多卡（tensor-parallel / pipeline stage）

- 52 个文件引用 TP/stage，其中 15 个在 `qwen3_5`（`execution/text.cpp` 的 rank/layer 分支、
  planning、program），删除等于改模型执行主路径。
- 按你的规则：**建议不动**。如果只是"别让多卡代码影响单卡"，它现在是运行期选择，不影响 89 单卡。

### B5 Docker/多架构打包、Windows 路径

- `Dockerfile`、`compose.yaml`、`docker/`、`docs/release-archive-*.md`、`docs/build-windows.md`、
  MSVC/DirectStorage 分支、`tests/release/`。
- 不影响本机构建时间（它们不在 `build.sh` 的目标里）。**建议保留**，除非你要彻底只留本机脚本。

---

## 5. 执行顺序与验证（A 组）

**步 0（建议先做，它本身不删任何模型代码）**：写一个自清理 smoke 脚本
`scripts/scope-smoke.sh`：生成固定 prompt → 跑 8K greedy → 输出前 N 个 token 的指纹 → 退出时
删掉临时目录、确认无残留进程与显存。它属于新增测试工具，也请你批准后我再提交。

理由：你要“测试完毕环境自我清理保持干净”，那就把这个约定变成脚本，而不是靠我每次记得。
脚本内容草案（不依赖仓库内的任何私有接口）：

- `mktemp -d` 一个临时目录，`trap 'rm -rf' EXIT INT TERM`；
- 生成两个固定 prompt（短/中）写到临时目录，不写仓库、不写 `$HOME`；
- 用 `build-sm89/apps/ninfer` 跑：`--greedy --print-token-ids --max-new 64 --kv-dtype rk8v4`；
- 打印 `sha256` 指纹 + `--print-token-ids` 的前 16 个 token；
- 结束前：`pgrep -af apps/ninfer`（应为空）、`nvidia-smi memory.used`（应回到空载）；
- 退出码 = smoke 是否通过。

| 步 | 内容 | 验证（真实输出，贴给你） |
|---|---|---|
| 1 | A1a：删 sm_70（含无条件 include） | `./build.sh` + `scripts/scope-smoke.sh` token 一致 |
| 2 | A1b：删 sm_75 | 同上 |
| 3 | A1c：删 80/86 架构号与文档行 | 同上 |
| 4 | A1d：删 120a（TMA/PDL/blackwell/stub/CMake 约束） | 同上 |
| 5 | A2：删 Flash-Next + 4 个独占算子 | 同上 + `grep -r qwen4_exp src tests docs tools` 为空 |
| 6 | A3：Gemma 4 | 同上 |
| 7 | A4：Flint（含 frontend 方言） | 同上 + tool-call 测试仍过 |
| 8 | A5：DeepSeek-V4.1-Flash | 同上 |

**每步**：构建 → 8K smoke（greedy token 指纹与 `BASELINE.md` 比对）→ 测完清理（无残留进程、
`nvidia-smi` 回到 9 MiB、临时文件删除）。

## 6. 回滚

一条一个 commit，`git revert <sha>`；不 rebase、不 force-push。

## 7. 我需要你回答的

1. **A1–A5 是否批准**（按上面顺序执行）？如果只想先做一条，我建议从 **A1a（sm_70）** 开始。
2. **B1（FP8/NVFP4/T2/STQ 权重）删不删**？我建议删（编译时间最大收益），但它会少支持几种 27B artifact。
3. **B2（KV 的 fp8/k8v4/nvfp4）删不删**？建议删这三种。
4. B3/B4/B5 我按"不动"处理，除非你说改。

---

## 8. 附录：规则文件的改写建议（待你批准，附在执行第 1 步）

删族之后 `AGENTS.md` / `CONTRIBUTING.md` 若还写着"五个族、every platform"，以后会被当成
把删掉的代码加回来的理由。建议的改法（只改这两处，其余节不动）：

**`CONTRIBUTING.md` 的 `### Every family`** → 替换为：

```text
### One model, one card

The product is Qwen3.8-27B Dense on an RTX 4080 SUPER (`sm_89`). A change covers that route: the
Text model, its MTP and DFlash2 drafters, Vision, every `--kv-dtype` format it accepts, the Legacy
and Hybrid prefix caches, serving, CUDA Graphs and the per-card device profile.

- Other model families are removed from this line. Adding one back is a product change
  (`SCOPE.md`) and needs the owner's agreement before implementation.
- Multi-GPU code stays as it is: it is coupled to the model execution path and is not a target.
```

**`CONTRIBUTING.md` 的 `### Every platform`** → 替换为：

```text
### One platform

A change builds and runs on `sm_89` (RTX 40 series) with CUDA 12.8 or newer, on Linux with GCC.
Other compute capabilities (`70`, `75`, `80`, `86`, `120a`) are not built in this line; their code
is removed, not merely unbuilt.
```

**`AGENTS.md`** 的"Product and architecture"段落：删掉五个族的描述与 sm 列表里除 89 以外的项，
保留"single GPU"、"pipeline stages 与 tensor parallelism 存在但不参与验收"这两句，
其余（所有权边界、请求生命周期、KV/缓存权威、验证规则）不动。

其余四节（Lossless by default / Best default / Comparable measurements / Documentation /
Tests and tooling）**保持不变**——它们与模型族、平台无关，是仍然生效的工程规则。
