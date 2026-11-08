# CUTLIST：阶段 1 删减清单（待所有者确认）

状态：**提案**。本文件不删除任何东西。每一条编号（A1、A2…、B1…、C…）等待逐项确认：

- **A 组** = 建议本轮删（整族 / 整卡 / 整格式），删除顺序与依赖已排好；
- **B 组** = 待定，需要你决定或先测再定；
- **C 组** = 明确保留（并写明保留理由，防止以后被当成死代码再提一次）。

口径与证据：

- LOC = `git ls-files <path> | xargs wc -l`，含 `.cu/.cpp/.h/.cuh/.cmake`，不含生成文件。
- "编译单元" ≈ 该组内 `.cu` + `.cpp` 数量，用来估计构建时间，不作为运行时收益。
- **收益不承诺运行时性能**：删代码的直接收益是构建时间、二进制体积、迭代与心智负担；
  显存与 tok/s 只由阶段 2/3 的功能改动产生。唯一的间接运行时影响是路由表体积与指令缓存压力，未测。
- 所有"依赖"都按代码实测（`grep`/`include`/CMake include），不是推测；不确定的写在 §7。

---

## 1. 汇总

| 编号 | 内容 | 建议 | src LOC | tests | 文档 | 编译单元 |
|---|---|---|---:|---:|---:|---:|
| A1 | Gemma 4 族 | 删 | 8 014 + 2 953(core) | 3 740 | 622 | ~35 |
| A2 | Flint 族 | 删 | 4 085 + 2 668(core) | 1 991 | 279 | ~25 |
| A3 | DeepSeek-V4.1-Flash 族 | 删 | 8 201 + 3 147(core) | 4 939 | 1 148 | ~40 |
| A4 | A1–A3 的登记点（engine/registry/CMake/测试清单/tools） | 删 | — | 交叉 | 多篇 | — |
| A5 | 权重格式 FP8 / NVFP4 / T2 / STQ | 删 | 22 268 − 3 074(sm70 归 A6) = **19 194** | ~5 000 | tensor-formats 相关节 | ~140 |
| A6 | sm_70（Volta）+ sm_75（Turing）目标 | 删 | 3 074(sm70 目录) + 仿真/门控 ~1 500 | 少量 | build-system/gpus | ~25 |
| A7 | 120a / Blackwell 专属路径残留（TMA、PDL、blackwell_routes、sm120 stubs） | 删 | ~2 500（与 A5 有重叠） | 少量 | build-system | ~15 |
| B1 | Vision（图像/视频） | 待定 | ~3 200 + media decode | 多个 | 多篇 | ~20 |
| B2 | KV 格式 int8 / fp8 / rk4v4-e8 / k8v4 / nvfp4 | 待定 | ~7 000（含 attention 变体） | ~4 000 | cli/performance | ~20 |
| B3 | qwen4_exp（Flash-Next）族 | 阶段 2 后删 | 15 277 + 3 680(core) | 7 151 | 1 288+471 | ~50 |
| B4 | qwen3_5 内部：MoE + expert tier | 阶段 2 后删 | 614(model) + ~10 000(ops) | ~3 000 | 多篇 | ~30 |
| B5 | qwen3_5 内部：tensor-parallel / pipeline stage | 阶段 2 后删 | 3 536(core) + 模型内 TP 分支 | 1 951 | 980 | ~15 |
| B6 | DFlash（仅 35B-A3B 用） | 阶段 2 后删 | 少量（与 DFlash2 共享） | 少量 | dflash.md 374 | ~5 |
| B7 | Docker / 多架构打包 / Windows 专用路径 | 待定 | — | 2 555(release) | 多篇 | — |

---

## 2. A 组：建议本轮删

原则：**先砍整族、整卡、整格式；不要先砍 qwen3_5 内部**。每一条独立提交、独立回滚。

### A1 Gemma 4 族

| 项 | 内容 |
|---|---|
| 文件 | `src/models/gemma4/`（24 文件，8 014 行，含 `sources.cmake`）、`src/runtime/engine/gemma4_core.{h,cpp}`（142+2 811） |
| 测试 | `tests/models/gemma4/`（`test_kernels.cpp`、`test_kv.cpp`、`test_vision_{cpu_,}kernels.cpp`、`test_assistant_real.cpp`、`test_vision_real.cpp`）、`tests/convert/test_gemma4.py`、`tests/models/qwen3_5/tests.cmake:51`（`ninfer_gemma4_output_test`） |
| 文档 | `docs/gemma4.md`（622）、`docs/cli.md`/`docs/serving.md`/`docs/recipes.md`/`docs/performance.md` 的 Gemma 段落、`README.md` 的 artifact 行 |
| tools | `tools/convert/gemma4.py`、`tools/reference/gemma4{,_vision}.py`、`tools/convert/__main__.py` 注册行 |
| 依赖方向 | **单向**：gemma4 `#include "models/qwen3_5/{frontend,load,model,weights}.h"`，qwen3_5 不反向依赖 gemma4（已 grep 确认）；无其它族引用 gemma4 |
| 风险 | 中。① `tests/models/qwen3_5/tests.cmake` 里登记了跨族测试，必须同一步改清单；② `tools/expert_profile_inc.py` 被 gemma4/flint 共用（保留，MoE 还在）；③ docs 的多处段落要同步删，否则留死链 |
| 收益 | 8 014 + 2 953 行源码、~35 编译单元；`ninfer_serving`/`ninfer_engine` 少一个链接分支 |
| 回滚 | 单 commit，`git revert <sha>` |

### A2 Flint 族

| 项 | 内容 |
|---|---|
| 文件 | `src/models/flint/`（11 文件，4 085）、`src/runtime/engine/flint_core.{h,cpp}`（136+2 532） |
| 测试 | `tests/models/flint/`（`test_kernels.cpp`、`test_branches_real.cpp`）、`tests/models/qwen3_5/test_flint_frontend.cpp`、`tests/models/qwen3_5/tests.cmake:55`、`tests/fixtures/flint/`、`tests/convert/test_flint.py` |
| 文档 | `docs/flint.md`（279）+ 各文档 Flint 段落 |
| tools | `tools/convert/flint.py`、`tools/reference/flint.py`、注册行 |
| **附带** | `src/models/qwen3_5/frontend/flint_output.{h,cpp}`（Flint 方言的 tool-call 解析，被 `frontend.cpp` 在 `ToolCallSyntax::Flint` 下选用）随族一起删，否则留下无人调用的方言代码 |
| 依赖方向 | 单向（同 A1） |
| 风险 | 中。`frontend.cpp` 里的 `ToolCallSyntax::Flint` 分支与 `flint_output` 要一起处理；结构化输出/tool-call 的既有测试要确认只走 Qwen 语法 |
| 收益 | 4 085 + 2 668 行、~25 编译单元 |
| 回滚 | 单 commit |

### A3 DeepSeek-V4.1-Flash 族

| 项 | 内容 |
|---|---|
| 文件 | `src/models/deepseek_v41/`（17 文件，8 201）、`src/runtime/engine/deepseek_v41_core.{h,cpp}`（139+3 008） |
| 测试 | `tests/models/deepseek_v41/`（`test_kernels.cpp`、`test_vision_{,real_}kernels.cpp`、`test_vision_real.cpp`）、`tests/models/qwen3_5/test_deepseek_frontend.cpp`、`tests/models/qwen3_5/tests.cmake:60`、`tests/convert/test_deepseek_v41.py` |
| 文档 | `docs/deepseek-v41.md`（1 148）+ 各文档段落 |
| tools | `tools/convert/deepseek_v41.py`、`tools/reference/deepseek_v41{,_vision}.py`、注册行 |
| 注意 | 该族自带 host/disk expert 与 Engram 表；**它的 host expert 传输实现不共享给 qwen3_5 的 KV 流式**（阶段 2 复用对象是 `core/host_kv_arena` 与 `ops/kv_stream`），因此删它不影响阶段 2 |
| 风险 | 中（同 A1） |
| 收益 | 8 201 + 3 147 行、~40 编译单元 |
| 回滚 | 单 commit |

### A4 A1–A3 的登记点

改这些文件（不是删）：

| 位置 | 改什么 |
|---|---|
| `src/models/registry.h` / `registry.cpp` | `enum class Architecture` 去掉 `Gemma4, Flint, DeepSeekV41`；`resolve_architecture` 的映射 |
| `src/runtime/engine/engine.cpp` | 删 4 个 `#include`（16–19 行）、`is_*_artifact` 判定、`Engine::Impl` 的 variant 分支（427–500 行）、`CoreState` 的 `is_same_v` 链（867–869 行） |
| `src/models/CMakeLists.txt` | 删 `include(gemma4/sources.cmake)` 等 3 行（21–23）；`qwen4_exp` 行留到 B3 |
| `tests/CMakeLists.txt` | 删 4 个 `include(models/<family>/tests.cmake)` |
| `tests/models/qwen3_5/tests.cmake` | 删 3 个跨族测试登记 |
| `docker/`、`compose.yaml`、`scripts/` | 检查是否按族构建镜像/跑测试（与 B7 一起处理） |
| `README.md`、`docs/README.md`、`docs/recipes.md`、`docs/weight-conversion.md`、`docs/maintainer/engine-architecture.md`、`AGENTS.md` | 删族描述；`engine-architecture` 的族清单与"第二个/第三个显式族"表述要按 `SCOPE.md` 重写 |

风险：低（机械），但要一次性地把文档改到不撒谎，否则 `AGENTS.md` 描述的"五个族"与现实不符。

### A5 权重格式 FP8 / NVFP4 / T2 / STQ

依据：目标 artifact 的实际格式已核实（见 `SCOPE.md` §2）——只有 GGUF 块与原生
`q4/q5/q6/q8`，**没有任何 FP8 / NVFP4 / T2 / STQ 权重**。`--kv-dtype fp8|nvfp4|k8v4` 是 KV
存储编码（B2），与这里的**权重**单元不是同一套代码。

| 格式 | 目录 | LOC | 编译单元 |
|---|---|---:|---:|
| NVFP4 | `src/ops/linear/nvfp4/`（含 `shapes/`、TMA、int8 grouped） | 9 980 | 72 |
| FP8 | `src/ops/linear/fp8/` | 5 474 | 48 |
| T2（ternary） | `src/ops/linear/t2/` | 3 397 | 20 |
| STQ | `src/ops/linear/stq/` | 343 | 3 |
| 合计 | | **19 194** | **143** |

必须同步改：

- `src/ops/CMakeLists.txt`（`ninfer_nvfp4_non_rdc` 目标、A5 相关 PUBLIC 定义）；
- `src/ops/linear/sources.cmake`、`src/ops/linear/linear.cpp`（dispatcher）、`linear/part_projection.*`；
- `src/ops/fp8_sm86_stubs.cpp`、`src/ops/nvfp4_sm86_stubs.cpp`（stub 单元）；
- `src/ops/launcher/*`、`src/ops/wrapper/*` 中 FP8/NVFP4/T2/STQ 的入口；
- `src/models/qwen3_5/execution/parameters.cpp`、`load/prepare.cpp`（格式到原生参数的映射）、
  `src/ops/weight_input.cpp`；
- CLI/serve：`--nvfp4-w4a8-int`、`--nvfp4-w4a16`（删除选项）与 `docs/cli.md`；
- 测试：`tests/ops/linear/test_nvfp4*.cpp`、`test_fp8*.cpp`、`test_t2*`、`test_stq*`、
  `tests/ops/quantized_weight.h` 的对应 fixture、`tests/ops/tests.cmake` 登记；
- 文档：`docs/maintainer/tensor-formats.md`（777 行的格式表）、`storage-layouts.md`、
  `docs/weight-conversion.md`、`docs/performance.md` 的 MXFP8/NVFP4 段。

风险：**高**，是本轮最需要验证的一步：

1. 删格式后 loader 对"合法 artifact"的校验要仍然接受目标 artifact（`ninfer` 启动 + 8K smoke）；
2. capacity query / workspace 规划按实际绑定推导（文档声称如此），删掉单元后要确认不再有
   FP8/NVFP4 workspace 计入；
3. dispatcher 删分支后，`bf16`/`gguf`/`q4/q5/q6/q8` 的路由不能变（用固定 prompt token IDs 对比）。

建议拆成 4 个小 commit（NVFP4 → FP8 → T2 → STQ），每个都 build + smoke。

### A6 sm_70 / sm_75 目标

| 项 | 内容 |
|---|---|
| 构建系统 | `CMakeLists.txt` 的 70/75 分支与 CUDA 版本约束（`NINFER_ARCH_VOLTA/TURING`、`NINFER_TIER_VOLTA/TURING`、`cmake/PruneCudaPtx.cmake` 的架构白名单） |
| 源码 | `src/ops/linear/sm70/`（14 文件 3 074）、`src/ops/common/sm70_emulation.cuh`、`pre_sm80.cuh`、`sm75_mma.cuh`、`mma.cuh` 的 Volta/Turing 分支 |
| 注意力 | `softmax_attention/dense/causal_cache/prompt_sm70.{cu,h}`、`small_t_sm70.cuh`、`prompt_bf16_sm75.cuh`、`small_t_bf16_sm75.cuh` |
| 线性 | `q4/q4_ksplit_mma.cuh` 等文件里的 `NINFER_SM75` 分支 |
| 测试/文档 | `docs/build-linux.md` 的 V100/Turing 节、`docs/gpus.md` 的 70/75 行、`docs/maintainer/build-system.md` |

风险：中。`NINFER_SM8X_COMPAT`（sm_89 实际走的兼容路径）与 sm_80/86 共享，**不能**在删
70/75 时顺手动它。建议顺序：先删 sm_70（Volta 仿真最独立），再评估 sm_75。

收益：~25 编译单元 + 构建矩阵简化（`cmake/PruneCudaPtx.cmake` 白名单、CUDA 12/13 双版本约束）。

### A7 120a / Blackwell 残留

在 A5 之后，Blackwell 专属代码只剩：`src/ops/linear/bf16/bf16_a16_tma_mma.cuh`、
`src/core/pdl.cuh`、`src/ops/common/blackwell_routes.h`、`sparse_moe/prefill` 的 TMA 分支、
`softmax_attention` 的 `NINFER_TIER_BLACKWELL` 分支、`NINFER_SM120_FP8/NVFP4` 定义。

风险：中。`NINFER_SM8X_COMPAT` 与 Blackwell 路径是两条独立分支，删 Blackwell 不影响 89；
但 `cudaGraph` 的 PDL 选项与 `ops/CMakeLists.txt` 的目标划分要一起清。

收益：~15 编译单元，并让"哪些代码在 89 上编译但永不执行"的陷阱单元消失（构建更诚实）。

---

## 3. B 组：待定（需要你决定或先测）

### B1 Vision（图像/视频）

- 现状：`start-mtp.sh` 明确启用了 `--vision --vision-residency overlay --vision-max-merged 1024`，
  目标 artifact 带 `vision` 组件。你的任务书说"Vision/图片/视频不在目标内"。
- 体量：`src/models/qwen3_5` 内 Vision ≈ 3 200 行（`execution/vision_{cpu,overlay}.*`、
  `load/vision*`、`program/vision_control.*`）+ `src/media/decode/` + `ops/` 的 Vision 入口 +
  `docs/serving.md` 的媒体节 + 测试。
- **风险**：删 Vision 会改变 `start-mtp.sh` 能否原样启动（不是我们今天要改脚本，但它会失败）。
- 决策点：① 保留 Vision（目标路径之外的第二个功能面）；② 删除 Vision，并把启动参数改为纯文本。
  建议：**阶段 2 之后再删**（Vision 不占目标路径的显存——overlay 模式平时在 host——但它占用
  workspace 规划与启动时间；先不扩大阶段 1 的验证面）。

### B2 KV 格式

你要保留 `bf16 / rk8v4 / rk4v4 / rk2v4-e8`。可删候选：

| 格式 | 源码 | 说明 |
|---|---|---|
| `int8` | `kv_cache/int8_g64_codec.cuh`(491)、`softmax_attention` 的 i8 系列 | **注意**：`rk8v4` 的 key plane 就是 `int8` 的 rotated G64 编码，i8 注意力文件同时服务 rk8v4/rk4v4。删"int8 选项"只删 dispatcher 与测试行，省不了多少代码 |
| `fp8` | `kv_cache/fp8_e4m3_row_codec.cuh`(153)、`prompt_fp8.{cu,cuh}`、`small_t_fp8.{cu,cuh}` | ~1 484 行，独立 |
| `nvfp4` | `kv_cache/nvfp4_group16_codec.cuh`(193)、`prompt_nvfp4*`、`small_t_nvfp4*` | ~2 571 行，独立；且与 A5 的 NVFP4 权重共享部分格式化代码 |
| `k8v4` | `kv_cache/append/k8v4_*`、`prompt_k8v4*`、`small_t_k8v4*` | ~1 568 行，独立 |
| `rk4v4-e8` | `e8_root_codec` 的兄弟分支 | 与 rk4v4 共享 4-bit value plane；删除只省一个分支 |

收益：~5 600 行 + 一批 `.cu`（每个都是模板重单元）。风险：中——每个格式都有独立的 Op 测试与
golden 行，删错会静默改变 `--kv-dtype` 的行为。建议：**A 组之后单独一步**，先删 fp8/k8v4/nvfp4
（三个完全独立的），`int8`/`rk4v4-e8` 保持现状（删它们省不了代码，只是隐藏选项）。

### B3 qwen4_exp（Qwen3.8-Flash-Next）

- 15 277 src + 3 546 core + 7 151 tests。**它的 `--kv-resident` 是本仓库唯一可用的
  "host 权威 KV + device 池 + GPU 内 residency map"生产实现**，是阶段 2 的直接参考。
- 建议：阶段 2 移植完成、`ops/kv_stream` 在 qwen3_5 dense 路径跑通之后，再作为一整族删除。
- 删它时必须一起删：`ops/{qsa_indexer, sparse_attention, ple_inject, hyper_connection}`（合计 2 498 行）、
  `runtime/engine/qwen4_exp_core.*`、ngram 相关 CLI（`--ngram-*` 的大部分）、`docs/qwen3-8-flash-next.md`、
  `docs/maintainer/{qwen4-exp-model,dflash}.md` 的对应部分。

### B4 qwen3_5 内部：MoE

- 模型层只有 ~614 行（`execution/expert_runtime.*`、`load/expert_sources.cpp`、`expert_*.h`），
  但 `ops/{sparse_moe, moe_experts, moe_route}` 9 511 行、`core/{expert_delivery, expert_tier,
  expert_profile}` 约 3 000 行、`program/planning` 与 `execution` 的 MoE 分支、CLI 的
  `--expert-*` 一整族选项、`docs/maintainer/qwen3_5-model.md` 的 MoE 节都依赖它。
- 建议：阶段 2 之后再动。它不是"删掉一个文件"级别的改动，而是"删掉半个 program"。
- 顺带：`Qwen3_5MoeForCausalLM` 与 Dense 共用同一份 block 数学，删 MoE 需要把 Dense 路径的
  条件分支收敛成单一路径，属于第二阶段之后的独立重构。

### B5 tensor-parallel / pipeline stage

- `core/tensor_parallel.{cu,h}` 2 940、`core/stage_link.cu` 267、`core/stage_plan.h` 329、
  `docs/maintainer/pipeline-stages.md` 980、测试 1 951 行、CLI 的 `--devices/--stage-layers/
  --tensor-parallel/--tensor-split/--tensor-attention/--tensor-parallel-transport/
  --tensor-parallel-prompt-partials` 七个选项。
- 你的目标是单卡。但 `text.cpp` 的 layer pipeline 与 KV 流式的"按层搬运"在所有权上相邻，
  阶段 2 会读它的 `layer_ready_` 事件模式。建议阶段 2 之后再删。

### B6 DFlash（非 DFlash2）

- `--spec dflash` 只用于 35B-A3B（MoE），目标 artifact 只有 `dflash2`。
- 与 DFlash2 共享 `program/speculative` 与验证路径，删除收益小、耦合大；建议随 B4 一起评估。

### B7 Docker / 打包 / Windows

- `Dockerfile`、`compose.yaml`、`docker/`、`docs/release-archive-*.md`、
  `docs/build-windows.md`、`cmake` 的 MSVC/DirectStorage 路径、`tests/release/`（2 555 行）。
- 你的目标是 Linux 单机自用。但 `build.sh` 已在用现成 build 目录，删打包不产生收益；
  建议只删**多架构镜像矩阵**，保留 `Dockerfile` 与本机无关的构建文档。**待你决定。**

---

## 4. C 组：明确保留（不要当成死代码删）

| 项 | 理由 |
|---|---|
| `src/models/qwen3_5/**`（Dense 路径） | 唯一目标模型；`program/` 是阶段 2 的落点 |
| `src/core/paged_kv_cache.*`、`paged_kv_storage.h`、`host_kv_arena.*`、`transfer_work.*`、`evictable_kv_pool.*`、`vmm.*` | KV 流式的物理基础 |
| `src/ops/kv_stream/*` | 阶段 2 的 residency map / 可捕获拷贝内核来源（Strata 归属见 `CONTRIBUTORS.md`） |
| `src/ops/softmax_attention/dense/**`（bf16 + 4 个 KV 格式） | 目标模型的 attention；`context`/`packed`/`prompt` 路由全部在用 |
| `src/ops/kv_cache/**`（保留格式的 codec）、`linear_attention/**`、`dynamic_grouped_conv/**`、`gdn_*` | GDN 与 KV append |
| `src/ops/linear/{bf16,gguf,q4,q5,q6,q8,common}` | artifact 实际使用的权重格式 |
| `src/runtime/engine/context_cache/**`、`scheduler.h`、`admission_policy.*`、`kv_capacity.*`、`device_profile.*` | 调度、容量与每卡 profile（4080 SUPER 已内建） |
| `src/serve/**`、`src/product/**`、`src/models/qwen3_5/frontend/**` | 外部协议契约与文本前端 |
| `src/calibration/**`、`docs/device-profiles.md`、`docs/gpus.md` | 4080 SUPER profile 的再标定能力 |
| `apps/{cli,serve,calibrate,perplexity}` | 产品入口 |
| `tests/{artifact,convert,models/qwen3_5,ops}` 的目标子集、`tests/host` | 正确性保护 |
| `LICENSE`、`NOTICE`、`CONTRIBUTORS.md` | 法务 + Strata 归属 |

---

## 5. Phase 1 执行顺序（每步一个小 commit）

| 步 | 内容 | 完成后验证 |
|---|---|---|
| 1 | A4 的文档/SCOPE 改动 + `AGENTS.md`/`CONTRIBUTING.md` 单目标化 | `git diff --check`；文档链接检查 |
| 2 | A1 Gemma 4 | `./build.sh` + 8K smoke + token IDs |
| 3 | A2 Flint | 同上 |
| 4 | A3 DeepSeek-V4.1 | 同上 |
| 5 | A4 登记点收尾（registry/engine/CMake/tests 清单） | 同上 + `--help` 无族名 |
| 6 | A5a NVFP4 权重 | 同上 |
| 7 | A5b FP8 权重 | 同上 |
| 8 | A5c T2 + STQ | 同上 |
| 9 | A6 sm_70 | 同上 |
| 10 | A7 120a/Blackwell 残留 | 同上 + 确认 89 构建不再定义 `NINFER_SM120_*` |
| 11 | （可选）A6 sm_75 | 同上 |

每步的 smoke 固定为：同一 prompt、`--greedy --print-token-ids`、`--max-context 8192`、
`--kv-dtype rk8v4`、`--spec mtp --draft-tokens 3`，取前 64 个 token ID 与 `BASELINE.md` 比对。

**验收标准**：

1. 每步 `./build.sh` 成功（真实日志）；
2. 每步 token ID 与基线完全一致（exact）；
3. 每步 `ninfer_tests` 的目标子集通过（若测试套件已构建，见 B 组之外的"测试构建"决定）；
4. 没有留下指向已删族的 CMake 目标、测试登记、文档链接；
5. `docs/README.md` 的文档地图与 `SCOPE.md` 一致。

---

## 6. 回滚

- 每个条目一个 commit（A5 拆 4 个），提交信息写"改了什么/为什么/测试结果"；
- 回滚 = `git revert <sha>`，不 rebase、不 force-push；
- 删除是向前提交，历史里仍可取回文件；**不**使用 `git clean`/`reset --hard`；
- 若某步 smoke 失败且 30 分钟内无法定位，直接 revert 该步，保留其余步骤。

---

## 7. 未核实 / 需要你确认的点

1. **编译收益的量化**：我没有在删除前测"每个族的增量构建时间"。建议第 2 步开始记录
   `ninja` 的 `-j16` 全量构建耗时作为对照（一次全量构建即可，不必每步全量）。
2. **`--prefill-cublas`**：它把宽 prefill GEMM 交给 cuBLAS，属于 bf16 路径的替代实现，
   与 A5 无关，默认保留（未测其与流式的交互）。
3. **`ops/linear/part_projection.*`** 是否被 A5 的格式共用，删格式时要逐文件确认。
4. **`docs/performance.md` / `docs/performance/reference-2026-09.md`** 里各族的表：删族后要
   同步删行，否则文档描述与现实不符（`CONTRIBUTING.md` 的文档规则）。
5. **`AGENTS.md` / `CONTRIBUTING.md` 的改法**：我建议把 "Every family"/"Every platform"
   两节替换为"单目标契约"+"新增范围必须改 `SCOPE.md`"，并保留"lossless by default"、
   "best default"、"comparable measurements"、"documentation"、"tests and tooling" 五节不变。
   这需要你确认（这是规则本身的变化，不只是文档措辞）。
