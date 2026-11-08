# CUTLIST：阶段 1 删减清单（待所有者逐项审核）

状态：**提案，未执行**。本轮分支到这份文件为止只新增了根目录文档，没有删除或修改任何源码、
测试、脚本、构建文件。

## 0. 规则（来自 `SCOPE.md` §0，最高优先级）

1. 删的是**别的模型的支持**；27B 的功能、KV 格式、Vision、量化格式、并发、TP/pipeline **不动**。
2. **RTX 30 / 40 / 50（sm_86 / 89 / 120a）的构建与路由保留。**
3. 每条都要**你先审核**；未经审核的删除视为越界，可被整体回滚。
4. 一条一个 commit，独立可 revert；提交信息写明删除范围与验证结果。

## 1. 结论（诚实摘要）

按新规则，**可删面很小**：

| 类别 | 是否可删 | 体量 |
|---|---|---:|
| Gemma 4 + Flint + DeepSeek-V4.1-Flash（三个别的模型） | 建议删，待你批准 | src 20 300 行 + core 8 768 行 + tests ~10 700 行 + docs 2 049 行 |
| Qwen3.8-Flash-Next（另一个模型） | 待定，建议阶段 2 之后再决定 | src 15 277 + core 3 680 + tests 7 151 |
| Qwen3.5 MoE 35B-A3B（另一个模型，共用 program） | 待定，不建议现在动 | 模型层 ~614 行，但牵动 ops ~9 500、core ~3 000 与半个 program |
| GPU 目标（sm_70/75、sm_86/89/120a） | **保留**（30/40/50 是你要的）；70/75 待你决定 | — |
| 权重格式 FP8/NVFP4/T2/STQ | **保留**（同一个 27B 的其它 artifact、120a 构建） | — |
| KV 格式 int8/fp8/rk4v4-e8/k8v4/nvfp4 | **保留**（27B 的可选特性） | — |
| Vision | **永久保留** | — |
| TP / pipeline / 多卡相关 CLI | **保留**（功能，不是别的模型） | — |

因此这次删减的直接收益是：**少三个族的编译单元（约 100 个 .cu/.cpp）、测试面与文档面**，
不是"大幅瘦身"。编译时间的大头（`ops/linear` 51 919 行、`softmax_attention` 19 494 行）**不会**
因为这次删减而减少，因为它们是目标路径与 30/40/50 平台支持本身。这一点我先说清楚，避免数字误导。

---

## 2. A 组：建议删（三个别的模型族）

### A1 Gemma 4

| 项 | 内容 |
|---|---|
| 源码 | `src/models/gemma4/`（24 文件，8 014 行，含 `sources.cmake`）、`src/runtime/engine/gemma4_core.{h,cpp}`（142 + 2 811） |
| 测试 | `tests/models/gemma4/`（6 文件）、`tests/convert/test_gemma4.py`、`tests/models/qwen3_5/tests.cmake:51` 的 `ninfer_gemma4_output_test` |
| 文档 | `docs/gemma4.md`（622）+ `docs/{cli,serving,recipes,performance}.md` 与 `README.md` 的 Gemma 段落 |
| tools | `tools/convert/gemma4.py`、`tools/reference/gemma4{,_vision}.py`、`tools/convert/__main__.py` 注册 |
| 依赖方向 | **单向**：gemma4 `#include "models/qwen3_5/{frontend,load,model,weights}.h"`；qwen3_5 不引用 gemma4（已 grep 确认）；其它族不引用它 |
| 风险 | 中。① 一个跨族测试登记在 `tests/models/qwen3_5/tests.cmake`，必须同一步改；② `tools/expert_profile_inc.py` 被 gemma4/flint 共用（保留，MoE 还在）；③ 文档要多处同步删，否则留死链 |
| 收益 | 8 014 + 2 953 行源码、约 35 个编译单元 |
| 验证 | `./build.sh`；`ninfer_tests` 目标子集（若已建）；8K smoke 的 greedy token 与基线一致 |
| 回滚 | 单 commit `git revert` |

### A2 Flint

| 项 | 内容 |
|---|---|
| 源码 | `src/models/flint/`（11 文件，4 085）、`src/runtime/engine/flint_core.{h,cpp}`（136 + 2 532） |
| 附带 | `src/models/qwen3_5/frontend/flint_output.{h,cpp}`（Flint 方言的 tool-call 解析）与 `frontend.cpp` 里的 `ToolCallSyntax::Flint` 分支——**只删族会留下无人调用的方言代码**，要一起处理 |
| 测试 | `tests/models/flint/`（2 文件）、`tests/models/qwen3_5/test_flint_frontend.cpp`、`tests/models/qwen3_5/tests.cmake:55`、`tests/fixtures/flint/`、`tests/convert/test_flint.py` |
| 文档 | `docs/flint.md`（279）+ 各文档段落 |
| tools | `tools/convert/flint.py`、`tools/reference/flint.py`、注册行 |
| 风险 | 中。`frontend.cpp` 的方言选择与 `flint_output` 必须同步处理；结构化输出/tool-call 测试要确认只走 Qwen 语法 |
| 收益 | 4 085 + 2 668 行、约 25 个编译单元 |
| 回滚 | 单 commit |

### A3 DeepSeek-V4.1-Flash

| 项 | 内容 |
|---|---|
| 源码 | `src/models/deepseek_v41/`（17 文件，8 201）、`src/runtime/engine/deepseek_v41_core.{h,cpp}`（139 + 3 008） |
| 测试 | `tests/models/deepseek_v41/`（4 文件）、`tests/models/qwen3_5/test_deepseek_frontend.cpp`、`tests/models/qwen3_5/tests.cmake:60`、`tests/convert/test_deepseek_v41.py` |
| 文档 | `docs/deepseek-v41.md`（1 148）+ 各文档段落 |
| tools | `tools/convert/deepseek_v41.py`、`tools/reference/deepseek_v41{,_vision}.py`、注册行 |
| 注意 | 它的 host/disk expert 传输**不是**阶段 2 的复用对象（复用对象是 `core/host_kv_arena` 与 `ops/kv_stream`），删它不影响 KV 流式 |
| 风险 | 中（同 A1） |
| 收益 | 8 201 + 3 147 行、约 40 个编译单元 |
| 回滚 | 单 commit |

### A4 登记点收尾（改文件，不删目录）

| 位置 | 改什么 |
|---|---|
| `src/models/registry.h` / `registry.cpp` | `Architecture` 去掉 `Gemma4, Flint, DeepSeekV41` 与映射 |
| `src/runtime/engine/engine.cpp` | 4 个 `#include`、`is_*_artifact` 判定、`Impl` 的 variant 分支、`CoreState` 的 `is_same_v` 链 |
| `src/models/CMakeLists.txt` | 删 `include(gemma4/sources.cmake)`、`include(flint/sources.cmake)`、`include(deepseek_v41/sources.cmake)` |
| `tests/CMakeLists.txt` | 删 3 个 `include(models/<family>/tests.cmake)` |
| `tests/models/qwen3_5/tests.cmake` | 删 3 个跨族测试登记 |
| `README.md`、`docs/README.md`、`docs/recipes.md`、`docs/weight-conversion.md`、`docs/maintainer/engine-architecture.md` | 删族描述，`AGENTS.md`/`CONTRIBUTING.md` 的族清单按 `SCOPE.md` 改写 |

风险：低（机械），但必须一次把文档改到不撒谎。

---

## 3. B 组：待定（默认不动，等你单独决定）

### B1 Qwen3.8-Flash-Next（`qwen4_exp`）

- 另一个模型（Flash-Next），15 277 src + 3 680 core + 7 151 tests。
- **但它同时是本仓库唯一可运行的"host 权威 KV + device 池 + GPU 内 residency map + 可捕获拷贝"**
  的生产实现（`--kv-resident`，12 个 sparse 层），也是阶段 2 的直接参考。
- 建议：**阶段 2 移植完成后再决定**。若现在删，阶段 2 要自己重建这套机制。
- 若决定删，需同时处理 `ops/{qsa_indexer, sparse_attention, ple_inject, hyper_connection}`（2 498 行）
  与 `docs/qwen3-8-flash-next.md`。

### B2 Qwen3.5 MoE（35B-A3B）

- 另一个模型（`Qwen3_5MoeForCausalLM`），模型层只有 ~614 行，但 `ops/{sparse_moe, moe_experts,
  moe_route}` 9 511 行、`core/{expert_delivery, expert_tier, expert_profile}` 约 3 000 行、
  CLI 的 `--expert-*` 一整族、program 的 MoE 分支都依赖它。
- 与 Dense 共用同一份 block 数学与 program；删它等于把 program 收敛成单一路径，属于结构性重构。
- 建议：阶段 2 之后单独评估；现在不动。

### B3 DFlash（`--spec dflash`，非 DFlash2）

- 只服务 35B-A3B；与 DFlash2 共享 `program/speculative` 与验证路径。
- 建议随 B2 一起评估。

### B4 sm_70 / sm_75（Tesla V100 / Turing）

- **不属于** RTX 30/40/50，删了不影响目标卡；但也是"平台支持"这一功能。
- 体量：`src/ops/linear/sm70/`（14 文件 3 074 行）+ `ops/common/{sm70_emulation,pre_sm80,sm75_mma}.cuh`
  + 注意力/线性的 `NINFER_SM70/SM75` 分支 ≈ 5 000 行、约 25 个编译单元。
- 建议：**默认保留**（收益小、验证面大）。你若要删，我单独提一条执行计划。

### B5 权重格式 FP8 / NVFP4 / T2 / STQ

- 它们**不是别的模型**：官方 Qwen3.8-27B 就有 groupwise-int / MXFP8 / NVFP4 artifact，
  且 120a 构建需要 FP8/NVFP4 的 native 单元才能加载这些权重。
- 体量 19 194 行、143 个编译单元（是编译时间大头之一）。
- 建议：**保留**。删了就无法加载同一 27B 的其它 artifact，属于功能删减。

### B6 KV 格式 int8 / fp8 / rk4v4-e8 / k8v4 / nvfp4

- 27B 的可选特性（`--kv-dtype`），文档里有实测质量/速度数据；`rk8v4` 与 `int8` 共享 key 编码。
- 建议：**保留**。你早期说只留四种，与"27B 特性不变"冲突，我按后者执行；要删请单列。

### B7 多架构打包与 Windows 路径

- `Dockerfile`、`compose.yaml`、`docker/`（多架构镜像矩阵）、`docs/build-windows.md`、
  MSVC/DirectStorage 路径、`tests/release/`（2 555 行）。
- 这是"平台支持"而非"别的模型"，且不属于本机路线；**默认保留**，等你决定。

---

## 4. C 组：明确保留（本轮不动）

| 项 | 理由 |
|---|---|
| `src/models/qwen3_5/**`（Dense） | 目标模型；`program/` 是阶段 2 的落点 |
| `src/core/paged_kv_cache.*`、`paged_kv_storage.h`、`host_kv_arena.*`、`transfer_work.*`、`evictable_kv_pool.*`、`vmm.*` | KV 流式的物理基础 |
| `src/ops/kv_stream/*` + `CONTRIBUTORS.md` 的 Strata 归属 | 阶段 2 的机制来源 |
| `src/ops/softmax_attention/**`（九种 KV 格式） | 27B 的 attention 与全部 KV 选项 |
| `src/ops/kv_cache/**`、`linear_attention/**`、`dynamic_grouped_conv/**`、`gdn_*` | GDN 与 KV append |
| `src/ops/linear/**`（含 fp8/nvfp4/t2/stq/sm70 子目录） | 见 B5/B4：27B 其它 artifact 与平台支持 |
| `src/runtime/engine/context_cache/**`、`src/runtime/prefix_cache/**` | Legacy + Hybrid 两种 prefix cache（serving 的默认与可选模式） |
| `src/runtime/engine/{device_profile,device_profiles.json,calibration}` | 30/40/50 每卡 profile 与重标定 |
| `src/serve/**`、`src/product/**`、`src/models/qwen3_5/frontend/**`（含 Vision 相关） | 外部协议契约与文本/媒体前端 |
| `apps/{cli,serve,calibrate,perplexity}` | 产品入口 |
| `tests/{artifact,convert,models/qwen3_5,ops,host}` 的目标子集 | 正确性保护 |
| Vision（`execution/vision*`、`load/vision*`、`program/vision_control*`、`src/media/decode`） | 你已确认永久保留 |
| `LICENSE`、`NOTICE`、`CONTRIBUTORS.md` | 法务与归属 |

---

## 5. 若 A 组获批：执行顺序与验收

| 步 | 内容 | 验证 |
|---|---|---|
| 1 | 文档与规则（`AGENTS.md`/`CONTRIBUTING.md` 的族清单按 `SCOPE.md` 改写；`docs/README.md` 地图） | `git diff --check`，链接检查 |
| 2 | A1 Gemma 4 | `./build.sh` + 8K smoke + greedy token 与基线一致 |
| 3 | A2 Flint（含 `flint_output`） | 同上 |
| 4 | A3 DeepSeek-V4.1-Flash | 同上 |
| 5 | A4 登记点收尾 | 同上 + `--help` 不再出现族名 |

每步一个 commit；**每一步等你确认后再执行下一步**（或你一次性批准 A1–A4，我按序执行并在每步
贴出真实构建/运行输出）。

**验收标准**：构建成功；8K smoke 的 greedy token 与 `BASELINE.md` 一致；没有残留的族引用
（`grep`）；文档不再描述已删的族；`git diff --check` 干净。

## 6. 回滚

`git revert <sha>`（不 rebase、不 force-push）。每个族一个 commit，互不依赖。

## 7. 需要你回答的问题

1. **A1–A4 是否批准**（三个族 + 登记点）？
2. **B4（sm_70/75）是否删**？默认保留。
3. **B5/B6/B7 是否保留**？我按"保留"处理，除非你说删。
4. `AGENTS.md`/`CONTRIBUTING.md` 的改法：把 "Every family" 收成"Qwen3.5 Dense 27B（MoE/Flash-Next
   的去留见 CUTLIST）"，"Every platform" 收成"sm_86/89/120a（RTX 30/40/50）+ 现有的 sm_70/75 回退"，
   其余四节（lossless / best default / comparable measurements / documentation / tests）**不变**。
   这样可以吗？
