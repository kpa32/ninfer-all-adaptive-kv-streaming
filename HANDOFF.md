# 交接文档：fork-qwen38-27b-16gb 越界代码删除（A3 完成，待审核清单已列）

> 本文件是工作交接记录，不纳入 git（与 `REMOVALS.txt` 一样）。全部完成后删除。
> 最后更新：A3（T0 纯删除）完成（42074842，远端构建 0 错误）。待审核清单见 §5，等用户评审。

## 0. 一句话状态

分支 `fork-qwen38-27b-16gb` 已删完：四个越界模型家族（A1）、`qwen3_5/frontend/`
内嵌家族方言 + registry 枚举值（A2）、孤儿 ops + 三条死路由选项链（A3/T0）。
全部已提交、推送、远端构建 0 错误。
**下一步：§5 待审核清单由用户慢慢评审；Pass B（MoE）在评审后执行。**

## 1. 任务背景

- 这是 NInfer-all 的硬 fork，唯一支持模型：**Qwen3.8-27B dense**（16 GB 卡，
  开发/测量卡 RTX 4080 SUPER sm_89）。
- 删除目标：越界模型家族 → MoE 基础设施 → 孤儿 ops/CLI 选项 → serve 外围功能。
- 策略：**先删，编译不过再补**；每一步必须保持可编译，远端增量验证后才提交。
- 纯删除原则：不得改变 Qwen3.5 dense 或 MTP/DFlash2 的行为。
- **评估区（不盲删）**：MTP/DFlash2 drafting（Qwen3.8-27B 保留能力）；
  家族专用 engine/serve 选项——先确认 dense 是否使用再删。
- 用户本地工作流问题（是否用 WebUI/llama.cpp 端点/LoRA 等）决定 §5 各项去留；
  无回复默认删除。

## 2. 已完成并提交（远端已验证）

| 提交 | 内容 |
|---|---|
| 初始裁剪（1211 文件） | `.github/`、`tests/`、`docs/`、`bench/`、`tools/`、`examples/`、`eval/`、`scripts/` 内容、`LICENSE`、`README.md`、`CONTRIBUTING.md`、flake、docker 等；CMake 去掉 BUILD_TESTING/BENCHMARKS/tools；`CMakePresets.json` 重写；`build.sh`（16 线程、ccache、sm_89、Ninja Release） |
| `21d0d65e` A1 | 四个家族模型目录、四个 engine core、`engine.cpp` 分发、`apps/ngram_profile/`、`divided_head.cpp/.h`、各 CMake 条目、`rope_scaling.h` 家族条目、`grammar_masks.h` 注释。99 文件，-48798 行 |
| `6e64dc19` + `ec6dab81` A2 | `qwen3_5/frontend/` 家族方言（3 对 output 文件、processor/tokenizer/chat_template/prompt_layout/output_session/frontend 的家族分支）、`resources.h` 字段、`tool_call_parser.h` 注释、registry 4 个枚举值、frontend_sources.cmake 条目。21 文件，-2837 行 |
| `42074842` A3 (T0) | 4 个孤儿 ops 目录 + 4 个 ops 头文件；`prompt_route_options.h`（--prefill-tail/--iq1m-experts）、`ngram_table_options.h`（--ngram-table 家族）、`kv_resident_options.h`（--kv-resident）三条死选项链：engine 字段、枚举、可观测性统计、CLI/serve/perplexity/launcher 全链路。32 文件，-3261 行 |

各步远端增量构建均 0 错误，4 产物重新链接：`ninfer`、`ninfer-serve`、
`ninfer-calibrate`、`ninfer-perplexity`。

A3 保留项（已确认）：
- `src/ops/kv_stream/`（410 行 + 86 行头文件）——活跃 KV 流线的机制实现（host 常驻
  pinned KV + 设备页池 + residency map），当前无消费者，但正对该功能线；
  列入 §5.1 待用户定夺。
- Qwen ngram-copy drafting（`ngram_sources.h`、`ngram_proposer.h`、`draft.cpp`）与
  ngram-archive 会话识别（`ngram_native_sessions`）——Qwen 自身能力，非死代码。
- ops 层 IQ1_M 内核（gguf/moe_experts 内的 `iq1m_*`）——MoE 属 Pass B。

## 3. Pass B（MoE 基础设施，用户已确认删除，待 §5 评审后执行）

Qwen3.8-27B 是 dense，MoE 全部越界。规模（行）：

| 位置 | 规模 |
|---|---|
| `src/ops/moe_experts/` | 3605 |
| `src/ops/sparse_moe/` | 5595 |
| `src/ops/moe_route/` | 278 |
| `src/core/expert_delivery.cu/.h` | 2491 |
| `src/core/expert_parallel.cpp/.h` | 167 |
| `src/models/expert_profile.cpp/.h` + `expert_profiles/builtin.inc` | 3886 |
| `src/product/hybrid_expert_options.h` + `--expert-residency`/`--expert-cache-mib` 链 | ~200 |
| `src/models/qwen3_5/` 内 MoE 分支（config/ffn/program/mtp） | 待定位 |
| `src/models/registry.h/.cpp` — `Qwen3_5Moe` 枚举值 + 分派 | 小 |

## 4. 深水（用户明确缓后处理，不在此阶段动）

- 多卡/流水线：`--devices`、`--stage-layers`、`src/artifact/{binder,materializer}`、
  `src/core/{device,stage_link,stage_plan}`、`auxiliary_replicas`。
- tensor-parallel：`--tensor-parallel*` 选项链。
- 并发/多请求调度：engine 的 1-8 并发、concurrent prefill、FIFO ingress。

## 5. 待审核清单（用户慢慢评审；每项无回复默认删除）

每项都标注了规模与"删了会失去什么"。评审格式：逐项回答"留/删/不确定"即可。

### 5.1 `src/ops/kv_stream/`（496 行）— 边界项，最需先定

- 内容：paged KV 全部常驻 pinned host 内存、设备保留 `slots` 页池、每轮按
  residency map 把选中页搬上设备——**正是活跃 KV 流线的机制实现**
  （Strata 的 kv_stream，MIT 派生）。
- 现状：无消费者（唯一使用者 `sparse_attention` 已删）。
- 风险：若 KV 流线计划在此 op 上构建，现在删等于把唯一现成实现推回 git 历史
  （可恢复，但重做成本）。
- 建议：默认**保留**到 KV 流线方案定型，方案定型后再定去留。请用户确认。

### 5.2 MCP proxy（`src/serve/mcp_proxy.cpp/.h`，382 行 + `/cors-proxy` 路由 + `--webui-mcp-proxy`）

- 内容：同源 CORS 中继，给内置 WebUI 的 MCP 客户端转发第三方 http MCP 服务器。
- 依附：WebUI。若 WebUI 删，此项自动无意义。
- 问题：你在浏览器 WebUI 里配过外部 MCP 服务器吗？

### 5.3 typed decisions（`src/serve/systemone.cpp` 1190 行 + `src/serve/decisions_http.cpp` 366 行 + `/v1/systemone`、`/v1/decisions`）

- 内容：厂商扩展的"类型化决策"端点（非 OpenAI/Anthropic 契约）。
- 问题：你的工作流调用过这两个端点吗？

### 5.4 llama.cpp API 子集（`text_completion.cpp` 611 + `llamacpp_http.cpp` 535 + `rerank.cpp` 176 行 + `/completion`、`/tokenize`、`/detokenize`、`/apply-template`、`/v1/rerank`）

- 内容：llama.cpp server 兼容端点子集。
- 依附：WebUI 部分功能依赖这些端点（与 5.6 联动评审）。
- 问题：你用 llama.cpp 客户端/脚本打这些端点吗？

### 5.5 WebUI（`src/serve/webui/embed.cpp` 144 行 + 页面资产 + 相关路由）

- 内容：内置网页界面（llama.cpp 发布版 WebUI 资产 + 自研嵌入器）。
- 问题：你用浏览器界面吗？还是纯 API/CLI？

### 5.6 可观测性套件（`request_log.cpp` 1403 + `operational_log.cpp` 709 + `console_stats.cpp` 281 行 + `/metrics`、`/stats`、`/slots`、`/props`）

- 内容：serve 请求日志、操作日志、控制台统计与 Prometheus 风格端点。
- 问题：你监控 serve 的指标吗（Prometheus 抓取、日志文件）？

### 5.7 disk KV tier（`src/core/disk_kv_bridge.cpp/.h` 513 + `disk_kv_store.cpp` 566 + `src/models/qwen3_5/program/storage/disk_tier.cpp` 472 行 + `--disk-kv-path`/`--disk-kv-restore`）

- 内容：前缀缓存的磁盘层（把冷 KV 页落到盘上，重启可恢复）。
- 注意：与 5.1 的 KV 流线是**不同**机制（disk tier 是持久缓存层，不是长上下文流式）。
- 问题：你的长上下文场景需要重启恢复缓存吗？

### 5.8 LoRA / control vectors（`frontend/steering.cpp/.h` 672 + `execution/steering.h` 55 + `ops::steer_residual` + `/lora-adapters` + `--lora` 等）

- 内容：运行时加载 GGUF LoRA 适配器与 control vector（abliteration 式去对齐等）。
- 注意：范围内模型含 abliterated fine-tune——若那些模型是**独立转换的 .ninfer 工件**
  （已内化），则运行期 LoRA 可能无人用；若你靠 LoRA 文件叠加行为，则必须留。
- 问题：你会加载 `.lora`/control vector 文件吗？

### 5.9 model suspend / 多模型路由（`Engine::suspend/resume/residency` 公共 API + `--models-dir`/`--models-preset`/`--models-max`）

- 内容：多模型目录路由、请求间挂起/恢复模型。
- 现状：单模型部署下全是死路径。
- 问题：确认你只跑一个模型文件？

### 5.10 prompt grafts（`frontend/graft.cpp/.h` 477 + `program/graft_injection.cpp` 355 + `--graft`/`--default-graft`）

- 内容：把已算好的上下文"嫁接"进请求（共享前缀的另一种入口）。
- 注意：与 disk KV tier（5.7）、prefix cache 有功能重叠；prefix cache 是默认路径。
- 问题：你用过 `--graft`/`--default-graft` 吗？

### 评审约定

- 每项回答"留 / 删 / 不确定"即可；"不确定"按删除处理（可在后续恢复）。
- 5.1 建议保留到 KV 流线方案定型；其余默认删除。
- 评审完成后按批次执行：每批一个提交，远端构建 0 错误后推送；
  与 §4 深水项无关。

## 6. 远端验证流程

- 远端：`kpa@192.168.2.13`（Debian 13，16 核，62 GB RAM，RTX 4080 SUPER 16,376 MiB，
  driver 615.71.09，CUDA 13.4：`/usr/local/cuda/bin/nvcc`，g++ 14.2，ccache，ninja，cmake 3.31.6）。
- 专用树：`/home/kpa/ninfer-16/ninfer-all-adaptive-kv-streaming`，构建目录 `build-sm89`。
- 同步（本地 push 后）：
  `git fetch -q origin && git reset --hard -q origin/fork-qwen38-27b-16gb`
- 构建：`./build.sh > /tmp/build.log 2>&1`（16 线程；首次自动 configure）
- 检查：`echo BUILD_EXIT=$?`（0 即通过）+ `grep -cE "error:|FAILED" /tmp/build.log`（0）。
  **注意**：`| grep | head` 管道后 `$?` 是 head 的退出码，不可靠，必须单独取。
- **禁止**：杀/重启其他树的进程。`~/ninfer/` 下有别的 `ninfer-serve` 在跑，
  **绝不能碰**；`~/ninfer/` 是禁区。`~/ninfer-fork/` 旧树也不许动。
- 跑 CUDA 测试/bench 前先 `nvidia-smi --query-compute-apps` 确认卡空闲；
  编译不需要 GPU。
- 模型：`/home/kpa/ninfer-16/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（15 GB）。
- WebUI 资产：`third_party/webui/b11165/llama-b11165`（gitignored）。

## 7. 本地（Windows）约束与坑

- 仓库：`D:/GithubRepo/ninfer-all-adaptive-kv-streaming`，分支 `fork-qwen38-27b-16gb`，
  远端 `https://github.com/kpa32/ninfer-all-adaptive-kv-streaming`。
- **Windows 没有 python/g++**（python 是 Store 占位符，exit 49）。
  文件操作用 `edit` 工具、`sed`、`grep`、`awk`；本地无法编译，只能远端验证。
- `edit` 工具要求 oldText 精确匹配（含空白/对齐空格）；大段删除先 `sed -n 'N,Mp'`
  验证边界行号再动手；对齐不一致时按行号替换。
- 提交只写 Conventional Commits；提交信息如实记录删了什么、留了什么。
- 每次删除后先本地 grep 验证残留为 0（**含 help 常量、apply() 调用、include**——
  A3 两轮补漏都是这类），再提交推送。
- 工作记录文件：根目录 `REMOVALS.txt` 与本文件，均不纳入 git。
- 时钟偏差：Windows 与 Linux 差 10-25 s，`tar` 报未来时间戳属正常。

## 8. 行为规则（用户约定）

1. 删优先，编译错误后补；但每步保持可编译、远端验证后才提交。
2. 纯删除：不得改 Qwen3.5 dense / MTP/DFlash2 行为。
3. MTP、DFlash/DFlash2 是评估区，不盲删；删模型家族 ≠ 删所有相关能力。
4. 现在只需支持 Qwen 3.8 27B dense；MoE 要删。
5. 不确定的模块列进 §5 清单给用户评估；无回复默认删除。
6. 远端 GPU 被占时只做编译，不跑测试/bench。
7. 报告要如实披露有利和不利发现。
8. 多卡/多流/流水线/tensor-parallel/并发是深水，缓后处理（§4）。
9. 后续沟通用中文。
