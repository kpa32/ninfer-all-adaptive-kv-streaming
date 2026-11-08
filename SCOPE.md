# 产品范围：单目标 NInfer（RTX 4080 SUPER · Qwen3.8-27B Dense）

本文件是这个仓库唯一的**产品范围契约**，也是删减的许可边界。代码、测试、文档、构建目标都以
本文件为准；`AGENTS.md` 与 `CONTRIBUTING.md` 的描述不得宽于本文件。

---

## 0. 删减规则（所有者 2026-11-08 确认，最高优先级）

1. **删减的对象是"产品范围"（其它模型的支持），不是"功能"。**
   27B 上正在使用、或可以使用的特性必须**保持不变**——包括 Vision、全部 KV 格式、
   MTP/DFlash2、context/prefix cache、并发、CUDA Graph、结构化输出、量化权重格式。
2. **支持 RTX 30 / 40 / 50 系列（sm_86 / sm_89 / 120a）的代码保留。**
3. 任何删除都必须先写进 `CUTLIST.md`，由所有者逐条审核通过后执行；**不得静默删除**。
4. 未经审核的删除视为越界：所有者可直接回滚，相关工作作废。
5. 每个删除条目一个独立 commit，可单独 revert；提交信息写明"删了什么/为什么/测试结果"。
6. 拿不准的一律归入 `CUTLIST.md` 的 B 组（待定），不默认删除。

审计期间的事实：本分支（`target/4080s-27b`）到本文件为止只**新增**了根目录文档
（`SCOPE.md`/`SPEC.md`/`CUTLIST.md`/`BASELINE.md`/`RISKS.md`），没有改动任何源码、测试、脚本。

---

## 1. 硬件

| 项 | 值 |
|---|---|
| 唯一目标机 | NVIDIA GeForce RTX 4080 SUPER，16 376 MiB，计算能力 8.9（Ada），80 SM |
| 主机链路 | PCIe 4.0 x16（最大值 16.0 GT/s ×16；空闲回落 2.5 GT/s，加载时实测升到 16.0 GT/s） |
| 主机内存 | 64 GB 级（`MemTotal` 64 213 MiB，Debian 13） |
| 本机构建目标 | `CMAKE_CUDA_ARCHITECTURES=89` |

**保留（不删）**：sm_86（RTX 30 系列）、sm_89（RTX 40 系列）、120a（RTX 50 系列）的构建与路由，
以及它们各自需要的回退路径、device profile 与标定工具。删掉它们等于删功能。

**待定**：sm_70（Tesla V100）与 sm_75（Turing）——不属于 RTX 30/40/50，但删它们也不影响目标卡；
见 `CUTLIST.md` B1，由所有者决定。

**非目标（产品范围）**：与非目标模型绑定的多卡形态（`--devices` pipeline stage、
`--tensor-parallel` 等）属于"功能"而非"别的模型"，因此**保留**，直到所有者明确要求删除。

## 2. 模型与权重

- 目标模型：架构 `Qwen3_5ForCausalLM` / `model_type: qwen3_5_text`，**Dense 27B**。
- artifact（当前使用）：`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（15 017 456 896 B = 13.99 GiB，
  文本权重 10.9 GiB），组件 `text`、`proposal`(131 072 行)、`vision`、`mtp`、`dflash2`。
- 几何（自 artifact 目录读回，2026-11-08）：

  | 量 | 值 |
  |---|---:|
  | 层数 | 64（48 `linear_attention` + 16 `full_attention`，full 层在 3,7,…,63） |
  | H / dense FFN / vocab | 5 120 / 17 408 / 248 320 |
  | Q / KV 头 / head_dim | 24 / 4 / 256 |
  | GDN | key 16×128，value 48×128，conv 宽度 4 |
  | 位置 | 262 144 原生；`rope_theta=1e7`，partial rotary 0.25，mrope `[11,11,10]`，`rms_norm_eps=1e-6` |
  | DFlash2 | 5 层 sliding(2048)，target layers `[5,19,33,47,61]` |

- 权重格式：**保留全部格式**。IQ3_S artifact 用到 `bf16`/`fp32`/`int32`/GGUF 块/原生 row-split
  `q4/q5/q6/q8`；FP8、NVFP4、T2、STQ 是**同一个 27B 架构**的其它 artifact（官方 groupwise-int、
  MXFP8、NVFP4 等）以及 120a 构建需要的格式，属于 27B 的可选特性，不删。
- 其它模型族（Gemma 4、Flint、DeepSeek-V4.1-Flash、Qwen3.8-Flash-Next、Qwen3.5 MoE 35B-A3B）
  是**别的模型的支持**，是本次删减的唯一对象；逐项见 `CUTLIST.md`。

## 3. 功能（全部保留）

文本 chat（`ninfer` CLI）与 HTTP serving（`ninfer-serve`，OpenAI/Anthropic 协议契约）、
`--spec mtp` 与 `--spec dflash2`、`--lookup-ngram`/`--ngram-draft-tokens`、paged KV + host KV arena +
context/prefix cache（Legacy 与 Hybrid 两种模式）、GDN、dense attention、CUDA Graph、
`--invariant-attention`、`--gdn-state-fp16`、结构化输出、离线 CausalScoring（`ninfer-perplexity`）、
**Vision（图片/视频，永久保留）**、每卡 device profile 与 `ninfer-calibrate`、
`--max-concurrency 1..8`、pipeline stages 与 tensor parallelism。

KV 存储格式：**九种全部保留**（`bf16`、`int8`、`fp8`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`、
`k8v4`、`nvfp4`）。关于"只要 bf16/rk8v4/rk4v4/rk2v4-e8"的早期指示，与"27B 特性保持不变"
冲突，按后者执行；如果所有者要删其余五种，那是功能删减，走 `CUTLIST.md` B 组单独确认。

**非目标**：`.ninfer` 之外的模型格式、Python 推理路线、C++ SDK 导出、主动抢占、优先级/QoS、
跨 Engine context store、大规模 continuous batching。

## 4. 新增能力：自适应 KV 流式（本分叉的唯一新增产品能力）

- 目标：262 144 原生上下文在 16 GiB 上不 OOM，decode 稳定，正确性与全驻留路径逐 token 一致。
- 形态：host（pinned）为权威 KV，设备持有 resident 区 + 有界 ring，按 full-attention 层流式预取，
  resident/streaming 划分按可用显存自适应。
- 约束：
  - **不得**改变 attention 数学、KV codec、页语义；同格式 streaming on/off 的 greedy token IDs 与
    logprobs 必须一致；
  - 近似/丢页（lossy）只能作为单独选项、默认关闭、需所有者事先同意；
  - 默认仍是全驻留；`--kv-stream` 一类开关默认 off，稳定后才讨论默认值；
  - 单卡、单 GPU stream 语义优先；多流 event 节点是优化，不是 MVP 前提。

## 5. 构建与发布

- 本机唯一构建：`build.sh` 的 `sm_89`（Ninja、Release、`NINFER_BUILD_APPS=ON`）。
- 其它架构（sm_86/80/120a、sm_70/75）的构建支持按 §1 保留，不因本分叉删除。
- `LICENSE`、`NOTICE`、`CONTRIBUTORS.md` 保留（`CONTRIBUTORS.md` 中 `kv_stream` 的 Strata 归属
  必须保留，该实现仍在用）。

## 6. 出口条件

| 阶段 | 出口条件 |
|---|---|
| 1 删减 | `CUTLIST.md` 逐项确认；每步 `./build.sh` 通过、8K smoke 通过、固定 prompt greedy token 序列与基线一致；`git diff --check` 干净 |
| 2 流式 | 262 144 不 OOM；8K/32K/64K/128K/200K/262K decode 有真实数字；streaming on/off 同格式 token IDs 一致；resident 路径性能不回退；失败可回退（默认 off） |
| 3 优化 | 每个变量单独 A/B，同一主机同一 build，报告均值与范围；最差变化同样报告 |

## 7. 变更规则

1. 本文件是范围的唯一权威；`AGENTS.md`、`CONTRIBUTING.md`、`README.md` 的描述不得宽于本文件。
2. 扩范围或缩功能都需要所有者显式同意，并在同一次改动中更新本文件与受影响的构建/测试/文档。
3. `.ninfer` 仍是唯一 C++ 产品 artifact；CLI/serve/bench 只走公共 `ninfer::Engine`。
4. 代码所有权边界（Core / Artifact / Ops / Model / Runtime / Product）不变。
