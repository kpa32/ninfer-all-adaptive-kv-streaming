# 产品范围：单目标硬分叉（RTX 4080 SUPER · Qwen3.8-27B Dense）

本文件是这个仓库唯一的**产品范围契约**。`AGENTS.md` 与 `CONTRIBUTING.md` 中的
"every model family / every platform" 规则由本文件取代：本仓库只交付下列一条路线，
改动本文件必须由仓库所有者显式决定，代码、测试、文档、构建目标都以本文件为准。

来源：所有者 2026-11-08 的任务书（阶段 1 审计期间确认）。本文档在阶段 1 交付时仍是
**草稿**，删除清单确认后升级为 `active`。

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，16 376 MiB，计算能力 8.9（Ada），80 SM |
| 主机链路 | PCIe 4.0 x16（`max_link_speed = 16.0 GT/s`，`width = 16`）；空闲时链路回落到 2.5 GT/s，加载时须复核 |
| 卡数 | 1 |
| 主机内存 | 64 GB 级（`MemTotal` 64 213 MiB） |
| 构建目标 | `CMAKE_CUDA_ARCHITECTURES=89` |

**非目标**：sm_70 / sm_75 / 120a 构建、Turing/Volta 仿真、Blackwell 专属指令路径、任何多卡形态
（`--devices` pipeline stage、`--tensor-parallel`、`--tensor-split`、`--stage-layers`）、NVLink/P2P。
sm_86 / sm_80 只在目标路径确实共享同一份回退实现时保留，且不作为验收对象。

## 2. 模型与权重

- 架构：`Qwen3_5ForCausalLM` / `model_type: qwen3_5_text`，**Dense**。
- artifact：`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（15 017 456 896 B = 13.99 GiB 整包，
  文本权重 10.9 GiB），组件：`text`、`proposal`(131 072 行索引头)、`vision`、`mtp`、`dflash2`。
- 几何（自 artifact 目录读回，2026-11-08）：

  | 量 | 值 |
  |---|---:|
  | 层数 | 64（48 `linear_attention` + 16 `full_attention`，full 层在 3,7,…,63） |
  | H / dense FFN / vocab | 5 120 / 17 408 / 248 320 |
  | Q / KV 头 / head_dim | 24 / 4 / 256 |
  | GDN | key 16×128，value 48×128，conv 宽度 4 |
  | 位置 | 262 144 原生；`rope_theta=1e7`，partial rotary 0.25，mrope `[11,11,10]`，`rms_norm_eps=1e-6` |
  | DFlash2 | 5 层 sliding(2048)，target layers `[5,19,33,47,61]` |

- 权重格式（artifact 内实际使用，必须全部保留）：
  `bf16`(627 tensors)、`fp32`(96)、`int32`(2)、GGUF 块 `gguf_blocks_v1`(386：`iq1_m` 1、`iq2_s` 17、
  `iq2_xs` 9、`iq2_xxs` 6、`iq3_s` 131、`iq3_xxs` 68、`iq4_xs` 89、`q2_k` 19、`q4_k` 41、`q6_k` 5）、
  原生 row-split `row_split_k128_v1`(132：`q4_g64_fp16` 54、`q5_g64_fp16` 54、`q6_g64_fp16` 1、
  `q8_g32_fp16` 23)。
  artifact 中**不存在** FP8、NVFP4、T2、STQ、SM70 格式的权重。

**非目标**：MoE（`Qwen3_5MoeForCausalLM`）、Qwen3.8-Flash-Next、Gemma 4、Flint、
DeepSeek-V4.1-Flash、Vision 图像/视频输入、ngram/PLE/Engram、其它 artifact 家族。
Vision 与 MoE 代码的**保留期限**见 `CUTLIST.md`（Vision 取决于所有者对 `--vision` 的决定）。

## 3. 功能

保留：

- 文本 chat（`ninfer` CLI）与 HTTP serving（`ninfer-serve`，含已承诺的 OpenAI/Anthropic 协议行为）；
- `--spec mtp`（当前 `start-mtp.sh` 使用）与 `--spec dflash2`（artifact 带组件，可选保留）；
- `--lookup-ngram` / `--ngram-draft-tokens` 复制式起草（与 `--spec` 并行，默认开启，不属于额外的模型族）；
- paged KV + host KV arena + context/prefix cache（`--slot-*`、`--auto-long-anchors`、`--auto-prefix-grid` 等）；
- GDN、dense attention、CUDA Graph、`--invariant-attention`、`--gdn-state-fp16`、结构化输出、离线
  CausalScoring（`ninfer-perplexity`）；
- 每卡 device profile 与 `ninfer-calibrate`（4080 SUPER 的 profile 已内建）。

KV 存储格式：`bf16`（基线/调试）、`rk8v4`、`rk4v4`、`rk2v4-e8`。**不得**只保留一种。
`int8`、`fp8`、`rk4v4-e8`、`k8v4`、`nvfp4` 的去留见 `CUTLIST.md`（待所有者确认）。

并发：保留现有 `--max-concurrency 1..8` 语义，不以并发为目标也不删除（KV 流式首先在 C=1 上成立）。

**非目标**：图片/视频（除非所有者决定保留 Vision）、主动抢占、优先级/QoS、跨 Engine context store、
大规模 continuous batching、Python 推理路线、C++ SDK 导出、非 `.ninfer` 的模型格式。

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

- 唯一构建目标：`build.sh` 的 `sm_89`（Ninja、Release、`NINFER_BUILD_APPS=ON`）。
- 不保留 Volta/Turing/Blackwell 的构建分支与打包矩阵（Docker 的 `cuda12.9`/`cuda13.4` 多架构镜像、
  `docker/`、`compose.yaml` 的去留见 `CUTLIST.md`）。
- `LICENSE`、`NOTICE`、`CONTRIBUTORS.md` 保留（`CONTRIBUTORS.md` 中的 `kv_stream` Strata 归属
  必须保留，因为该实现仍在用）。

## 6. 出口条件

| 阶段 | 出口条件 |
|---|---|
| 1 删减 | `CUTLIST.md` 逐项确认；每步 `./build.sh` 通过、8K smoke 通过、固定 prompt greedy token 序列与基线一致；`git diff --check` 干净 |
| 2 流式 | 262 144 不 OOM；8K/32K/64K/128K/200K/262K decode 有真实数字；streaming on/off 同格式 token IDs 一致；resident 路径性能不回退；失败可回退（默认 off） |
| 3 优化 | 每个变量单独 A/B，同一主机同一 build，报告均值与范围；最差变化同样报告 |

## 7. 变更规则

1. 本文件是范围的唯一权威；`AGENTS.md`、`CONTRIBUTING.md`、`README.md` 的描述不得宽于本文件。
2. 扩范围需要所有者显式同意，并在同一次改动中更新本文件与受影响的构建/测试/文档。
3. `.ninfer` 仍是唯一 C++ 产品 artifact；CLI/serve/bench 只走公共 `ninfer::Engine`。
4. 代码所有权边界（Core / Artifact / Ops / Model / Runtime / Product）不变。
