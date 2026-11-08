---
id: adaptive-kv-streaming-inputs
type: architecture-design
title: 自适应 KV 流式的外部资料（参考文章与 CUDA Graph 说明）
status: active
---

# 自适应 KV 流式：外部输入资料

**这份文件是设计输入，不是 NInfer 当前状态的描述。** 里面每一条关于 NInfer 的说法都必须在
`src/core/decode_graph.*`、`src/core/paged_kv_*`、`src/models/qwen3_5/state/` 里核实过才能当事实用；
核实结果记在 [kv-streaming-design](kv-streaming-design.md)。文章的作者环境是 RTX 5070 Ti（PCIe 5.0、
Blackwell），本机是 RTX 4080 SUPER（PCIe 4.0、sm_89），带宽和内核都不同，数字不可直接搬。

---

## 附录 A：参考文章全文

标题：在 16G 显存上运行 Qwen 27B 并实现完整上下文长度：为 llama.cpp 构建自适应 KV 缓存流
作者：Raymond Huang
日期：2026-08-28
原文：https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9

说明：以下为从桌面 HTML 提取的机器翻译全文，已去掉导航、广告、评论，并修正明显错译；
技术内容以原文和作者代码为准。文章中也写作 Qwen3.5-27B，与 Qwen3.8-27B 是同一 Dense 27B 几何。

### 头痛

自托管一个强大的 LLM 需要昂贵的机器。DGX Spark 约 4000 美元，36GB 统一内存的 Mac 约 2500 美元，
二手 24GB RTX 4090 也可能超过 2000 美元。如果你只有 16GB 或更低显存的显卡，选择基本只有
10B 级 dense 或 30B 级 MoE。想把 300 亿参数塞进 16GB，通常要降到 Q2、牺牲质量、缩短上下文。

但作者在 RTX 5070 Ti 16GB 上跑 Qwen3.8-27B，UD-Q3-XL 量化，262K 上下文，Q8 K cache / Q4 V cache：

- 8K 上下文：约 50 tok/s
- 49K：约 40 tok/s
- 114K：约 30 tok/s
- 176K：约 20 tok/s
- 253K：仍有约 10 tok/s

### 显存超额订阅

作者的 NAS 里有一块 RTX 5070 Ti，平时还要跑视频转码和人脸识别。LLM、转码、识别同时抢显存会 OOM。
他想要类似操作系统分页的机制：进程可以占用较多显存，其他进程需要时自动驱逐，而不是崩溃。

llama.cpp 已经通过 `cudaMallocManaged` 支持 CUDA 统一内存（UVM）：
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 时，显存不够会溢出到系统内存。作者把层卸载回 GPU，
让 llama-server 使用近 15.8GB 显存，转码/识别启动时也不会崩，但预填充和解码会暂时明显变慢。

### 以 UVM 模式运行 27B 模型和完整上下文

第一次用 Qwen3.8-27B Q4_K_XL、262,144 上下文启 UVM：模型权重本身超过 16GB，预热就崩溃。

换成 UD-Q3_K_XL（约 13GB）后能启动，预填充/解码正常。但上下文增长后显存逐渐被占满，
解码速度明显下降。大约 120K 上下文时，模型权重和完整 KV 无法同时放进显存，UVM 页面抖动开始。
UVM 不知道哪些页对解码更重要，可能驱逐 GPU 马上要用的权重或 KV 页；这些页又要迁回来，
继续驱逐别的页，形成来回迁移，解码性能崩溃。

作者试过 `cudaMemAdvise` 给 CUDA 提示，但没有实质效果。结论：普通 UVM 不适合“大 dense 模型 + 长上下文”。

### 在 VRAM 中定义驱逐策略

先看 LLM 解码的特点：

- Qwen3 系列有些层是线性注意力，KV 不随上下文增长；有些层是 full attention，KV 随上下文线性增长。
- 解码时层按顺序执行，后一层要等前一层。
- 每一层的 KV 只在该层计算时被访问；该层算完，KV 到下一个 token 之前都闲置。
- 因此，一层算完就可以把它的 KV 从 VRAM 移除，给别的层腾空间。
- 但重新加载需要时间；如果等 GPU 需要时才拷贝，GPU 会停等。必须提前预取。

结论：在 VRAM 里放一个环形缓冲区。

### VRAM 中的环形缓冲区用于 KV 缓存流式传输

以 Qwen3.8-27B 为例，生成一个 token 要经过 16 个 full-attention 层。
假设环形缓冲区有 4 个 slot：P1、P2、P3、P4，每个 slot 足够放一个 full-attention 层的 KV。

- 第 1 层开始时，P1 已经放着 C1，P2/P3/P4 分别放着 C2/C3/C4。
- 第 1 层算完，转到第 2 层，P2 的 C2 在使用。C1 到下一个 token 之前不再需要，
  可以把它从 P1 移除，并把第 5 层的 C5 预取到空出的 P1。
- 第 2、3、4 层同理。理想情况下，第 5 层开始时 C5 已经加载好，GPU 不用停等。

看起来很理想：只在 VRAM 保留一个小工作集，通过 ring 流式传输每一层的 KV。

但 PCIe 带宽很快成为瓶颈。PCIe 5.0 x16 理论约 64GB/s 每方向，作者实测有效约 50GB/s。
如果每个 token 都把系统内存里的全部 KV 流一遍，最大 tok/s 约为：

    PCIe 有效带宽 / 每 token 要传的 KV 字节数

Q8 K / Q4 V、100K 上下文时，Qwen3.8-27B 的 KV 约 2.5GB。按 50GB/s：

    50 GB/s ÷ 2.5 GB ≈ 20 tok/s

这还没算额外传输和同步开销。而 GPU 本身计算速度远高于此。如果每个 token 都流全部 KV，
GPU 只会等 PCIe。解决了显存容量，又引入带宽问题。

所以不能把整个 KV 变成一个大 ring、每 token 全走 PCIe。必须仔细决定每个上下文长度下
“流多少、留多少”：只有一部分 KV 在系统内存和显存之间移动，其余驻留显存。
这样较小的预取就能跟上 GPU 计算，而不是让 GPU 等 PCIe。

### 具有自动重新分区功能的自适应 KV 缓存流

- 上下文还小、整个 KV 能放进固定显存空间时：不流式传输，ring 保持禁用。
- 上下文超出固定空间时：需要为 ring 腾空间。扩大 ring 意味着从显存移除一部分 KV，
  这些 KV 每次对应层计算时都要从系统内存加载。
- 权衡：ring 要足够大，提供足够预取槽，让 KV 在 GPU 到达该层前完成预取；
  ring 也要尽可能小，因为每个被驱逐的驻留 KV 都会在每个 token 上增加 PCIe 流量。

作者实现的自适应重分区策略：

1. 尽可能保持所有数据驻留。只要固定显存空间能容纳整个 KV，ring 就禁用。
2. 需要流式时，从小规模开始。KV 不再容纳时，从驻留缓存释放一小部分，
   公平分配到各个 full-attention 层；释放出的 VRAM 成为流式层共享的 ring。
3. 只在必要时扩展 ring。上下文继续增长，每个流式区域变大、预取时间变长；
   现有 ring 可能不足以隐藏延迟。此时再从驻留 KV 移除一小部分，扩大 ring。
4. 但有一个限制：上下文越长，每 token 要流的 KV 越多，最终 PCIe 本身成为瓶颈。
   预取变慢不是因为 ring 太小，而是已经传了太多数据。
5. 如果继续扩大 ring，就会驱逐更多驻留 KV，每 token 传更多数据，PCIe 压力更大，
   形成恶性循环：
   更大的 ring → 更少驻留 KV → 更多流式传输 → 更高 PCIe 利用率 → 更长预取延迟 → 更大的 ring。
6. 因此增加保障：只有当扩大 ring 仍有助于隐藏预取延迟时才扩大；
   如果 PCIe 带宽已饱和，就停止扩大 ring。

### 结果

基准设置：

- 操作系统：Ubuntu 24.04
- 内核：Linux 6.14.0，内核参数 `iommu=pt`
- 显卡：RTX 5070 Ti 16GB
- 模型：Unsloth Qwen3.8-27B
- 量化：UD-Q3-K-XL
- KV 缓存：Q8 K / Q4 V
- 上下文：`--ctx-size = 预填充长度 + 256`
- 解码基准：256 个生成 token 的平均解码速度

结果：

- 上下文约 120K 之前，修改版和标准 llama.cpp 几乎一样：整个 KV 仍能放进显存，
  流式机制不需要工作；解码速度随上下文增加而下降是正常计算负载。
- 约 120K 之后，标准 llama.cpp 开始过度占用显存，UVM 页面抖动，解码速度快速下降；
  修改版靠 ring + 前瞻预取继续沿稳定坡度下降。
- 200K 上下文时，修改版仍约 20 tok/s。
- 推到 Qwen3.8-27B 完整原生上下文（不用 YaRN）时，仍约 13 tok/s。
- 换成更大的 UD-IQ4-XS 量化后，权重多占约 1GB，流式机制更早启动、要处理更多 KV，
  但解码速度曲线仍保持稳定斜率。

### 结论

- 这套改进主要利于用中端消费级 GPU 托管本地 LLM、跑 agent 工作负载的用户。
- 如果 15 tok/s 是可接受的流畅聊天下限，用这套实现可以把上下文压缩阈值放到约 200K token。
- 以前可用上下文基本受限于显存中 KV 的物理容量（Q8 K/Q4 V 约 100K token）；
  只开 UVM 不能真正解决，因为显存超额后页面抖动会严重拖慢解码。
- 自适应 KV 流式让 VRAM 不再是硬墙。
- 作者现在用 RTX 5070 Ti 跑 Qwen3.8-27B UD-Q3-K-XL，完整原生上下文，开 UVM，
  Hermes agent 配置在约 200K token 处压缩，运行得很好。

### 信用

特别感谢朱雯雯绘制封面图片和制作技术动画。

---

## 附录 B：CUDA Graph 与流式 KV 的动态替换（所有者提供，待代码验证）

以下为所有者提供的技术资料。它是设计假设和实现方向，不是 NInfer 现状的证明；
实现前必须在 NInfer 的 `decode_graph`、paged KV、block table、graph capture 代码中逐条核实。

### 一、CUDA Graph 的静态与动态边界

很多人以为“图一旦捕获，里面所有东西都焊死了，必须从头到尾一成不变”。事实并非如此。
要理解流式 KV 能否在 CUDA Graph 下动态替换，需要分清 CUDA Graph 固化了什么、允许动态改变什么。

| 维度 | 在 CUDA Graph 中是否静态？ | 为什么 |
|---|---|---|
| 拓扑结构（DAG） | 绝对静态 | 算子执行顺序、依赖边在捕获后固化，驱动借此消除 CPU 调度延迟 |
| Kernel 内部参数指针 | 默认静态 | 捕获时传给 kernel 的参数列表（如 `void*` 地址）被写进节点描述符 |
| 指针指向的显存数据 | 完全动态（运行时可变） | GPU 执行内核时，会去该指针指向的全局显存中现读现算 |

流式 KV 不需要改动图的拓扑，它只需要内核“去读取流式进来的那一页”。这可以通过以下机制实现。

### 二、支持流式 KV 动态替换的核心原理

#### 1. 核心机制：内存间接寻址（PagedAttention 的 Block Table）

这是大模型推理引擎配合 CUDA Graph 的核心范式：

- 捕获阶段：Attention 内核的签名不是直接传 KV 显存首地址，而是传入一个预先分配好的页表指针：
  `paged_attention_kernel(..., int* block_table, ...);`
  在 CUDA Graph 建立时，`block_table` 的显存虚拟地址是固定不变的。
- 执行阶段：Graph 不知道也不在乎 `block_table` 数组里填的具体数字是多少。
  - 常规 decode：随着 token 增加，CPU 将新申请的页号写入这个数组。
  - 流式 KV 替换：假设第 12 层的第 50 个历史页被卸载到 Host，当前解码步它被拉回了显存的
    Ring Slot #2。你只需要在 launch 之前（或异步 H2D）将页表里的这一项改写：
    `block_table[layer=12][page=50] = Slot_2_Physical_Page_ID`
  - 当 CUDA Graph 里的 Attention 内核执行时，它解引用 block_table，动态查出当前这一页该去
    Slot #2 读。整个图既不需要重新捕获，也不需要任何驱动级开销，完成动态物理替换。

用户注：ninfer 文档提到 page IDs 是稳定输入数据、不是 graph key，且 block table publication
通过 pinned shadow 逐步 H2D 发布；这一点需要在 `docs/maintainer/paged-kv-cache.md` 和
`src/core/decode_graph.*` 中核实。

#### 2. CUDA 原生机制：节点参数原地打补丁（`cudaGraphExecKernelNodeSetParams`）

如果某种流式设计不走页表，而是想把 Attention 内核直接绑定的输入显存指针换掉，CUDA 原生也支持：

- NVIDIA 提供 `cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams)`。
- 它允许直接修改已实例化图中某节点的入参（如更换 buffer 指针、更换 scalar 参数），
  不需要重新捕获，也不需要重新实例化整张图。
- 开销：单次节点参数修改约 0.5–1 微秒，远低于重新 capture 或普通 eager launch 的开销（5–20 微秒）。
- 不过在推理引擎中，机制 1（改 Block Table 显存）通常比机制 2 更快，因为改显存数据
  可以完全通过 GPU 异步流完成，连这 1 微秒的 CPU 驱动 API 调用都省了。

#### 3. 动态虚拟内存重映射（CUDA VMM：`cuMemMap`）

ninfer 现有的 `--vision-residency overlay`（平时在内存，用时借显存）底层用的就是 CUDA VMM：

- 原理：虚拟地址在 CUDA Graph 里始终固定，但底层绑定的物理页（`CUmemGenericAllocationHandle`）
  可以通过 `cuMemUnmap` / `cuMemMap` 动态切换。
- 局限：VMM 是操作系统和 GPU MMU 驱动级的映射，每次调用需要几十微秒，
  适合粗粒度的整图/整层借还（如视觉编码阶段）；对于每一步 decode 的微秒级循环，
  改页表（机制 1）更合理。

### 三、流式 KV 最关键的“另一半”：Graph 内跨流异步同步

流式传输不仅要求“换页”，还要求“算到第 l 层时，第 l+2 层必须刚好在后台从 Host 拷贝到显存完成，
不能发生数据竞争”。

普通 stream 模式下很容易（`cudaStreamWaitEvent`）。捕获进 CUDA Graph 后，图还能等外部流的传输吗？

答案：完全支持，依赖 `cudaGraphAddEventWaitNode`。

- 原理：CUDA Graph 原生支持多流捕获与事件节点（Event Node）。
  1. 捕获图时，Compute Stream 在执行到第 l+2 层的 Attention 计算节点之前，捕获一个 WaitEvent 依赖。
  2. 这会在 Graph DAG 内部生成一个 `cudaGraphEventWaitNode`。
  3. 执行时：专门的 Restore Stream 异步把第 l+2 层 KV 搬进 Ring Buffer，搬完后执行
     `cudaEventRecord`；Graph 内部运行到该节点时，由 GPU 硬件工作分发器自动等待该事件触发，
     随后立刻启动计算。
- 行业实战：vLLM、TensorRT-LLM 在处理 DeepSeek MLA / MoE 权重的重叠通信、双流并行时，
  都是把跨流事件捕获进 CUDA Graph 来掩盖传输延迟。

### 总结

回到问题：CUDA Graph 原理上支持流式 KV 的动态替换吗？

1. **寻址层面：完全支持。** 通过 Block Table 间接寻址，CUDA Graph 锁死的是页表指针，
   页表里的物理页号是纯显存数据，每步改写即可让内核读到流式页，完全规避图重捕获。
2. **流水线同步层面：完全支持。** 通过图内的 Event Wait Node，可以让 Graph 执行到特定层时
   硬件级等待外部 H2D 拷贝流完成，实现流式重叠。

因此，从 CUDA 底层原理看，CUDA Graph 不是流式 KV 的绊脚石；相反，正是它的 Block Table 机制和
Event Node 机制，让流式 KV 能以零 CPU 调度开销跑在微秒级 decode 循环里。
