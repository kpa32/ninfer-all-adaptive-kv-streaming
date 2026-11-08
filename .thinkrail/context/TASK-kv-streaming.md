# TASK: 自适应 KV 流式（--kv-stream）

status: draft

## Request

在 RTX 4080 SUPER（16 376 MiB）上让 Qwen3.8-27B Dense（IQ3_S，权重 11.6 GiB）跑满
**262 144 原生上下文**，`rk8v4` 为最低质量档、稳定后支持 `int8`；`--kv-stream` opt-in，
关闭时与今天逐字节一致；同格式开/关的 greedy token ID 与 logprobs 完全一致。

完整设计见 `docs/maintainer/kv-streaming-design.md`（496 行，已获所有者批准）。本文件只记录
本次要执行的范围、已定的决定、以及施工中确认/推翻的事实。

## Product

已定（所有者多轮确认，不再重问）：

| 维度 | 结论 |
|---|---|
| 为什么 | 16 GB 装不下 262 144（`rk8v4` 要 7.33 GB，预算 3.07 GB）；降 KV 质量被否决；UVM 被文章证明会崩 |
| 用户看到什么 | 同一个 CLI，多一个 `--kv-stream`；262 144 能启动并 decode，tok/s 与解释一起报告 |
| 完成判据 | 见 `SPEC.md` 的 5 条完成条件（262 144 `rk8v4`/`int8` 稳定、逐 token 一致、默认关、诚实速度、不回退 120K 全驻留） |
| 不做 | 删模型族/架构；UVM；降 KV 质量；预填充阶段流式 |
| 失败时 | 启动即拒绝并说明原因，不静默降级 |

## Design（已批准，摘要）

**核心算术**：per-(layer,rank,kind) plane 下环不能跨层复用 → `16(R+S) ≥ 16P`，流式不可能省内存。
→ 新建**扁平槽池**（一槽 = 一层一页，K/V/Kscale/Vscale 四平面）+ **每层一张表**；
注意力 `(pages, table)` 签名不变。

**映射**：`p < active−R → slot = 16R + p`（恒等，永不变）；`p ≥ active−R → slot = L·R + ((p−active+R) mod R)`
（常驻窗口，每 64 token 转一格）。

**容量**：`16R + S ≤ 29 392`，`S ≥ P − R` ⇒ `15R ≥ 16P − 29 392`。262 144：R=1686/层（f=0.41），
环 2410 页（246 MB），每轮搬 4.03 GB。

**每轮序列（M1 单流）**：每层 1–3 条 2D H2D（命令数 = 层×plane，不是页数）+ `ready`/`consumed`
事件；图内 `cudaEventWaitExternal`；无重叠。

**上界**：MVP 16.4 tok/s；分段流水 20.5；`int8` 10.8/12.8。带宽 25 GB/s 待实测。

## 执行顺序

M0 核实（只读）→ M1 扁平槽池/表/写入路径/单流序列 → M2 逐 token 一致 → M3 控制器 →
M4 分段流水 → M5 共存。详见设计文档 §10 与 TODO 计划。

## 未确认的假设

1. 块表在多次图重放间保持同一设备地址（M0-1）。
2. `cudaEventWaitExternal` 支持每轮重录同一事件（M0-2）。
3. `copy_from_host` 可在图外投递、图内配对等待（M0-3）。
4. PCIe 4.0 x16 有效带宽 ≈ 25 GB/s（M0-5，需实测）。
