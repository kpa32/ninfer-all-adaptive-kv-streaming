# KV 流式：运行记录

**每条记录都是一次真实运行留下的证据**：时间、git、命令行、二进制与模型摘要、显卡与链路、
程序自己写的 JSON 报告、以及生成的 token ids。跑一次就落盘；之后的对比读这些文件，而不是
把两边再跑一遍。工具是 [`tools/bench/run_kv_stream.py`](../../../tools/bench/run_kv_stream.py)。

人工结论仍写在 [`docs/maintainer/kv-streaming-measurements.md`](../../../docs/maintainer/kv-streaming-measurements.md)；
这里只放可机器比较的事实。

## 目录

```
bench/results/kv-stream/
  calibration.json                  prompt 配方 → token 数的标定（每个模型一条）
  golden/<case>.ids                 oracle（原版二进制，或回归基准）的 token ids
  golden/<case>.json                那次录制的命令行、摘要、prompt、时间
  runs/<UTC>-<suite>[-<tag>]-<sha8>/
    campaign.json                   这次调用的全部元数据 + 每个 case 的每次重复与判定
    <case>/<variant>/rep-NN/
      report.json                   程序自己写的运行报告（数值的唯一来源）
      stdout.log / stderr.log       原始输出（含 summary 与 generated ids）
      prompt.json                   prompt 原文（体积大，不入库；配方与 sha256 在 campaign 里）
  INDEX.md                          自动生成的索引（`report` 子命令）
```

## 用法

```bash
# 一次：标定 prompt 配方（约 1 分钟，两条短跑）
python tools/bench/run_kv_stream.py calibrate

# 等价性：先录原版金标，再用我们的构建比对
python tools/bench/run_kv_stream.py golden --suite equiv
python tools/bench/run_kv_stream.py check  --suite equiv

# 回归基准（原版答不了的长上下文，只能由我们自己录）
python tools/bench/run_kv_stream.py golden --suite perf

# 性能：一条命令跑完并落盘
python tools/bench/run_kv_stream.py bench --case stream-251k --reps 3 --tag ring2

# 索引与对比
python tools/bench/run_kv_stream.py report
python tools/bench/run_kv_stream.py compare runs/<A> runs/<B>
```

公共参数：`--binary`（我们的构建，默认 `build-sm89/apps/ninfer`）、`--oracle`（原版，默认
`/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer`）、`--model`、`--device`、`--timeout`、
`--allow-busy`。卡不是空闲的（>512 MiB 占用或有计算进程）时工具拒绝开始：不空闲的卡测不出
可比的数。

## case

| case | 目的 | prompt token | 容量 | 新增 | 变体 | oracle |
|---|---|---:|---:|---:|---|---|
| `equiv-8k` | 小上下文逐 id 回归 | 8 000 | 8 192 | 16 | 全驻留 | 原版 |
| `equiv-100k` | 中上下文与 100 K prefill 形状 | 100 000 | 100 096 | 16 | 全驻留 | 原版 |
| `equiv-183k` | 两种布局都成立的唯一区间：原版全驻留 vs 我们流式，逐 id 相同 | 183 000 | 184 064 | 512 | 全驻留 + 流式 | 原版 |
| `stream-251k` | 优化目标：262 144 原生上下文、MTP3、流式窗口 | 251 000 | 262 144 | 64 | 流式 | 自己（回归） |
| `resident-8k` | 同形状不开流式的卡速参照 | 8 000 | 8 192 | 64 | 全驻留 | 自己（回归） |

为什么是 183 K：原版带 MTP3 在 ~186 K 以上起不来；`rk4v4`+MTP3 的流式窗口是 2 835 页 =
181 440 token。所以只有 181 440–184 064 这个区间里，一边能跑、另一边确实在流式。512 个新增
token 让 decode 跨过每层 8 页，环复用与跨轮预取都被覆盖，而不只是 prefill 期的流式。

所有 case 固定：`--kv-capacity auto --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 256
--greedy --seed 42 --invariant-attention --spec mtp --draft-tokens 3`。`--prefill-chunk` 会移动
token（M-08），所以比对从不混两个问题。

## 判定

- `equiv-*`：生成 ids 必须与 `golden/<case>.ids` **逐 id 相同**；不同则记录第一处差异的位置与
  两个 id，判定 FAIL。
- `streamed` 变体额外要求：`stream.enabled` 且 `prompt_tokens > 窗口 token`（否则流式根本没参与，
  这次运行不算证据）。
- `stream.rounds_series` 与 `decode_rounds.samples` 必须等长（两边的逐轮序列按轮对齐）；不等长
  记录为 FAIL。
- 报告里的 decode 速度必须与 stderr summary 的 `decode speed` 一致（2% 内），防止 JSON 与文本
  两条路径给出不同的数。
- `timing_lost > 0` 是警示不是失败：它表示有事件对在被覆盖前没来得及读取，拷贝时间会偏低，
  而不是拷贝更便宜。

## 对比

`compare` 打印两边共同出现过的 case/variant 的指标差：`decode_tok_s`、逐轮时间（中位数与 p90）、
接受长度、每轮 H2D/D2H、拷贝占用、暴露等待、有效链路带宽、主机与设备占用。同时打印两边的
HEAD、GPU、驱动、PCIe、SM 时钟——跨进程比较时这些是可比性的前提，工具不会替你判断时钟漂移。

真正要证明「这次改动更快」时，仍需交错 A/B：见
[`tools/bench/README.md`](../../../tools/bench/README.md) 的 `run_interleaved_ab.py`。本工具负责
把每次运行完整留痕，供跨天、跨提交观测。

## 指标去哪读

程序的 JSON 报告是数值的唯一来源（`--report-json`）：

| 段 | 内容 |
|---|---|
| `run` / `target` / `devices` | 时间、argv、build、模型与容量、GPU 名称/驱动/算力/显存 |
| `timings` / `decode_rounds` | prepare/prefill/decode/total、逐轮 `{seconds, tokens}` |
| `memory` / `stream` | KV 池、窗口、环、主机权威副本、逐轮 `{H2D, D2H, copy, exposed}` 与逐层样本 |
| `speculative` | 草稿/接受、接受率与接受长度、逐位接受 |
| `kv_transfer` | 主/后端 KV 与状态的 D2H/H2D/D2D 页数与字节 |
