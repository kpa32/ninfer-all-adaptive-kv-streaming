# KV 流式：运行记录

**每条记录都是一次真实运行留下的证据**：时间、git、命令行、二进制与模型摘要、显卡与链路、
程序自己写的 JSON 报告、以及生成的 token ids 与 logprobs。跑一次就落盘；之后的对比读这些
文件，而不是把两边再跑一遍。工具是
[`tools/bench/run_kv_stream.py`](../../../tools/bench/run_kv_stream.py)。

人工结论仍写在 [`docs/maintainer/kv-streaming-measurements.md`](../../../docs/maintainer/kv-streaming-measurements.md)；
这里只放可机器比较的事实。

## 金标是谁

| 角色 | 二进制 | 用途 |
|---|---|---|
| **等价性 oracle** | `/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer`（所有者自己编译的原版，2026-11-04.3+g…） | 只**执行**，不写、不编、不 git。它没有流式特性，所以它给出的答案就是"流式必须与之逐 token 相同"的答案 |
| 回归基线 | 我们的 `build-sm89/apps/ninfer` | 长上下文（>186 K）原版起不来，只能用我们自己录的 ids 作**回归**基准；它**不是**等价性证据，`golden/<case>.json` 里 `oracle` 字段写的是 `self` |

`check` 只与 `oracle: original` 的金标比对。回归基线的行在 INDEX 里同样标 `golden`，
但元数据区分得开。

## 目录

```
bench/results/kv-stream/
  calibration.json                  prompt 配方 → token 数的标定（每个模型一条）
  window-scan.json                  (容量, headroom) → 流式窗口，来自 `window` 子命令
  golden/<case>.ids                 oracle 的 token ids（逐 id 比对的那个文件）
  golden/<case>.json                那次录制的命令行、版本、摘要、prompt/tail 的 sha256、时间
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
python tools/bench/run_kv_stream.py calibrate                    # 一次：prompt 配方 → token 数
python tools/bench/run_kv_stream.py window --capacity 184064 --headroom-mib 1008
python tools/bench/run_kv_stream.py golden --suite equiv         # 原版录 ids
python tools/bench/run_kv_stream.py check  --suite equiv         # 我们的构建比对（逐 id + logprobs）
python tools/bench/run_kv_stream.py golden --suite perf          # 长上下文回归基线（我们自己的）
python tools/bench/run_kv_stream.py bench  --case stream-251k --reps 3 --tag ring2
python tools/bench/run_kv_stream.py report
python tools/bench/run_kv_stream.py compare runs/<A> runs/<B>
```

公共参数：`--binary`、`--oracle`、`--model`、`--device`、`--timeout`、`--allow-busy`、
`--no-logprobs`。卡不空闲（>512 MiB 占用或有计算进程）时工具拒绝开始。

## case

| case | 目的 | prompt token | 容量 | 新增 | 变体 | oracle |
|---|---|---:|---:|---:|---|---|
| `equiv-8k` | 小上下文逐 id 回归 | 8 014 | 8 192 | 16 | 全驻留 | 原版 |
| `equiv-100k` | 中上下文与 100 K prefill 形状 | 99 984 | 100 096 | 16 | 全驻留 | 原版 |
| `equiv-183k` | 两种布局都成立的唯一区间 | 183 012 | 184 064 | 512 | 全驻留 + 流式 | 原版 |
| `stream-251k` | 优化目标：262 144 原生上下文、MTP3 | 251 012 | 262 144 | 64 | 流式 | 自己（回归） |
| `resident-8k` | 同形状不开流式的卡速参照 | 8 014 | 8 192 | 64 | 全驻留 | 自己（回归） |
| `prompt-shape` | prompt 配方本身的守卫：8 K 下必须真的生成满 512 token | 8 014 | 8 192 | 512 | 全驻留 | 自己（回归） |

为什么是 183 K：原版带 MTP3 时 1 GiB 自动 headroom 下装不下 184 064 的预留（3.99 GB vs 可用
4.03 GB），要给 `--kv-headroom-mib 16`；流式默认 64 MiB 头寸又会把 183 K 整个装进窗口而不流式。
`window-scan.json` 记录：容量 184 064 + headroom 1 008 MiB → 窗口 2 111 页 = 135 104 token。
所以 183 K 确实在流，且 decode 的 512 个 token 会跨过 8 页/层。

`prompt-shape` 存在的原因：最初用 "The capital of France is" 收尾，贪婪解码在第 57 个 token
就命中 stop token，183 K 的 decode 只跑了 19 轮、一页没跨——那个 case 名义上测流式，实际只测了
prefill。现在写法是让模型接着数数，并把"生成数 == --max-new"变成硬检查。

所有 case 固定：`--kv-capacity auto --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 256
--greedy --seed 42 --invariant-attention --spec mtp --draft-tokens 3`。`--prefill-chunk` 会移动
token（M-08），所以比对从不混两个问题。

## 判定（这就是"标注"）

判定不靠印象，全部由记录推出，任一条不满足即 FAIL：

| 检查 | 含义 |
|---|---|
| `ids` | 生成 ids 与 `golden/<case>.ids` **逐 id 相同**；不同则记录第一处差异位置与两个 id |
| `logprobs_match_across_layouts` | 同一二进制、同一 prompt，全驻留与流式的 logprob 数组**逐位相同**（SPEC 完成条件 2） |
| `generated_all_requested` | 真的生成了 `--max-new` 个 token（没被 stop token 提前收尾） |
| `stream_engaged` | 流式变体确实在流：`stream.enabled` 且 `prompt_tokens > 窗口 token` |
| `stream_series_aligned` | `stream.rounds_series` 与 `decode_rounds.samples` 等长，逐轮对齐 |
| `stream_series_matches_total` | 逐轮上载之和 == 总上载 − prefill 上载（1% 内），防止"每轮"数里混进 prefill |
| `report_matches_summary` | 报告里的 decode 速度与 stderr summary 的 `decode speed` 一致（2% 内） |
| `printed_ids_match_report` | `--print-token-ids` 打印的 ids 与报告里的 ids 相同 |

`timing_lost > 0` 是警示不是失败：表示有事件对在被覆盖前没来得及读取，拷贝时间会偏低，而不是
拷贝更便宜。

## 可复现性

- **参数**：全部固定在 `FIXED_FLAGS`，每次运行的完整 `argv` 写进记录；case 的容量、变体、额外
  开关（headroom）也在记录里。
- **prompt**：不是一个文件，而是一个配方（句子 + 重复次数 + 尾巴）。原文与 sha256 都入库；
  token 数由 `calibrate` 的线性拟合放置，**实际 token 数**在每个 case 的 `metrics.prompt_tokens`
  里，并受容量与窗口检查约束。
- **数值**：greedy + `--seed 42` + 固定 `--prefill-chunk`；`--invariant-attention` 让注意力路线
  与宽度无关。ids 一旦与金标相同，后续每次 `check` 都重验，不依赖"上次好像是这个"。
- **速度**：会随 GPU 时钟、温度、功耗漂移，所以每次重复单独记录，索引里用中位数；记录里带
  GPU 名称/驱动/PCIe/时钟，`compare` 会把它们一起打印。要证明"这次改动更快"，仍用
  [`run_interleaved_ab.py`](../../../tools/bench/README.md) 做交错 A/B；本工具负责留痕与跨天观测。
- **不覆盖**：同一条命令再跑一次就是新目录，旧记录不会被改写。

## 与项目其它工具的关系

| 工具 | 什么时候用它 |
|---|---|
| `bench/` 的 `ninfer_bench` | 产品吞吐（pp/tg 矩阵、JSON schema）。它没有 `--kv-stream`，所以流式路径走本工具；它用 `bench/fixtures/bench_corpus.ids` 切**精确 token 数**，比文本配方更硬 |
| `tools/bench/run_interleaved_ab.py` | 两个构建/两套参数的可比 A/B（配对中位数、漂移对照） |
| `tools/bench/speed_of_light.py --baseline` | 已有 JSON 报告的事后归因（与 roofline 比），思路与本工具 `compare` 相同 |
| `tests/models/qwen3_5/*_real.cpp` | Engine 级行为测试（需要 `BUILD_TESTING=ON` 的构建树；本机暂无） |
| `docs/maintainer/kv-streaming-measurements.md` | 人工结论与"这条数据决定了什么" |

## 指标去哪读

程序的 JSON 报告是数值的唯一来源（`--report-json`）：

| 段 | 内容 |
|---|---|
| `run` / `target` / `devices` | 时间、argv、build、模型与容量、GPU 名称/驱动/算力/显存 |
| `timings` / `decode_rounds` | prepare/prefill/decode/total、逐轮 `{seconds, tokens}` |
| `memory` / `stream` | KV 池、窗口、环、主机权威副本、逐轮 `{H2D, D2H, copy, exposed}` 与逐层样本 |
| `speculative` | 草稿/接受、接受率与接受长度、逐位接受 |
| `generation` | ids 与（`--logprobs` 时）每个 token 的 logprob 及 top-k |
| `kv_transfer` | 主/后端 KV 与状态的 D2H/H2D/D2D 页数与字节 |
