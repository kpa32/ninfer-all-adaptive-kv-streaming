# KV 流式：测量存根

每条记录都是真实运行的输出，用来说明"当时为什么这么改"。设计结论在
[`kv-streaming-design.md`](kv-streaming-design.md)，这里只存可复现的原始证据：时间、提交、命令、
结果、以及这条数据被用来决定什么。**新测量追加到末尾，不覆盖旧的**；结论变了就再写一条并指明它
推翻了哪一条。

## 环境

| 项 | 值 |
|---|---|
| 主机 | `kpa@192.168.2.13`，Linux，16 核 |
| 仓库 | `~/ninfer-fork/ninfer-all-adaptive-kv-streaming`，分支 `kv-streaming` |
| 构建 | `build-sm89`（Release，`BUILD_TESTING=OFF`），`cmake --build build-sm89 -j 14` |
| GPU | NVIDIA GeForce RTX 4080 SUPER，16 376 MiB，驱动 615.71.09，sm_89 |
| 模型 | `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（11.6 GiB 权重） |
| KV 格式 | `rk4v4`（每 token 17 920 B，每页 71 680 B，页 = 64 token） |

## 通用命令

除注明外，每次测量都用同一组参数：

```bash
# 251K 提示：246 652 token 的 α 文本，几乎吃满 262 144 的容量
python3 -c 'import json;json.dump([{"role":"user","content":("alpha "*246600).strip()}],open("/tmp/v_p.json","w"))'

./build-sm89/apps/ninfer ~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --messages /tmp/v_p.json --max-context 262144 --kv-capacity auto --kv-stream \
  --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 256 --greedy --invariant-attention \
  --max-new 64 --spec mtp --draft-tokens 3 [--kv-stream-report]
```

`--kv-stream-report` 打开拷贝/等待计时（默认关；实测开关代价在噪声内，见 M-04）。
`decode speed` 除以 `mtp acceptance length` 得到**每轮时间**——接受长度随采样文本浮动，轮时间是
可比的量，tok/s 不是。

---

## M-01 池子按实测空闲分配，头寸从 256 MiB 降到 64 MiB 地板

- 时间：2026-11-08（本会话）
- 提交：`987ef5fd`（头寸 64 MiB）+ `e2f61ed8`（清掉失败的 `cudaMalloc` 留下的错误）+ `3a545b40`（graft 计数作用域）
- 命令：同上，无 `--kv-stream-report`；另加一次 `--kv-headroom-mib 1` 强制触发回退
- 结果：

| 场景 | 池 | 启动后余 | decode |
|---|---:|---:|---:|
| 默认（64 MiB 地板），8K 提示 | 47 894 / 65 536 页组 | 53.5 MiB | 83.7 tok/s |
| `--kv-headroom-mib 1`（故意装不下） | 首试 48 816 失败 → 重试 47 880 | — | 83.8 tok/s |

- 日志证据（`--kv-headroom-mib 1`）：`KV capacity: 48816 page groups did not fit … retrying with 47880`
- 用来决定：头寸不再是安全余量，而是"计划误差的下限"，真正的兜底是这 6 次 64 MiB 回退。

## M-02 仪表：k=1（环深一层）时的拷贝与暴露

- 时间：2026-11-08
- 提交：`a434767f`（`--kv-stream-report`），运行于 `cab7c33e` 之后
- 命令：通用命令 + `--kv-stream-report`
- 结果（251K，池 47 894，窗口 2 919 页/层，环 936 页 × 16 层）：

| 量 | 值 |
|---|---:|
| decode | 36.4 tok/s（接受长度 2.33）→ **每轮 64.0 ms** |
| 每轮上载 | 1 023.7 MiB |
| 拷贝流占用 | 42.57 ms |
| 计算流等拷贝（暴露） | **22.65 ms** |
| 每层 | 拷贝 2.66 ms，只藏住 1.24 ms，露出 1.42 ms |

- 同一配置关掉报告：36.5 tok/s（开关代价在噪声内）
- 用来决定：调度还有 ~50% 的空间（完美重叠 = 拷贝本身 42.57 ms → 54.7 tok/s），于是做环深 2。

## M-03 环深 k=2：拷贝提前一层

- 时间：2026-11-09T00:2x+08:00（本会话）
- 提交：`1adfea6b`（环分区 + 提前一层）+ `db4c3f3e`（零窗口锚点修复）
- 命令：通用命令，报告开/关各跑多次
- 结果（251K，池 47 894，窗口 **2 835** 页/层，环 **1 020** 页 × 16 层）：

| 运行 | decode | 接受长度 | 每轮 | 拷贝占用 | 暴露 | 每轮上载 |
|---|---:|---:|---:|---:|---:|---:|
| 报告关 ① | 46.7 | 2.62 | 56.1 ms | — | — | — |
| 报告开 ① | 48.2 | 2.74 | 56.8 ms | 47.57 ms | 11.60 ms | 1.09 GiB |
| 报告开 ② | 53.0 | 3.00 | 56.6 ms | 47.37 ms | 11.32 ms | 1.09 GiB |
| 报告开 ③ | 50.3 | 2.86 | 56.9 ms | 47.76 ms | 11.78 ms | 1.09 GiB |
| 报告关 ② | 50.7 | 2.95 | 58.2 ms | — | — | — |

- 与 M-02 比：**每轮 64.0 → 56.9 ms（−11%）**，暴露 22.65 → 11.4 ms（减半），代价是窗口
  2 919 → 2 835 页/层，上载 1 024 → 1 090 MiB（拷贝占用 42.6 → 47.5 ms）。
- tok/s 从 36.4 涨到 48–53，其中一部分来自接受长度（2.33 → 2.74–3.00），所以按轮时间比较。
- 用来决定：保留 k=2。

## M-04 环深 k=3：更深的环不减少暴露

- 时间：2026-11-09T00:4x+08:00
- 提交：`1e1d73a4`（临时把 `kKVStreamRingDepth` 改成 3）
- 命令：通用命令 + `--kv-stream-report`，跑两次
- 结果（窗口 2 738 页/层，环 1 117 页 × 16 层）：

| 运行 | decode | 接受长度 | 每轮 | 拷贝占用 | 暴露 | 每轮上载 |
|---|---:|---:|---:|---:|---:|---:|
| ① | 46.9 | 2.86 | 61.0 ms | 51.86 ms | 11.67 ms | 1.19 GiB |
| ② | 46.9 | 2.95 | 62.9 ms | 51.92 ms | 11.77 ms | 1.19 GiB |

- 关键否定证据：**暴露时间不随环深变（11.4 → 11.7 ms）**，而拷贝量涨了 9%。所以残余的
  0.7 ms/层 不是"环不够深、拷贝赶不上"，而是每轮固定的起/停：轮首前 `ring_depth` 层没有可重叠的
  计算，轮末最后一层的拷贝无处可藏。
- 用来决定：环深回到 2（`a60647b1`）；要再拿回这 11 ms，需要**跨轮预取**（在上一轮的末尾为下一轮
  的头几层发上载），不是加深环。

## M-05 深环的历史测量（M4a，结论已作废但留档）

- 时间：更早，提交 `94ad5f54` → 撤回 `6570f794`
- 结果（rk8v4 120K / rk4v4 190K）：depth 1/2/14 = 81.7/80.4/65.6；73.9/73.0/71.9/68.1
- 为什么作废：那两个上下文**环是空的**（全驻留，根本不拷），加深环只剩窗口变小，必然更慢。
  这条记录留在原因是防止把"深环没用"当成一般结论——251K 处它是有效的（M-03）。

## 待测（下一步的候选，尚无数据）

| 候选 | 预期 | 依据 |
|---|---|---|
| 跨轮预取（轮末发下一轮头 `depth` 层的上载） | 每轮最多 −11 ms（−19%） | M-04 的暴露与环深无关，是轮首/轮末 |
| 流式段更小的页（质量合同变更，需批准） | 最多 +40% | 拷贝量与页大小成正比，M-03 里拷贝已占 84% 的轮时间 |
| 更大窗口（拿回环的区域代价） | 每页/层 = 0.24 ms/轮 | 每页 71 680 B ÷ 25.2 GB/s |

## M-06 跨轮预取：轮末为下一轮头几层发上载

- 时间：2026-11-09T01:50±+08:00
- 提交：`47f14a7c`
- 命令：通用命令 + `--kv-stream-report`（246 652 token 提示、容量 262 144、环深 2）
- 结果：每轮 54.1 ms、上载 1.09 GiB、接受长度 2.24。同一提示下环深 2 无预取为 56.9 ms。
- **未验证**：等价性（逐 token 与原版一致）。这是 1 号待办，见交接文档 §4。

## M-07 环深 2 的重复性（3 次）

- 时间：2026-11-09T00:5x
- 提交：`a60647b1`
- 结果：每轮 56.8 / 56.6 / 56.9 ms，拷贝占用 47.57 / 47.37 / 47.76 ms，
  暴露 11.60 / 11.32 / 11.78 ms。轮时间稳定在 ±0.3%，tok/s 的浮动全部来自接受长度
  （2.74 / 3.00 / 2.86）。

## M-08 数值脆弱性探针：`--prefill-chunk` 会改变 token

- 时间：2026-11-09T02:5x
- 提交：`47f14a7c`（同一二进制、同一 `--seed 42 --greedy --invariant-attention`）
- 命令：通用命令 + `--print-token-ids`，只改 `--prefill-chunk 256` ↔ `128`
- 结果：两次的 `generated ids` md5 不同（`6f616f42…` vs `fdd87141…`）。
- 含义两条：
  1. **比较必须固定 `--prefill-chunk`**，否则问的不是同一个问题；
  2. 246 K 的 α 重复提示词对数值扰动极其敏感（近 tie 的贪心选择），**不能用作等价性 oracle**。
     等价性金标改用 `scripts/kv-stream-golden.sh` 的 8k / 100k / 184k 三个 case。
- 同时确认：同一二进制、同一命令行连跑两次，ids 完全相同（确定性成立）。

## M-09 全驻留 190K + MTP3 在这张卡上装不下（复现 M-5.9）

- 时间：2026-11-09T02:2x
- 命令：`--max-context 190016 --kv-capacity auto --kv-headroom-mib 8 --spec mtp --draft-tokens 3`
- 结果：`error: minimum Engine runtime reservation requires 4103599616 bytes in addition to
  8388608 bytes of automatic headroom, but only 4032299008 bytes are available after weights`
- 含义：全驻留 + MTP 的上限约 186 K；流式窗口在 `rk4v4`+MTP3 下是 2 835 页 = 181 440 token。
  两者重叠区间只有 ~184 K，**等价性 case 只能放在那里**。

## M-10 原版金标、逐 token 等价性与 251K 基线（2026-11-08）

- 提交：`73c3ae14`（工具 `cf90eaaf` 之后）
- 工具：`tools/bench/run_kv_stream.py`；记录目录 `bench/results/kv-stream/`
- 前置：`calibrate`（本配方 34.0 token/句 + 100 token 尾巴）、`window --capacity 184064
  --headroom-mib 1008`（窗口 135 104 token）

命令（GPU 空闲，同一模型/同一卡）：

```bash
python3 tools/bench/run_kv_stream.py --model <artifact> calibrate --force
python3 tools/bench/run_kv_stream.py --model <artifact> golden --suite equiv --force
python3 tools/bench/run_kv_stream.py --model <artifact> check  --suite equiv
python3 tools/bench/run_kv_stream.py --model <artifact> golden --suite perf  --force
python3 tools/bench/run_kv_stream.py --model <artifact> bench  --case stream-251k --reps 3 --tag baseline
```

**等价性**（`runs/20261108T195936Z-equiv-73c3ae14`，四行全 PASS）：

| case | 布局 | prompt | 生成 | 轮 | decode | 每轮 | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| equiv-8k | 全驻留 | 7 948 | 16 | 4 | 150.0 | 25.0 ms | ids 相同 |
| equiv-100k | 全驻留 | 99 952 | 16 | 4 | 119.0 | 31.6 ms | ids 相同 |
| equiv-183k | 全驻留 | 182 946 | 512 | 129 | 106.1 | 37.4 ms | ids + logprob 相同 |
| equiv-183k | 流式 | 182 946 | 512 | 129 | 91.4 | 43.4 ms | ids + logprob 相同 |

- 金标由**原版二进制**录制：`/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer`，
  `ninfer 2026-11-04.3+…-gf6928e00`。它是唯一允许执行的参照，不写、不编。
- 183 K 的 512 个 token 跨 8 页/层（窗口 135 104 token、750 页），环复用与滑窗都在 decode 中发生。
- 流式与全驻留的 `logprobs_sha256` 都是 `feb518b5…`，**逐位相同**。
- 流式每轮上载 823 MiB、暴露 2.4 ms；`stream_series_matches_total` 通过（逐轮之和 == 总 − prefill）。
- 旧记录里 3 593 MiB/轮是工具的错（把 prefill 的 55.2 GB 摊进了 decode 轮），已改并作废。

**251K 基线**（`runs/20261108T201204Z-bench-baseline-73c3ae14`，3 次）：

| 指标 | 值 |
|---|---:|
| prompt token | 250 946 |
| decode | 69.2 tok/s |
| 每轮 | 56.7 ms |
| 每轮上载 | 1 193.6 MiB |
| 暴露（计算等拷贝） | 15.8 ms |
| 接受长度 | 3.94 |
| 主机权威副本 / 设备窗口 | 4 480 / 3 274 MiB |

与 M-03 的 56.9 ms/轮几乎相同；tok/s 高来自接受长度（3.94 对 2.24–3.0），不是调度改动。

**过程发现（都已在本提交里修掉）**：

1. 思考请求发布的是 reasoning，Engine 只为 content token 保留 logprob ⇒ 开着思考时
   `--logprobs` 得到空数组（183 K 那次的 512 个 token 全是 reasoning、stdout 为空）。
   基准固定 `--no-thinking`，才有可逐位比较的分布。
2. 收尾句必须"活过"预算：`1 2 3 …` 列表在第 51 个 token 命中 stop token；改成"续写数字到 800"
   后 8 K 与 183 K 都能生成满 512。为此加了 `prompt-shape` 哨兵 case（8 K、512 token）。
3. 8 192 容量装不下 8 K 提示 + 512 输出（引擎报 `context-capacity`，只给 245 个），
   哨兵 case 用 16 384 容量。

**未验证**：跨版本 logprob（原版 CLI 没有 `--logprobs`，只有 `--print-token-ids`）；
8k/100k 只生成 16 个 token（跨版本 ids 已比，分布未比）。
