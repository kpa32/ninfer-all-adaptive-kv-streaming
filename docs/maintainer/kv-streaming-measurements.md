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

## M-11 投机后端的占用与草稿成本：DFlash2 的代价是窗口，不是速度（2026-11-08）

- 提交：`1b9e7dd6`（`footprint` 子命令）
- 记录：`bench/results/kv-stream/footprint.json` + `runs/20261108T214346Z-footprint-ac0f4d53/`
- 产物：`models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（自带 `text/vision/mtp/dflash2`；
  drafter 投影是 `q8_g32_fp16 / row_split_k128_v1`）

```bash
python3 tools/bench/run_kv_stream.py --model <artifact> footprint \
    --context 262144 --config none:0 mtp:3 mtp:7 dflash2:3 dflash2:7
```

**实测占用**（容量 262 144、`--kv-stream`、8 K 提示、`--max-new 8`）：

| 配置 | 权重 | 上下文池 | 流式窗口 | 窗口 token |
|---|---:|---:|---:|---:|
| none/0 | 10.90 GiB | 3 995 页 | 3 895 页 | 249 280 |
| mtp/3 | 11.60 GiB | 3 362 页 | 2 869 页 | 183 616 |
| mtp/7 | 11.60 GiB | 3 362 页 | 2 858 页 | 182 912 |
| dflash2/3 | 13.40 GiB | 1 724 页 | 1 228 页 | 78 592 |
| dflash2/7 | 13.40 GiB | 1 724 页 | 1 217 页 | 77 888 |

派生：DFlash2 权重 **+1.80 GiB**（相对 MTP；Q8 drafter 本体 1.70 + 选择器/状态 0.10），
窗口 **−1 641 页 = −105 024 token**。汇率核对：1.80 GiB ÷ 1 146 880 B = 1 685 页，
实测 1 641，差 44 页是 drafter 自己的滑窗 KV 与状态。

**这份 footprint 的 accept / decode 两列不可用**：`--max-new 8` 只有 1–2 轮，
`mtp/7 89.0 < mtp/3 118.2`、`dflash2/7 75.6` 都是采样噪声，不是速率。

**草稿成本**（已发布实测：`docs/performance/reference-2026-09.md` §Draft length，RTX 4090、
Qwen3.8-27B、`rk8v4`、单请求；tok/s，括号是接受率）：

| 草稿器 | k | 短聊 | 32 768 | 131 072 |
|---|---:|---:|---:|---:|
| none | – | 54.8 | 52.0 | 45.6 |
| MTP | 3 | 110.6 (0.58) | 114.7 (0.66) | 80.3 (0.45) |
| MTP | 7 | 106.8 (0.36) | 102.8 (0.37) | 64.3 (0.22) |
| DFlash2 | 3 | 120.6 (0.58) | 120.6 (0.63) | 88.7 (0.45) |
| DFlash2 | 7 | 148.8 (0.35) | 151.9 (0.39) | 102.3 (0.27) |

还原成每轮毫秒（`轮 = (1 + k·接受) / tok/s`，短聊列）：

| | k=1 | k=3 | k=5 | k=7 |
|---|---:|---:|---:|---:|
| MTP | 22.8 | 24.8 | 27.7 | 33.0 |
| DFlash2 | 22.7 | 22.7 | 23.1 | 23.7 |

**DFlash2 的每轮成本对 k 近乎平坦**（22.7 → 23.7），MTP 每加一个草稿 +1.7 ms；7 草稿时
DFlash2 每轮便宜 9.3 ms（131 K 时 11.2 ms，且接受率 0.27 vs 0.22）。接受率本身两者几乎相同，
所以优势来自**每轮成本**，不是更会猜。

**结论 1（最重要）**：窗口固定时，流式搬运的惩罚是所有后端共享的**沉没成本**。以 183 K 上下文、
本卡实测的 77 888 token 窗口（惩罚 17.7 ms/轮）代入：

| 配置 | 每轮计算 | + 惩罚 | 合计 | 每轮 token | tok/s |
|---|---:|---:|---:|---:|---:|
| DFlash2-7 | 23.7 ms | 17.7 | 41.4 | 3.45 | 83 |
| MTP-3 | 24.8 ms | 17.7 | 42.5 | 2.74 | 64 |
| MTP-7 | 33.0 ms | 17.7 | 50.7 | 2.54 | 50 |

⇒ **窗口不变时 DFlash2-7 全程领先约 29%，后端切换没有收益。**

**结论 2**：收益来源是把 DFlash2 的 1.80 GiB 还给窗口（+105 024 token ⇒ 183 K 时惩罚归零），
不是换草稿器：MTP-3 拿到自己的窗口后 183 K 上是 24.8 ms/轮 ≈ 110 tok/s。

**结论 3**：`--adaptive-mtp` 在长上下文会自己退回窄宽度（131 K：MTP-3 80.3 > MTP-7 64.3，
接受率 0.45 vs 0.22）。尾部该切的是 **MTP-3**，不是关掉投机——`none` 在 131 K 只有 45.6。

**没有测的**：本卡（RTX 4080 SUPER 16 GB）上 DFlash2 的绝对速度。上表是 4090、24 GB 卡、
无流式；比值可用于判断方向，绝对值不能引用。

### M-11 补充：`--embedding-residency host` 是这份产物上最便宜的一笔显存（2026-11-08，外推）

产物核对（`python3 -m tools.artifact.inspect <artifact> --bindings --objects`）：

| 张量 | 格式 | 形状 | 字节 | 结论 |
|---|---|---|---:|---|
| `text/token_embedding` | `gguf_iq2_s` | 248 320 × 5 120 | 407 244 800 = **388.4 MiB** | 可搬主机（`tie_word_embeddings: false`） |
| `text/output_head` | `gguf_q4_k` | 248 320 × 5 120 | 715 161 600 = 682 MiB | 必须留设备 |

按 M-11 的汇率（1 页 = 1 146 880 B = 64 token）：388.4 MiB = **355 页 = 22 720 token** 的窗口；
251 K 上下文下尾部短 22 720 token，每轮少搬 407 MB ≈ 15.4 ms，**预测**每轮 56.7 → ≈41 ms、
decode 69.2 → ≈95 tok/s（+38%）。

前提：`--kv-capacity` 必须自动（写死容量则搬走的内存不会进窗口）；prefill 的 gather 多走
≈640 MB 总线（251 K 提示，一次性，≈+24 ms）。**以上两行加速是外推，没有实测**——测法只差
一个 footprint 变体，见 `bench/results/kv-stream/README.md`「白拿的显存」。

## M-12 帧（span）流水：机制成立、逐位相同，但在 251 K 上比整层拷贝慢（2026-11-09）

**目的**：把一层内的拷贝与发射重叠，消掉每轮 9.2–15.8 ms 的计算流空等（设计 §5.13.2 第一步）。

**做了什么**：`small_t_i8` 家族（`rk8v4`/`rk4v4`/`rk2v4`/`int8`）的 small-T 发射接受 split 区间
（`CausalSmallTInvocation::split_begin/split_end`），一次调用内按 split 升序分帧发射、每帧
`wait → launch → consumed`、层末仍只 reduce 一次；帧宽 = `splits_per_frame × split_keys ÷ 64`，
`split_keys` 与内核 `causal_small_t_partition_range` 同规则。默认路径改为整层拷贝
（`NINFER_STREAM_FRAMES=1` 才开帧）。

**同一构建内的 A/B**（251 K、容量 262 144、`rk4v4` + MTP3、`--invariant-attention`、RTX 4080 SUPER）：

| 变体 | tok/s | 每轮 ms | 暴露 ms/轮 | 上载 MiB/轮 | 记录 |
|---|---:|---:|---:|---:|---|
| 整层拷贝（历史基线，`73c3ae14`） | 69.2 | 56.7 | 15.8 | 1193.6 | `20261108T201204Z-bench-baseline-73c3ae14` |
| 整层拷贝（本次构建，`5731011d`） | 69.2 | 56.7 | 15.6 | 1193.6 | `20261109T043626Z-bench-m12-wave-noframes-5731011d` |
| 帧 spf=20、帧内 64 页分块 | 67.5–67.7 | 57.9–58.2 | 11.1 | 1191.2 | `20261109T040336Z-bench-m12-framed-251k-28ade87e` |
| 帧 spf=20、一帧一次拷贝 | 65.1–65.8 | 59.3–60.5 | 11.3 | 1191.0 | `20261109T041142Z-bench-m12-framed-onerun-2cf5d0c3` |
| 帧 spf=17（按波对齐）、一帧一次 | 64.0–64.1 | 61.1–61.5 | 8.7 | 1191.2 | `20261109T042835Z-bench-m12-wave-aligned-5731011d` |
| 帧 spf=17 + 只拷流式前缀（空帧不记事件） | 48.9–50.8 | 77.5–91.3 | 1.5 | 1191.2 | `20261109T044203Z-bench-m12-prefix-only-c941d6ca` |

**结论 1：暴露确实降了，但每轮没有降。** 分帧把暴露从 15.6 砍到 8.7 ms，同时每轮涨了 4.4 ms；
再往前切（空帧不记事件）暴露掉到 1.5 ms，每轮反而涨到 78 ms。251 K 下链路本身是瓶颈：
每轮 1 191 MiB ÷ 25.4 GB/s ≈ 47 ms，占每轮 56.7 ms 的 83%——**能不暴露的上限只有那么多**，
分帧只是把等待在层内重新排列，额外的发射与流事件（每层 5–17 次）反而收费。

**结论 2：帧要"按一个整段调用所需的波数"分，而不是按一个波。** `spf = sms / kv_heads = 20` 时
85 个 split 变成 5 次发射 = 5 个整波，而整段调用只要 4.25 波；改成
`frames = ceil(active × kv_heads / sms)`、`spf = ceil(active / frames)`（= 17，5 段 × 0.85 波
= 4.25 波）把这份浪费去掉，暴露 11.1 → 8.7 ms，但每轮仍因事件开销多 3 ms。

**结论 3：机制本身是对的，可以逐位相同。** 同一构建、`--invariant-attention`、183 K 流式下，
开启分帧后 `check --suite equiv` 的 `equiv-183k/streamed` 行 **PASS**，ids 与 logprobs 都和全驻留
逐位相同，同轮 `stream_upload_mib_per_round = 824.7`（证明 decode 真在搬，不是少搬变快）
（`20261109T035348Z-equiv-dbg8-cb327a1a`）。默认路径（整层拷贝）的等价性全集 **4/4 PASS**：
`20261109T045055Z-equiv-m12-equiv-default-e25152e0`。

**因此**：这一轮不交付"更快"，交付的是"分帧机制可用、可行、已验证逐位相同，但在 251 K 上不赚钱"。
下一步要动的是**每轮字节数**（帧级环多给窗口 ≈ −6%，或更小的流式页），不是继续在层内切细。

**没有测的**：`rk8v4` 上的帧路径（窗口更小、尾部更大，量级应当更不利，但不能外推）。

### M-12b 交付格式 `rk8v4` 在 251 K 的实测（2026-11-09，默认路径）

`--kv-dtype rk8v4`、容量 262 144、MTP3、`--invariant-attention`、单次（`reps 1`）：

| 项 | 值 |
|---|---:|
| 流式窗口 | **1 670 页 = 106 880 token**（自动容量实测，不是设计表里的 1 789 页） |
| 每轮 | **151.0 ms** |
| decode | **26.1 tok/s** |
| 每轮上载 | **3 603.0 MiB** |
| 暴露（计算等拷贝） | **114.3 ms / 轮 = 76%** |
| 接受长度 | 3.9 |

记录：`20261109T051316Z-bench-m12b-rk8v4-real-b509e872`（argv 里 `--kv-dtype rk8v4`，
运行摘要 `kv cache dtype rotated-k8g64-v4g32` 即 `rk8v4`，80 SMs）。

**读法**：`rk8v4` 每页 104 448 B，且窗口比 `rk4v4` 小 610 页 ⇒ 尾部大 630 页/层 ⇒ 每轮字节是
`rk4v4` 的 3.0 倍（3 603 vs 1 194 MiB）。链路耗 3 603 MiB ÷ 25.4 GB/s ≈ 142 ms，占每轮的 94%，
所以暴露 114 ms 是“链路来不及”，不是调度没重叠。**结论：`rk8v4` 上唯一有效的方向是减字节**
（更大窗口 / 更小页 / 把权重挤出显存），层内重叠无论怎么切都不会有收益。

**没有测的**：`rk8v4` 的等价性金标（原版二进制在这个容量下装不下 `rk8v4` 的全驻留布局，
所以 `golden` 无法录）。因此 `rk8v4` 目前只有“能稳定 decode”的证据，没有“逐 token 与原版相同”
的证据——后者只能在 `int8`/`rk4v4` 上做，或在更大显存的卡上录 `rk8v4` 金标。

### M-13 保真移植后的首次上卡（2026-11-09，`equiv-183k`）

同一构建、同一条用例，`resident` 与 `streamed` 两个变体：

| 项 | resident | streamed（新框架） | streamed（移植前） |
|---|---:|---:|---:|
| greedy ids vs 金标 | 512/512 相同 | **512/512 相同** | 512/512 相同 |
| 每轮上载 | — | **779.9 MiB** | 822.9 MiB |
| 暴露（计算等拷贝） | — | **1.0 ms/轮** | 15.6 ms/轮 |
| decode | 106.0 tok/s | 93.8 tok/s | 89.9 tok/s |
| 接受长度 | 4.0 | 4.0 | 3.94 |
| 环 | — | 7 runs = 896 槽（≈1.26× 流式量） | 2 × 756 页 = 1512 槽 |
| 窗口 | — | **2 150 页/层** | 2 111 页/层 |

记录：`20261109T075100Z-equiv-s6-equiv7-baae1682`（campaign 判 FAIL，仅因
`logprobs_match_across_layouts=false`；ids 判据通过）。

**结论与未决**：暴露从 15.6 ms 降到 1.0 ms（15×），窗口 +39 页/层，每轮字节 −5%。
唯一没过的判据是"两布局 logprobs 逐位相同"（SHA-256 摘要不同）：ids 完全一致说明 KV 内容正确，
差异在低位。可疑来源按优先级：① `run_pages=128` 让 bounded 派发的 split 边界与整层派发不同，
若归约顺序随帧而变则该判据必然失败；② 新窗口把 split 边界移动了一页（2111→2150 页 ⇒ split 数
不同），累加顺序随之变化。下一步：先确认 `causal_attention_small_t_launch_for` 是否"层末单次归约"
（若按帧归约，就是①），再决定是修归约还是把该判据改成"同布局内逐位、跨布局用容差"。

### M-14 保真移植的 logprob 回归：三组定点实验（2026-11-09）

`equiv-183k` 的 `streamed` 变体与 `resident` 基线比：**ids 512/512 相同，logprobs 最大差 0.118 nat
（508/512 有差）**。为定位，跑了三组定点实验（都用记录过的 argv，只改一个参数）：

| 实验 | 配置 | 最大差 | 结论 |
|---|---|---:|---|
| E1 | `--kv-stream` + `--kv-headroom-mib 16`（池装得下全上下文 ⇒ 窗口 2876 页/层、环 0 run） | **0.000000** | 平铺池的**窗口寻址、块表发布、append、advance 全部逐位正确** |
| E2 | 流式 + `--no-cuda-graph` | 0.606 | 无图时更差 ⇒ 图不是病因 |
| E3 | 流式 + `NINFER_STREAM_VERIFY=1`（每层拷完立即把环槽 D2H 回来与主机权威逐字节比） | — | 1949 次检查中 **1923 次不符** |
| E4 | 同上，并打印"该槽实际持有哪一页" | — | `layer=7 page=0 slot=34400 … holds=layer=0 page=0`；下一轮 `layer=0 slot=34656 holds=layer=9` |

**结论**：错误定位在**环槽复用**这条路径上，且只在"该 run 之前被别的层用过"时出现（首次使用的
run 全部 ok）。指纹证明槽里躺着的是**上一个使用者**的页，不是本层的页——即本层这次拷贝对消费者而言
没有生效。环形缓冲共 7 个 run，而一层需要 6 个，于是同一轮内第 7 层起就回卷复用到第 0 层的 run。

**下一步（已确定）**：在 `upload()` 末尾记录一个"本层全部拷贝完成"的事件，探针先等它再比对；若等完后
仍不符，说明块表指向的槽与拷贝写入的槽不同（那么 `slot_of_page` 与拷贝循环的下标换算不一致）；若等完
后相符，则证明消费者真的在拷贝落地前就读了——即 `consumed_run_` 的等待在**同一轮内**是失效的
（事件对象持有的是更早一条记录），修法是给 run 记"本轮已被消费者释放"的代际，而不是只看事件。
