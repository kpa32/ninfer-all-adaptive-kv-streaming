# KV 流式：决策与缺陷台账（历史记录，不是当前状态文档）

**这个文件是记录区，不是产品文档**：它保存"哪个决定由哪条数据支撑""哪些数字作废""哪些路被实测否掉"
以及已修缺陷的账本。当前状态与判据在 `docs/maintainer/`（设计、策略、验证机制）；这里出现的日期、
提交号与运行目录名只用于追溯，**不要把这里的结论当规格**。

规则：新的决策先写进 `docs/maintainer/kv-streaming-strategy.md` 的 P 表，再由这里追加"它当时凭哪条
数据"；被推翻的条目保留原文并标注"已被 X 取代"，不删改历史。

---

## 决策索引（先读这张表，能省掉重做实验）

| 结论 | 决定 | 章节 | 运行记录 |
|---|---|---|---|
| 262 144 靠提高精度装不下（`rk8v4` 7.33 GB > 3.07 GB 预算） | 必须页粒度流式 | M-1…M-6 | — |
| 帧式流水在 251 K 更慢（67.5 / 64.0 vs 69.2 tok/s） | 当时回退整层拷贝 | M-12 | `…045828Z-bench-m12-default-251k-e25152e0` |
| `rk8v4` 251 K 实测 26.1 tok/s、3603 MiB/轮、暴露 114 ms | 该格式唯一有效方向是减字节 | M-12b | `…051316Z-bench-m12b-rk8v4-real-b509e872` |
| 环从 `2×` 改 `1.10×` 后 251 K 慢 17% | 控制器不能只看字节 | M-13/M-17 | `…075837Z-equiv-s6-equiv8-0e688e0c` |
| logprob 偏 0.118 nat 的根因是**层号映射两次** | 修 `stream_layer = fidx` | M-16 | `…091551Z-equiv-s6-equiv10-48746c74` |
| 首轮重分区把 126 ms 塞进被测窗口 | decode 前不迁移 | M-18 | `s10-nospike` |
| 迁移实验把迁移轮算进均值 ⇒ 任何实验都"变慢" | 试-run 跳过迁移轮 + 基线取稳态 + 被否不重试 | M-19 | `s16-fair` |
| 缩环让掉了前视：1.375 → 1.125 只省 1.6% 字节 | 门限按**几何前视比例**判，不看字节绝对值 | M-19 | `s16-fair` |
| 251 K `rk4v4` 稳态 54.4 ms / 72.8 tok/s，`rk8v4` 147.9 ms / 26.8 tok/s | 速度线收口，**超过移植前** | M-19/M-20 | `s16-fair`、`s18-rk8v4` |
| 池装得下时（150 K/180 K）零字节、等于全驻留速度 | "零搬"是正确期望，不是失败 | M-21 | `s20-range` |
| 150 K 紧池：控制器提出缩环（省 37% 字节），实验测出慢 85% ⇒ 回退 | 实验是裁判，门限只是预筛 | M-21 | `s20-range` |
| run 宽度 49/96/128/192 对时间影响 ±0.7%、暴露恒 6.0 ms | 默认保持 128，旋钮只作测量接口 | M-22 | `s22-rp49/96/128/192` |
| 183 K 每轮 1410 MiB / 45.4 ms ≈ 31 GB/s，已达链路峰值 | 该档**带宽受限**，调度侧没有余量 | M-22 | `s22-rp128` |
| graph 在流式下不可捕获（捕获行无页，380 µs 失败），收益仅 0.65 ms/轮（251 K ≈1.3%） | 维持关图，不做图内换页 | — | `…192253Z` on / `…192831Z`、`…193412Z` off |
| `--embedding-residency host` 在流式下是纯正收益、0 代价（用户判定） | 不做 A/B，直接采用；数仍是模型推算 | — | — |

## 已被实测否掉的路（不要再试，除非硬件变了）

| 尝试 | 结果 | 依据 |
|---|---|---|
| 整层内再切帧（48 帧/轮） | 251 K 上 67.5 / 64.0 tok/s，比整层 69.2 慢 | M-12 |
| 把环加大到 2× 换重叠 | 251 K 上 65.2 ms/轮（1.10× 时为 60.4），字节 +9% | M-17、M-19 |
| 用字节模型判断要不要迁移 | 只看得见字节、看不见前视，且结构上只允许缩环 | M-18 |
| 分帧流水（每 run 一次发射） | logprobs 逐位正确、暴露 0.1 ms，但每轮 102.6 ms（2.2×） | M-17、M-22 |
| 导出 run 宽度为 `segment_keys/page_size = 49` | 暴露不变、时间 ±0.7%，无收益 | M-22 |
| 设备侧等待（让 kernel 轮询 ready 标志） | **未实现**：183 K 已带宽受限（31 GB/s），能把暴露压到 0 的分帧也只省 6 ms 却多付 56 ms | M-17、M-22 |
| 首轮 decode 就按预测重分区 | 126 ms 迁移塞进被测窗口，首轮 189.8 ms | M-18 |


## 作废的数字（引用旧记录前先核对这张表）

| 数字 | 出处 | 为什么作废 |
|---|---|---|
| 102.6 tok/s | `runs/20261109T003445Z-bench-frames-224d057b` | 那一版"帧 hook 拿不到 ⇒ 该格式 decode 一次都不搬"（逐轮 `upload_bytes=0`）。速度来自少搬的 ~1.1 GB/轮，不是跳过优化 |
| 94.2 tok/s / 暴露 0.9 ms（183 K） | M-13 | 层号映射两次的版本：`consume` 等的都是错的 run，等于没有任何等待，拷贝可以任意抢跑 |
| 57.7 tok/s / 60.4 ms（251 K） | M-17 | 首轮 126 ms 重分区 + 迁移实验把迁移轮算进均值，两者都在被测窗口内 |
| 50.8 tok/s（251 K 环比 2.00） | M-17 | 控制器中途又改了布局，该次不是"2.00 稳态"；且当次 ids 与金标不同 |
| 67.2 tok/s（251 K 稳态，修前） | M-18 | 4 次 ~170 ms 实验尖峰摊在 129 轮里；修好后同口径为 72.8 |
| 26.1 tok/s（`rk8v4` 251 K） | M-12b | **不作废，但口径不同**：64 token 窗口；512 token 稳态为 26.8 / 147.9 ms |


## 环境

## 关键不变量（实现层的合同，复核时逐条对照代码）

## 4. 怎么实现的（关键不变量，复核时逐条对照代码）

1. **池恒等式**：`Σ layer_resident[i] + ring_pages ≤ slot_count`。
   规划器 `plan_kv_stream_uniform()` 用**二分**求最大可行窗口（左侧随 R 单调增），而不是定点迭代；
   环按 run 取整；`ratio` 装不下时退到 1.0（深度一）；`run_pages` 从请求值逐次减半直到能装下。
2. **窗口槽不变式**：`slot = layer_offset(layer) + page % resident(layer)`。
   纯函数 ⇒ 一个驻留页在窗口期内槽位不变，块表几乎不用改。超集情形（全部驻留）与今天的
   paged 缓存布局逐字节一致（单测 `all_resident_matches_today`）。
3. **流式前缀寻址**：页 `p` 落在环 run `(first_run + p / run_pages) % ring_runs` 的第 `p % run_pages`
   个偏移 ⇒ 一个 run 在设备上是**连续**的，一次 `copy_from_host_pages` 就能整批搬。
4. **两条事件链**：`ready_run_[r]`（该 run 的拷贝落地）与 `consumed_run_[r]`（该 run 已被读完）。
   写者在覆盖 run 前等待 `consumed_run_[r]`；消费者等待 `ready_run_[last]` **和** `ready_[layer]`
   （后者在**块表发布之后**记录 —— 缺这一条会有数值错误，见 §7）。
5. **块表发布**：每层每轮**一次** `publish_slots_at(layer, 0, [0, streamed))`（一个 strided H2D）。
   这是"每轮重新寻址还能负担得起"的关键。
6. **重分区**（`repartition`）：进出窗口的页各搬一次（出：D2H 到主机权威；进：从主机权威 H2D 回窗口槽），
   其余页只重发布表项；随后作废环记账、`run_cursor_ = 0`、`++layout_generation_`（让调优器重置）。
7. **控制器**（`evaluate_partition`）：每轮一次，输入是**上一窗口**的 deadline 比例、占空比、环峰值；
   没有新样本且非首轮 decode 就不评估；决策把槽位数交给 `resize_kv_stream_split()` 转成本布局能表达的
   (每层窗口, 环)。
8. **计量**：窗口开在 `begin_round`、关在下一次 `begin_round`；窗口没跑完就保持打开，绝不覆盖未读样本；
   每轮字节**无条件**记账（campaign 靠它判"是否真的在搬"）。

---


## 已修复的缺陷（逐条可验证）

## 6. 已修复的缺陷（复核时可逐条验证）

1. **`ahead == 0` 时整层一次拷贝都没发**：旧逻辑只发"提前 ahead 层"的拷贝。修复后 183K 从
   `upload=0.0` 变成 779.9 MiB/轮。
2. **每轮字节只在 `--kv-stream-report` 时记账**：campaign 的 `stream_engaged` 因此误判 FAIL。
   现在每轮都记账，只有计时事件是可选的。
3. **规划器拒绝"只装一层流式量"的探针候选**：容量扫描第一个候选就失败 ⇒ 启动直接报错。
   现在零窗口是合法计划（全部流式），扫描得以继续。
4. **规划器的定点迭代在预算不足时下溢**：改为二分求解 + `ratio` 1.0 回退 + `run_pages` 逐次减半。
5. **池校验签名**：`kv_stream_pool_pages(split)` 不再需要 layers。
6. **绝对层号 vs full-attention 序号**：旧帧路径用绝对层号索引按 full-attention 序号编号的空间，
   后半模型的流式层会静默失配。现在的规则是：调用方映射一次 `compact_layer_indices[layer]`，
   空间内部一律用这个序数，**不得再映射**（§7 的根因就是多映了一次）。
7. **消费者没等块表发布**：`wait_ready`/`wait_run` 现在同时等 `ready_[layer]`（发布之后记录）。
   效果：logprob 差 0.118 → 0.079 nat（真实但不足）。
8. **层号被映射两次**（根因，见 §7）：`stream_layer` 直接用 `fidx`。
9. **`ahead` 没有上限**：夹到 `count − 1`，`issue()` 自带边界检查（`rk8v4`/251K 的崩溃）。
10. **span 流水按帧序而非页区间寻址**：改为用 hook 报告的 `(first_page, page_count)` 换算 run 区间
    （`wait_run_range`/`consume_run_range`），bounded 路径因此逐位正确（仍太贵，见 §8）。

---

## 7. 已修复的主缺陷（这轮的核心）

## 决策记录：深环（两层前视）在本机实测更慢，已由试验机制自动回退

**假设**：251K 时环深 1.38 个 extent ⇒ 没有任何重叠（每轮 = 46.1 ms 拷贝 + ~8.3 ms 注意力 = 54.4 ms）。
把环加深到 2.0 个 extent（18 runs，窗口 2905 → 2849 页/层，每轮字节 +7%）应能吃掉注意力那 8.3 ms。

**实现**：`plan_kv_stream_lookahead()` 规划"两层前视"的最小环；`evaluate_partition` 在没有被否目标、
且当前 lookahead < 2.0 时，把它当作**一次试验**提交（沿用已有的"跑 4 轮比中位数、慢 0.5% 回退、被否
目标粘性拒绝"机制）。不是常数、不是开关。

**结果**（251K / `rk4v4` / MTP3 / `--max-new 128`，`NINFER_STREAM_DEBUG=1`）：

```
[partition] lookahead trial: ring 1408 -> 2304 pages (1.38 -> 2.00)
[partition] trial done mean=116.02 baseline=75.25 revert
最终：ring_runs=11 window=2905 中位 54.8 ms/轮（min 53.2 / max 172.9），1114.5 MiB/轮，63.4 tok/s
首轮序列：[75.1, 54.7, 172.4, 56.6, 56.6, 56.6, 56.9, 172.9]  ← 两次迁移各一次尖峰
```

**判定**：假设**被证伪**（深环 −54%），机制正确（提案→实测→回退→粘性拒绝，两处尖峰就是两次迁移）。
**原因尚未归因**：候选是"所有 run 共用一条 FIFO 拷贝流，`issue(current+ahead)` 把下一层的拷贝排在本层
读之前，却又要等上一轮该 run 的消费事件"，于是每层退化成一次完整的往返。归因是 P3 的第一件事。

**顺带修掉一个潜伏崩溃**：`repartition()` 改了 `ring_runs_` 却没有扩 `ready_run_`/`consumed_run_`
两个事件数组，环一变大就索引越界（实测 SIGSEGV，rc=139）。修复：`grow_events()`。此前控制器只缩环，
所以这条路径从未被走到。

## 决策记录（续）：深环无收益，提案已从出货路径移除

**稳态对比**（251K / `rk4v4` / 512 token 口径，三轮独立运行一致）：

| 布局 | 每轮字节 | 每轮时间 | 计算侧暴露 |
|---|---:|---:|---:|
| 浅环（ring 11，lookahead 1.38） | 1112–1113 MiB | 54.5–54.8 ms | 12.3 ms |
| 深环（ring 18，lookahead 2.00） | 1174 MiB（+5.6%） | 56.4–56.8 ms（+1.9 ms） | 14.0 ms |

**归因（有数据的部分）**：拷贝流已经在链路速度上——1174 MiB ÷ 49.2 ms = **24.4 GB/s**，标定峰值 25.4 GB/s；
`copy_busy_ratio` = 1.00、`deadline_misses` ≈ 30/32 在两种布局下都一样。所以环深买不到"不存在的时间"，
只剩它多付的 +5.6% 字节。**尚未归因**：每轮约 8 ms 的链路空转出现在轮边界（`ahead=1` 只在轮内前视，
而轮末预取是被删掉的那半），这才是下一个要查的东西。

**试验统计本身有两个 bug（已修）**：
1. `window_seconds_` 在下一窗关闭前保持不变，而试验每次评估都给它定价 ⇒ 同一轮被计多次；
2. 用均值 + 单一跳过计数无法排除迁移窗 ⇒ 一个 170 ms 的迁移把 +3.5% 读成 +54%。

修法：只给"本次评估新关闭"的窗定价（`fresh_window`），并按**窗口中位数**判定（与基准工具同口径）。
即便如此，深环仍然只值 +1.9 ms 的代价、没有任何暴露收益，所以**提案从出货路径移除**
（`plan_kv_stream_lookahead` 保留，因为调度器会用到它）。移除后 251K 的逐轮序列为
`[74.5, 54.9, 54.3, 54.9, 54.3, 54.9, 54.3, 54.6 …]`，中位 54.7 ms、最大 85.2 ms，**不再出现两次 170 ms 迁移**。

**顺带修掉的潜伏崩溃**：`repartition()` 改 `ring_runs_` 却不扩 `ready_run_`/`consumed_run_` 事件数组，
环一变大即越界（SIGSEGV）。已加 `grow_events()`。

## 决策记录：251K 的池已经是卡的上限，剩余空间只有"跨轮重叠"那 7 ms

给每轮加"轮边界空档"字段（compute 起点到本轮首个 copy 的设备时间差）时得到的真实结论，与预期不同：

| 量 | 值 | 出处 |
|---|---:|---|
| 每轮拷贝时间 | 47.2 ms | `rounds_series[].copy_seconds` |
| 每轮时间（中位） | 54.4 ms | `decode_rounds` |
| 计算侧暴露 | 14.4 ms | `rounds_series[].exposed_seconds` |
| 启动后剩余显存 | **53.5 MiB** | `memory.available_after_startup_bytes` |
| 规划留白 | 64 MiB | `memory.planned_slack_bytes` |
| 池负载 | 3.55 GiB | `memory.kv_payload_bytes` |

推论（都不需要新探针）：

1. **池已经是上限**：3.85 GiB 可用里 3.55 GiB 全给池，只剩 53.5 MiB。`kDefaultKvStreamHeadroomBytes` 默认
   就是 64 MiB（`include/ninfer/base.h:70`），bench 用例里 1008 MiB 只用于 183K 的对齐。**没有可回收的窗口**。
2. **拷贝已在链路速度**：1113 MiB ÷ 47.2 ms = 24.7 GB/s（标定 25.4）。⇒ 每轮字节是物理下限。
3. **重叠已经做了大半**：`copy 47.2 + compute ≈ 40 − round 54.4 ≈ 33 ms` 已经重叠；剩下 7.2 ms 未重叠。
4. 因此 251K 的天花板就是 `max(拷贝 47.2, 计算 40) ≈ 47.2 ms`（≈ 86 tok/s），**只剩跨轮预取那 7 ms 可拿**。

**仪表的处置**：该字段用 `cudaEventElapsedTime` 读"上一轮的 compute 起点 ↔ 本轮首个 copy"，主机跑在
设备前面时事件未完成，读到的是 0 —— 而它想证的东西已被表中的四个数界住，所以整套探针**撤回**
（不进树）。要证"跨轮预取有用"，直接看每轮时间是否从 54.4 掉到 47–50 即可。

## 决策记录：跨轮重叠在这个容量上不划算（P3 结案，否决）

要"下一轮的头几层在本轮计算期间就拷完"，需要给环多留出**一个层的 extent**（8 runs = 1024 页 = 73 MB）
作为下一轮首层的着陆区。这笔账两侧都能算：

| 项 | 量 | 依据 |
|---|---:|---|
| 多留 8 runs 的代价（窗口让出 64 页/层） | 每轮字节 **+6%** | 深环试验实测 +5.6%（1174 vs 1113 MiB） |
| 换成时间 | +2.8 ms/轮 | 1174 MiB ÷ 25.4 GB/s − 47.2 ms |
| 能买回的时间 | **≤3 ms**（首层 extent 的传输 74 MB ÷ 25.4 GB/s = 2.9 ms） | 冷启动就是这一层的传输 |
| 实测净效果 | **+1.9 ms（更慢）** | 深环试验：56.5 vs 54.6 ms，暴露 12.3 → 14.0 ms |

**判定：不划算，结案否决。** 不是"没做"，是"做了并且测了，代价大于收益"：环退回到字节最省的规划
（11 runs / 1.38 extent），`plan_kv_stream_lookahead` 与其单测保留，等换到更快的链路或更大的窗口
（比如 24 GB 卡）时这笔账会翻过来，届时不需要重写。

**251K decode 的结论**：池已是卡上限（启动后剩 53.5 MiB）、拷贝 47.2 ms 已在 24.7 GB/s、
重叠已有约 33 ms ⇒ 剩余 7.2 ms 属于“首层冷启动 + 尾部”，而其唯一兑现方式（多留一个 extent）
实测更贵。**这一档的速度结论到此为止**：54.4 ms / 72.8 tok/s，天花板 47.2 ms / 86 tok/s。

## 决策记录：graph 在流式下不可捕获，且就算可捕获也只值 1.3%（P0 结案）

两个问题分开回答，都是实测：

**能不能捕获**——不能，而且是结构性的。图路径为每个并发捕获行预留一页
（`graphs.cpp reserve_capture_rows` 的 `physical_page(alloc, 0)`），而流式地址在
`commit_activation` 里把 `page_count` 置 0（页由 `StreamedKVSpace` 发布），所以捕获在 380 µs 内
以 `KV logical page is outside the address space` 失败，引擎把它当容量不足重试 5 次全失败。

**值不值得为它拆**——不值得。183 K 全驻留（`equiv-183k` resident，512 token）实测：

| 臂 | 每轮时间 | tok/s |
|---|---:|---:|
| graph on | 37.1 ms | 106.9 |
| graph off（两次） | 37.8 / 37.7 ms | 105.0 / 105.1 |
| 流式两臂（都关图，噪声基准） | 46.4 / 46.7 / 46.2 ms | — |

差 0.65 ms/轮（1.75%），与“约 500–700 个 kernel × 1.3 µs 间隙”同量级：CPU 提交 1.5–2 ms 被
37 ms 的 GPU 工作完全掩盖，graph 只能消 kernel 之间的 GPU 间隙，**上限就这么大**。251 K 的
54.4 ms/轮把它稀释到约 1.3%。

要让它覆盖流式，图内必须同时包含每层的 copy issue 与其 `consumed` record，并每次 replay
更新 memcpy 源。而 251 K 的环只有 11 run、一层要 46 run（`ahead = 0` 的串行档），把 issue
移到图外一次性发全部层，第二个层就会等一个只有 replay 才会 record 的事件——**死锁**。

**判定：维持关图（`startup.cpp` 的注释即此依据），不做图内换页。** 诊断开关已删。

## 决策记录：`--embedding-residency host` 在流式下免测采用（用户判定）

embedding 常驻占 388.4 MiB；换成 host 后这部分显存回到 KV 池（约 +355 页/层），按字节模型
每轮搬运少约 400 MiB（1116 → 约 710 MiB），代价是每轮一次几 KB 的 H2D 读。用户判定这是
**纯正收益、0 代价**，不需要一次 A/B 去确认，因此这一项不再排测量。

**引用时必须标注：上面两个数（−400 MiB、+355 页/层）是模型推算，不是实测。** 要把它变成流式的
默认（现在默认仍是 `device`）则是一次产品决定，需要先实测再改默认。

---

## 缺陷记录：布局改名不搬字节（流式槽位的正确性通道）

- 现象：同一 prompt、同一构建，只改"建引擎前预留多少显存"，答案就变（0.19–0.63 nat、top-1 翻转）；
  serve 上表现为首个请求 4 token 后 EOS、`mtp 0/9`。
- 根因：槽号由布局推导（`kv_stream_layout.h:156`），`repartition` 的"两边都常驻"分支只重发表项不搬字节，
  表项于是指向别的页。责任链：`85f95f6e` → `60dad291` → `0b46796f` → `1159d91a`（均 2026-11-09，分支本地）。
- 处置：`c8bdb380` 先修正确性（回写 host 权威再读入新槽）；随后移除请求中途换形（唯一收益 decode 0.2%，
  P1 记录），池在启动时按 decode 形规划、进程内不变。
- 回退点：tag `kv-stream/layout-switch-before-removal`、分支 `kv-stream/preserved-layout-switch`。
- 完整档案（证据表、被排除的假设、复现配方、不变量）：`bench/results/kv-stream/layout-rename-defect.md`。

## 决策记录：请求中途换布局已移除（`68941676`）

- 换形的唯一实测收益是 decode 轮 0.2%（P1 记录），代价是"布局参与页命名 ⇒ 换形即改名 ⇒ 必须搬字节"，
  以及本文件上一节记录的正确性缺陷（`layout-rename-defect.md`）。
- 现在的形态：池在**启动时**按 decode 形（深环）一次规划整上下文，`enter_layout`/`evaluate_partition`
  恒返回 false，请求生命周期内布局不变；环太小装不下深环时回退 prompt 形。
- 证据（同一构建）：六组几何（`resident` 1413→40、`ring` 24→35）答案逐位相同；保留臂与冷跑臂几何不同
  也逐位相同；8 K 流式仍 `all-resident`；`layout_generation=0`（此前首轮 decode 会 +1）。
- 下一步若删除"死代码"（`repartition` 的窗口迁移分支、`kv_stream_partition.*` 及其单测），验收要求：
  251 K 轮时间 ≤ 基线 +1%、183 K 等价 PASS、prefill tok/s ≤ −1%。回退点：
  tag `kv-stream/layout-switch-before-removal` / 分支 `kv-stream/preserved-layout-switch`。

## 测量记录：布局一次装配后的验收（`68941676` / `898fca00`）

- 等价性套件（本机、同一构建 `898fca00`）：`check --suite equiv` 全 PASS——8 K/100 K/183 K 的
  resident 与 183 K 的 streamed（`decode_tok_s=100.3`、`round_ms_median=39.6`、
  `stream_upload_mib_per_round=356.6`、`stream_exposed_ms_per_round=0.0`）。
  记录：`bench/results/kv-stream/runs/20261111T001855Z-equiv-layout-once-c14-898fca00`。
- 251 K（`bench --case stream-251k`）：`decode_tok_s=95.0`、`round_ms_median=41.4`、
  `acceptance_length=3.9`、`upload=689.1 MiB/轮`、`exposed=1.0 ms`。
  记录：`bench/results/kv-stream/runs/20261111T001056Z-bench-layout-once-step1-898fca00`。
  与文档记录的同一配方 ~96.5 tok/s 相差 ~1.5%，但记录区没有相邻基线可归因（配置不同），
  归因需要与保留 tag 做交错 A/B（另一套 build 目录，属 Gate 2 的剩余项）。
- 正确性：六组几何（resident 1413→40、ring 24→35）答案逐位相同；保留臂与冷跑臂几何不同也逐位相同；
  8 K 流式 `all-resident`、`layout_generation=0`。

## 状态指针（交接用）：`68941676` + 文档 `259b1e26`

- 已修：`repartition` 改名不搬字节（`c8bdb380`）；已移除请求中途换布局（`68941676`，池启动即 decode 形）。
- 已建：C14 几何不变性（`898fca00`）；缺陷档案 `layout-rename-defect.md`（`4844f2a6`）。
- 回退点：tag `kv-stream/layout-switch-before-removal` / 分支 `kv-stream/preserved-layout-switch`（`c8bdb380`）。
- 未完成与复跑命令：见 `handoff.md` §3（U1 交错 A/B 归因、U2 死代码删除、U3 in-repo 驱动未验证编译、
  U4 自动化、U5 长会话/int8、U6 测试机 8 处临时补丁、U7 文档收尾）。

## 移除落地：控制器与换形代码已删（本次提交）

- 删除：`src/core/kv_stream_partition.{h,cpp}`（232 行，零调用者）、`StreamedKVSpace::enter_layout`
  /`evaluate_partition`/`repartition`（约 170 行）、`advance()` 的全驻留加宽分支（数学上不可达，
  现改为显式报错）、控制器状态（试迁移/冷却/拒绝环/窗口信号）、`kv_stream_ring_peak_ratio`
  （控制器的第三个输入，随控制器移除）、报告字段 `stream.layout_generation`（含 CLI/report/
  observability 三处）、`test_kv_stream_partition.cpp`。
- 保留：环形本体与 run 拷贝、消费/事件顺序、`kv_stream_signal.*`、`kv_stream_feedback` 的
  delta/busy 两项（报告仍用）、`kv_stream_span_tuner.*`、保留/复用通路、规划族。
- 新增回归：`tests/test_kv_stream_layout.cpp::every_page_of_a_plan_is_named_once`
  （一层窗口内槽号唯一、环为整 run、不越池）；`tests/models/qwen3_5/test_engine_kv_stream_geometry.cpp`
  （同进程两 Engine、容量不同 ⇒ 池形状不同，ids 与 logprobs 必须位相等；几何相同则 skip 而非假通过）。
- 定性（写进 `layout-rename-defect.md` §8）：架构选择错误而非笔误；收益实测为零；保真度不等于收益。
- 文档同步：strategy §0.1/§0.2/§2 R7/§4/§7.2/P1/§9.2、fidelity #4/#6/T2/§11.3、design §2.3/§4.8/M3、
  verification C5（退役）/C6（改为规划期保证）、measurements 表 6（改为"启动时规划一次"）、
  SPEC 完成条件 #2（新增几何不变性）。

## 验证记录：全新树 `build-verify`（`bf0554da`）

删除后的验收全部在一棵**全新配置/全新目录**的树上做（`build-sm89`、`build-sm89-tests` 已删除，
它们含诊断补丁时代的对象文件）；测试机源码与本地同一提交、零未提交改动，8 处诊断补丁已清除。

**特性层（流式开）**

- `ctest -R kv_stream`：`layout` / `feedback` / `span_tuner` / `signal` / `guards` / `ninfer_kv_stream_test` 全 PASS；
  `serial_real` **PASS 749 s**（190 K，冷跑 + 保留两臂；冷跑臂 `rounds 7→14`、字节 oracle
  `736 checks / 0 mismatches`）；`geometry_real`（C14，新增）**PASS 86 s**：
  `2889 vs 1866 pages/layer, 11 vs 20 ring runs` → `8 ids and 6 logprobs identical`。
- `geometry_driver` 预留 0/64/512 MiB：`resident` 1413/1372/1075、`ring` 24/24/26，
  三组 `ids equal=1 logprobs equal=1`；`reuse_driver`：`frontier=4614 → reused=4614`、`equal_ids=1`、`OK`。
- `check --suite equiv`：**4/4 PASS**，`20261111T073620Z-equiv-verify-clean-bf0554da`
  （183 K streamed 100.2 tok/s、`upload 356.6 MiB/轮`、`exposed 0.1 ms`）。
- 251 K `bench --case stream-251k --reps 2`：**PASS**，`95.0 tok/s`、`41.4 ms/轮`、`upload 689.1 MiB/轮`、
  `exposed 1.0 ms`，`20261111T074344Z-bench-verify-clean-bf0554da`；与删除前同构测量**完全一致**（性能中性）。
- 8 K CLI：`shape=all-resident`、`ring_runs=0`、报告 `stream` 无 `layout_generation`（其余 25 键齐全）。

**两个真机测试曾经是"错误的合格证"**

- `serial_real` 的保留臂把前沿写死成 `190008`，正确值是 `prompt+generated−1 = 190007`；该臂从加入起
  从未在测试树跑过，所以错的常量没暴露。现改为按请求算术并打印 `(executed E)`。
- 几何用例最初两处都错：没有清空流式拒绝的 `context_cache` 字段（启动即报错，**根本没进行比较**），
  以及用显式容量造几何（`--kv-stream` 要求 `auto`）。现改为同一 `auto`、headroom 1 GiB vs 2 GiB。

**共享面回归（全量 ctest，`-j 1`，`2026-11-11 15:51`–`17:30`）**

- **378/385 PASS**。7 个失败全部与本次改动无关，逐条有原文：
  `chat_templates`（缺 `jinja2`）、`qwen4_exp_ngram_writer_interop`（缺 `pytest`）、
  `tma_shared_card`（`requires an sm_100a or sm_120a GPU`）、`qwen3_5_moe_real`（需 35B MoE 产物，
  机器只有 27B 稠密）、`qwen3_5_dflash_real`（产物无 `dflash` 组件）、
  `qwen3_5_prefix_real`（`requires 5778971136 bytes, but only 3975675904 bytes are available`）、
  `qwen3_5_hybrid_prefix_real`（`3297600000` vs `2109210624`，显存不足）。
- 支撑"无关"的三条：这 7 个测试文件不在改动面（`git diff --name-only` 命中 0）；全仓只有
  `serial_real`、`geometry_real` 两个测试文件打开 `kv_stream`，所以这些用例不构造 `StreamedKVSpace`；
  其余是依赖/架构/产物的前置校验失败，不是断言。
- 未做：回到改动前的提交再跑这 7 个做双跑反证（需另建一棵树，约 45 分钟）。现结论是推断，不是对比。

## 缺陷记录：#12 保留的上下文被当成租约（`2026-11-11`，serve 实测）

- **触发**：`--kv-stream`（保留默认开）下，服务端第二段对话进来。用户实测 req#7 `cache 37,412 (98.5%)`
  之后 req#8 抛 `KV activation reservation is stale` → 未捕获 → `terminate` → **进程 abort**（上下文仅 20%）。
- **Engine 级 30 秒复现**：`A(901 tok)` 保留后，任意不同前缀都抛
  `context cache exhausted: an idle Engine found no placement`；`NINFER_ADMISSION_DEBUG=1` 打印
  `no materialization plan`。
- **根因**：私有 continuation 目录容量 = `max_private_continuations` = 1。保留的上下文占住唯一槽位，
  新请求的规划器无法建立计划；若规划器已取用激活，句柄作废后再被使用就是 stale。base 的 resident
  缓存不是这个语义——它会把旧条目挤掉，冷请求照常接纳。
- **修复**（`f8f3d6c…` 实际提交见 log）：放置失败且原因是"无物化计划"时，丢弃队头**无法完整复用**的
  已发布 continuation，并**重做整次放置**（单次放置内部重试会让规划器握着已释放句柄 ⇒
  `pressure planning candidate is stale`，第一版踩过）。丢弃不搬字节（host 页即权威副本）。
- **验收**：Engine 级五步全接纳、无关/更长两臂 ids 与无缓存逐位（C15 测试 75 s PASS）；
  serve `A → B → A` 三条 200，同会话继续仍 `cache 78 (60.5%)`、TTFT 126 ms；`reuse_driver` 仍
  `reused=4614 equal_ids=1`、`serial_real` 750 s PASS、9/9 kv_stream 用例 PASS。
- **已知边界（容量，不是缺陷）**：只有 1 个私有槽 ⇒ 第二段对话一定会挤掉第一段；旧版看起来"保住了
  60.5%"是因为它把第二段**拒了**。要多会话同时留住上下文，需要更多私有槽（host RAM + 地址空间），
  属容量决策。
- **未关闭**：只共享开头的请求现在冷跑（不复用命中前缀）——要恢复需"替换式发布"（复用旧条目前缀的
  同时把它挤掉）。另：`NINFER_ADMISSION_DEBUG` 是本次新增的诊断口。
- **过程教训**：serve 的 429 复现第一次是"只重编了 `ninfer`、没重编 `ninfer-serve`"造成的假象；
  以后每次改引擎/运行时库，都要把 serve 与驱动一起重链。

## 复验（同一构建，`2026-11-11 19:2x`）

- 放置检查恢复只读后，15 个放置/缓存/流式用例 **全 PASS**（含 `ninfer_resource_manager_test`、
  `serial_real` 749 s、`geometry_real` 76 s、`eviction_real`（C15）76 s）。
- serve `A → B → A`：三条 **HTTP 200**，进程存活（修复前 B 是 429，或 `stale` → abort）。
- serve `A → A2`（同一会话继续）：`cache 78 (60.5%, turn closure)`、TTFT 126 ms——复用未受影响。
- 过程失误（记录在案）：第一版把"丢弃"放进只读的放置检查里，功能可过、契约被破坏，
  由 `resource_manager_test` 的 `retained source protection` 断言抓住；现改为在放置循环里、
  两次检查之间执行，每请求最多一次。

## 缺陷 #13：规划成功、执行才发现不可行（`2026-11-11`，serve 实测）

- **触发**：`worker crash: streamed KV frontier is beyond the space's context` → req#6 500、req#7 503、
  随后 `FATAL engine unavailable after an engine-wide failure`，进程退出（exit 2）。上下文约 20%。
- **成因**：`sequence.text_kv_valid = reuse_base` 声称"KV 有效到该前沿"，但**流式空间只有一个活世系**：
  中间任何"不延续"的请求都会把空间倒回到自己的大小，之后复用旧 continuation 就要求比空间持有的更多页。
  规划期不校验、`enter_request` 才抛 ⇒ 在 worker 里抛 ⇒ 整个进程被带走。
- **修复**：`KVAddressSpaceStore::stream_can_serve(tokens)` + 规划期校验（`request_plan.cpp`：
  任何非 Root 复用若空间装不下，直接返回"无候选" ⇒ 请求走冷路径），并配合准入期的驱逐让陈旧条目消失。
- **验收**（同一构建）：15 个放置/缓存/流式用例 **14/14 PASS**（`resource_manager_test`、`serial_real` 749 s、
  `geometry_real` 76 s、`eviction_real`（C15）76 s、`prefix_cache_index`、`context_cache_defaults` 等）；
  serve `A → B → A` 与 `A → A2` 共 5 条请求 **全 200**、进程存活；日志中 `worker crash` / `terminate` /
  `FATAL engine` 计数 = **0**；同一会话继续仍 `cache 78 (60.5%, turn closure)`。
- **纪律**：这一族三个缺陷（#11 保留被当租约、#12 空间倒回后声明仍有效、#13 规划期不校验可行性）都是
  "声明比字节活得久"。任何"保存的上下文"都必须由**持有跨度的那一层**回答"还装得下吗"。
