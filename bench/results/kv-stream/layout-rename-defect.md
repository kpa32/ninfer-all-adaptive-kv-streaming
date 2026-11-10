# 缺陷档案：布局改名不搬字节（流式槽位的正确性通道）

本文件是历史与排查记录，不是当前状态文档；当前状态见 `docs/maintainer/kv-streaming-verification.md`
与 `docs/cli.md`。把它写下来是因为**这个 bug 找了很久**，而且它的表现（"同一 prompt 换一台引擎/换一点
空闲显存就换答案"）极易被误判成环境噪声或"允许的布局敏感"。

## 1. 一句话

流式 KV 把一个页的**物理槽号**定义成布局的算术函数
（`kv_stream_window_slot = layer_offset(L) + page % resident(L)`，`src/core/kv_stream_layout.h:156`），
而 `repartition` 在迁移常驻窗口与环的边界时**只重发块表、不搬字节**，于是表的项指向了**别的页**：
同一个 prompt 用同样的权重算出不同的 logits，差异大小 0.19–0.63 nat，足以翻转 top-1。

## 2. 现象与暴露路径

1. 最初在 serve 上暴露：`--kv-stream` + 前缀复用，第二请求复用正常
   （`cache 4,614 (100.0%, private endpoint)`、TTFT 2 ms），但把 `device_state_slots` 由 0 改成 1 后，
   **首个请求自己**就崩了：4 token 后 EOS、`mtp accepted 0/9`（对照 8 token、`3/10`）。
2. 打开 `logprobs` 做位级比对后清楚：position 0 逐位相同，position 1 分布量级不同（top-1 翻转）。
3. 最终最小复现与缓存、投机、采样全部无关：**同进程两台完全相同的冷跑引擎**，只要池几何不同，
   答案就不同；同几何则逐位一致。

## 3. 决定性证据（数字均为实测，可复跑）

| # | 配置 | 结果 |
|---|---|---|
| E1 | 同进程 3 台冷引擎，几何相同（`resident=1411 ring=24`） | 三臂 ids 全等、logprob 逐位相同 |
| E2 | 同进程 2 台冷引擎，几何不同（`1431/ring 23` vs `1411/ring 24`） | ids 不等；pos1 `271/-0.8109` vs `8366/-0.1798` |
| E3 | 受控几何：建引擎前 `cudaMalloc` 预留 0 / 64 / 128 MiB | `resident 1411 / 1371 / 1331`；0 → `8366`，64 与 128 → 都是 `271`（答案由几何区间决定，稳定可复现） |
| E4 | 全驻留池（`max_context 8192`）两台引擎 | 逐位相同 |
| E5 | 预填块数二分（prompt 2048）：chunk 2048(1)/1024(2)/512(4) | 1、2 块相同；4 块发散（块越多，`advance` 触发的边界更多） |
| E6 | 两臂 `[stream] dispatch` 行 | 逐行相同（`route/width/span/keys/segment`）⇒ 不是注意力路由差异 |
| E7 | `DeviceArena` 全零、`prefill_a8` 开/关 | 数字完全相同 ⇒ 不是未初始化显存、不是 A8 |
| E8 | 修复后同 E3 的几何对照 | 两臂 `8366 / -0.00117492676` **逐位一致** |

E3 的量级说明：0.19–0.63 nat 远超"归约顺序"级别的 1e-4（对照：本仓库曾因 1e-4 的布局差异而
**撤回**一次布局改动，`519c02a7`），同量级参照是本仓库记录过的真 bug（层号二次映射 0.118 nat）。

## 4. 被排除的假设（不要再重复走）

1. **device StateImage 槽 / DeviceFork**：插桩显示全程 `fork=0`、两臂同一物理槽；`device/host_state_slots`
   各种组合单独验证都不改变现象形态。device 槽只是**间接改变池几何**，于是踩中同一个 bug。
2. **采样器（presence_penalty 1.5）**：惩罚归零后现象不变。
3. **prefix identity / publish / anchor**：`allow_prefix_identity=false`、`endpoint_anchors=false`、
   `progress_anchor_tokens=0` 单独关闭，差异不变。
4. **未初始化 workspace**：`DeviceArena` 全零后数字不变；`compute-sanitizer --tool initcheck` 0 errors。
5. **prefill A8（int8 激活）**：开关前后数字完全一致。
6. **注意力路由/包络**：两臂 `[stream] dispatch` 逐行相同。
7. **上游 kernel**：base（`f6928e00`）里没有 `kv_stream_*`（`git grep plan_kv_stream|window_slot|ring_pages`
   在 base 上 0 命中），页有稳定物理 ID，布局不参与命名 ⇒ 该通道在 base 上不存在。

## 5. 根因与责任链（全部是分支本地提交，2026-11-09，作者 kpa）

| commit | 主题 | 引入/依赖 |
|---|---|---|
| `85f95f6e` | *support a per-layer window so a round can concentrate its streams* | 引入推导式槽号 `layer_offset + page % resident` |
| `60dad291` | *make the ring a flat pool sized by the work it covers* | 注释断言"页在窗口内槽号不变，所以表项写一次即可" |
| `0b46796f` | *copy and consume the ring in runs instead of whole layers* | 写成硬注释"A page keeps its bytes where they are and changes only the slot it is named by" |
| `1159d91a` | *move the resident/ring boundary mid-run from measurement* | 让布局在请求中途变化 —— 上一条注释**当场失效**，未重新推导 |

机制：页的地址由布局推出 ⇒ 布局一变，所有常驻页改名；`repartition` 的"两边都常驻"分支只
`publish_slot`，字节留在旧槽 ⇒ 表项指向别人占着的槽。

## 6. 测试为什么没抓到

- `tests/test_kv_stream_layout.cpp::a_resident_page_keeps_its_slot` 验证的是**同一布局内**前沿推进时
  槽号稳定，不是**两个布局之间**；`grep repartition tests/` 没有任何字节级用例。
- 全部等价性验收（`check --suite equiv`、C1–C13）都是"同 capacity、固定池、单进程"的配置对照，
  **几何从未被当成变量**。而池大小由 `--kv-capacity auto` 从空闲显存算出，天然随环境变化。

## 7. 复现配方（10 分钟，不依赖任何诊断补丁）

```bash
# 同一构建、同一 prompt，只改"建引擎前预留多少显存"⇒ 池几何不同
LP_RESERVE1=0 LP_RESERVE2=64 LP_PROMPT_TOKENS=1536 LP_CHUNK=512 \
  ./lpg_driver <artifact.ninfer>     # 源码 debian:/tmp/lpg_driver.cpp
# 期望（修复后）：两臂 resident 不同，ids 与 logprobs 逐位相同
```

`lpg_driver` 用 `Engine::generate(..., logprobs=true)` 比较模型自己的 log p，不经采样器、不受 stop
token 影响；`cudaMalloc` 预留用来在**同一进程内**制造几何差。

## 8. 处置

- **先修正确性**（`c8bdb380`）：`repartition` 现在先把常驻窗口回写到 host 权威，再读入新形状命名的槽；
  所有回写都在任何回填之前，所以"离开窗口的页"与"接管其槽的页"不会交叉。
- **再砍掉换形**（后续提交）：`enter_layout(true)` 的 prompt→decode 换形与控制器试探性 repartition
  的唯一收益是 decode 轮 0.2%（记录见 strategy P1），却提供了本缺陷的唯一可达路径。池在**启动时**
  就按 decode 形规划整上下文，进程内不再变化。
- **回退点**：tag `kv-stream/layout-switch-before-removal` 与分支 `kv-stream/preserved-layout-switch`
  （均指向 `c8bdb380`，即"已修好、尚未砍"的状态）：
  ```bash
  git checkout kv-stream/layout-switch-before-removal -- \
    src/core/kv_stream_layout.h src/core/kv_stream_partition.h src/core/kv_stream_partition.cpp \
    src/models/qwen3_5/program/storage
  ```

## 9. 由此确立的不变量（回归必须守住）

1. **改名必须搬字节**：页的槽号若随布局变化，则该页的字节必须随之移动；不存在"只改表"的第三条路。
2. **逐位结论必须带几何指纹**：`main_page_groups` / `ring_runs` / `layer_resident` 必须进报告与记录，
   否则"逐位相同"不可复现（本次所有误判的根）。
3. **同一个 prompt 在两种池几何下必须给出相同的 ids 与 logprobs**：这是本仓库缺失的那一格测试。

## 10. 仍未关闭

- 冷跑 vs 保留跑在几何 `1431/1411`（ring 23/24）下仍发散一次（修复后复现），怀疑同类问题在
  **环寻址**（`slot_of_page` 的 ring 分支 / `upload()` 的 run 指派）而非窗口改名；需在同一受控几何下
  分离"只差 ring"与"只差 window"，然后按第 9 条的不变量处置。
- 该结论不依赖本分支的任何设计文档：全部证据来自 base 代码、本分支代码与实测数字。
