# 交接文档：KV 流式 + 前缀复用（当前状态）

本文件是**接手第一份要读的东西**：现在到哪一步、哪些已验证、哪些没做完、怎么复跑、怎么回退。
历史决策与作废数字在 `ledger.md`；单个缺陷的证据档案在 `layout-rename-defect.md`。

## 0. 当前状态（一句话）

前缀复用在流式下**已实现并逐位验证**；一个"布局改名不搬字节"的**正确性缺陷已修复**，并据此**移除了请求中途换布局**
（唯一收益 0.2%）；**几何不变性（C14）已建立**。剩余工作全是收尾与归因测量，见 §3。

| 项 | 值 |
|---|---|
| 分支 | `kv-stream-27b-16g` |
| HEAD | `259b1e26`（文档）；代码 HEAD `68941676` |
| 回退点（换形在、已修好） | tag `kv-stream/layout-switch-before-removal`、分支 `kv-stream/preserved-layout-switch`（都指向 `c8bdb380`） |
| 测试机 | `kpa@192.168.2.13`（debian），仓库 `~/ninfer-fork/ninfer-all-adaptive-kv-streaming` |
| 模型 | `~/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` |
| 构建 | `build-sm89`（Release、sm_89、`BUILD_TESTING=OFF`） |

## 1. 本轮做了什么（每条都能复验）

| 提交 | 内容 | 证据 |
|---|---|---|
| `03879140` | 流式请求结束保留上下文（frontier + 一张 **host** StateImage，KV 不复制），下一请求按前缀复用 | 第二请求 `reuse=1 base=4614`、`cache 4,614 (100.0%, private endpoint)`、TTFT 2 ms |
| `8561809e` | `--kv-stream` **默认保留并复用**（去掉开发旗标）；端点固定 host shelf；guard/启动拒绝 device 槽 | `--no-prefix-reuse` 仍可关；两请求 ids 与冷跑一致 |
| `884cfc66` | serve 不再静默覆盖 `--device-state-slots`，改为带修复 flag 的拒绝 | `--device-state-slots 1` 实测报错退出 |
| `c8bdb380` | **修正确性**：`repartition` 改名必须搬字节（先回写 host 权威，再读入新槽；所有回写在任何回填之前） | 几何 1413 vs 1372 两臂 `8366 / -0.00117492676` 逐位一致（修复前 `-0.1798` vs `-0.8109`） |
| `4844f2a6` | 缺陷档案 `layout-rename-defect.md` + ledger 条目 | 含 8 组实测证据、7 条已排除假设、复现配方、回退命令 |
| `68941676` | **移除请求中途换布局**：池在启动时按 decode 形（深环）一次规划整上下文；`enter_layout`/`evaluate_partition` 恒 false | 六组几何（`resident` 1413→40、`ring` 24→35）答案逐位相同；8 K 仍 `all-resident`、`layout_generation=0` |
| `898fca00` | `docs/cli.md` 改为"一次规划、请求期内不变"；**新增 C14 几何不变性** | verification C14 |
| `259b1e26` | 台账：等价套件与 251 K 的测量记录 | 见 §2 数字 |

**等价套件（本项目自带门禁）全 PASS**：`check --suite equiv` — 8 K / 100 K resident、183 K resident 与 streamed
（streamed `100.3 tok/s`、`39.6 ms/轮`、`upload 356.6 MiB/轮`、`exposed 0.0 ms`）。
记录 `bench/results/kv-stream/runs/20261111T001855Z-equiv-layout-once-c14-898fca00`。

**251 K**（`bench --case stream-251k`）：`95.0 tok/s`、`41.4 ms/轮`、`acceptance 3.9`、
`upload 689.1 MiB/轮`、`exposed 1.0 ms`。
记录 `bench/results/kv-stream/runs/20261111T001056Z-bench-layout-once-step1-898fca00`。

## 2. 缺陷档案（必读）

`bench/results/kv-stream/layout-rename-defect.md`。三条由此确立的不变量：

1. **改名必须搬字节**：页的槽号若随布局变化，字节必须同时移动；没有"只改表"的第三条路。
2. **逐位结论必须带几何指纹**：`main_page_groups` / `ring_runs` / `layer_resident` / `layout_generation`
   进报告与记录，否则"逐位相同"不可复现（本轮所有误判的根）。
3. **同一 prompt 在两种池几何下必须给出相同 ids 与 logprobs**（C14）。

## 3. 未完成事项（按优先级，含判据与成本）

### U1 251 K 的交错 A/B 归因（Gate 2 剩余项）
- 问题：251 K 现在 `95.0 tok/s`，同配方历史记录 ~96.5 tok/s，差 ~1.5%；记录区**没有相邻基线**可归因。
- 做法：与保留 tag `kv-stream/layout-switch-before-removal` 做**交错** A/B（同一提示、交替 reps），
  需要**第二套 build 目录 = 一次全量编译（约 25–30 分钟）**，须先获批。
- 判据：轮时间中位差 ≤ +1%、prefill tok/s ≥ −1%；否则回到 §5 的 C 方案（分配式槽位）。

### U2 Step 2：删除死代码（纯简化，性能中性，可延后）
- 删除：`StreamedKVSpace::repartition` 的窗口迁移分支（`advance` 的全驻留加宽已不可达）、
  `kv_stream_partition.{h,cpp}`、`test_kv_stream_partition.cpp`、`NINFER_STREAM_WINDOW_PAGES`
  （改为启动期规划生效）、`layout_generation` 的递增语义。
- 风险：`kv_stream_feedback` 被 span tuner 复用，删除前先核对引用。
- 判据：编译通过；`test_kv_stream_layout`/`feedback`/`span_tuner` 绿；两个驱动仍通过；251 K 不变。

### U3 in-repo 驱动尚未验证编译（**上次命令被中止**）
- 新加：`tools/bench/kv_stream/{geometry_driver,reuse_driver}.cpp` + `README.md`。
- 未做：从仓库路径编译并各跑一次。**这是接手第一件小事**（命令见 README 与 §4）。
- 注意：改动后必须**重新链接**驱动（链接在修复前的旧驱动会给出错误的"仍发散"读数，本轮因此误判过一次）。

### U4 自动化（C14 目前只有人工驱动）
- C14 应成为自动化用例：两臂几何不同 ⇒ ids/logprobs 逐位（可用 `LP_RESERVE` 手法或显式 pool 页数）。
- `test_engine_kv_stream_serial_real` 的 cold + retained 两臂仍未在测试树上跑（需 `BUILD_TESTING=ON`，约 30 分钟，须获批）。
- `test_kv_stream_span_tuner.cpp` 至今未在 `tests/cmake/CoreTests.cmake` 注册。

### U5 端到端长会话与格式覆盖（产品面）
- `--kv-stream` 下 192 K / 251 K 的**多轮对话**实测（看 `cache N (x%)`、TTFT、`stream.retained`）未做。
- `int8` 262 144 启动 + 稳态未跑（P6 的另一条验收）。
- serve 的 `--kv-stream` 与 4 个模型族（Qwen3.5-core / Flash-Next / Gemma 4 / Flint / DeepSeek-V4.1）交叉冒烟未做。

### U6 测试机上的临时改动（**必须先清理**）
`debian:~/ninfer-fork/ninfer-all-adaptive-kv-streaming` 有 **8 处未提交**的诊断/放开守卫补丁：
`src/core/arena.cu`（`NINFER_ZERO_ARENA`）、`src/models/qwen3_5/program/decode.cpp`、`prefill.cpp`、
`storage/context.cpp`、`transactions/capture.cpp`（`NINFER_STATE_DEBUG` 打印）、
`planning/startup.cpp`（放开 device/host 槽位守卫 + 强制聚合值）、
`src/ops/softmax_attention/.../causal_softmax_attention.cpp`（打印上限 32→4096）、
`src/serve/serve_options.cpp`（`device_state_slots = 1` TEMP）。
- 回滚：`git checkout -- <file>`；诊断脚本与手法见缺陷档案 §7，需要时重加，不要留在树里。
- 另：`/tmp` 下的驱动对象文件/二进制可丢，源码已在仓库。

### U7 文档收尾
- `docs/maintainer/kv-streaming-strategy.md` 的 P1 行与 §0.2 仍描述"首轮进入 decode 形"，需按"启动即 decode 形"改写。
- `docs/maintainer/kv-streaming-design.md` 若提到双布局/阶段切换，同样改写。
- `docs/maintainer/kv-streaming-measurements.md` 增一行新形状的 251 K/183 K 数字（含几何指纹）。
- `bench/results/kv-stream/handoff.md`（本文件）在 U1 完成后补 A/B 结论。

## 4. 环境与常用命令

**机器**：`kpa@192.168.2.13`，RTX 4080 SUPER 16 376 MiB，CUDA 13.4.92（`nvcc` 在 `/usr/local/cuda/bin`），
cmake 3.31.6，`python3` 3.13.5（本项目要求 Python 3.11+，脚本用系统 3.13 可跑）。

```bash
# 同步仓库（本地 push 后，构建期间不改源码）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && git checkout -- . && \
  git fetch -q origin kv-stream-27b-16g && git merge --ff-only FETCH_HEAD -q && git log --oneline -1'

# 增量构建（默认；不要全量）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  CCACHE_DISABLE=1 cmake --build build-sm89 --target ninfer ninfer-serve -j 12'

# 守卫单测（不走测试树，手工编译）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  g++ -std=c++20 -O1 -Iinclude -Isrc tests/test_kv_stream_guards.cpp -o /tmp/tg && /tmp/tg'

# 等价性门禁
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  python3 tools/bench/run_kv_stream.py check --suite equiv --tag <tag>'

# 251 K 稳态（约 4 分钟）
ssh debian 'cd ~/ninfer-fork/ninfer-all-adaptive-kv-streaming && \
  python3 tools/bench/run_kv_stream.py bench --case stream-251k --reps 2 --tag <tag>'

# 手工驱动（先按 tools/bench/kv_stream/README.md 编译并**重新链接**）
/tmp/geometry_driver <artifact>              # 同几何对照
LP_RESERVE2=64 /tmp/geometry_driver <artifact>   # 不同几何，必须仍逐位相同
LP_ARM2_RETAINED=1 /tmp/geometry_driver <artifact>
/tmp/reuse_driver <artifact>                 # 两请求复用
```

## 5. 决策与回退

- **换形已移除**，理由与数据：`ledger.md`（收益 0.2% × 正确性通道）。
- **回退**（如需恢复换形）：
  ```bash
  git checkout kv-stream/layout-switch-before-removal -- \
    src/core/kv_stream_layout.h src/core/kv_stream_partition.h src/core/kv_stream_partition.cpp \
    src/models/qwen3_5/program/storage
  ```
- 若 U1 测出 decode 退化 > 2%，改走**分配式槽位**（run 粒度 ID + 发布表）：换形只搬"跨边界"页，
  比当前"整窗回写 host"更快；这是唯一预留的备选方案。

## 6. 硬性工作约定（沿用）

- 动手前先说明；先搜项目已有基建；不租 pod（测试机一直可用）；只做增量构建。
- 提交/推送后同步测试机；Conventional Commits；结论要真（没跑写"未跑"）。
- 任何"逐位"结论必须带几何指纹；任何布局变更必须二选一（字节等价，或不存在该旋钮）。
