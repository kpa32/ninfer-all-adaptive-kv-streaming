# BASELINE：分叉时的环境事实与测量协议

状态：**§1–§3 已核实（只读命令，2026-11-08）；§5 的测量结果待填**。
原因：审计期间 Linux 机器上有一个正在进行的 `./build.sh`（PID 16123），所有者要求先不要动，
因此我没有并发构建、没有启动服务。等你放行后按 §4 的命令执行并回填 §5。

所有数字都必须来自真实命令输出；未执行的格子写 `pending`，不写推测值。

---

## 1. 环境事实（已核实）

| 项 | 值 | 命令 |
|---|---|---|
| 主机 | Debian 13，16 核，64 GiB RAM（`MemTotal` 64 213 MiB） | `nproc`、`free -m` |
| GPU | NVIDIA GeForce RTX 4080 SUPER，16 376 MiB，320 W 上限 | `nvidia-smi --query-gpu=...` |
| 驱动 | 615.71.09 | 同上 |
| CUDA | 13.4（V13.4.92） | `/usr/local/cuda/bin/nvcc --version` |
| 主机编译器 | gcc-13 / g++-13（Debian 13.3.0-16） | `build.sh` 自动选择 |
| CMake / Ninja | 3.31.6 / 1.12.1 | `build.sh` |
| PCIe | 最大 16.0 GT/s ×16（Gen4 x16）；**空闲时当前 2.5 GT/s** | `/sys/bus/pci/devices/0000:08:00.0/current_link_speed` |
| 计算能力 / SM | 8.9 / 80 | `device_profiles.json` 的 `nvidia-geforce-rtx-4080-super-sm89`（`origin: built in: RTX 4080 SUPER at 320 W, 2026-10-16`） |
| 仓库（Linux） | `/home/kpa/ninfer-fork/ninfer-all-adaptive-kv-streaming`，`master`，`6eb311fd`，工作区干净 | `git rev-parse HEAD`、`git status --short` |
| 构建目录 | `build-sm89`（审计时 929 MiB，构建进行中） | `du -sh` |
| 模型 | `/home/kpa/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（15 017 456 896 B） | `ls -la` |
| 聊天模板 | `/home/kpa/ninfer-fork/models/Qwen-UD-Chat-Templates.jinja`（`start-mtp.sh` 用相对路径，见 §3） | `ls` |

### artifact 事实（`python3 -m tools.artifact.inspect`，只读）

```
version: 3
name: qwen3.8-27b
components: text (Qwen3_5ForCausalLM / qwen3_5_text), proposal (131072 rows),
            vision (qwen3_5_vision, depth 27), mtp (Qwen3_5MTP), dflash2 (5 sliding layers, window 2048)
objects: 1249 (tensors 1243, resources 6), bindings 1513, uses 844
formats: bf16 627, fp32 96, int32 2,
         gguf: iq1_m 1, iq2_s 17, iq2_xs 9, iq2_xxs 6, iq3_s 131, iq3_xxs 68, iq4_xs 89,
               q2_k 19, q4_k 41, q6_k 5,
         native row-split: q4_g64_fp16 54, q5_g64_fp16 54, q6_g64_fp16 1, q8_g32_fp16 23
layouts: contiguous_le_v1 725, gguf_blocks_v1 386, row_split_k128_v1 132
```

结论：目标 artifact **不含** FP8 / NVFP4 / T2 / STQ / SM70 权重（支撑 `CUTLIST.md` A5）。

---

## 2. 构建事实（已核实）

`/home/kpa/ninfer-fork/build.sh`（不改）：

```
CUDA_ARCH=89           → -DCMAKE_CUDA_ARCHITECTURES=89
BUILD_DIR=build-sm89
JOBS=16
-DCMAKE_BUILD_TYPE=Release -G Ninja
-DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
-DNINFER_WEBUI_DIR=<third_party/webui/*>   （若无则 scripts/fetch-webui.sh）
可选 ccache launcher（机器上有 ccache 时）
```

实际编译命令（审计期间从进程表读到，sm_89 走兼容路径）：

```
nvcc ... -O3 -DNDEBUG -std=c++20
  --generate-code=arch=compute_89,code=[compute_89,sm_89]
  -DNINFER_SM8X_COMPAT=1 -DNINFER_TIER_AMPERE=1
  -lineinfo -Xfatbin=-compress-all -compress-mode=size
```

注意：`BUILD_TESTING=OFF` / `NINFER_BUILD_BENCHMARKS=OFF` 意味着**当前 build 目录里没有
`ninfer_tests` 与 `ninfer-bench`**。要跑测试/基准需要另建一个配置（例如 `build-sm89-tests`
加 `-DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON`），这会是一次接近全量的重新编译。
**这需要你决定是否现在做**（见 §6）。

---

## 3. 启动参数事实与两处不一致（已核实）

`/home/kpa/ninfer-fork/start-mtp.sh` 的实际参数（不改）：

```
model      models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer
--model-id Qwen3.8-27B-IQ3_S-Ninfer
--host 0.0.0.0 --port 11222 --stats-port 11223
--chat-template models/Qwen-UD-Chat-Templates.jinja
--max-concurrency 1 --log-level debug --log-stats-panel on
--preserve-thinking --default-reasoning-effort xhigh
--vision --vision-residency overlay --vision-max-merged 1024
--device-profile auto
--kv-dtype rk8v4 --max-context 120000 --kv-capacity 120000
--spec mtp --draft-tokens 3
--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--post-thinking-sampler temp=0.7,top_p=0.95,top_k=20,min_p=0 --post-thinking
--prefill-chunk 256
--gdn-state-fp16
--cuda-graph-allowance-mib 72
```

不一致（**不修，只报告**）：

1. `SERVER="$SCRIPT_DIR/ninfer-all/build-sm89/apps/ninfer-serve"` —— 实际产物在
   `$SCRIPT_DIR/ninfer-all-adaptive-kv-streaming/build-sm89/apps/ninfer-serve`，该脚本今天启动不了。
2. 脚本开头 `sudo nvidia-smi -pm 1` 与 `sudo nvidia-smi -i 0 -pl 320` —— 与"不要 sudo"冲突。
3. 工作目录假设 `$SCRIPT_DIR` 下有 `models/`（成立）与 `ninfer-all/`（不成立）。
4. `--max-context 120000` 小于 262 144 原生窗口；这正是阶段 2 要突破的限制
   （`rk8v4` 在 262 144 需要约 6.4 GiB KV，与 10.9 GiB 权重合计超过 16 GiB）。

---

## 4. 测量协议（放行后执行；每条命令都记完整输出）

环境前缀（每个 shell 都要，`build.sh` 里有但交互 shell 没有）：

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
REPO=/home/kpa/ninfer-fork/ninfer-all-adaptive-kv-streaming
BIN=$REPO/build-sm89/apps/ninfer
MODEL=/home/kpa/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer
TEMPLATE=/home/kpa/ninfer-fork/models/Qwen-UD-Chat-Templates.jinja
```

前置检查：

```bash
git -C $REPO rev-parse HEAD && git -C $REPO branch --show-current && git -C $REPO status --short
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
cat /sys/bus/pci/devices/0000:08:00.0/current_link_speed
$BIN --help | head -40
```

固定 prompt（确定性生成，不用模型推理）。**不要**把长 prompt 当命令行参数传：Linux 单参数上限
`MAX_ARG_STRLEN` = 128 KiB，120K token 的文本会直接 `E2BIG`，所以长 prompt 一律走 `--messages` 文件。

```bash
python3 - <<'PY'
import json, random
random.seed(20261108)
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
# 约 4 char/token -> 1.1M chars ~ 270K tokens，够 262K 那一档
with open('/tmp/base_long.txt','w') as f:
    for i in range(40000):
        f.write(f"[{i:05d}] " + " ".join(random.choice(words) for _ in range(12)) + "\n")
for ctx in (8192, 32768, 65536, 120000, 200000, 262000):
    chars = int(ctx * 3.6)          # 偏保守，实际 token 数以 CLI 报告为准
    text = open('/tmp/base_long.txt').read(chars)
    json.dump({"messages": [{"role": "user", "content": text + "\n\nWrite one sentence."}]},
              open(f'/tmp/base_ctx_{ctx}.json','w'))
open('/tmp/base_short.json','w').write(json.dumps(
    {"messages": [{"role": "user", "content": "Explain in one paragraph why paged KV caches help long-context inference."}]}))
PY
ls -l /tmp/base_ctx_*.json
```

A. 短上下文 smoke（每个 commit 都跑，与基线比对 token）：

```bash
cd /tmp   # 避免仓库相对路径干扰
/usr/bin/time -v $BIN "$MODEL" --chat-template "$TEMPLATE" \
  --messages /tmp/base_short.json \
  --max-context 8192 --max-new 64 --kv-dtype rk8v4 \
  --spec mtp --draft-tokens 3 --greedy --print-token-ids \
  --log-level info > /tmp/run_smoke.out 2> /tmp/run_smoke.err
echo rc=$?
sha256sum /tmp/run_smoke.out /tmp/run_smoke.err
grep -E "tok/s|GPU|memory|tokens" /tmp/run_smoke.err
```

B. 上下文阶梯（`rk8v4`，MTP3，greedy）。每档用对应长度的 messages 文件：

```bash
: > /tmp/baseline_ctx.txt
for CTX in 8192 32768 65536 120000; do
  ( /usr/bin/time -v $BIN "$MODEL" --chat-template "$TEMPLATE" \
      --messages /tmp/base_ctx_${CTX}.json \
      --max-context $CTX --kv-capacity $CTX --max-new 64 \
      --kv-dtype rk8v4 --spec mtp --draft-tokens 3 --greedy --print-token-ids \
      --log-level info > /tmp/run_${CTX}.out 2> /tmp/run_${CTX}.err ) &
  PID=$!
  peak=0
  while kill -0 $PID 2>/dev/null; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.5
  done
  wait $PID; rc=$?
  { echo "ctx=$CTX rc=$rc peak_vram_MiB=$peak";
    grep -E "prompt|tokens|tok/s|GPU memory|elapsed" /tmp/run_${CTX}.err; } >> /tmp/baseline_ctx.txt
done
cat /tmp/baseline_ctx.txt
```

注意：`nvidia-smi memory.used` 是**设备级**占用（含驱动/桌面），不是进程级；报告里两个都给
（CLI 自报的 GPU memory + 采样峰值）。如果 `prompt tokens` 超过 `--max-context`，CLI 会拒绝，
按它报的实际 token 数重生成更短的 `messages` 文件再跑（记录实际值，不记期望值）。

C. KV 格式对照（32K）：

```bash
for KV in bf16 rk8v4 rk4v4 rk2v4-e8; do
  $BIN "$MODEL" --chat-template "$TEMPLATE" --messages /tmp/base_ctx_32768.json \
    --max-context 32768 --kv-capacity 32768 --max-new 64 --kv-dtype $KV \
    --spec mtp --draft-tokens 3 --greedy --print-token-ids \
    --log-level info > /tmp/run_kv_${KV}.out 2> /tmp/run_kv_${KV}.err
  echo "kv=$KV rc=$?"
done
```

D. 服务端 smoke（PID 管理，结束后停掉；不杀别人的进程）：

```bash
SERVE=$REPO/build-sm89/apps/ninfer-serve
$SERVE "$MODEL" --model-id Qwen3.8-27B-IQ3_S-Ninfer \
  --host 127.0.0.1 --port 11222 --stats-port 11223 \
  --chat-template "$TEMPLATE" --max-concurrency 1 \
  --kv-dtype rk8v4 --max-context 120000 --kv-capacity 120000 \
  --spec mtp --draft-tokens 3 --prefill-chunk 256 --gdn-state-fp16 \
  --device-profile auto --cuda-graph-allowance-mib 72 \
  --log-level info > /tmp/serve.log 2>&1 &
PID=$!
for i in $(seq 60); do curl -sf http://127.0.0.1:11222/health && break; sleep 2; done
curl -s http://127.0.0.1:11222/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3.8-27B-IQ3_S-Ninfer","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}'
kill $PID; wait $PID 2>/dev/null
```

（命令里去掉了 `start-mtp.sh` 的两行 `sudo`；`--vision` 代码永久保留，带不带只取决于启动参数。
我这侧默认**不**带 `--vision`，以便与 CLI 测量用同一份显存布局；是否带请告知。）

E. 带宽与链路（阶段 2 前必须做，脚本自建，不改仓库）：

```bash
# 推理进行中读链路
cat /sys/bus/pci/devices/0000:08:00.0/current_link_speed
# pinned H2D 有效带宽：用 ninfer 自己的 copy 路径（tools/ 下的微基准或一次大块 cudaMemcpyAsync）
```

---

## 5. 测量结果

### 5.1 短上下文 smoke（服务端，`rk8v4`，MTP3，`--max-context 120000`）

来源：所有者 2026-11-08 15:06–15:08 的 `./start-mtp.sh` 启动日志（`build untagged+6eb311fd`，
`nvcc 13.4.92, 89`）。这是**服务端**数据，不是 CLI 阶梯，也不是我复现的值。

| 项 | 值 |
|---|---|
| 权重就绪 | 11.9 GiB 读取 / 11.6 GiB 上卡，23.7 s（513.5 MiB/s） |
| 启动总时长 | 27.8 s（含 pinned 8.00 GiB host KV、CUDA graphs 635 ms、warmup 97 ms） |
| KV 容量 | 120 000 tokens，`rk8v4`，explicit，1 875 pages；`runtime 3.62 GiB`；**启动后空闲 145.5 MiB** |
| 内存台账 | 权重后空闲 3.74 GiB → 启动后 145.5 MiB；headroom 0 B；slack 127.0 MiB；CUDA graphs 允许 72 MiB、用 36 MiB |
| req#1（371 prompt，openai-chat，tools 2） | prefill **1.31k tok/s**，TTFT 284 ms，decode **108.9 tok/s**，MTP 接受 23/42 (54.8%) |
| req#2（421 prompt，共享前缀命中 364 = 86.5%） | TTFT 45.6 ms，decode **127.8 tok/s**，MTP 接受 90/129 (69.8%) |
| req#3（长回复，5 s 窗口） | decode **112.0 tok/s**（560 tok/窗口），batch 1.00 |
| 加载时 PCIe 链路 | **16.0 GT/s**（空闲时 2.5 GT/s） |
| 峰值 VRAM | 15 894 / 16 376 MiB（进程 15 860 MiB） |

结论：

- 16 GiB 上 `rk8v4` 的实际上限就是 120K 量级（启动后只剩 145 MiB），阶段 2 要突破的不是权重，
  而是 **KV 占用**：`rk8v4` 每 token 约 25.5 KiB 的理论值与 `runtime 3.62 GiB / 120K`
  （含 State/workspace）吻合到 20% 以内。
- 短上下文 decode 112–128 tok/s 已经接近 `docs/performance.md` 里 RTX 4090 的 MTP3 数字
  （146/109），说明本卡当前没有被构建或功耗问题拖累，后续流式优化的对照基线是可信的。
- `~/.cache/ninfer/device-profiles.json` 里的 4080 SUPER 标定被**拒绝**：
  `calibrated against route catalog a0d4d93c979588e1 (build 2026-11-04.3-15-gf6928e00), this build's is e614262150e40eaf; it is ignored`。
  当前用的是内建 profile（78 routed keys, 2026-10-16）。是否花 GPU 时间重标定（`ninfer-calibrate`）
  是阶段 3 的决定，不是基线缺陷。

### 5.2 上下文阶梯

| ctx | prefill tok/s | decode tok/s | 峰值 VRAM MiB | 备注 |
|---:|---:|---:|---:|---|
| 8 192 | pending | pending | pending | |
| 32 768 | pending | pending | pending | |
| 65 536 | pending | pending | pending | |
| 120 000 | pending | pending | pending | 与 `start-mtp.sh` 同容量；服务端短上下文数字见 §5.1 |

### 5.3 KV 格式（32K）

| KV | prefill tok/s | decode tok/s | 峰值 VRAM | 每 token KV 字节（理论） |
|---|---:|---:|---:|---:|
| bf16 | pending | pending | pending | 64 KiB |
| rk8v4 | pending | pending | pending | 25.5 KiB |
| rk4v4 | pending | pending | pending | 17.5 KiB |
| rk2v4-e8 | pending | pending | pending | 13.5 KiB |

### 5.4 长文 planted-code recall

| ctx | code | 命中 |
|---:|---|---|
| | pending | pending |

### 5.5 带宽与链路

| 项 | 值 |
|---|---|
| 空闲链路 | 2.5 GT/s |
| 加载时链路 | **16.0 GT/s（已测，服务端运行时）** |
| pinned H2D 有效带宽 | pending（阶段 2 第一件事） |

---

## 6. 需要你决定的两件事（影响基线完整性）

1. **测试套件是否现在构建**：当前 build 关掉了 `BUILD_TESTING` 与 benchmarks。
   要跑 `ninfer_tests`（含 KV/attention/GDN/linear 的 oracle）与 `ninfer-bench`，需要另建
   `build-sm89-tests` 并接近全量重编（这台机器上大概是一次数小时的构建）。
   替代方案：先只做 CLI 基线与 golden token，测试套件等阶段 1 删除动作开始后再按需构建。
2. **基线要不要包含 262 144**：`rk8v4` 在 262 144 需要约 6.4 GiB KV，与 10.9 GiB 权重合计
   超过 16 GiB，**今天必然 OOM**。这是阶段 2 要解决的问题，不是基线缺陷。
   建议基线只到"能跑到的最大"（预计 `rk2v4-e8` 约 240K，`rk8v4` 约 120K），
   并把 OOM 的完整日志作为"目标差距"的证据保留。

---

## 7. 参考数字（**不是本机测量**，仅用于 sanity check）

来自本仓库 `docs/performance.md`（Qwen3.8-27B GSQ-RCO IQ3_S，`rk8v4`，服务端；每格的两个数字
是 **IQ3_S / 官方 groupwise-int artifact** 在同一张卡同一时段的对照）与 `docs/gpus.md`：

| 机器 | 短聊无起草 | 短聊 MTP3 | 32 768 上下文后 decode |
|---|---:|---:|---:|
| RTX 4090（24 GB，128 SM） | 70.6 / 55.0 | 146.4 / 109.3 | 66.0 / 52.2 |
| RTX 3090（24 GB，82 SM） | 59.9 / 40.3 | 108.7 / 82.7 | 50.4 / 38.4 |

4080 SUPER 是 80 SM、256-bit 736 GB/s 显存带宽（4090 为 128 SM / 1008 GB/s）。
本机基线落在 4090 与 3090 之间偏 3090 一侧是合理的；若明显低于 3090 那一列，
先查链路/功耗/构建参数，而不是先怀疑模型或 KV 格式。
