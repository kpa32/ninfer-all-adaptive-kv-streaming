# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

Every change follows the [engineering rules](CONTRIBUTING.md#engineering-rules) scoped to this
fork: lossless by default, the supported model (Qwen3.8-27B) on every supported platform,
with optimization and defaults tuned for the 16 GB configuration, the best measured default,
comparable measurements, documentation of the current state, and the test and tooling
conventions. Where an inherited document (CONTRIBUTING.md's "Every family", the other model
guides) names families outside the scope below, this document governs until it is updated.
When a task cannot meet a rule, its report names the rule and the reason.

## Product and architecture

This is a hard fork of NInfer-all (lineage: NInfer → NInfer-3090 → NInfer-all), cut from the
`kv-stream-27b-16g` branch after the adaptive KV streaming experiment. It maintains one
model, **Qwen3.8-27B**, on the NVIDIA cards the project supports, and optimizes and tunes
every feature for the **16 GB configuration**.

- **Model.** The supported model is Qwen3.8-27B (`Qwen3_5ForCausalLM`, the dense 27B of the
  Qwen3.5 family) and the artifacts built on it that share its architecture, binding and
  execution path: the official `groupwise-int` release, the GSQ-RCO IQ3_S release and the
  fine-tunes (abliterated, Uncensored, Ternary Bonsai 2). It keeps MTP and DFlash2 drafting,
  Vision, the nine `--kv-dtype` formats, YaRN, up to eight concurrent requests and the full
  serving surface. No other model family is in scope.
- **Cards.** Every supported NVIDIA build target stays supported: `70`, `75`, `80`, `86`,
  `89` and `120a` (CUDA 12.8 or newer; `120a` needs 13.1 or newer except 13.2.0 and 13.2.1).
  The **16 GB configuration is the optimization target**: new features, kernel routes,
  schedules, memory budgets and defaults are designed for 16 GB cards and are measured and
  tuned there. A feature that also runs on 24 GB and larger cards is not maintained or
  verified there; it inherits the 16 GB behavior and no separate tuning on the bigger cards
  is owed. The development, reference and only measured card is the **RTX 4080 SUPER 16 GB**
  (sm_89, 16,376 MiB, PCIe 4.0 x16, on a 64 GB RAM Debian 13 host). The binary carries a
  built-in device profile for the RTX 4080 SUPER (measured 2026-10-16 at its default 320 W
  power limit); like every inherited constant it is a hypothesis until verified on your card,
  and `ninfer-calibrate` measures again after a driver or clock change
  ([device profiles](docs/device-profiles.md)). Upstream schedules, route tables and published
  measurements come from other cards (RTX 3090, 4090, 5090, RTX PRO 6000); treat them, and any
  tuning constant, as a hypothesis until measured on the 4080 SUPER.
- **Active work line.** Adaptive KV streaming (`--kv-stream`): each full-attention layer keeps
  its newest pages on the device and streams the rest from pinned host memory every round, so
  Qwen3.8-27B reaches its 262,144-token context on a 16 GB card. The goal is
  [SPEC.md](SPEC.md); the single plan authority is
  [docs/maintainer/kv-streaming-strategy.md](docs/maintainer/kv-streaming-strategy.md) with its
  design, fidelity, measurements and verification companions.
- **Out of scope, kept in the tree.** Everything else inherited from upstream — Qwen3.6 dense and
  MoE, Qwen3.8-Flash-Next, Gemma 4, Flint, DeepSeek-V4.1-Flash, MoE expert residency and the CPU
  share, pipeline stages (`--devices`), tensor parallelism (`--tensor-parallel`) and the Windows
  builds — remains in the tree but is not maintained: no new features, no tuning, bug fixes only
  where they block the in-scope path. It is the fork's pruning target; removing a piece to
  shrink the tree toward the scope above is in scope work.

Generation uses one resident model on one GPU. It runs startup-fixed concurrency of one to eight
requests, bounded FIFO ingress (whose one-token reads trade places by remaining work), no
active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads, and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

Documentation rules, including README's What's New, are in
[CONTRIBUTING.md](CONTRIBUTING.md#documentation-describes-the-current-state).

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| The fork's scope, and the KV-streaming goal and plan | `SPEC.md`, `docs/maintainer/kv-streaming-strategy.md` |
| Engineering rules for every change: lossless, families, platforms, defaults, measurements, docs, tests | `CONTRIBUTING.md` (scoped by this file's product section until it is updated) |
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`; types in the headers `include/ninfer/types.h` lists |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use a Python 3.11 interpreter explicitly; the default shell's `python3` may be a different
version, so check it before relying on it. Build trees, artifacts and profiles live where the host
keeps them (by convention `build/`, `models/`, `profiles/ncu/`, `profiles/nsys/`,
`profiles/bench/`); the toolchain is CUDA 12.8 or newer, and 13.1 or newer for `120a` (not 13.2.0 or
13.2.1).
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Compilation needs no GPU. Executing CUDA tests, benchmarks or perplexity runs needs the local card
free: the test host is the RTX 4080 SUPER 16 GB with 64 GB of RAM on Debian 13, and this fork has
no rented hosts or pods path. Run there when the card is idle, and say so in the report when a
check could not run because it was not free.

## Commits

Create commits only when requested. Commit subjects follow the
[tests and tooling rules](CONTRIBUTING.md#tests-and-tooling).

## Remote test host (192.168.2.13)

The measurement card is reached as `kpa@192.168.2.13` (Debian 13, 16 cores, 62 GB RAM,
RTX 4080 SUPER 16,376 MiB, driver 615.71.09). CUDA 13.4: `nvcc` is not on the non-login
PATH, use `/usr/local/cuda/bin/nvcc`; cmake 3.31.6; Python 3.13.5.

- **Dedicated directory.** `/home/kpa/ninfer-16` is used exclusively for this fork branch,
  isolated from the other trees on the host. The repo clone is
  `/home/kpa/ninfer-16/ninfer-all-adaptive-kv-streaming`; its build tree is `build-sm89`
  (Release, sm_89, Ninja, ccache). `models/` is already in place
  (`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` plus `Qwen-UD-Chat-Templates.jinja`).
- **Never touch the other trees' processes.** A model server from another tree may be
  running on the card (e.g. `~/ninfer/ninfer-all/build-sm89/apps/ninfer-serve`); never
  kill, restart, or otherwise interfere with it. Compiling needs no GPU; running
  benchmarks does — check `nvidia-smi --query-compute-apps` first.
- **Forbidden zone.** `~/ninfer/` on the host: do not read, write, create, or build there;
  the stock binaries inside are executed only by the bench tool.
- **Old tree.** `~/ninfer-fork/ninfer-all-adaptive-kv-streaming` belongs to the
  `kv-stream-27b-16g` branch; never build or reset it for this fork.

Typical commands (sync only after pushing; never edit sources during a build):

```bash
# sync the dedicated tree (after local push)
ssh kpa@192.168.2.13 'cd /home/kpa/ninfer-16/ninfer-all-adaptive-kv-streaming && git fetch -q origin && git reset --hard -q origin/fork-qwen38-27b-16gb'
# configure once, then build incrementally (16 threads; sm_89)
ssh kpa@192.168.2.13 'cd /home/kpa/ninfer-16/ninfer-all-adaptive-kv-streaming && ./build.sh'
# single-file syntax check (not a substitute for running)
g++ -std=c++20 -fsyntax-only -Iinclude -Isrc -I/usr/local/cuda/include <file>.cpp
```

Pitfalls recorded from prior work on this host:

- `pgrep -f` self-matches its own pattern; use a bracketed form such as `[r]un_kv_stream[.]py`.
- The Windows and Linux clocks skew 10–25 s; `tar` reports future timestamps — `touch` a
  common header before building after editing it.
- Killing the local SSH client does not stop remote processes; confirm with `pgrep`.
- The bench tool refuses a busy card; check `nvidia-smi --query-compute-apps` before running.
- A full `BUILD_TESTING=ON` tree build takes about 30 minutes; never trigger it for one test.
