#!/usr/bin/env python3
"""Record, replay and compare KV-streaming runs.

One command runs the CLI, and everything the run was and did lands in
`bench/results/kv-stream/`: the command line, the git state, the binary's and the model's digests,
the card and its clocks, the prompt's recipe, the program's own JSON report (every phase's time,
both sides of the PCIe traffic, the per-round and per-layer copy series, speculative acceptance)
and the generated ids. A later run is compared against those files instead of re-running both sides
of a comparison.

    python tools/bench/run_kv_stream.py calibrate
    python tools/bench/run_kv_stream.py golden --suite equiv      # the original binary's ids
    python tools/bench/run_kv_stream.py check  --suite equiv      # our build against them
    python tools/bench/run_kv_stream.py bench  --case stream-251k --reps 3 --tag ring2
    python tools/bench/run_kv_stream.py report                    # regenerate INDEX.md
    python tools/bench/run_kv_stream.py compare runs/<a> runs/<b>

`--oracle` names the original build the streamed layout must answer like. It is only executed,
never written: `golden` records its token ids, so a later change is judged against them.

Everything is pinned: all flags that move the arithmetic are in FIXED_FLAGS, the prompt is a
recipe rather than a file, and a run refuses to start on a busy card. A record is never
overwritten; a new run gets a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "bench" / "results" / "kv-stream"
DEFAULT_MODEL = Path.home() / "ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
DEFAULT_BINARY = REPO_ROOT / "build-sm89/apps/ninfer"
DEFAULT_ORACLE = Path("/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer")

# The prompt is a fixed recipe: the same sentence repeated, then a tail that makes the model keep
# writing. A fixed tail also makes the token count linear in the repetition count, which is what lets
# two short calibration runs place a 183 K-token prompt exactly. The tail asks for a numbered list:
# greedy decoding then continues the list instead of closing the answer, so a case that wants 512
# generated tokens really gets them (an assistant-style tail hits the stop token after a few dozen).
PROMPT_SENTENCE = (
    "NInfer keeps every KV page of the context in pinned host memory and streams the newest "
    "window onto the device, so a round copies only the share the window lacks."
)
PROMPT_TAIL = (
    "\n\nContinue this list of positive integers, one number per line, from 21 up to 800. "
    "Each line holds one number and nothing else: no commentary, no summary, and no line "
    "before 800.\n"
)
CALIBRATION_POINTS = (100, 200)

# Every flag that moves the arithmetic. --prefill-chunk changes the batched shapes and was
# measured to move tokens (measurements M-08), so a comparison never mixes two questions.
FIXED_FLAGS: tuple[str, ...] = (
    "--kv-capacity",
    "auto",
    "--kv-dtype",
    "rk4v4",
    "--gdn-state-fp16",
    "--prefill-chunk",
    "256",
    "--greedy",
    "--seed",
    "42",
    "--invariant-attention",
    "--spec",
    "mtp",
    "--draft-tokens",
    "3",
    # A thinking request publishes reasoning, and the Engine keeps log probabilities for content
    # tokens only: the ids would still compare, but SPEC wants the distributions too, so the
    # benchmark asks for the answer instead of the reasoning.
    "--no-thinking",
)


@dataclass(frozen=True)
class Case:
    name: str
    purpose: str
    kind: str  # "equivalence" | "perf"
    target_tokens: int
    capacity: int
    max_new: int
    variants: tuple[str, ...]
    oracle: str = "original"  # "original" records the base binary's ids, "self" ours
    resident_flags: tuple[str, ...] = ()
    streamed_flags: tuple[str, ...] = ()
    # A case whose purpose names a full-length generation must not silently stop early: the record
    # then shows the model closed the answer instead of the window sliding.
    require_full_generation: bool = False
    # A case that pins its own cache format: the delivery format `rk8v4` needs a context the all-resident
    # layout can still hold, which is a different case from the `rk4v4` range the suite uses by default.
    kv_dtype: str = ""
    # Whether the case's argv must actually stream. Where the pool holds the whole context the correct
    # answer is that nothing streams and the adapter costs nothing, so a case measuring that region
    # asserts the opposite: the window covers the prompt and no page is copied during decode.
    expect_streaming: bool = True


# The contexts both layouts can serve: the original with MTP3 stops starting above ~186 K, and the
# streamed layout's window at rk4v4+MTP3 is 2 835 pages (181 440 tokens), so 183 K is the range
# where one layout is possible and the other engaged. -183k generates 512 tokens so decode crosses
# eight pages per layer and the ring reuse and cross-round prefetch are exercised, not only the
# prompt's own streaming.
CASES: dict[str, Case] = {
    "equiv-8k": Case(
        name="equiv-8k",
        purpose="Small-context regression: our build must answer exactly like the original binary.",
        kind="equivalence",
        target_tokens=8000,
        capacity=8192,
        max_new=16,
        variants=("resident",),
        require_full_generation=True,
    ),
    "equiv-100k": Case(
        name="equiv-100k",
        purpose="Mid-context regression and the prefill shapes of a real 100 K prompt.",
        kind="equivalence",
        target_tokens=100_000,
        capacity=100_096,
        max_new=16,
        variants=("resident",),
        require_full_generation=True,
    ),
    "equiv-100k-rk8v4": Case(
        name="equiv-100k-rk8v4",
        purpose=(
            "The delivery format at the largest context both layouts serve: rk8v4's all-resident "
            "layout holds ~110K on a 16 GB card (2.67 GiB of KV beside 11.6 GiB of weights), so this "
            "is where a streamed rk8v4 run can be compared bit for bit against a resident one. Above "
            "it only the streamed layout starts, which is why the suite's other range is rk4v4."
        ),
        kind="equivalence",
        target_tokens=100_000,
        capacity=101_376,
        max_new=512,
        variants=("resident", "streamed"),
        resident_flags=("--kv-headroom-mib", "16"),
        streamed_flags=("--kv-headroom-mib", "1008"),
        require_full_generation=True,
        kv_dtype="rk8v4",
    ),
    "equiv-183k": Case(
        name="equiv-183k",
        purpose=(
            "The only range both layouts serve: the original all-resident and our streamed layout "
            "must emit the same 512 ids while decode slides the window through the ring."
        ),
        kind="equivalence",
        target_tokens=183_000,
        capacity=184_064,
        max_new=512,
        variants=("resident", "streamed"),
        # The all-resident exception needs the whole 1 GiB automatic headroom cut down to 16 MiB
        # to fit its 3.99 GB reservation. The streamed run keeps the gigabyte, which is what a
        # measured scan (`window-scan.json`) says leaves a 2 111-page (135 104-token) window: the
        # prompt is past it, so ~749 pages per layer really do travel every round. Both are
        # memory-layout knobs; the arithmetic is the same either way.
        resident_flags=("--kv-headroom-mib", "16"),
        streamed_flags=("--kv-headroom-mib", "1008"),
        require_full_generation=True,
    ),
    "stream-251k": Case(
        name="stream-251k",
        purpose=(
            "The optimization target: a 251 K-token prompt at the 262 144 native capacity on a "
            "16 GB card, decode with MTP3 over the streamed window."
        ),
        kind="perf",
        target_tokens=251_000,
        capacity=262_144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
    ),
    # The same shape with a long enough decode that the one-time costs - the pool's first layout
    # experiment, the first round's warm-up - are a rounding error. A 64-token decode reports a
    # median that hides them and a mean that over-weights them; this case is what a session sees.
    "stream-251k-steady": Case(
        name="stream-251k-steady",
        purpose=(
            "Steady-state 251 K decode: 512 tokens, so the first-round and first-experiment costs "
            "amortise the way they do in a session."
        ),
        kind="perf",
        target_tokens=251_000,
        capacity=262_144,
        max_new=512,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
    ),
    # The interval the feature exists for: a pool that holds most - and at the low end all - of one
    # layer's context, so the ring can collapse and a round carries far less. These three points are
    # the measured answer to "does the adaptive boundary actually move here", which 183 K and 251 K
    # cannot show because their ring is already the right size.
    "stream-150k": Case(
        name="stream-150k",
        purpose=(
            "A pool that holds the whole context: --kv-stream must cost nothing, moving no page and "
            "decoding at the resident rate."
        ),
        kind="perf",
        target_tokens=150_000,
        capacity=262_144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        expect_streaming=False,
    ),
    # The same context with a pool too small to hold it: here the boundary has to be found by the
    # controller, which is the path 183 K and 251 K never exercise (their ring is already right).
    "stream-150k-tight": Case(
        name="stream-150k-tight",
        purpose=(
            "A 150 K context in a pool that cannot hold it: the controller must move the boundary and "
            "a round must carry far less than at 251 K."
        ),
        kind="perf",
        target_tokens=150_000,
        capacity=262_144,
        max_new=512,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        streamed_flags=("--kv-headroom-mib", "1008"),
    ),
    # The same 183 K shape with a pool that cannot hold it: this is where the copies cannot hide behind
    # the compute (one run of margin against six runs of extent), so it is the case that prices the
    # launch width and the run width on the exact shape the equivalence suite uses.
    "stream-180k-tight": Case(
        name="stream-180k-tight",
        purpose=(
            "183 K in a pool that cannot hold it: the copies have one run of margin, so the launch "
            "width and the run width are what decide whether they hide."
        ),
        kind="perf",
        target_tokens=180_000,
        capacity=262_144,
        max_new=512,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        streamed_flags=("--kv-headroom-mib", "1008"),
    ),
    "stream-180k": Case(
        name="stream-180k",
        purpose=(
            "The largest context this pool still holds whole: streaming must stay idle and the "
            "decode rate must match the resident one."
        ),
        kind="perf",
        target_tokens=180_000,
        capacity=262_144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        expect_streaming=False,
    ),
    "resident-8k": Case(
        name="resident-8k",
        purpose="Card-speed reference: the same shape without streaming, to price the streamed path.",
        kind="perf",
        target_tokens=8000,
        capacity=8192,
        max_new=64,
        variants=("resident",),
        oracle="self",
        require_full_generation=True,
    ),
    # The speculative arm of the 251 K comparison: one flag apart from stream-251k, because the
    # n-gram copier would otherwise hide what the drafter itself accepts.
    "spec-251k-none": Case(
        name="spec-251k-none",
        purpose="251 K without speculation: the window the drafters take memory from.",
        kind="perf",
        target_tokens=251000,
        capacity=262144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        streamed_flags=("--spec", "none", "--ngram-draft-tokens", "0"),
    ),
    "spec-251k-mtp3": Case(
        name="spec-251k-mtp3",
        purpose="251 K with MTP and three drafts, the same drafting the other arms get.",
        kind="perf",
        target_tokens=251000,
        capacity=262144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        streamed_flags=("--draft-tokens", "3", "--ngram-draft-tokens", "0"),
    ),
    "spec-251k-mtp7": Case(
        name="spec-251k-mtp7",
        purpose="251 K with seven drafts: the widest window the MTP block verifies.",
        kind="perf",
        target_tokens=251000,
        capacity=262144,
        max_new=64,
        variants=("streamed",),
        oracle="self",
        require_full_generation=True,
        streamed_flags=("--draft-tokens", "7", "--ngram-draft-tokens", "0"),
    ),
    "prompt-shape": Case(
        name="prompt-shape",
        purpose=(
            "Cheap guard on the prompt recipe itself: at 8 K it must generate the full 512 tokens "
            "without hitting the stop token, which is what makes the 183 K case exercise page "
            "slides during decode."
        ),
        kind="perf",
        target_tokens=8000,
        # The prompt itself stays 8 K, but the case must have room for its 512 output tokens:
        # an 8 192-token capacity answers 245 of them and reports context-capacity.
        capacity=16384,
        max_new=512,
        variants=("resident",),
        oracle="self",
        require_full_generation=True,
    ),
}
EQUIVALENCE_SUITE = ("equiv-8k", "equiv-100k", "equiv-183k")
PERF_SUITE = ("prompt-shape", "stream-251k", "resident-8k")
SPEC_SUITE = ("spec-251k-none", "spec-251k-mtp3", "spec-251k-mtp7")


# ---------------------------------------------------------------------------------------------
# Facts about the run


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(point: datetime) -> str:
    return point.strftime("%Y%m%dT%H%M%SZ")


def iso(point: datetime) -> str:
    return point.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_output(command: Sequence[str]) -> Optional[str]:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def git_state() -> dict[str, Any]:
    head = command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]) or ""
    branch = command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    status = command_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]) or ""
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "HEAD"], capture_output=True
    ).stdout
    return {
        "head": head,
        "short_head": head[:8],
        "branch": branch,
        "dirty_files": dirty,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else "",
    }


def host_state() -> dict[str, Any]:
    memory_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "kernel": platform.platform(),
        "cpu": cpu,
        "cpu_threads": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "python": sys.version.split()[0],
    }


GPU_QUERY = (
    "index,name,driver_version,memory.total,memory.used,"
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
    "clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,"
    "clocks_throttle_reasons.active,utilization.gpu"
)

GPU_FIELDS = (
    "index",
    "name",
    "driver_version",
    "memory_total_mib",
    "memory_used_mib",
    "pcie_gen_current",
    "pcie_gen_max",
    "pcie_width_current",
    "pcie_width_max",
    "clock_sm_mhz",
    "clock_memory_mhz",
    "temperature_c",
    "power_w",
    "power_limit_w",
    "throttle_reasons",
    "utilization_percent",
)


def _number(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def gpu_state() -> dict[str, Any]:
    output = command_output(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
    )
    devices: list[dict[str, Any]] = []
    if output:
        for line in output.splitlines():
            values = [_number(part) for part in line.split(",")]
            if len(values) == len(GPU_FIELDS):
                devices.append(dict(zip(GPU_FIELDS, values)))
    compute = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    return {
        "devices": devices,
        "compute_apps": [line for line in (compute or "").splitlines() if line.strip()],
    }


def require_idle_gpu(allow_busy: bool) -> None:
    state = gpu_state()
    if not state["devices"]:
        raise SystemExit("error: nvidia-smi reported no GPU")
    busy: list[str] = []
    for device in state["devices"]:
        used = device.get("memory_used_mib")
        if isinstance(used, (int, float)) and used > 512:
            busy.append(f"gpu {device['index']} has {used} MiB in use")
    busy.extend(state["compute_apps"])
    if busy and not allow_busy:
        raise SystemExit(
            "error: the card is not idle, so the run would not be comparable:\n  "
            + "\n  ".join(busy)
            + "\n  (stop the other process, or pass --allow-busy to record anyway)"
        )


# ---------------------------------------------------------------------------------------------
# The prompt


def prompt_text(sentences: int) -> str:
    return (PROMPT_SENTENCE + " ") * sentences + PROMPT_TAIL


def prompt_payload(sentences: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt_text(sentences)}]


def write_prompt(directory: Path, sentences: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompt.json"
    path.write_text(json.dumps(prompt_payload(sentences), ensure_ascii=False))
    return path


def prompt_recipe_digest() -> str:
    return sha256_text(PROMPT_SENTENCE + "\n" + PROMPT_TAIL)


@dataclass(frozen=True)
class Calibration:
    model_sha256: str
    tokens_per_sentence: float
    tail_tokens: float
    points: list[dict[str, int]]
    measured_utc: str
    argv: list[str]

    def sentences_for(self, target_tokens: int) -> int:
        sentences = int(round((target_tokens - self.tail_tokens) / self.tokens_per_sentence))
        return max(1, sentences)

    def tokens_for(self, sentences: int) -> float:
        return self.tokens_per_sentence * sentences + self.tail_tokens


def load_calibration(path: Path) -> Optional[Calibration]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    # A calibration belongs to one model and one prompt recipe: changing the sentence or the tail
    # changes the offset, and a stale fit would silently place the prompt somewhere else.
    if data.get("recipe_sha256") != prompt_recipe_digest():
        return None
    return Calibration(
        model_sha256=data["model_sha256"],
        tokens_per_sentence=data["fit"]["tokens_per_sentence"],
        tail_tokens=data["fit"]["tail_tokens"],
        points=data["points"],
        measured_utc=data["measured_utc"],
        argv=data["argv"],
    )


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "kv-stream-prompt-calibration",
                "model_sha256": calibration.model_sha256,
                "recipe_sha256": prompt_recipe_digest(),
                "sentence": PROMPT_SENTENCE,
                "tail": PROMPT_TAIL,
                "points": calibration.points,
                "fit": {
                    "tokens_per_sentence": calibration.tokens_per_sentence,
                    "tail_tokens": calibration.tail_tokens,
                },
                "measured_utc": calibration.measured_utc,
                "argv": calibration.argv,
            },
            indent=2,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------------------------
# Running one request


SUMMARY_LINE = re.compile(r"^summary\s+(.+?)\s{2,}(.+?)\s*$", re.MULTILINE)
GENERATED_IDS = re.compile(r"^tokens\s+generated ids\s+([0-9 ]+)\s*$", re.MULTILINE)
RATE = re.compile(r"^([0-9.]+)\s*([kKmM]?)", re.ASCII)
STARTUP_WEIGHTS = re.compile(r"weights ready \| ([0-9.]+) GiB")


@dataclass
class RepResult:
    argv: list[str]
    returncode: int
    seconds: float
    directory: Path
    report: dict[str, Any]
    summary: dict[str, str]
    ids: list[int]
    stderr_tail: str


def parse_summary(text: str) -> dict[str, str]:
    return {name.strip(): value.strip() for name, value in SUMMARY_LINE.findall(text)}


def parse_ids(text: str) -> list[int]:
    matches = GENERATED_IDS.findall(text)
    if not matches:
        return []
    return [int(token) for token in matches[-1].split()]


def parse_rate(text: str) -> Optional[float]:
    match = RATE.match(text)
    if not match:
        return None
    value = float(match.group(1))
    return value * {"": 1.0, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}[match.group(2)]


def gib(value: Optional[float]) -> str:
    return f"{value / 2**30:.2f} GiB" if value else "n/a"


def startup_weight_bytes(stderr: str) -> int:
    """The loader prints what it put on the card before anything else can fail."""
    match = STARTUP_WEIGHTS.search(stderr)
    return int(float(match.group(1)) * 2**30) if match else 0


def page_group_bytes(report: Mapping[str, Any]) -> int:
    """One page of one layer: the unit a pool's bytes and a window's pages are counted in."""
    stream = report.get("stream", {})
    layers = stream.get("layers") or 0
    page = stream.get("page_bytes") or 0
    return int(layers) * int(page)


def first_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.startswith("error:"):
            return line.strip()
    return ""


def apply_spec_override(argv: list[str], extra: Sequence[str]) -> list[str]:
    """Rewrite the pinned speculation flags when a config names its own backend.

    `--spec none` rejects a stray `--draft-tokens`, so an override replaces both rather than
    appending a second `--spec` after a draft count the backend cannot take.
    """
    if "--spec" not in extra:
        return argv
    spec = extra[list(extra).index("--spec") + 1]
    draft = extra[list(extra).index("--draft-tokens") + 1] if "--draft-tokens" in extra else None
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in ("--spec", "--draft-tokens"):
            if token == "--spec":
                out += ["--spec", spec]
            elif draft is not None:
                out += ["--draft-tokens", draft]
            skip = True
            continue
        out.append(token)
    return out


def with_pinned(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a pinned flag's value instead of appending a second one.

    `rk4v4` is the measured test stand-in; the delivery format is `rk8v4` (handoff 0.0), and a
    record has to say which one it ran, so the override replaces the value in place and the argv
    the record stores stays the one the process read.
    """
    out: list[str] = []
    replaced = False
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == flag:
            out += [flag, value]
            replaced = True
            skip = True
            continue
        out.append(token)
    if not replaced:
        out += [flag, value]
    return out


def request_argv(
    binary: Path,
    model: Path,
    prompt_path: Path,
    case: Case,
    variant: str,
    device: int,
    extra: Sequence[str] = (),
    dtype: str = "",
) -> list[str]:
    argv = [str(binary), str(model), "--messages", str(prompt_path)]
    argv += ["--max-context", str(case.capacity), *FIXED_FLAGS, "--max-new", str(case.max_new)]
    if dtype:
        argv = with_pinned(argv, "--kv-dtype", dtype)
    argv += list(case.streamed_flags if variant == "streamed" else case.resident_flags)
    argv = apply_spec_override(argv, extra)
    # A later flag wins in the CLI's parser, so a config override goes after the pinned flags and
    # the argv still records exactly what the run used.
    argv += list(extra)
    if variant == "streamed":
        argv.append("--kv-stream")
    return argv + ["--device", str(device)]


def run_once(
    binary: Path,
    model: Path,
    prompt_path: Path,
    case: Case,
    variant: str,
    device: int,
    directory: Path,
    timeout: float,
    with_report: bool,
    with_logprobs: bool = False,
    extra: Sequence[str] = (),
    dtype: str = "",
) -> RepResult:
    directory.mkdir(parents=True, exist_ok=False)
    argv = request_argv(binary, model, prompt_path, case, variant, device, extra, dtype)
    argv.append("--print-token-ids")
    if with_logprobs:
        argv.append("--logprobs")
    report_path = directory / "report.json"
    if with_report:
        argv += ["--report-json", str(report_path)]
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        process = subprocess.Popen(
            argv, stdout=out, stderr=err, start_new_session=True, env=os.environ.copy()
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            returncode = -9
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
    seconds = time.monotonic() - started
    stderr = stderr_path.read_text(errors="replace")
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    ids = [int(token) for token in report.get("generation", {}).get("token_ids", [])]
    if not ids:
        ids = parse_ids(stderr)
    return RepResult(
        argv=argv,
        returncode=returncode,
        seconds=seconds,
        directory=directory,
        report=report,
        summary=parse_summary(stderr),
        ids=ids,
        stderr_tail="\n".join(stderr.strip().splitlines()[-3:]),
    )


# ---------------------------------------------------------------------------------------------
# Metrics


def median(values: Iterable[float]) -> Optional[float]:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def metrics_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    timings = report.get("timings", {})
    generation = report.get("generation", {})
    prompt = report.get("prompt", {})
    rounds = report.get("decode_rounds", {}).get("samples", [])
    speculative = report.get("speculative", {})
    stream = report.get("stream", {})
    device = (report.get("devices") or [{}])[0]

    round_seconds = [float(sample["seconds"]) for sample in rounds]
    round_tokens = [float(sample["tokens"]) for sample in rounds]

    metrics: dict[str, Any] = {
        "prompt_tokens": prompt.get("tokens"),
        "generated_tokens": generation.get("tokens"),
        "prefill_seconds": timings.get("prefill_seconds"),
        "prefill_tok_s": timings.get("prefill_tokens_per_second"),
        "decode_seconds": timings.get("decode_seconds"),
        "decode_tok_s": timings.get("decode_tokens_per_second"),
        "total_seconds": timings.get("total_seconds"),
        "load_seconds": report.get("target", {}).get("load_seconds"),
        "kv_capacity": report.get("target", {}).get("kv_capacity"),
        "kv_page_groups": report.get("target", {}).get("kv_capacity_page_groups"),
        "device_name": device.get("name"),
        "rounds": report.get("decode_rounds", {}).get("count"),
        "rounds_sampled": len(rounds),
        "round_ms_mean": 1e3 * statistics.fmean(round_seconds) if round_seconds else None,
        "round_ms_median": 1e3 * median(round_seconds) if round_seconds else None,
        "round_ms_p90": 1e3 * percentile(round_seconds, 0.9) if round_seconds else None,
        "tokens_per_round": statistics.fmean(round_tokens) if round_tokens else None,
        "acceptance_length": speculative.get("acceptance_length"),
        "acceptance_rate": speculative.get("acceptance_rate"),
        "drafted_tokens": speculative.get("drafted_tokens"),
        "stream_series_len": len(stream.get("rounds_series", [])),
    }

    if stream.get("enabled"):
        count = stream.get("rounds") or 0
        upload = stream.get("upload_bytes") or 0
        download = stream.get("download_bytes") or 0
        prompt = stream.get("prefill", {})
        prompt_upload = prompt.get("upload_bytes") or 0
        prompt_download = prompt.get("download_bytes") or 0
        busy = stream.get("copy_busy_seconds") or 0.0
        exposed = stream.get("exposed_seconds") or 0.0
        series = stream.get("rounds_series", [])
        metrics.update(
            {
                "stream_host_mib": (stream.get("host_bytes") or 0) / 2**20,
                "stream_device_mib": (stream.get("device_bytes") or 0) / 2**20,
                "stream_window_mib": (stream.get("window_bytes") or 0) / 2**20,
                "stream_ring_mib": (stream.get("ring_bytes") or 0) / 2**20,
                "stream_window_pages": stream.get("resident_pages"),
                "stream_page_bytes": stream.get("page_bytes"),
                "stream_upload_mib": upload / 2**20,
                "stream_download_mib": download / 2**20,
                "stream_decode_upload_mib": (upload - prompt_upload) / 2**20,
                "stream_decode_download_mib": (download - prompt_download) / 2**20,
                "stream_copy_busy_s": busy,
                "stream_exposed_s": exposed,
                "stream_prefill_upload_mib": prompt_upload / 2**20,
                "stream_prefill_download_mib": prompt_download / 2**20,
                "stream_series_upload_mib": sum(int(row["upload_bytes"]) for row in series) / 2**20,
                "stream_timing_lost": stream.get("timing", {}).get("lost"),
            }
        )
        if count:
            # Per-round figures are the decode loop's, never the grand total divided by its rounds:
            # the prompt's own traffic has no round and would otherwise inflate every one of them.
            metrics.update(
                {
                    "stream_upload_mib_per_round": (upload - prompt_upload) / 2**20 / count,
                    "stream_download_mib_per_round": (download - prompt_download) / 2**20 / count,
                    "stream_copy_ms_per_round": 1e3 * busy / count,
                    "stream_exposed_ms_per_round": 1e3 * exposed / count,
                }
            )
        if busy > 0.0:
            metrics["stream_link_gib_s"] = (upload - prompt_upload) / busy / 2**30
    return metrics


def median_metrics(reps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {key for rep in reps for key, value in rep.items() if isinstance(value, (int, float))}
    )
    out: dict[str, Any] = {}
    for key in keys:
        values = [float(rep[key]) for rep in reps if isinstance(rep.get(key), (int, float))]
        out[key] = median(values) if values else None
    for key in ("device_name",):
        values = [rep[key] for rep in reps if rep.get(key)]
        if values:
            out[key] = values[0]
    return out


# ---------------------------------------------------------------------------------------------
# A campaign: one invocation of the tool, every run it made


def binary_facts(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "version": command_output([str(path), "--version"]) or "",
    }


def model_facts(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": iso(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
        "sha256": sha256_file(path),
    }


@dataclass
class Campaign:
    identifier: str
    directory: Path
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def open(results: Path, suite: str, tag: str) -> "Campaign":
        state = git_state()
        stamp = timestamp(utc_now())
        identifier = f"{stamp}-{suite}"
        if tag:
            identifier += f"-{tag}"
        identifier += f"-{state['short_head'] or 'nogit'}"
        directory = results / "runs" / identifier
        suffix = 1
        while directory.exists():
            suffix += 1
            directory = results / "runs" / f"{identifier}-{suffix}"
        directory.mkdir(parents=True)
        campaign = Campaign(identifier=directory.name, directory=directory)
        campaign.data = {
            "schema": 1,
            "kind": "kv-stream-campaign",
            "id": campaign.identifier,
            "suite": suite,
            "tag": tag,
            "started_utc": iso(utc_now()),
            "git": state,
            "host": host_state(),
            "gpu_before": gpu_state(),
            "model": {},
            "binaries": {},
            "cases": [],
            "verdict": "running",
        }
        return campaign

    def relative(self) -> str:
        return str(self.directory.relative_to(REPO_ROOT))

    def save(self) -> None:
        (self.directory / "campaign.json").write_text(json.dumps(self.data, indent=2) + "\n")

    def close(self, verdict: str) -> None:
        self.data["finished_utc"] = iso(utc_now())
        self.data["verdict"] = verdict
        self.save()


def case_entry(case: Case, variant: str, program: str) -> dict[str, Any]:
    return {
        "case": case.name,
        "variant": variant,
        "program": program,
        "purpose": case.purpose,
        "target_tokens": case.target_tokens,
        "capacity": case.capacity,
        "max_new": case.max_new,
        "reps": [],
        "checks": {},
    }


def compare_ids(golden: Sequence[int], actual: Sequence[int]) -> dict[str, Any]:
    if not golden:
        return {"checked": False, "reason": "no golden ids recorded"}
    if list(golden) == list(actual):
        return {"checked": True, "match": True, "tokens": len(actual)}
    position = next(
        (index for index, (left, right) in enumerate(zip(golden, actual)) if left != right),
        min(len(golden), len(actual)),
    )
    return {
        "checked": True,
        "match": False,
        "tokens": len(actual),
        "first_difference": position,
        "golden_id": golden[position] if position < len(golden) else None,
        "actual_id": actual[position] if position < len(actual) else None,
    }


def logprobs_digest(report: Mapping[str, Any]) -> str:
    records = report.get("generation", {}).get("logprobs")
    if not records:
        return ""
    return sha256_text(json.dumps(records, sort_keys=True, separators=(",", ":")))


def record_rep(entry: dict[str, Any], result: RepResult, variant: str, case: Case) -> None:
    metrics = metrics_from_report(result.report) if result.report else {}
    rep = {
        "argv": result.argv,
        "returncode": result.returncode,
        "wall_seconds": result.seconds,
        "metrics": metrics,
        "summary": result.summary,
        "ids": result.ids,
        "ids_sha256": sha256_text(" ".join(str(token) for token in result.ids)),
        "ids_count": len(result.ids),
        "logprobs_sha256": logprobs_digest(result.report) if result.report else "",
        "stderr_tail": result.stderr_tail,
    }
    if result.report:
        checks = entry.setdefault("checks", {})
        generation = result.report.get("generation", {})
        round_series = result.report.get("decode_rounds", {}).get("samples", [])
        stream_series = result.report.get("stream", {}).get("rounds_series", [])
        checks["stream_series_aligned"] = not stream_series or len(stream_series) == len(
            round_series
        )
        if case.require_full_generation:
            checks["generated_all_requested"] = generation.get("tokens") == case.max_new
        stream = result.report.get("stream", {})
        if stream.get("enabled") and stream_series:
            decoded = (stream.get("upload_bytes") or 0) - (
                stream.get("prefill", {}).get("upload_bytes") or 0
            )
            series = sum(int(row["upload_bytes"]) for row in stream_series)
            # The per-round series must account for exactly the decode traffic: otherwise a
            # per-round figure would be hiding a share of the request's bytes.
            checks["stream_series_matches_total"] = abs(series - decoded) <= max(decoded * 0.01, 1)
        if variant == "streamed":
            window_tokens = (stream.get("resident_pages") or 0) * 64
            prompt_tokens = result.report.get("prompt", {}).get("tokens", 0)
            checks["stream_window_tokens"] = window_tokens
            if case.expect_streaming:
                checks["stream_engaged"] = bool(stream.get("enabled")) and (
                    prompt_tokens > window_tokens
                )
            else:
                # The pool holds the whole context: the correct answer is that no page stands in the
                # ring and a decode round copies nothing at all.
                decode_bytes = (stream.get("upload_bytes") or 0) - (
                    (stream.get("prefill") or {}).get("upload_bytes") or 0
                )
                checks["stream_idle"] = bool(stream.get("enabled")) and (
                    prompt_tokens <= window_tokens and decode_bytes == 0
                )
        summary_decode = result.summary.get("decode speed")
        reported = result.report.get("timings", {}).get("decode_tokens_per_second")
        if summary_decode and reported:
            parsed = parse_rate(summary_decode)
            if parsed:
                checks["report_matches_summary"] = (
                    abs(parsed - reported) / max(parsed, 1e-9) < 0.02
                )
        printed = parse_ids(result.directory.joinpath("stderr.log").read_text(errors="replace"))
        report_ids = [int(token) for token in generation.get("token_ids", [])]
        if printed and report_ids:
            checks["printed_ids_match_report"] = printed == report_ids
    entry["reps"].append(rep)


def entry_verdict(entry: dict[str, Any], golden: Sequence[int]) -> str:
    if any(rep["returncode"] != 0 for rep in entry["reps"]):
        return "FAIL"
    if golden:
        checked = entry["checks"].get("ids")
        if not checked or not checked.get("match"):
            return "FAIL"
    for name, value in entry["checks"].items():
        if isinstance(value, bool) and not value:
            return "FAIL"
    return "PASS"


def finish_entry(entry: dict[str, Any], golden: Sequence[int]) -> None:
    entry["metrics"] = median_metrics([rep["metrics"] for rep in entry["reps"] if rep["metrics"]])
    last = entry["reps"][-1] if entry["reps"] else {}
    if golden:
        entry["checks"]["ids"] = compare_ids(golden, last.get("ids", []))
    entry["verdict"] = entry_verdict(entry, golden)


def print_entry(entry: dict[str, Any]) -> None:
    metrics = entry.get("metrics", {})
    parts = []
    for key in (
        "prompt_tokens",
        "decode_tok_s",
        "round_ms_median",
        "acceptance_length",
        "stream_upload_mib_per_round",
        "stream_exposed_ms_per_round",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.1f}" if isinstance(value, float) else f"{key}={value}")
    print(f"  {entry['case']}/{entry['variant']}: " + ", ".join(parts) + f" -> {entry['verdict']}")


def golden_paths(results: Path, case: Case) -> tuple[Path, Path]:
    return results / "golden" / f"{case.name}.ids", results / "golden" / f"{case.name}.json"


def read_golden(results: Path, case: Case) -> list[int]:
    ids_path, _ = golden_paths(results, case)
    if not ids_path.is_file():
        return []
    text = ids_path.read_text().strip()
    return [int(token) for token in text.split()] if text else []


def read_golden_meta(results: Path, case: Case) -> dict[str, Any]:
    _, meta_path = golden_paths(results, case)
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def record_golden(results: Path, case: Case, entry: dict[str, Any], rep: Mapping[str, Any]) -> None:
    ids_path, meta_path = golden_paths(results, case)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(rep["ids"])
    ids_path.write_text(" ".join(str(token) for token in ids) + "\n")
    meta_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "kv-stream-golden",
                "case": case.name,
                "oracle": case.oracle,
                "program": entry["program"],
                "prompt_sentences": entry["sentences"],
                "prompt_sha256": entry["prompt_sha256"],
                "prompt_tokens": rep.get("metrics", {}).get("prompt_tokens")
                or rep["summary"].get("prompt tokens"),
                "capacity": case.capacity,
                "max_new": case.max_new,
                "ids_sha256": rep["ids_sha256"],
                "logprobs_sha256": rep.get("logprobs_sha256", ""),
                "recorded_utc": iso(utc_now()),
                "argv": rep["argv"],
                "summary": rep["summary"],
            },
            indent=2,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------------------------
# Running cases


def require_calibration(results: Path, model: Path) -> Calibration:
    calibration = load_calibration(results / "calibration.json")
    if calibration is None or calibration.model_sha256 != sha256_file(model):
        raise SystemExit(
            "error: no prompt calibration for this model; run `run_kv_stream.py calibrate` first"
        )
    return calibration


def sentences_for(case: Case, calibration: Calibration) -> int:
    sentences = calibration.sentences_for(case.target_tokens)
    estimated = calibration.tokens_for(sentences)
    if abs(estimated - case.target_tokens) / case.target_tokens > 0.01:
        print(
            f"warning: {case.name} wants {case.target_tokens} tokens but the recipe gives "
            f"{estimated:.0f}; recalibrate if this is unexpected",
            file=sys.stderr,
        )
    return sentences


def run_case(
    args: argparse.Namespace,
    campaign: Campaign,
    case: Case,
    sentences: int,
    reps: int,
    program: str,
    binary: Path,
) -> bool:
    prompt_path = write_prompt(campaign.directory / case.name, sentences)
    golden = read_golden(args.results, case)
    golden_meta = read_golden_meta(args.results, case)
    record = (
        args.record_golden
        and (not golden or args.force)
        and (
            (case.oracle == "original" and program == "oracle")
            or (case.oracle == "self" and program == "ours")
        )
    )
    ok = True
    entries: list[dict[str, Any]] = []
    for variant in case.variants:
        entry = case_entry(case, variant, program)
        entry["sentences"] = sentences
        entry["prompt_sha256"] = sha256_text(prompt_text(sentences))
        if golden_meta and not record:
            # A golden recorded for a different prompt is not evidence about this one: the recipe
            # changed and the ids must be recorded again instead of matching an older question.
            # While recording, this run *is* the source, so there is nothing older to match.
            entry["checks"]["golden_prompt_matches"] = (
                golden_meta.get("prompt_sha256") == entry["prompt_sha256"]
            )
        entries.append(entry)
    # Interleaved: rep 1 of every variant, then rep 2, and so on. A report compares variants, so they
    # have to be measured under the same conditions - a variant that runs after the other one's whole
    # block measures a different thermal, clock and page-cache state, and the difference lands in the
    # comparison (CONTRIBUTING.md, "Comparable measurements"). Each rep is a fresh process, so there is
    # no prefix cache between reps to reuse; the prompt is identical by design, because both variants
    # are compared against the same prompt ids and a per-rep nonce would invalidate the golden ids.
    for rep_index in range(1, reps + 1):
        for entry, variant in zip(entries, case.variants):
            print(f"{case.name}/{variant}: rep {rep_index}/{reps} ...", flush=True)
            result = run_once(
                binary,
                args.model,
                prompt_path,
                case,
                variant,
                args.device,
                campaign.directory / case.name / variant / f"rep-{rep_index:02d}",
                args.timeout,
                with_report=program == "ours",
                with_logprobs=program == "ours" and args.logprobs,
                dtype=(case.kv_dtype or args.kv_dtype),
            )
            record_rep(entry, result, variant, case)
            if result.returncode != 0:
                print(f"  exit {result.returncode}: {result.stderr_tail}")
    for entry in entries:
        finish_entry(entry, [] if record else golden)
        campaign.data["cases"].append(entry)
        campaign.save()
        print_entry(entry)
        if entry["verdict"] != "PASS":
            ok = False
        if record and entry["reps"] and entry["reps"][-1]["ids"]:
            record_golden(args.results, case, entry, entry["reps"][-1])
            print(f"  recorded golden ids for {case.name}")

    # SPEC completion condition 2: the streamed layout's greedy ids and log probabilities must be
    # the same as the resident layout's, exactly - not within a tolerance. Both are in this build,
    # so the comparison is between two runs of the same binary with one flag apart.
    if len(entries) > 1 and program == "ours" and args.logprobs:
        digests = {entry["variant"]: entry["reps"][-1].get("logprobs_sha256", "") for entry in entries}
        if all(digests.values()):
            matched = len(set(digests.values())) == 1
            for entry in entries:
                entry["checks"]["logprobs_match_across_layouts"] = matched
                if not matched:
                    entry["verdict"] = "FAIL"
                    ok = False
                campaign.save()
    return ok


# ---------------------------------------------------------------------------------------------
# Subcommands


def command_calibrate(args: argparse.Namespace) -> int:
    model_sha = sha256_file(args.model)
    store = args.results / "calibration.json"
    calibration = load_calibration(store)
    if calibration and calibration.model_sha256 == model_sha and not args.force:
        print(
            "calibration: already measured for this model "
            f"({calibration.tokens_per_sentence:.4f} tokens/sentence)"
        )
        return 0
    require_idle_gpu(args.allow_busy)
    work = args.results / "runs" / f"{timestamp(utc_now())}-calibrate-{git_state()['short_head']}"
    work.mkdir(parents=True, exist_ok=False)
    points: list[dict[str, int]] = []
    argv_log: list[str] = []
    for sentences in CALIBRATION_POINTS:
        capacity = 8192 if sentences <= 100 else 16384
        directory = work / f"sentences-{sentences}"
        prompt_path = write_prompt(directory, sentences)
        argv = [
            str(args.binary),
            str(args.model),
            "--messages",
            str(prompt_path),
            "--max-context",
            str(capacity),
            "--kv-capacity",
            "auto",
            "--kv-dtype",
            "rk4v4",
            "--gdn-state-fp16",
            "--prefill-chunk",
            "256",
            "--greedy",
            "--seed",
            "42",
            "--invariant-attention",
            "--spec",
            "none",
            "--max-new",
            "1",
            "--device",
            str(args.device),
            "--report-json",
            str(directory / "report.json"),
        ]
        print(f"calibrate: {sentences} sentences ...", flush=True)
        started = time.monotonic()
        done = subprocess.run(argv, capture_output=True, text=True)
        seconds = time.monotonic() - started
        (directory / "stdout.log").write_text(done.stdout)
        (directory / "stderr.log").write_text(done.stderr)
        if done.returncode != 0:
            print(done.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"error: calibration run failed (exit {done.returncode})")
        report = json.loads((directory / "report.json").read_text())
        tokens = int(report["prompt"]["tokens"])
        points.append({"sentences": sentences, "tokens": tokens})
        argv_log = argv
        print(f"  {tokens} prompt tokens in {seconds:.1f} s", flush=True)

    first, second = points[0], points[1]
    per_sentence = (second["tokens"] - first["tokens"]) / (second["sentences"] - first["sentences"])
    tail = first["tokens"] - per_sentence * first["sentences"]
    save_calibration(
        store,
        Calibration(
            model_sha256=model_sha,
            tokens_per_sentence=per_sentence,
            tail_tokens=tail,
            points=points,
            measured_utc=iso(utc_now()),
            argv=argv_log,
        ),
    )
    print(
        f"calibration: {per_sentence:.4f} tokens/sentence + {tail:.1f} tail tokens -> "
        f"{store.relative_to(REPO_ROOT)}"
    )
    return 0


def selected_cases(names: Sequence[str]) -> list[Case]:
    cases = []
    for name in names:
        if name not in CASES:
            raise SystemExit(f"error: unknown case {name!r}; known: {', '.join(CASES)}")
        cases.append(CASES[name])
    return cases


def command_golden(args: argparse.Namespace) -> int:
    calibration = require_calibration(args.results, args.model)
    cases = selected_cases(args.case)
    require_idle_gpu(args.allow_busy)
    campaign = Campaign.open(args.results, "golden", args.tag)
    campaign.data["binaries"]["oracle"] = binary_facts(args.oracle)
    campaign.data["binaries"]["ours"] = binary_facts(args.binary)
    campaign.data["model"] = model_facts(args.model)
    ok = True
    try:
        for case in cases:
            sentences = sentences_for(case, calibration)
            if case.oracle == "original":
                # The original has no streamed layout: the ids it answers with are the golden for
                # every variant, so it is run once, all resident.
                print(f"golden: {case.name} from the original binary ({sentences} sentences)")
                ok = (
                    run_case(
                        args,
                        campaign,
                        replace(case, variants=("resident",)),
                        sentences,
                        1,
                        "oracle",
                        args.oracle,
                    )
                    and ok
                )
            else:
                print(f"golden: {case.name} from our build (regression baseline)")
                ok = run_case(args, campaign, case, sentences, 1, "ours", args.binary) and ok
    finally:
        campaign.data["gpu_after"] = gpu_state()
        campaign.close("PASS" if ok else "FAIL")
    print(f"campaign: {campaign.relative()} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def command_check(args: argparse.Namespace) -> int:
    calibration = require_calibration(args.results, args.model)
    cases = selected_cases(args.case)
    absent = [case.name for case in cases if not read_golden(args.results, case)]
    if absent:
        raise SystemExit(
            "error: no golden ids for " + ", ".join(absent) + "; run `golden` first"
        )
    require_idle_gpu(args.allow_busy)
    campaign = Campaign.open(args.results, "equiv", args.tag)
    campaign.data["binaries"]["ours"] = binary_facts(args.binary)
    campaign.data["model"] = model_facts(args.model)
    ok = True
    try:
        for case in cases:
            sentences = sentences_for(case, calibration)
            ok = run_case(args, campaign, case, sentences, 1, "ours", args.binary) and ok
    finally:
        campaign.data["gpu_after"] = gpu_state()
        campaign.close("PASS" if ok else "FAIL")
    print(f"campaign: {campaign.relative()} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def command_bench(args: argparse.Namespace) -> int:
    calibration = require_calibration(args.results, args.model)
    cases = selected_cases(args.case or list(PERF_SUITE))
    require_idle_gpu(args.allow_busy)
    campaign = Campaign.open(args.results, "bench", args.tag)
    campaign.data["binaries"]["ours"] = binary_facts(args.binary)
    campaign.data["model"] = model_facts(args.model)
    ok = True
    try:
        for case in cases:
            sentences = sentences_for(case, calibration)
            ok = run_case(args, campaign, case, sentences, args.reps, "ours", args.binary) and ok
    finally:
        campaign.data["gpu_after"] = gpu_state()
        campaign.close("PASS" if ok else "FAIL")
    print(f"campaign: {campaign.relative()} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def command_footprint(args: argparse.Namespace) -> int:
    """Measure what each speculative backend costs on this card, and what it takes from the window.

    The same artifact runs once per backend with everything else pinned, so the difference in
    weights and in the context pool is that backend's footprint; the pool difference is also the
    window a streamed request loses, and therefore the extra traffic it copies every round.
    """
    require_idle_gpu(args.allow_busy)
    calibration = require_calibration(args.results, args.model)
    work = args.results / "runs" / f"{timestamp(utc_now())}-footprint-{git_state()['short_head']}"
    work.mkdir(parents=True, exist_ok=False)
    prompt_tokens = min(args.prompt_tokens, args.context - args.max_new - 1)
    sentences = calibration.sentences_for(prompt_tokens)
    prompt_path = write_prompt(work / "prompt", sentences)
    model_sha = sha256_file(args.model)
    ledger_path = args.results / "footprint.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
    configs = [tuple(row.split(":", 1)) for row in args.config]
    ok = True
    for spec, draft in configs:
        draft_tokens = int(draft)
        extra = ["--spec", spec]
        if draft_tokens:
            extra += ["--draft-tokens", str(draft_tokens)]
        # The drafters are compared on their own: a copied n-gram would hide their acceptance.
        extra += ["--ngram-draft-tokens", "0"]
        case = Case(
            name=f"footprint-{spec}-{draft_tokens}",
            purpose="Read one backend's weights and context pool on this card.",
            kind="calibration",
            target_tokens=prompt_tokens,
            capacity=args.context,
            max_new=args.max_new,
            variants=("streamed",),
        )
        print(f"footprint: {spec}/{draft_tokens} at {args.context} ...", flush=True)
        result = run_once(
            args.binary,
            args.model,
            prompt_path,
            case,
            "streamed",
            args.device,
            work / f"{spec}-{draft_tokens}",
            args.timeout,
            with_report=True,
            extra=extra,
            dtype=args.kv_dtype,
        )
        memory = result.report.get("memory", {}) if result.report else {}
        stream = result.report.get("stream", {}) if result.report else {}
        speculative = result.report.get("speculative", {}) if result.report else {}
        timings = result.report.get("timings", {}) if result.report else {}
        stderr = (result.directory / "stderr.log").read_text(errors="replace")
        weights = startup_weight_bytes(stderr)
        pages = page_group_bytes(result.report) if result.report else 0
        pool_bytes = memory.get("sequence", {}).get("capacity_bytes", 0)
        entry = {
            "measured_utc": iso(utc_now()),
            "model_sha256": model_sha,
            "git_head": git_state()["short_head"],
            "context": args.context,
            "spec": spec,
            "draft_tokens": draft_tokens,
            "returncode": result.returncode,
            "started": result.returncode == 0,
            "error": first_error(stderr) if result.returncode != 0 else "",
            "weights_bytes": weights,
            "pool_bytes": pool_bytes,
            "pool_pages": pool_bytes // pages if pages else 0,
            "minimum_reservation_bytes": memory.get("minimum_runtime_reservation_bytes", 0),
            "window_pages": stream.get("resident_pages"),
            "window_tokens": (stream.get("resident_pages") or 0) * 64,
            "stream_host_bytes": stream.get("host_bytes"),
            "stream_device_bytes": stream.get("device_bytes"),
            "acceptance_length": speculative.get("acceptance_length"),
            "decode_tokens_per_second": timings.get("decode_tokens_per_second"),
            "seconds": result.seconds,
            "argv": result.argv,
            "campaign": str(work.relative_to(REPO_ROOT)),
        }
        ledger = [
            row
            for row in ledger
            if not (
                row.get("model_sha256") == model_sha
                and row.get("context") == args.context
                and row.get("spec") == spec
                and row.get("draft_tokens") == draft_tokens
            )
        ]
        ledger.append(entry)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        if result.returncode != 0:
            ok = False
            print(f"  {spec}/{draft_tokens}: failed: {entry['error']}")
        else:
            print(
                f"  {spec}/{draft_tokens}: weights {gib(entry['weights_bytes'])}, "
                f"pool {entry['pool_pages']} pages, window {entry['window_pages']} pages, "
                f"accept {entry['acceptance_length']}, decode {entry['decode_tokens_per_second']}"
            )
    print(f"footprint: {ledger_path.relative_to(REPO_ROOT)}")
    return 0 if ok else 1


def command_window(args: argparse.Namespace) -> int:
    """Measure what window a pool leaves, so a streamed case can be given a pool it streams over.

    The window is a startup fact (the pool minus the ring, divided over the layers), so a short
    prompt is enough to read it. The scan is kept next to the calibration because a later case
    picks its headroom from it instead of guessing.
    """
    require_idle_gpu(args.allow_busy)
    work = args.results / "runs" / f"{timestamp(utc_now())}-window-{git_state()['short_head']}"
    work.mkdir(parents=True, exist_ok=False)
    prompt_path = write_prompt(work / "prompt", 64)
    scan_path = args.results / "window-scan.json"
    scan = json.loads(scan_path.read_text()) if scan_path.is_file() else []
    model_sha = sha256_file(args.model)
    ok = True
    for headroom in args.headroom_mib:
        case = Case(
            name=f"window-{headroom}",
            purpose="Read the stream window a pool leaves.",
            kind="calibration",
            target_tokens=0,
            capacity=args.capacity,
            max_new=1,
            variants=("streamed",),
            streamed_flags=("--kv-headroom-mib", str(headroom)),
        )
        print(f"window: capacity {args.capacity}, headroom {headroom} MiB ...", flush=True)
        result = run_once(
            args.binary,
            args.model,
            prompt_path,
            case,
            "streamed",
            args.device,
            work / f"headroom-{headroom}",
            args.timeout,
            with_report=True,
            dtype=args.kv_dtype,
        )
        stream = result.report.get("stream", {}) if result.report else {}
        entry = {
            "measured_utc": iso(utc_now()),
            "model_sha256": model_sha,
            "git_head": git_state()["short_head"],
            "capacity": args.capacity,
            "headroom_mib": headroom,
            "returncode": result.returncode,
            "enabled": bool(stream.get("enabled")),
            "resident_pages": stream.get("resident_pages"),
            "pages_per_layer": stream.get("pages_per_layer"),
            "window_tokens": (stream.get("resident_pages") or 0) * 64,
            "host_bytes": stream.get("host_bytes"),
            "device_bytes": stream.get("device_bytes"),
            "page_groups": result.report.get("target", {}).get("kv_capacity_page_groups"),
            "seconds": result.seconds,
            "argv": result.argv,
            "campaign": str(work.relative_to(REPO_ROOT)),
        }
        scan = [
            row
            for row in scan
            if not (
                row.get("model_sha256") == model_sha
                and row.get("capacity") == args.capacity
                and row.get("headroom_mib") == headroom
            )
        ]
        scan.append(entry)
        scan_path.write_text(json.dumps(scan, indent=2) + "\n")
        if result.returncode != 0:
            ok = False
            print(f"  failed: {result.stderr_tail}")
        else:
            print(
                f"  window {entry['window_tokens']} tokens ({entry['resident_pages']} pages x 64), "
                f"{entry['pages_per_layer']} pages/layer, pool {entry['device_bytes']} B"
            )
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# The index and the comparison


def load_campaigns(results: Path) -> list[dict[str, Any]]:
    campaigns: list[dict[str, Any]] = []
    for path in sorted((results / "runs").glob("*/campaign.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["_path"] = str(path.parent)
        campaigns.append(data)
    return campaigns


INDEX_HEADER = (
    "| run | suite | case | variant | decode tok/s | accept | round ms | H2D MiB/round | "
    "exposed ms/round | host MiB | device MiB | verdict |"
)
INDEX_RULE = "|" + "---|" * 12


def metric_cell(metrics: Mapping[str, Any], key: str, digits: int = 2) -> str:
    value = metrics.get(key) if metrics else None
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def index_rows(results: Path, campaigns: Sequence[Mapping[str, Any]]) -> list[str]:
    rows: list[str] = []
    for campaign in sorted(campaigns, key=lambda item: item.get("started_utc", ""), reverse=True):
        link = Path(campaign["_path"]).relative_to(results) / "campaign.json"
        for entry in campaign.get("cases", []):
            metrics = entry.get("metrics") or median_metrics(
                [rep.get("metrics", {}) for rep in entry.get("reps", []) if rep.get("metrics")]
            )
            rows.append(
                "| [{}]({}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    campaign.get("started_utc", ""),
                    link.as_posix(),
                    campaign.get("suite", ""),
                    entry.get("case", ""),
                    entry.get("variant", ""),
                    metric_cell(metrics, "decode_tok_s"),
                    metric_cell(metrics, "acceptance_length"),
                    metric_cell(metrics, "round_ms_median"),
                    metric_cell(metrics, "stream_upload_mib_per_round", 0),
                    metric_cell(metrics, "stream_exposed_ms_per_round"),
                    metric_cell(metrics, "stream_host_mib", 0),
                    metric_cell(metrics, "stream_device_mib", 0),
                    entry.get("verdict", ""),
                )
            )
    return rows


def command_report(args: argparse.Namespace) -> int:
    campaigns = load_campaigns(args.results)
    rows = index_rows(args.results, campaigns)
    index = args.results / "INDEX.md"
    index.write_text(
        "\n".join(
            [
                "# KV streaming runs",
                "",
                "Generated by `python tools/bench/run_kv_stream.py report`; do not edit by hand.",
                "One row per case and variant, a campaign's medians over its repetitions. The",
                "protocol, the cases and the fields' meaning are in [README.md](README.md).",
                "",
                INDEX_HEADER,
                INDEX_RULE,
                *rows,
                "",
            ]
        )
    )
    print(f"index: {len(rows)} rows -> {index.relative_to(REPO_ROOT)}")
    return 0


COMPARE_KEYS = (
    "decode_tok_s",
    "round_ms_median",
    "round_ms_p90",
    "acceptance_length",
    "acceptance_rate",
    "tokens_per_round",
    "prefill_tok_s",
    "stream_upload_mib_per_round",
    "stream_download_mib_per_round",
    "stream_copy_ms_per_round",
    "stream_exposed_ms_per_round",
    "stream_link_gib_s",
    "stream_host_mib",
    "stream_device_mib",
    "stream_window_mib",
    "stream_ring_mib",
)


def campaign_metrics(campaign: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in campaign.get("cases", []):
        metrics = entry.get("metrics") or median_metrics(
            [rep.get("metrics", {}) for rep in entry.get("reps", []) if rep.get("metrics")]
        )
        out[(entry.get("case", ""), entry.get("variant", ""))] = metrics or {}
    return out


def command_compare(args: argparse.Namespace) -> int:
    left = load_campaign(Path(args.left))
    right = load_campaign(Path(args.right))
    print(
        f"A: {left.get('id')}  {left.get('started_utc')}  "
        f"head {left.get('git', {}).get('short_head')}"
    )
    print(
        f"B: {right.get('id')}  {right.get('started_utc')}  "
        f"head {right.get('git', {}).get('short_head')}"
    )
    for name, campaign in (("A", left), ("B", right)):
        devices = campaign.get("gpu_before", {}).get("devices", [])
        if devices:
            device = devices[0]
            print(
                f"{name} gpu: {device.get('name')} driver {device.get('driver_version')} "
                f"pcie {device.get('pcie_gen_current')}x{device.get('pcie_width_current')} "
                f"sm {device.get('clock_sm_mhz')} MHz"
            )
    left_metrics = campaign_metrics(left)
    right_metrics = campaign_metrics(right)
    shared = [key for key in left_metrics if key in right_metrics]
    if not shared:
        print("no case and variant appear in both campaigns", file=sys.stderr)
        return 2
    for case, variant in shared:
        print(f"\n## {case} / {variant}")
        print("| metric | A | B | delta |")
        print("|---|---:|---:|---:|")
        a = left_metrics[(case, variant)]
        b = right_metrics[(case, variant)]
        for key in COMPARE_KEYS:
            if key not in a or key not in b or a[key] is None or b[key] is None:
                continue
            before, after = float(a[key]), float(b[key])
            delta = (after - before) / before * 100.0 if before else float("nan")
            print(f"| {key} | {before:.3f} | {after:.3f} | {delta:+.1f}% |")
    return 0


def load_campaign(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "campaign.json"
    if not path.is_file():
        raise SystemExit(f"error: no campaign at {path}")
    data = json.loads(path.read_text())
    data["_path"] = str(path.parent)
    return data


# ---------------------------------------------------------------------------------------------
# Entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record, replay and compare KV-streaming runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--allow-busy", action="store_true", help="run even if the card is in use")
    parser.add_argument(
        "--kv-dtype",
        default="",
        help=(
            "override the pinned --kv-dtype rk4v4 (e.g. rk8v4, the delivery format); the record "
            "keeps the value the run actually read"
        ),
    )
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help=(
            "skip the log-probability comparison of `check`/`golden` (faster; the ids are still "
            "compared). SPEC condition 2 wants the streamed and resident layouts' log "
            "probabilities identical, so it is on by default there and off for `bench`."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="measure the prompt recipe's tokens per sentence")
    calibrate.add_argument("--force", action="store_true")
    calibrate.set_defaults(func=command_calibrate, record_golden=False, force=False)

    golden = sub.add_parser("golden", help="record the oracle's ids (or a regression baseline)")
    golden.add_argument("--suite", choices=("equiv", "perf"), default="equiv")
    golden.add_argument("--case", nargs="*", default=None)
    golden.add_argument("--tag", default="")
    golden.add_argument(
        "--force", action="store_true", help="record again even when golden ids already exist"
    )
    golden.set_defaults(func=command_golden, record_golden=True)

    check = sub.add_parser("check", help="run our build and compare it with the golden ids")
    check.add_argument("--suite", choices=("equiv",), default="equiv")
    check.add_argument("--case", nargs="*", default=list(EQUIVALENCE_SUITE))
    check.add_argument("--tag", default="")
    check.set_defaults(func=command_check, record_golden=False, force=False)

    bench = sub.add_parser("bench", help="record performance repetitions of a case")
    bench.add_argument("--case", nargs="*", default=None)
    bench.add_argument("--reps", type=int, default=1)
    bench.add_argument("--tag", default="")
    bench.set_defaults(func=command_bench, record_golden=False, force=False)

    report = sub.add_parser("report", help="regenerate INDEX.md from the records")
    report.set_defaults(func=command_report, record_golden=False, force=False)

    window = sub.add_parser("window", help="measure the stream window a capacity and headroom give")
    window.add_argument("--capacity", type=int, required=True)
    window.add_argument("--headroom-mib", type=int, nargs="+", required=True)
    window.set_defaults(func=command_window, record_golden=False, force=False)

    footprint = sub.add_parser(
        "footprint", help="measure each speculative backend's weights, context pool and window"
    )
    footprint.add_argument("--context", type=int, default=262144)
    footprint.add_argument("--prompt-tokens", type=int, default=8000)
    footprint.add_argument("--max-new", type=int, default=8)
    footprint.add_argument(
        "--config",
        nargs="+",
        default=["none:0", "mtp:3", "mtp:7", "dflash2:3", "dflash2:7"],
        help="spec:draft-tokens entries, run once each",
    )
    footprint.set_defaults(func=command_footprint, record_golden=False, force=False)

    compare = sub.add_parser("compare", help="print the metric deltas of two campaigns")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(func=command_compare, record_golden=False, force=False)

    args = parser.parse_args(argv)
    # Correctness commands compare the distribution as well as the ids; a timing run must not pay
    # for log probabilities it does not read.
    args.logprobs = args.command in ("check", "golden") and not args.no_logprobs
    if args.command == "golden" and args.case is None:
        args.case = list(EQUIVALENCE_SUITE if args.suite == "equiv" else PERF_SUITE)
    args.results = args.results.resolve()
    args.binary = args.binary.resolve()
    args.oracle = args.oracle.resolve()
    args.model = args.model.resolve()
    if args.command in ("golden", "check", "bench", "calibrate", "window", "footprint"):
        if not args.binary.is_file():
            raise SystemExit(f"error: no binary at {args.binary}")
        if not args.model.is_file():
            raise SystemExit(f"error: no model at {args.model}")
    if args.command == "golden" and not args.oracle.is_file():
        raise SystemExit(f"error: no oracle binary at {args.oracle}")
    if not args.results.is_dir():
        args.results.mkdir(parents=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
