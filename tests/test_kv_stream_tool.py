from __future__ import annotations

import math
import sys
import tempfile
import traceback
from pathlib import Path

from tools.bench import run_kv_stream
from tools.bench.run_kv_stream import (
    Case,
    compare_ids,
    index_rows,
    median_metrics,
    metrics_from_report,
    parse_ids,
    parse_rate,
    parse_summary,
    prompt_text,
)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=1e-12)


def sample_report() -> dict:
    """The shape of a streamed run's report, with numbers that make the derived metrics checkable."""
    return {
        "schema": 1,
        "kind": "ninfer-run-report",
        "target": {"load_seconds": 25.0, "kv_capacity": 184064, "kv_capacity_page_groups": 2876},
        "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 4080 SUPER"}],
        "prompt": {"tokens": 183000, "reused_tokens": 0},
        "generation": {"tokens": 512, "finish_reason": "output-limit", "token_ids": [1, 2, 3]},
        "timings": {
            "prefill_seconds": 100.0,
            "prefill_tokens_per_second": 1830.0,
            "decode_seconds": 10.0,
            "decode_tokens_per_second": 51.1,
            "total_seconds": 110.0,
        },
        "decode_rounds": {
            "count": 200,
            "sample_cap": 2048,
            "samples": [{"seconds": 0.05, "tokens": 2.5}] * 200,
        },
        "speculative": {"acceptance_length": 2.5, "acceptance_rate": 0.5, "drafted_tokens": 750},
        "stream": {
            "enabled": True,
            "resident_pages": 2835,
            "layers": 16,
            "ring_depth": 2,
            "page_bytes": 71680,
            "host_bytes": 4_700_000_000,
            "device_bytes": 2_100_000_000,
            "window_bytes": 3_250_000_000,
            "ring_bytes": 100_000_000,
            "rounds": 200,
            "upload_bytes": 2**30 + 200 * 2**20 * 100,
            "download_bytes": 2**30 + 200 * 2**20 * 10,
            "copy_busy_seconds": 4.0,
            "exposed_seconds": 2.0,
            "prefill": {"upload_bytes": 2**30, "download_bytes": 2**30},
            "timing": {"samples": 3200, "lost": 0},
            "rounds_series": [{"upload_bytes": 0, "download_bytes": 0}] * 200,
        },
    }


def test_prompt_text_repeats_a_fixed_recipe() -> None:
    one = prompt_text(1)
    three = prompt_text(3)
    assert one.endswith(run_kv_stream.PROMPT_TAIL)
    assert three == (run_kv_stream.PROMPT_SENTENCE + " ") * 3 + run_kv_stream.PROMPT_TAIL
    # The count is linear in the repetition count, which is what calibration relies on.
    assert prompt_text(2) == run_kv_stream.PROMPT_SENTENCE + " " + one


def test_summary_and_ids_are_read_back_from_the_cli_text() -> None:
    text = (
        "summary     prompt tokens             183000\n"
        "summary     decode speed              51.1 tok/s\n"
        "tokens      generated ids             1596 1144 4087\n"
    )
    summary = parse_summary(text)
    assert summary["prompt tokens"] == "183000"
    assert close(parse_rate(summary["decode speed"]), 51.1)
    assert close(parse_rate("1.56k tok/s") or 0.0, 1560.0)
    assert parse_ids(text) == [1596, 1144, 4087]


def test_metrics_carry_the_round_and_link_accounting() -> None:
    metrics = metrics_from_report(sample_report())
    assert metrics["prompt_tokens"] == 183000
    assert metrics["rounds_sampled"] == 200
    assert close(metrics["round_ms_median"], 50.0)
    assert close(metrics["tokens_per_round"], 2.5)
    # The prompt's own traffic has no round; per-round figures must be the decode loop's only.
    assert close(metrics["stream_prefill_upload_mib"], 1024.0)
    assert close(metrics["stream_upload_mib_per_round"], 100.0)
    assert close(metrics["stream_download_mib_per_round"], 10.0)
    assert close(metrics["stream_copy_ms_per_round"], 20.0)
    assert close(metrics["stream_exposed_ms_per_round"], 10.0)
    assert close(metrics["stream_link_gib_s"], 100.0 * 200 / 4.0 / 1024.0)
    assert metrics["stream_timing_lost"] == 0
    assert metrics["stream_series_len"] == 200
    assert metrics["device_name"] == "NVIDIA GeForce RTX 4080 SUPER"


def test_metrics_without_a_round_report_no_per_round_figure() -> None:
    report = sample_report()
    report["stream"]["rounds"] = 0
    report["stream"]["rounds_series"] = []
    metrics = metrics_from_report(report)
    assert "stream_upload_mib_per_round" not in metrics
    assert metrics["stream_decode_upload_mib"] > 0


def test_logprobability_digest_is_order_independent_and_content_sensitive() -> None:
    report = {"generation": {"logprobs": [{"id": 1, "logprob": -0.5, "top_ids": [1, 2]}]}}
    same = {"generation": {"logprobs": [{"top_ids": [1, 2], "logprob": -0.5, "id": 1}]}}
    other = {"generation": {"logprobs": [{"id": 1, "logprob": -0.50000006, "top_ids": [1, 2]}]}}
    empty = {"generation": {}}
    assert run_kv_stream.logprobs_digest(report) == run_kv_stream.logprobs_digest(same)
    assert run_kv_stream.logprobs_digest(report) != run_kv_stream.logprobs_digest(other)
    assert run_kv_stream.logprobs_digest(empty) == ""


def test_median_metrics_ignores_labels_and_takes_the_median() -> None:
    reps = [
        {"decode_tok_s": 50.0, "device_name": "a", "rounds": 10},
        {"decode_tok_s": 60.0, "device_name": "a", "rounds": 12},
        {"decode_tok_s": 70.0, "device_name": "a", "rounds": 14},
    ]
    median = median_metrics(reps)
    assert close(median["decode_tok_s"], 60.0)
    assert close(median["rounds"], 12.0)
    assert median["device_name"] == "a"


def test_id_comparison_names_the_first_difference() -> None:
    match = compare_ids([1, 2, 3], [1, 2, 3])
    assert match["checked"] and match["match"]
    mismatch = compare_ids([1, 2, 3], [1, 9, 3])
    assert mismatch["checked"] and not mismatch["match"]
    assert mismatch["first_difference"] == 1
    assert mismatch["golden_id"] == 2
    assert mismatch["actual_id"] == 9
    short = compare_ids([1, 2, 3], [1, 2])
    assert short["first_difference"] == 2
    assert short["actual_id"] is None


def test_index_links_the_campaign_relative_to_the_results_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        results = Path(temporary) / "kv-stream"
        campaign_dir = results / "runs" / "20260101T000000Z-equiv-abcd1234"
        campaign_dir.mkdir(parents=True)
        campaign = {
            "id": campaign_dir.name,
            "suite": "equiv",
            "started_utc": "2026-01-01T00:00:00Z",
            "cases": [
                {
                    "case": "equiv-183k",
                    "variant": "streamed",
                    "verdict": "PASS",
                    "metrics": {"decode_tok_s": 51.1, "acceptance_length": 2.5},
                }
            ],
            "_path": str(campaign_dir),
        }
        rows = index_rows(results, [campaign])
        assert len(rows) == 1
        assert "runs/20260101T000000Z-equiv-abcd1234/campaign.json" in rows[0]
        assert "| equiv | equiv-183k | streamed | 51.10 | 2.50 |" in rows[0]


def test_case_table_pins_the_two_layout_range() -> None:
    case = run_kv_stream.CASES["equiv-183k"]
    assert case.max_new == 512
    assert case.variants == ("resident", "streamed")
    # The streamed window at rk4v4 + MTP3 is 2 835 pages of 64 tokens; the target must be past it,
    # and the prompt plus its generation must fit the capacity the original can still start with.
    assert 2835 * 64 < case.target_tokens
    assert case.target_tokens + case.max_new <= case.capacity


def test_golden_files_are_keyed_by_case() -> None:
    case = Case(
        name="equiv-8k",
        purpose="unit",
        kind="equivalence",
        target_tokens=8000,
        capacity=8192,
        max_new=16,
        variants=("resident",),
    )
    with tempfile.TemporaryDirectory() as temporary:
        results = Path(temporary)
        ids_path, meta_path = run_kv_stream.golden_paths(results, case)
        assert ids_path == results / "golden" / "equiv-8k.ids"
        assert meta_path == results / "golden" / "equiv-8k.json"
        ids_path.parent.mkdir(parents=True)
        ids_path.write_text("1 2 3\n")
        assert run_kv_stream.read_golden(results, case) == [1, 2, 3]



def test_pinned_dtype_is_replaced_not_appended() -> None:
    """The delivery format is rk8v4 while every other case pins rk4v4: the override replaces the
    pinned value where the parser reads it last, so the recorded argv and the run agree."""
    argv = ["ninfer", "model", "--kv-dtype", "rk4v4", "--max-new", "64"]
    pinned = run_kv_stream.with_pinned(argv, "--kv-dtype", "rk8v4")
    assert pinned.count("--kv-dtype") == 1
    assert pinned[pinned.index("--kv-dtype") + 1] == "rk8v4"
    assert pinned[-2:] == ["--max-new", "64"]

    absent = run_kv_stream.with_pinned(["ninfer", "model"], "--kv-dtype", "rk8v4")
    assert absent == ["ninfer", "model", "--kv-dtype", "rk8v4"]

    case = Case(
        name="c",
        purpose="p",
        kind="perf",
        target_tokens=8,
        capacity=8192,
        max_new=8,
        variants=("streamed",),
    )
    default = run_kv_stream.request_argv(Path("b"), Path("m"), Path("p"), case, "streamed", 0)
    assert default[default.index("--kv-dtype") + 1] == "rk4v4"
    overridden = run_kv_stream.request_argv(
        Path("b"), Path("m"), Path("p"), case, "streamed", 0, (), "rk8v4"
    )
    assert overridden.count("--kv-dtype") == 1
    assert overridden[overridden.index("--kv-dtype") + 1] == "rk8v4"

    sys.exit(main())



def test_the_delivery_format_has_its_own_equivalence_case() -> None:
    """rk8v4 只在"全驻留装得下"的上下文上能与流式逐位比对，这个用例就是那个区间。"""
    case = run_kv_stream.CASES["equiv-100k-rk8v4"]
    assert case.kv_dtype == "rk8v4"
    assert case.variants == ("resident", "streamed")
    assert case.oracle == "original"
    assert case.max_new == 512
    # The resident arm needs a context the card can hold beside 11.6 GiB of weights: 16 layers of
    # 64-token pages of 104 448 B at 100K is 2.67 GiB, which is why this case stops at 100K.
    pages = -(-case.target_tokens // 64)
    kib = pages * 16 * 104_448
    assert kib < 3 * 1024**3
    assert case.target_tokens + case.max_new <= case.capacity


def test_a_case_without_a_dtype_follows_the_global_override() -> None:
    """没有自己的格式时，全局 --kv-dtype 仍然生效（rk4v4 走默认，交付格式靠用例钉住）。"""
    assert run_kv_stream.CASES["equiv-183k"].kv_dtype == ""


def main() -> int:
    """Run every test function without pytest, so the record can be checked where it was made."""
    failures = 0
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for name, function in tests:
        try:
            function()
        except Exception:  # noqa: BLE001 - the point is to report every failure
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
