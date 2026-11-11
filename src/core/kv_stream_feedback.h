#pragma once

#include <cstdint>

namespace ninfer {

// What a decode round's streaming actually did: how many copy runs missed their attention, and how
// much of the round the copy engine was busy (docs/maintainer/kv-streaming-design.md).
//
// Both are pure: the CUDA side records raw samples and this layer turns them into the ratios the
// report carries, refusing samples that cannot be true. Ported from the reference implementation
// (llama-kv-stream-plan.cpp `llama_kv_stream_feedback_delta_make` and fattn.cu
// `kv_stream_collect_timing`). The boundary they were originally consumed by - the adaptive
// partition controller - is gone: the resident window and the ring are planned once, when the Engine
// starts, because re-shaping a pool renames every resident page
// (bench/results/kv-stream/layout-rename-defect.md). The ratios remain the round's explanation.

// Deadline sampling: one sample per immutable copy run, counted on the device at the instant the
// attention that needs the run is launched, and a miss when the run's ready flag is still clear
// then (reference: `kv_stream_record_deadline`). Counters live in pinned host memory mapped to the
// device, so reading them costs no synchronisation.
struct KVStreamDeadlineCounters {
    std::uint64_t samples = 0;
    std::uint64_t misses  = 0;
};

struct KVStreamFeedbackDelta {
    bool valid          = false;
    bool has_evaluation = false;
    // Interprets `error` only when `valid` is false.
    const char* error   = nullptr;

    std::uint64_t samples = 0;
    std::uint64_t misses  = 0;
    // `misses / samples` of this evaluation window; zero without an evaluation.
    double miss_ratio = 0.0;
};

// The samples this evaluation added on top of `previous`.
//
// Counters only ever grow within a run, so a decrease means the run restarted and the delta is not
// a measurement of anything; more misses than samples is likewise impossible (a sample is one
// deadline observation). Both are rejected rather than clamped, because a controller fed a
// plausible-but-impossible ratio would move the partition on noise.
[[nodiscard]] KVStreamFeedbackDelta
kv_stream_feedback_delta(KVStreamDeadlineCounters current,
                         KVStreamDeadlineCounters previous) noexcept;

// Copy-engine occupancy for one evaluation window, estimated from a single sampled run and scaled
// to the window's uploads (reference: `kv_stream_collect_timing`, which samples the first copy of
// the window and multiplies by the window's upload count).
//
// The estimate exists because the alternative — timing every copy — costs device events on the
// very path being measured, and because the copies of a window are near-uniform: they move pages of
// one layer between the same two places.
struct KVStreamBusySample {
    // Wall time of the sampled copy run on the copy stream.
    double copy_seconds = 0.0;
    // Copy runs the sampled one stands for, and runs the window issued in total.
    std::uint64_t sampled_uploads = 0;
    std::uint64_t window_uploads  = 0;
    // Device time of the window the uploads belong to (the round's compute span).
    double window_seconds = 0.0;
    // False when no run was recorded in the window; the caller then has no busy estimate.
    bool recorded = false;
};

// `sampled copy time x window uploads / (sampled uploads x window time)`, clamped to one. A window
// with no device time, no uploads, or no recorded run yields zero, which says nothing about the
// window unless `recorded` is set.
[[nodiscard]] double kv_stream_copy_busy_ratio(const KVStreamBusySample& sample) noexcept;

} // namespace ninfer
