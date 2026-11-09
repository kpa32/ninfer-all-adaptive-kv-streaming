#pragma once

#include "core/device.h"
#include "core/kv_stream_feedback.h"

#include <cstdint>

namespace ninfer {

// The device-side half of the two feedback signals the partition controller runs on
// (docs/maintainer/kv-streaming-fidelity.md S3, ported from the reference implementation's
// `kv_stream_record_deadline` / `kv_stream_collect_timing`).
//
// Why the device has to answer this: "did this copy land before the attention needed it" is only
// true or false at the instant that attention launches, and the host cannot look at that instant
// without synchronising. So each copy run owns one flag in pinned host memory that the device can
// also address: the copy stream clears it before the copy and sets it after, and a one-thread kernel
// on the compute stream, enqueued immediately before the consumer's wait, counts the sample and
// whether the flag was still clear. The host reads the counters between rounds, never waits on
// them.
//
// The same object measures copy-engine occupancy the way the reference does, because timing every
// copy would charge the measured path for its own events: one copy run per window is timed, and the
// window's busy time is that run scaled by how many runs the window issued.
class KVStreamSignal {
public:
    // `runs` is one flag per copy run the caller distinguishes - one per streaming layer, since a
    // layer's streamed pages are copied as one run.
    explicit KVStreamSignal(std::uint32_t runs);
    ~KVStreamSignal();

    KVStreamSignal(const KVStreamSignal&)            = delete;
    KVStreamSignal& operator=(const KVStreamSignal&) = delete;
    KVStreamSignal(KVStreamSignal&&)                 = delete;
    KVStreamSignal& operator=(KVStreamSignal&&)      = delete;

    // Before a run's copies are issued, and after them: the flag says whether the consumer would
    // still have waited when it launched.
    void clear_run(std::uint32_t run, cudaStream_t copy);
    void mark_run(std::uint32_t run, cudaStream_t copy);

    // One deadline sample for `run`, taken at the consumer's launch. Enqueued on the compute stream
    // immediately before the wait, so it observes exactly what the wait is about to observe.
    void sample_deadline(std::uint32_t run, cudaStream_t compute);

    // The window one round's copies belong to, and the copy run that stands for them.
    void begin_window(cudaStream_t compute);
    void end_window(cudaStream_t compute);
    void begin_copy_sample(cudaStream_t copy);
    void end_copy_sample(cudaStream_t copy);

    // The counters the device wrote. Safe to call at any time: a round still in flight simply is not
    // included yet.
    [[nodiscard]] KVStreamDeadlineCounters deadline_counters() const noexcept;

    // The window's device time, or nothing while the device has not finished it. A window whose
    // events were never recorded, or whose sampled run was not recorded, reports `recorded = false`.
    struct WindowTime {
        bool recorded  = false;
        double seconds = 0.0;
    };
    [[nodiscard]] WindowTime window_time() const noexcept;

    // The sampled copy run of the last completed window, or nothing while it has not finished.
    struct CopyTime {
        bool recorded  = false;
        double seconds = 0.0;
        std::uint64_t uploads = 0;
    };
    [[nodiscard]] CopyTime copy_time() const noexcept;

    // Accounts for the run being re-recorded before its time was read: a sample that never completed
    // is lost rather than guessed.
    void note_window_restart() noexcept { window_lost_ = true; }

    [[nodiscard]] std::uint64_t windows_lost() const noexcept { return window_lost_; }

    [[nodiscard]] std::size_t pinned_bytes() const noexcept;

private:
    std::uint32_t runs_ = 0;
    std::uint32_t* ready_host_    = nullptr;
    std::uint32_t* ready_device_  = nullptr;
    std::uint64_t* counters_host_ = nullptr;
    std::uint64_t* counters_device_ = nullptr;
    cudaEvent_t window_begin_ = nullptr;
    cudaEvent_t window_end_   = nullptr;
    cudaEvent_t copy_begin_   = nullptr;
    cudaEvent_t copy_end_     = nullptr;
    bool window_pending_ = false;
    bool copy_pending_   = false;
    std::uint64_t copy_uploads_ = 0;
    std::uint64_t window_lost_  = 0;
};

} // namespace ninfer
