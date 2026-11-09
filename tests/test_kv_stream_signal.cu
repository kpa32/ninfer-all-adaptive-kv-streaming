// The device-side feedback kernel of the streaming KV cache (core/kv_stream_signal.h): one flag per
// copy run, one sample per run at the consumer's launch, and two window measurements.
//
// The oracle is the test's own bookkeeping, not another kernel: this file decides which runs were
// marked ready and which were not, so the exact sample and miss counts it predicts are the
// specification. A counter that drifts from that bookkeeping is a wrong feedback signal, and the
// partition controller runs on those two numbers.
//
// Runs on a GPU; without one it reports the skip and exits 77, as tests/README.md prescribes.

#include "core/kv_stream_signal.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ninfer::KVStreamDeadlineCounters;
using ninfer::KVStreamSignal;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "  failed: %s\n", message.c_str());
    }
}

void check_counters(const KVStreamDeadlineCounters& actual, std::uint64_t samples,
                    std::uint64_t misses, const std::string& what) {
    check(actual.samples == samples,
          what + ": samples " + std::to_string(actual.samples) + " expected " +
              std::to_string(samples));
    check(actual.misses == misses,
          what + ": misses " + std::to_string(actual.misses) + " expected " +
              std::to_string(misses));
}

// A page-sized transfer on the copy stream, so the window and copy timings have real work to price.
void copy_some_bytes(cudaStream_t stream, void* staging) {
    void* destination = static_cast<char*>(staging) + (1u << 20);
    (void)cudaMemsetAsync(staging, 0x5a, 1u << 20, stream);
    (void)cudaMemcpyAsync(destination, staging, 1u << 20, cudaMemcpyDeviceToDevice, stream);
}

// One clear/mark cycle for `run`, then one sample: the sequence a round performs for each run.
void sample_once(KVStreamSignal& signal, std::uint32_t run, bool ready, cudaStream_t copy,
                 cudaStream_t compute) {
    signal.clear_run(run, copy);
    if (ready) { signal.mark_run(run, copy); }
    (void)cudaStreamSynchronize(copy);
    signal.sample_deadline(run, compute);
    (void)cudaStreamSynchronize(compute);
}

void test_a_ready_run_is_a_sample_not_a_miss(cudaStream_t copy, cudaStream_t compute) {
    KVStreamSignal signal(4);
    sample_once(signal, 0, /*ready=*/true, copy, compute);
    check_counters(signal.deadline_counters(), 1, 0, "a ready run");
}

void test_an_unready_run_is_a_miss(cudaStream_t copy, cudaStream_t compute) {
    KVStreamSignal signal(4);
    sample_once(signal, 0, /*ready=*/false, copy, compute);
    check_counters(signal.deadline_counters(), 1, 1, "an unready run");
}

void test_each_run_keeps_its_own_flag(cudaStream_t copy, cudaStream_t compute) {
    // Runs 0 and 2 are ready, 1 and 3 are not: the split must land exactly on the bookkeeping, which
    // fails if one run's flag leaks into another's sample (the shape of a shared-ring mistake).
    KVStreamSignal signal(4);
    sample_once(signal, 0, /*ready=*/true, copy, compute);
    sample_once(signal, 1, /*ready=*/false, copy, compute);
    sample_once(signal, 2, /*ready=*/true, copy, compute);
    sample_once(signal, 3, /*ready=*/false, copy, compute);
    check_counters(signal.deadline_counters(), 4, 2, "mixed readiness across four runs");
}

void test_a_clear_cycle_reaches_the_ready_state_again(cudaStream_t copy, cudaStream_t compute) {
    // A run that missed once and is marked in the next round is a sample, not a miss: the flag is
    // cleared before each copy, so the previous round's value cannot carry over.
    KVStreamSignal signal(2);
    sample_once(signal, 0, /*ready=*/false, copy, compute);
    sample_once(signal, 0, /*ready=*/true, copy, compute);
    check_counters(signal.deadline_counters(), 2, 1, "a missed run that became ready");
}

void test_a_window_measures_its_device_time(cudaStream_t copy, cudaStream_t compute,
                                            void* staging) {
    // The order the space uses: the window opens, one run of copies is sampled around its transfers,
    // the run is marked ready, and the window closes. `copy_time` reports the sampled run and how many
    // runs the window issued (`uploads`), which the caller scales to the window's occupancy.
    KVStreamSignal signal(2);
    signal.begin_window(compute);
    signal.begin_copy_sample(copy);
    copy_some_bytes(copy, staging);
    signal.end_copy_sample(copy);
    signal.clear_run(0, copy);
    signal.mark_run(0, copy);
    signal.end_window(compute);
    signal.sample_deadline(0, compute);
    (void)cudaStreamSynchronize(compute);

    const KVStreamSignal::WindowTime window = signal.window_time();
    check(window.recorded, "a finished window reports its time");
    check(window.seconds > 0.0, "a finished window's time is positive");
    check(signal.windows_lost() == 0, "a window that was read is not lost");

    const KVStreamSignal::CopyTime timed = signal.copy_time();
    check(timed.recorded, "a window that sampled a copy run reports its copy time");
    check(timed.seconds > 0.0, "a sampled copy run's time is positive");
    check(timed.uploads == 1, "the sampled window issued one upload");
}

void test_a_window_without_a_copy_sample_reports_none(cudaStream_t copy, cudaStream_t compute) {
    // A window whose copies were never sampled says so instead of reporting zero seconds: the
    // controller must not read "no sample" as "the copy engine was idle".
    KVStreamSignal signal(2);
    signal.begin_window(compute);
    signal.clear_run(0, copy);
    signal.mark_run(0, copy);
    signal.end_window(compute);
    signal.sample_deadline(0, compute);
    (void)cudaStreamSynchronize(compute);
    check(signal.window_time().recorded, "a finished window reports its time");
    check(!signal.copy_time().recorded, "an unsampled window reports no copy time");
    check(signal.copy_time().uploads == 0, "an unsampled window reports no uploads");
}

void test_an_unfinished_window_is_not_reported(cudaStream_t compute) {
    // Between `begin_window` and `end_window` the device has not finished the window, so nothing is
    // reported: a guessed time is worse than a missing one.
    KVStreamSignal signal(2);
    signal.begin_window(compute);
    check(!signal.window_time().recorded, "an open window reports no time");
    signal.end_window(compute);
    (void)cudaStreamSynchronize(compute);
    check(signal.window_time().recorded, "a closed window reports its time");
}

void test_a_restarted_window_is_counted_as_lost(cudaStream_t compute) {
    KVStreamSignal signal(2);
    signal.begin_window(compute);
    signal.end_window(compute);
    (void)cudaStreamSynchronize(compute);
    check(signal.windows_lost() == 0, "no window is lost before anything restarts one");
    signal.note_window_restart();
    check(signal.windows_lost() == 1, "a restarted window is accounted as lost");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "kv stream signal: no CUDA device, skipping\n");
        return 77;
    }
    (void)cudaSetDevice(0);

    cudaStream_t copy    = nullptr;
    cudaStream_t compute = nullptr;
    void* staging        = nullptr;
    if (cudaStreamCreate(&copy) != cudaSuccess || cudaStreamCreate(&compute) != cudaSuccess ||
        cudaMalloc(&staging, 2u << 20) != cudaSuccess) {
        std::fprintf(stderr, "kv stream signal: cannot create the test's streams\n");
        return 1;
    }

    test_a_ready_run_is_a_sample_not_a_miss(copy, compute);
    test_an_unready_run_is_a_miss(copy, compute);
    test_each_run_keeps_its_own_flag(copy, compute);
    test_a_clear_cycle_reaches_the_ready_state_again(copy, compute);
    test_a_window_measures_its_device_time(copy, compute, staging);
    test_a_window_without_a_copy_sample_reports_none(copy, compute);
    test_an_unfinished_window_is_not_reported(compute);
    test_a_restarted_window_is_counted_as_lost(compute);

    (void)cudaFree(staging);
    (void)cudaStreamDestroy(copy);
    (void)cudaStreamDestroy(compute);

    if (failures != 0) {
        std::fprintf(stderr, "kv stream signal: %d checks failed\n", failures);
        return 1;
    }
    std::printf("kv stream signal: all checks passed\n");
    return 0;
}
