#include "core/kv_stream_signal.h"

#include <cstring>
#include <stdexcept>

namespace ninfer {
namespace {

// One thread, one sample: the counters are in pinned host memory the device writes through the same
// pointer, so the atomics are ordinary device atomics and the host sees them without a copy
// (reference: `kv_stream_record_deadline`).
__global__ void record_deadline(const std::uint32_t* ready_flag, std::uint64_t* samples,
                                std::uint64_t* misses) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(samples), 1ULL);
        if (*ready_flag == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(misses), 1ULL);
        }
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) { (void)cudaEventDestroy(event); }
    event = nullptr;
}

} // namespace

KVStreamSignal::KVStreamSignal(std::uint32_t runs) : runs_(runs) {
    if (runs_ == 0) { throw std::invalid_argument("A KV stream signal needs at least one run"); }
    // Mapped pinned memory: the device writes the flags and the counters through the same pages the
    // host reads, so no transfer and no synchronisation stands between a round and its numbers.
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ready_host_),
                             runs_ * sizeof(std::uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ready_device_), ready_host_, 0));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&counters_host_), 2 * sizeof(std::uint64_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&counters_device_), counters_host_,
                                        0));
    std::memset(ready_host_, 0, runs_ * sizeof(std::uint32_t));
    std::memset(counters_host_, 0, 2 * sizeof(std::uint64_t));

    // These keep their times, so they are created without the disable-timing flag.
    CUDA_CHECK(cudaEventCreate(&window_begin_));
    CUDA_CHECK(cudaEventCreate(&window_end_));
    CUDA_CHECK(cudaEventCreate(&copy_begin_));
    CUDA_CHECK(cudaEventCreate(&copy_end_));
}

KVStreamSignal::~KVStreamSignal() {
    destroy_event(window_begin_);
    destroy_event(window_end_);
    destroy_event(copy_begin_);
    destroy_event(copy_end_);
    if (ready_host_ != nullptr) { (void)cudaFreeHost(ready_host_); }
    if (counters_host_ != nullptr) { (void)cudaFreeHost(counters_host_); }
    ready_host_      = nullptr;
    ready_device_    = nullptr;
    counters_host_   = nullptr;
    counters_device_ = nullptr;
}

void KVStreamSignal::clear_run(std::uint32_t run, cudaStream_t copy) {
    if (run >= runs_) { throw std::out_of_range("KV stream signal run is out of range"); }
    CUDA_CHECK(cudaMemsetAsync(ready_device_ + run, 0, sizeof(std::uint32_t), copy));
}

void KVStreamSignal::mark_run(std::uint32_t run, cudaStream_t copy) {
    if (run >= runs_) { throw std::out_of_range("KV stream signal run is out of range"); }
    CUDA_CHECK(cudaMemsetAsync(ready_device_ + run, 1, sizeof(std::uint32_t), copy));
}

void KVStreamSignal::sample_deadline(std::uint32_t run, cudaStream_t compute) {
    if (run >= runs_) { throw std::out_of_range("KV stream signal run is out of range"); }
    record_deadline<<<1, 1, 0, compute>>>(ready_device_ + run, counters_device_ + 0,
                                          counters_device_ + 1);
    CUDA_CHECK(cudaGetLastError());
}

void KVStreamSignal::begin_window(cudaStream_t compute) {
    // The previous window's events are only overwritten once its time has been taken, so a re-record
    // is a measurement the caller dropped, not one it silently replaced.
    window_pending_ = true;
    copy_pending_   = false;
    copy_uploads_   = 0;
    CUDA_CHECK(cudaEventRecord(window_begin_, compute));
}

void KVStreamSignal::end_window(cudaStream_t compute) {
    if (!window_pending_) { return; }
    CUDA_CHECK(cudaEventRecord(window_end_, compute));
}

void KVStreamSignal::begin_copy_sample(cudaStream_t copy) {
    if (copy_pending_ || !window_pending_) { return; }
    copy_pending_ = true;
    CUDA_CHECK(cudaEventRecord(copy_begin_, copy));
}

void KVStreamSignal::end_copy_sample(cudaStream_t copy) {
    if (!copy_pending_ || copy_uploads_ != 0) { return; }
    copy_uploads_ = 1;
    CUDA_CHECK(cudaEventRecord(copy_end_, copy));
}

KVStreamDeadlineCounters KVStreamSignal::deadline_counters() const noexcept {
    KVStreamDeadlineCounters counters;
    if (counters_host_ == nullptr) { return counters; }
    counters.samples = counters_host_[0];
    counters.misses  = counters_host_[1];
    return counters;
}

KVStreamSignal::WindowTime KVStreamSignal::window_time() const noexcept {
    WindowTime time;
    if (!window_pending_) { return time; }
    const cudaError_t status = cudaEventQuery(window_end_);
    if (status == cudaErrorNotReady) { return time; }
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return time;
    }
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, window_begin_, window_end_) != cudaSuccess) {
        (void)cudaGetLastError();
        return time;
    }
    time.recorded = true;
    time.seconds  = static_cast<double>(milliseconds) / 1000.0;
    return time;
}

KVStreamSignal::CopyTime KVStreamSignal::copy_time() const noexcept {
    CopyTime time;
    if (!copy_pending_ || copy_uploads_ == 0) { return time; }
    const cudaError_t status = cudaEventQuery(copy_end_);
    if (status == cudaErrorNotReady) { return time; }
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return time;
    }
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, copy_begin_, copy_end_) != cudaSuccess) {
        (void)cudaGetLastError();
        return time;
    }
    time.recorded = true;
    time.seconds  = static_cast<double>(milliseconds) / 1000.0;
    time.uploads  = copy_uploads_;
    return time;
}

std::size_t KVStreamSignal::pinned_bytes() const noexcept {
    return runs_ * sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t);
}

} // namespace ninfer
