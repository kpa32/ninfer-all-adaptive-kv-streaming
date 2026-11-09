#pragma once

#include <cstdint>

namespace ninfer {

// Moves the boundary between resident pages and the transfer ring inside one fixed pool
// (docs/maintainer/kv-streaming-fidelity.md S2; ported from the reference implementation's
// `llama_kv_stream_partition_adapt`).
//
// The reference article asks for exactly this and for nothing more (inputs, appendix A, adaptive
// repartition):
//
//   "1. 尽可能保持所有数据驻留。只要固定的显存空间能够容纳整个键值缓存，环形缓冲区就保持禁用状态。
//    2. 当需要进行流式传输时，从小规模开始 … 从驻留缓存中释放一小部分 … 释放出来的显存将成为由流式
//    传输层共享的环形缓冲区。
//    3. 仅在必要时才扩展环形缓冲区 … 从驻留的键值缓存中移除一小部分，并使用新释放的空间来扩展环形
//    缓冲区。
//    6. 只有在增大环形缓冲区仍有助于隐藏预取延迟的情况下才进行增长。如果 PCIe 带宽已饱和，则停止
//    增大环形缓冲区。"
//
// One growth step demotes exactly one page from every layer, so the ring grows by `layers` slots and
// the window shrinks by one page per layer at once: the resident partition stays balanced, and the
// pages that leave are the ones every layer gives up equally. Promotion performs the inverse after a
// longer hysteresis, because a page that comes back can be demoted again by the next miss.
struct KVStreamPartitionParams {
    std::uint32_t total_pool_pages = 0;
    std::uint32_t layers           = 0;
    // Pages one layer reaches at the current context: the resident window can never exceed it.
    std::uint32_t active_pages_per_layer = 0;
    // The ring may never shrink below this: a ring too small to hold one copy batch cannot overlap
    // anything, and the pool keeps no useful prefetch room.
    std::uint32_t minimum_ring_pages = 0;

    std::uint32_t previous_resident_pages = 0;
    std::uint32_t previous_ring_pages     = 0;

    double deadline_miss_ratio       = 0.0;
    double copy_engine_busy_ratio    = 0.0;
    double ring_peak_occupancy_ratio = 0.0;

    // Ring slots relative to the streamed pages one attention layer consumes. Above one leaves room
    // to begin the next layer early, which is what makes the copy overlap attention instead of
    // serialising with it.
    double target_ring_working_set_ratio = 1.10;

    std::uint32_t starved_evaluations          = 0;
    std::uint32_t overprovisioned_evaluations  = 0;
    std::uint32_t grow_hysteresis_evaluations  = 3;
    std::uint32_t shrink_hysteresis_evaluations = 8;
    std::uint32_t evaluations_since_repartition = UINT32_MAX;
    std::uint32_t repartition_cooldown_evaluations = 64;

    // The first decode graph has no usable streaming feedback yet, but its active working set is
    // already known. Select the deterministic overlap target immediately instead of spending several
    // rounds in an undersized ring before the hysteresis can react.
    bool entering_decode_layout = false;
};

struct KVStreamPartition {
    bool valid = false;
    // Interpreted only when `valid` is false.
    const char* error = nullptr;
    bool changed      = false;

    std::uint32_t resident_pages = 0;
    std::uint32_t ring_pages     = 0;
    std::uint32_t starved_evaluations         = 0;
    std::uint32_t overprovisioned_evaluations = 0;
};

// The partition this evaluation selects. Every rejection leaves the caller's partition untouched,
// because a controller that moves on a sample it could not interpret is worse than one that waits.
[[nodiscard]] KVStreamPartition
kv_stream_partition_adapt(const KVStreamPartitionParams& params) noexcept;

// The largest equal per-layer window whose ring still holds `target_ring_working_set_ratio` times one
// layer's streamed pages, bounded by the layer's active pages and by `minimum_ring_pages`.
//
// With `streamed = active - resident` and `ring = pool - layers * resident`:
//
//     pool - layers * R >= ratio * (active - R)     =>     R <= (pool - ratio * active) / (layers - ratio)
//
// which is what the search below evaluates exactly (in integers, without the division).
[[nodiscard]] std::uint32_t
kv_stream_partition_target_resident(const KVStreamPartitionParams& params) noexcept;

} // namespace ninfer
