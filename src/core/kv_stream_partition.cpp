#include "core/kv_stream_partition.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ninfer {

namespace {

bool ratio_valid(double value) noexcept { return value >= 0.0 && value <= 1.0; }

} // namespace

std::uint32_t kv_stream_partition_target_resident(const KVStreamPartitionParams& params) noexcept {
    if (params.layers == 0) { return 0; }
    const std::uint32_t maximum = std::min<std::uint64_t>(
        params.active_pages_per_layer,
        params.total_pool_pages > params.minimum_ring_pages
            ? (params.total_pool_pages - params.minimum_ring_pages)/params.layers
            : 0U);
    for (std::uint32_t resident = maximum;; --resident) {
        const std::uint32_t ring = params.total_pool_pages - resident*params.layers;
        const std::uint32_t streamed = params.active_pages_per_layer - resident;
        if (streamed == 0 ||
            static_cast<double>(ring) >= params.target_ring_working_set_ratio*
                                         static_cast<double>(streamed)) {
            return resident;
        }
        if (resident == 0) { return 0; }
    }
}

KVStreamPartition kv_stream_partition_adapt(const KVStreamPartitionParams& params) noexcept {
    KVStreamPartition result;
    result.resident_pages = params.previous_resident_pages;
    result.ring_pages     = params.previous_ring_pages;

    const auto fail = [&](const char* message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    if (params.total_pool_pages == 0 || params.layers == 0) {
        return fail("KV stream partition geometry must be non-zero");
    }
    if (params.grow_hysteresis_evaluations == 0 || params.shrink_hysteresis_evaluations == 0 ||
        params.repartition_cooldown_evaluations == 0) {
        return fail("KV stream partition hysteresis must be non-zero");
    }
    if (!ratio_valid(params.deadline_miss_ratio) || !ratio_valid(params.copy_engine_busy_ratio) ||
        !ratio_valid(params.ring_peak_occupancy_ratio)) {
        return fail("KV stream partition metrics must be ratios");
    }
    if (!std::isfinite(params.target_ring_working_set_ratio) ||
        params.target_ring_working_set_ratio <= 0.0) {
        return fail("KV stream target working-set ratio must be positive");
    }
    if (params.previous_resident_pages > params.active_pages_per_layer) {
        return fail("resident KV pages exceed active pages");
    }
    // The two halves must still cover the pool exactly: a partition that lost or duplicated slots
    // would make every later count wrong without failing anywhere else.
    if (std::uint64_t(params.previous_resident_pages)*params.layers + params.previous_ring_pages !=
        params.total_pool_pages) {
        return fail("previous KV stream partition does not cover the fixed pool");
    }
    if (params.previous_ring_pages < params.minimum_ring_pages) {
        return fail("previous KV stream ring is below its minimum");
    }

    constexpr double kMissThreshold = 0.01;
    // A repartition invalidates the compact resident addresses and copies the pages that change
    // hands. Do not pay that transition when the copy engine already lacks the headroom for a
    // larger ring to raise sustained throughput: this is the article's safeguard, "如果 PCIe 带宽
    // 已饱和，则停止增大环形缓冲区".
    constexpr double kCopySaturated = 0.80;
    constexpr double kCopyLight     = 0.50;
    constexpr double kRingLight     = 0.50;
    // Feedback may demote at most this far past the overlap target; anything beyond it is a ring
    // that grew while chasing its own latency, which is the article's vicious cycle.
    constexpr std::uint32_t kFeedbackGrowthEpochs = 1;

    const std::uint32_t target = kv_stream_partition_target_resident(params);
    const std::uint32_t feedback_floor =
        target > kFeedbackGrowthEpochs ? target - kFeedbackGrowthEpochs : 0;
    const bool overgrown_ring = params.previous_resident_pages < feedback_floor;

    const bool below_overlap_target = params.previous_resident_pages > target;
    const bool feedback_starved = params.previous_resident_pages > feedback_floor &&
                                  params.deadline_miss_ratio > kMissThreshold &&
                                  params.copy_engine_busy_ratio < kCopySaturated;
    const bool starved = below_overlap_target || feedback_starved;
    const bool overprovisioned = params.deadline_miss_ratio <= kMissThreshold &&
                                 params.copy_engine_busy_ratio < kCopyLight &&
                                 params.ring_peak_occupancy_ratio < kRingLight;

    result.starved_evaluations =
        starved ? params.starved_evaluations + 1 : 0;
    result.overprovisioned_evaluations =
        overprovisioned ? params.overprovisioned_evaluations + 1 : 0;

    const bool cooldown_complete =
        params.evaluations_since_repartition >= params.repartition_cooldown_evaluations;
    if (params.entering_decode_layout && below_overlap_target) {
        // The first decode round knows its working set but has no feedback yet: jump straight to the
        // target instead of growing one page per epoch for several tokens.
        result.resident_pages = target;
        result.ring_pages     = params.total_pool_pages - result.resident_pages*params.layers;
        result.starved_evaluations         = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overgrown_ring) {
        result.resident_pages = feedback_floor;
        result.ring_pages     = params.total_pool_pages - result.resident_pages*params.layers;
        result.starved_evaluations         = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && starved &&
               result.starved_evaluations >= params.grow_hysteresis_evaluations &&
               result.resident_pages > 0) {
        result.resident_pages =
            below_overlap_target ? target : result.resident_pages - 1;
        result.ring_pages = params.total_pool_pages - result.resident_pages*params.layers;
        result.starved_evaluations         = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overprovisioned &&
               result.overprovisioned_evaluations >= params.shrink_hysteresis_evaluations &&
               result.resident_pages < target &&
               result.ring_pages >= params.minimum_ring_pages + params.layers) {
        ++result.resident_pages;
        result.ring_pages -= params.layers;
        result.starved_evaluations         = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    }

    result.valid = true;
    return result;
}

} // namespace ninfer
