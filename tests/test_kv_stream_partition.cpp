#include "core/kv_stream_partition.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamPartitionParams;
using ninfer::kv_stream_partition_adapt;
using ninfer::kv_stream_partition_target_resident;

// Qwen3.5/3.8-27B has 16 full-attention layers, which is the geometry every case below uses: one
// growth step moves exactly 16 slots between the window and the ring.
constexpr std::uint32_t kLayers = 16;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

void check_pages(std::uint32_t actual, std::uint32_t expected, const std::string& message) {
    check(actual == expected, message);
}

// The shape of both fixtures below: a pool that holds 12 pages of context per layer in 160 slots, of
// which 16 must stay in the ring, and a 630-slot pool that holds 257.
KVStreamPartitionParams a_pool_of_160() {
    KVStreamPartitionParams params;
    params.total_pool_pages        = 160;
    params.layers                  = kLayers;
    params.active_pages_per_layer  = 12;
    params.minimum_ring_pages      = 16;
    params.previous_resident_pages = 9;
    params.previous_ring_pages     = 16;
    return params;
}

KVStreamPartitionParams a_pool_of_630() {
    KVStreamPartitionParams params;
    params.total_pool_pages        = 630;
    params.layers                  = kLayers;
    params.active_pages_per_layer  = 257;
    params.minimum_ring_pages      = 22;
    params.previous_resident_pages = 38;
    params.previous_ring_pages     = 22;
    return params;
}

// The target is the largest equal window whose ring holds 1.1x one layer's streamed pages: 160 - 16R
// >= 1.1(12 - R) holds up to R = 9, and 630 - 16R >= 1.1(257 - R) up to R = 23.
void the_target_is_the_largest_window_the_ring_ratio_allows() {
    check_pages(kv_stream_partition_target_resident(a_pool_of_160()), 9,
                "a 160-slot pool keeps nine pages of window");
    check_pages(kv_stream_partition_target_resident(a_pool_of_630()), 23,
                "a 630-slot pool keeps twenty-three pages of window");
}

// Misses with a link that still has headroom mean the ring is too small to hide the copy: one epoch
// of growth, one page off every layer at once, not one page off one layer.
void repeated_deadline_misses_grow_the_ring_by_one_balanced_epoch() {
    auto params                       = a_pool_of_160();
    params.deadline_miss_ratio        = 0.12;
    params.copy_engine_busy_ratio     = 0.70;
    params.starved_evaluations        = 2;
    params.grow_hysteresis_evaluations = 3;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the growth evaluation is valid");
    check(partition.changed, "the third starved evaluation grows the ring");
    check_pages(partition.resident_pages, 8, "growth demotes one page per layer");
    check_pages(partition.ring_pages, 32, "the ring gains one slot per layer");
    check_pages(partition.starved_evaluations, 0, "growth resets the starved counter");
}

// The article's safeguard: at 80% link occupancy a bigger ring cannot shorten the prefetch, it only
// evicts more resident KV. Misses alone must not demote anything then.
void pcie_saturation_does_not_sacrifice_more_resident_pages() {
    auto params                        = a_pool_of_160();
    params.deadline_miss_ratio         = 0.20;
    params.copy_engine_busy_ratio      = 0.99;
    params.starved_evaluations         = 10;
    params.grow_hysteresis_evaluations = 3;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the saturated evaluation is valid");
    check(!partition.changed, "a saturated link keeps the partition");
    check_pages(partition.resident_pages, 9, "no page is demoted while the link is saturated");
    check_pages(partition.ring_pages, 16, "the ring does not grow while the link is saturated");
}

// The overlap target is a bound, not a suggestion: feedback may demote at most one epoch past it, so
// a partition already at the target with misses still cannot go further while the link has headroom
// but the ring is no longer the limit.
void feedback_cannot_demote_beyond_one_epoch_past_the_target() {
    KVStreamPartitionParams params;
    params.total_pool_pages         = 7010;
    params.layers                   = kLayers;
    params.active_pages_per_layer   = 716;
    params.minimum_ring_pages       = 18;
    params.previous_resident_pages  = 416;  // the target is 417, so one page below it
    params.previous_ring_pages      = 354;
    params.deadline_miss_ratio      = 0.10;
    params.copy_engine_busy_ratio   = 0.80;
    params.starved_evaluations      = 2;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the bounded evaluation is valid");
    check(!partition.changed, "one page past the target is inside the feedback floor");
    check_pages(partition.resident_pages, 416, "the window stays where it is");
    check_pages(partition.ring_pages, 354, "the ring stays where it is");
}

// A ring that grew past the floor - by chasing its own latency - is repaired in one step instead of
// waiting for the shrink hysteresis, which needs a link that is almost idle.
void a_runaway_partition_heals_to_the_bounded_floor() {
    KVStreamPartitionParams params;
    params.total_pool_pages         = 7010;
    params.layers                   = kLayers;
    params.active_pages_per_layer   = 716;
    params.minimum_ring_pages       = 18;
    params.previous_resident_pages  = 368;
    params.previous_ring_pages      = 1122;
    params.deadline_miss_ratio      = 0.10;
    params.copy_engine_busy_ratio   = 0.80;
    params.starved_evaluations      = 2;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the healing evaluation is valid");
    check(partition.changed, "an overgrown ring is repaired");
    check_pages(partition.resident_pages, 416, "the window returns to the feedback floor");
    check_pages(partition.ring_pages, 354, "the ring returns to the floor's size");
    check_pages(partition.starved_evaluations, 0, "healing resets the starved counter");
}

// An undersized ring is a structural defect, not a tuning knob: even with a saturated link the
// partition jumps to the overlap target, because a ring that cannot overlap at all is what made the
// misses. This is the one growth the saturation brake does not block.
void an_undersized_ring_jumps_to_the_overlap_target_despite_copy_pressure() {
    auto params                        = a_pool_of_630();
    params.deadline_miss_ratio         = 0.40;
    params.copy_engine_busy_ratio      = 0.99;
    params.starved_evaluations         = 2;
    params.grow_hysteresis_evaluations = 3;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the undersized evaluation is valid");
    check(partition.changed, "an undersized ring moves to the target");
    check_pages(partition.resident_pages, 23, "the window lands on the overlap target");
    check_pages(partition.ring_pages, 262, "the ring takes every freed slot");
}

// Once the window sits on the target, misses with a link above the saturation line cannot demote
// further: the copy is saturated, so the ring is no longer what is limiting the prefetch.
void copy_pressure_stops_demotion_at_the_overlap_target() {
    auto params                   = a_pool_of_630();
    params.previous_resident_pages = 23;
    params.previous_ring_pages     = 262;
    params.deadline_miss_ratio     = 0.20;
    params.copy_engine_busy_ratio  = 0.90;
    params.starved_evaluations     = 10;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the saturated-at-target evaluation is valid");
    check(!partition.changed, "no demotion past the target on a saturated link");
    check_pages(partition.resident_pages, 23, "the window stays on the target");
    check_pages(partition.ring_pages, 262, "the ring stays at its target size");
}

// The first decode round has no feedback yet but already knows its working set: it takes the target
// immediately rather than spending several tokens one page at a time.
void the_decode_transition_selects_its_target_before_feedback_exists() {
    KVStreamPartitionParams params;
    params.total_pool_pages          = 6971;
    params.layers                    = kLayers;
    params.active_pages_per_layer    = 673;
    params.minimum_ring_pages        = 11;
    params.previous_resident_pages   = 435;  // the uniform prefill layout: everything but the ring
    params.previous_ring_pages       = 11;
    params.evaluations_since_repartition = 0;
    params.entering_decode_layout    = true;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the decode transition is valid");
    check(partition.changed, "the decode transition changes the partition immediately");
    check_pages(partition.resident_pages, 418, "the decode layout lands on the overlap target");
    check_pages(partition.ring_pages, 283, "the ring takes the slots the window gave up");
}

// A window holding every active page streams nothing, so the ring may sit at its minimum: that is
// the article's first rule, "只要固定的显存空间能够容纳整个键值缓存，环形缓冲区就保持禁用状态".
void a_window_that_holds_the_context_keeps_the_ring_at_its_minimum() {
    auto params                   = a_pool_of_160();
    params.active_pages_per_layer = 9;
    params.previous_resident_pages = 9;
    params.previous_ring_pages     = 16;
    params.deadline_miss_ratio     = 0.0;
    params.copy_engine_busy_ratio  = 0.0;

    const auto partition = kv_stream_partition_adapt(params);
    check(partition.valid, "the all-resident evaluation is valid");
    check(!partition.changed, "an all-resident window does not move");
    check_pages(partition.resident_pages, 9, "every active page stays resident");
}

// A pool that no longer covers the partition, a ring below its floor, or a metric that is not a ratio
// are all samples the controller must refuse rather than act on.
void an_inconsistent_partition_is_rejected() {
    auto broken_pool = a_pool_of_160();
    broken_pool.previous_ring_pages = 15;
    check(!kv_stream_partition_adapt(broken_pool).valid,
          "a partition that does not cover the pool is rejected");

    auto below_floor = a_pool_of_160();
    below_floor.previous_ring_pages = 8;
    below_floor.previous_resident_pages = 9;
    check(!kv_stream_partition_adapt(below_floor).valid,
          "a ring below its minimum is rejected");

    auto bad_metric = a_pool_of_160();
    bad_metric.deadline_miss_ratio = 1.5;
    check(!kv_stream_partition_adapt(bad_metric).valid, "a miss ratio above one is rejected");

    auto resident_too_large = a_pool_of_160();
    resident_too_large.previous_resident_pages = 13;
    check(!kv_stream_partition_adapt(resident_too_large).valid,
          "a window larger than the context is rejected");

    auto zero_geometry = KVStreamPartitionParams{};
    check(!kv_stream_partition_adapt(zero_geometry).valid,
          "a partition without a pool is rejected");
}

} // namespace

int main() {
    the_target_is_the_largest_window_the_ring_ratio_allows();
    repeated_deadline_misses_grow_the_ring_by_one_balanced_epoch();
    pcie_saturation_does_not_sacrifice_more_resident_pages();
    feedback_cannot_demote_beyond_one_epoch_past_the_target();
    a_runaway_partition_heals_to_the_bounded_floor();
    an_undersized_ring_jumps_to_the_overlap_target_despite_copy_pressure();
    copy_pressure_stops_demotion_at_the_overlap_target();
    the_decode_transition_selects_its_target_before_feedback_exists();
    a_window_that_holds_the_context_keeps_the_ring_at_its_minimum();
    an_inconsistent_partition_is_rejected();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream partition: all checks passed\n";
    return 0;
}
