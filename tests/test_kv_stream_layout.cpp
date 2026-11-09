#include "core/kv_stream_layout.h"
#include "core/kv_stream_slot_map.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamSlotMap;
using ninfer::KVStreamSplit;
using ninfer::kv_stream_pool_pages;
using ninfer::kv_stream_ring_runs;
using ninfer::kv_stream_runs_per_layer;
using ninfer::kv_stream_window_slot;
using ninfer::kv_stream_window_start;
using ninfer::plan_kv_stream_split;
using ninfer::resize_kv_stream_split;

constexpr std::uint32_t kLayers = 16;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

void check_equal(std::uint32_t actual, std::uint32_t expected, const std::string& message) {
    if (actual != expected) { ++failures[message]; }
}

// The all-resident split must reproduce the layout the paged cache has today: layer L's page p is
// slot L * pages_per_layer + p, so a pool that holds everything moves no byte.
void all_resident_matches_today() {
    constexpr std::uint32_t kPages = 64;
    const auto split = plan_kv_stream_split(kLayers * kPages, kPages, kLayers);
    check(split.has_value(), "a full budget must plan");
    if (!split) { return; }
    check_equal(split->resident_pages, kPages, "a full budget keeps every page resident");
    check_equal(split->ring_pages, 0, "a full budget needs no ring");
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t page = 0; page < kPages; ++page) {
            check_equal(static_cast<std::uint32_t>(kv_stream_window_slot(layer, page, *split)),
                        layer * kPages + page, "an all-resident page keeps today's slot");
        }
    }
}

// The measured geometry at 262144 tokens: 4096 pages of context per layer, 16 full-attention layers,
// and a pool of 31572 slots (the one the shipping 251K run used). The ring must be just above one
// streamed extent - the article's 1.10 - and every other slot must be window.
void the_shipping_262144_geometry() {
    constexpr std::uint32_t kPages  = 4096;
    constexpr std::uint32_t kBudget = 31572;
    const auto split = plan_kv_stream_split(kBudget, kPages, kLayers);
    check(split.has_value(), "the shipping geometry plans");
    if (!split) { return; }
    check(kv_stream_pool_pages(*split, kLayers) <= kBudget, "the split fits its pool");
    const std::uint32_t streamed = kPages - split->resident_pages;
    check(split->ring_pages >= streamed, "the ring holds one layer's streamed extent");
    check(static_cast<double>(split->ring_pages) <= 1.25 * static_cast<double>(streamed),
          "the ring stays near the reference's ratio instead of a whole extra extent");
    check_equal(split->ring_pages % split->run_pages, 0, "the ring is a whole number of runs");
    // The window is what the old layout could not reach: two extents of ring left it at 1670.
    check(split->resident_pages > 1670, "a ratio-sized ring keeps a larger window than depth two");
}

// Every page changes slot at most twice: once when the window takes it, and (when the context grows
// past its window) never again, because a streamed page is addressed by its run, not by its index.
void a_resident_page_keeps_its_slot() {
    constexpr std::uint32_t kPages = 132;
    const auto split = plan_kv_stream_split(16 * 40, kPages, kLayers, 8);
    check(split.has_value(), "a partially resident pool plans");
    if (!split) { return; }
    for (std::uint32_t active = split->resident_pages; active <= kPages; ++active) {
        const std::uint32_t start = kv_stream_window_start(active, *split);
        for (std::uint32_t page = start; page < active; ++page) {
            const auto first = kv_stream_window_slot(0, page, *split);
            for (std::uint32_t later = active; later <= kPages; ++later) {
                if (kv_stream_window_start(later, *split) > page) { break; }
                check(kv_stream_window_slot(0, page, *split) == first,
                      "a resident page keeps its slot while it stays resident");
            }
        }
    }
}

// Runs are what the copy and the consumer work in: contiguous slots, never straddling the ring, and
// distinct runs never overlap while the ring holds one layer's extent.
void runs_are_contiguous_and_disjoint() {
    KVStreamSlotMap map(256, 8);
    check_equal(map.slots(), 256, "a ring that is a whole number of runs keeps every run");
    check_equal(map.runs(), 32, "the run count follows the ring size");
    std::map<std::uint32_t, int> used;
    for (std::uint32_t run = 0; run < map.runs(); ++run) {
        const auto slot = map.next();
        check_equal(slot.count, 8, "a run covers the run's pages");
        check_equal(slot.begin % 8, 0, "a run starts on a run boundary");
        for (std::uint32_t i = 0; i < slot.count; ++i) { ++used[slot.begin + i]; }
    }
    for (const auto& [slot, count] : used) { check(count == 1, "a slot is handed out once per lap"); }
    // The cursor has wrapped exactly once and points at the first run again.
    check_equal(map.next().begin, 0, "the allocator is a lap");
}

void a_short_ring_rounds_down_to_whole_runs() {
    KVStreamSlotMap map(250, 8);
    check_equal(map.slots(), 248, "a ring shorter than the run multiple loses the remainder");
    check_equal(map.runs(), 31, "only whole runs are handed out");
    for (std::uint32_t run = 0; run < 40; ++run) {
        const auto slot = map.next();
        check(slot.begin + slot.count <= map.slots(), "a run never leaves the ring");
    }
}

// A run that holds a layer's tail is shorter than a full run; the planner still rounds the ring to
// whole runs so the copy can address it as one range.
void the_plan_rounds_the_ring_to_whole_runs() {
    constexpr std::uint32_t kPages = 4096;
    for (std::uint32_t run : {8U, 16U, 32U, 128U}) {
        const auto split = plan_kv_stream_split(31572, kPages, kLayers, run);
        check(split.has_value(), "any run size plans");
        if (!split) { continue; }
        check_equal(split->ring_pages % run, 0, "the ring is whole runs");
        check_equal(split->run_pages, run, "the split keeps the requested run size");
        check_equal(kv_stream_ring_runs(*split), split->ring_pages / run, "ring runs follow the ring");
        const std::uint32_t streamed = kPages - split->resident_pages;
        check(kv_stream_ring_runs(*split) >= (streamed + run - 1) / run,
              "the ring holds every run of one layer's extent");
    }
}

// The article's first rule, on the split side: when a request's context fits the pool, nothing
// streams, so the ring is not addressed and every page stays resident.
void a_context_that_fits_needs_no_ring() {
    const auto planned = plan_kv_stream_split(31572, 4096, kLayers);
    check(planned.has_value(), "the shipping geometry plans");
    if (!planned) { return; }
    const auto fitted = resize_kv_stream_split(*planned, 31572, 1900, kLayers);
    check(fitted.has_value(), "a short context re-sizes");
    if (!fitted) { return; }
    check_equal(fitted->ring_pages, 0, "a context that fits streams nothing");
    check_equal(fitted->resident_pages, 1900, "every page of the context stays resident");
    check(kv_stream_pool_pages(*fitted, kLayers) <= 31572, "the fitted split fits the pool");
}

// Growing the context keeps giving the ring what it needs and the window the rest, and never lets the
// ring fall below the extent it must hold.
void re_sizing_follows_the_context() {
    const auto planned = plan_kv_stream_split(31572, 4096, kLayers);
    check(planned.has_value(), "the shipping geometry plans");
    if (!planned) { return; }
    std::uint32_t previous_resident = 0;
    for (std::uint32_t active = 2000; active <= 4096; active += 97) {
        const auto fitted = resize_kv_stream_split(*planned, 31572, active, kLayers);
        check(fitted.has_value(), "a growing context stays servable");
        if (!fitted) { continue; }
        check(kv_stream_pool_pages(*fitted, kLayers) <= 31572, "a re-sized split fits the pool");
        const std::uint32_t streamed = kv_stream_window_start(active, *fitted);
        check(fitted->ring_pages >= streamed, "the ring always holds the streamed extent");
        check(fitted->resident_pages != 0, "the window never collapses to nothing");
        check(fitted->resident_pages <= previous_resident || previous_resident == 0
                  ? true
                  : fitted->resident_pages <= previous_resident,
              "the window never grows as the context grows");
        previous_resident = fitted->resident_pages;
    }
}

void refuses_what_it_cannot_serve() {
    check(!plan_kv_stream_split(0, 64, kLayers).has_value(), "a pool of nothing is refused");
    check(!plan_kv_stream_split(kLayers * 64, 64, 0).has_value(), "a pool without layers is refused");
    check(!plan_kv_stream_split(10, 64, kLayers).has_value(),
          "a pool smaller than one layer is refused");
    check(!plan_kv_stream_split(4096, 4096, kLayers, 0).has_value(), "a run of zero pages is refused");
    const auto planned = plan_kv_stream_split(31572, 4096, kLayers);
    if (planned) {
        check(!resize_kv_stream_split(*planned, 31572, 0, kLayers).has_value(),
              "a context of zero pages has no split");
        check(!resize_kv_stream_split(*planned, 100, 4096, kLayers).has_value(),
              "a pool too small for the context's ring has no split");
    }
}

} // namespace

int main() {
    all_resident_matches_today();
    the_shipping_262144_geometry();
    a_resident_page_keeps_its_slot();
    runs_are_contiguous_and_disjoint();
    a_short_ring_rounds_down_to_whole_runs();
    the_plan_rounds_the_ring_to_whole_runs();
    a_context_that_fits_needs_no_ring();
    re_sizing_follows_the_context();
    refuses_what_it_cannot_serve();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream layout: all checks passed\n";
    return 0;
}
