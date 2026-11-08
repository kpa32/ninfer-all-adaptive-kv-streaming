#include "core/kv_stream_layout.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <set>

namespace {

using ninfer::KVStreamSplit;
using ninfer::kv_stream_pool_pages;
using ninfer::kv_stream_ring_extent;
using ninfer::kv_stream_ring_slot;
using ninfer::kv_stream_slot;
using ninfer::kv_stream_window_slot;
using ninfer::plan_kv_stream_split;

// One line per distinct failure, however many frontiers reach it: a broken mapping would otherwise
// print once per page and hide everything else.
std::map<std::string, int> failures;

void check(bool condition, const char* message) {
    if (!condition) { ++failures[message]; }
}

// The all-resident split must reproduce the layout the paged cache has today: layer L's page p is
// slot L * pages_per_layer + p, so switching to the flat pool with no streaming cannot move a byte.
void all_resident_matches_today() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 64;
    const auto split = plan_kv_stream_split(kLayers * kPages, kPages, kLayers);
    check(split.has_value(), "a full budget must plan");
    check(split->resident_pages == kPages && split->ring_pages == 0,
          "a full budget keeps every page resident");
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t page = 0; page < kPages; ++page) {
            const auto slot = kv_stream_slot(layer, page, kPages, *split, kLayers);
            check(slot == static_cast<std::int32_t>(layer * kPages + page),
                  "an all-resident slot is the paged layout's page index");
        }
    }
}

// The 262 144 shape measured on the card: 4096 pages per layer, 16 full-attention layers, and the
// 2.64 GB a 16 GB card has left after weights and the MTP pool. The pool holds 6.2 contexts, so the
// ring buys its second region and costs 13.5% of the window for it.
void the_measured_262144_split() {
    constexpr std::uint32_t kBudget = 25300;
    constexpr std::uint32_t kPages  = 4096;
    constexpr std::uint32_t kLayers = 16;
    const auto split = plan_kv_stream_split(kBudget, kPages, kLayers);
    check(split.has_value(), "262 144 must plan");
    check(split->ring_depth == 2, "262 144 keeps two layers' streamed pages in flight");
    check(split->resident_pages == 1222, "262 144 keeps 1222 pages of each layer resident");
    check(split->ring_pages == 5748, "262 144 shares a 5748-slot ring, two regions deep");
    check(kv_stream_ring_extent(*split, kLayers) == kPages - split->resident_pages,
          "a region is exactly one layer's streamed pages");
    check(kv_stream_pool_pages(*split, kLayers) <= kBudget, "the split fits the budget");
    // The largest feasible resident window for two regions: one more page per layer would not fit.
    const std::uint64_t one_more = kv_stream_pool_pages(
        KVStreamSplit{split->resident_pages + 1,
                      split->ring_depth * (kPages - split->resident_pages - 1), split->ring_depth},
        kLayers);
    check(one_more > kBudget, "the split takes every page the budget can keep resident");
}

// Two properties the attention depends on, at every frontier: one layer's pages never share a
// slot, resident slots never overlap the ring, and no slot leaves the pool.
// A page keeps the slot it was given for as long as it is resident, and takes the ring slot for
// good once it leaves: two table writes in a page's life, not one per page boundary.
void a_page_changes_slot_at_most_twice() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 256;
    constexpr std::uint32_t kBudget = 3000;
    const auto split                = plan_kv_stream_split(kBudget, kPages, kLayers);
    check(split.has_value(), "the streaming budget must plan");
    if (!split) { return; }
    const std::uint32_t resident = split->resident_pages;
    for (std::uint32_t layer = 0; layer < 1; ++layer) {
        for (std::uint32_t page = 0; page < kPages; ++page) {
            std::set<std::int32_t> seen;
            for (std::uint32_t active = page + 1; active <= kPages; ++active) {
                const auto slot = kv_stream_slot(layer, page, active, *split, kLayers);
                seen.insert(slot);
                const bool resident_now = page + resident >= active;
                if (resident_now) {
                    check(slot == kv_stream_window_slot(layer, page, *split),
                          "a resident page keeps its window slot as the context grows");
                } else {
                    check(slot == kv_stream_ring_slot(layer, page, *split, kLayers),
                          "a streamed page maps to its ring slot for good");
                }
            }
            check(seen.size() <= 2, "a page changes slot at most once");
        }
    }
}

void slots_stay_inside_their_regions() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 96;
    for (std::uint32_t budget = kPages; budget <= kLayers * kPages; budget += 7) {
        const auto split = plan_kv_stream_split(budget, kPages, kLayers);
        check(split.has_value(), "any budget that holds one layer must plan");
        const std::uint32_t pool = kv_stream_pool_pages(*split, kLayers);
        check(pool <= budget, "the split never exceeds its budget");
        for (std::uint32_t active = 1; active <= kPages; ++active) {
            const std::uint32_t window_start =
                active > split->resident_pages ? active - split->resident_pages : 0;
            check(window_start <= kv_stream_ring_extent(*split, kLayers),
                  "a layer's streamed pages fit one ring region");
            for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
                std::set<std::int32_t> slots;
                for (std::uint32_t page = 0; page < active; ++page) {
                    const auto slot = kv_stream_slot(layer, page, active, *split, kLayers);
                    check(slot >= 0 && static_cast<std::uint32_t>(slot) < pool,
                          "a slot stays inside the pool");
                    check(slots.insert(slot).second, "two pages of one layer never share a slot");
                    if (page >= window_start) {
                        check(slot >= static_cast<std::int32_t>(layer * split->resident_pages) &&
                                  slot < static_cast<std::int32_t>((layer + 1) * split->resident_pages),
                              "a resident page lands in its layer's own region");
                    } else {
                        check(slot >= static_cast<std::int32_t>(kLayers * split->resident_pages),
                              "a streamed page lands in the shared ring");
                        // Two layers attend at once only if the ring is deeper than the distance
                        // between them: a page's region is its layer's, so the regions in flight
                        // never coincide.
                        const std::uint32_t extent = kv_stream_ring_extent(*split, kLayers);
                        const std::uint32_t region =
                            (static_cast<std::uint32_t>(slot) - kLayers * split->resident_pages) /
                            extent;
                        check(region == layer % split->ring_depth,
                              "a streamed page lands in its layer's own ring region");
                    }
                }
                // The identity mapping is what keeps the streamed half of every table static.
                if (split->resident_pages == 0) {
                    for (std::uint32_t page = 0; page < active; ++page) {
                        check(kv_stream_slot(layer, page, active, *split, kLayers) ==
                                  static_cast<std::int32_t>(page),
                              "with no resident pages a page is its own slot");
                    }
                }
            }
        }
    }
}

void refuses_what_it_cannot_serve() {
    check(!plan_kv_stream_split(4095, 4096, 16).has_value(),
          "a budget below one layer is refused, not degraded");
    check(!plan_kv_stream_split(25300, 4096, 0).has_value(), "no layers is refused");
    check(!plan_kv_stream_split(25300, 0, 16).has_value(), "no pages is refused");
    const auto single = plan_kv_stream_split(4096, 4096, 1);
    check(single.has_value() && single->resident_pages == 4096 && single->ring_pages == 0,
          "one layer that fits needs no ring");
    // No budget above one layer's pages can be refused for a single layer.
    check(plan_kv_stream_split(5000, 4096, 1).has_value(), "one layer above its pages still plans");
}

void a_bigger_budget_never_residents_less() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 512;
    std::uint32_t previous       = 0;
    std::uint32_t previous_depth = 1;
    for (std::uint32_t budget = kPages; budget <= kLayers * kPages; ++budget) {
        const auto split = plan_kv_stream_split(budget, kPages, kLayers);
        check(split.has_value(), "every budget above one layer plans");
        if (!split) { continue; }
        // A deeper ring spends window on its extra region; that one step is the only one allowed
        // to shrink it, and it must not cost more than a fifth of what it had.
        if (split->ring_depth == previous_depth || previous == 0) {
            check(split->resident_pages >= previous, "resident pages never shrink as the budget grows");
        } else {
            check(split->resident_pages * 5 >= previous * 4,
                  "a deeper ring costs at most a fifth of the window");
        }
        check(split->resident_pages * split->ring_depth + split->ring_pages ==
                  split->ring_depth * kPages,
              "every region plus the window is the layer's capacity");
        previous       = split->resident_pages;
        previous_depth = split->ring_depth;
    }
}

} // namespace

int main() {
    all_resident_matches_today();
    the_measured_262144_split();
    a_page_changes_slot_at_most_twice();
    slots_stay_inside_their_regions();
    refuses_what_it_cannot_serve();
    a_bigger_budget_never_residents_less();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream layout: all checks passed\n";
    return 0;
}
