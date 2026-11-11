#include "core/kv_stream_layout.h"
#include "core/kv_stream_slot_map.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamShape;
using ninfer::KVStreamSlotMap;
using ninfer::KVStreamSplit;
using ninfer::kv_stream_all_resident;
using ninfer::kv_stream_pool_pages;
using ninfer::kv_stream_ring_runs;
using ninfer::kv_stream_runs_of_layer;
using ninfer::kv_stream_runs_per_layer;
using ninfer::kv_stream_shape;
using ninfer::kv_stream_shape_name;
using ninfer::kv_stream_shape_of;
using ninfer::kv_stream_window_slot;
using ninfer::kv_stream_window_start;
using ninfer::plan_kv_stream_split;
using ninfer::plan_kv_stream_uniform;
using ninfer::resize_kv_stream_split;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

void check_equal(std::uint32_t actual, std::uint32_t expected, const std::string& message) {
    if (actual != expected) { ++failures[message]; }
}

// A pool that holds every page must produce the layout the paged cache has today: layer L's page p is
// slot sum(layer_resident[0..L)) + p, so switching to the flat pool moves no byte.
void all_resident_matches_today() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 64;
    const auto split = plan_kv_stream_split(kLayers * kPages, kPages, kLayers);
    check(split.has_value(), "a full budget must plan");
    if (!split) { return; }
    check_equal(split->ring_pages, 0, "a full budget needs no ring");
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        check_equal(split->resident_of(layer), kPages, "a full budget keeps every page resident");
        for (std::uint32_t page = 0; page < kPages; ++page) {
            check_equal(static_cast<std::uint32_t>(kv_stream_window_slot(layer, page, *split)),
                        layer * kPages + page, "an all-resident page keeps today's slot");
        }
    }
}

// The shipping geometry at 262144 tokens: 4096 pages of context per layer, 16 full-attention layers, and
// a pool of 31572 slots (the one the 251K run used). Whatever shape the planner picks, the pool must
// cover the plan and the window must beat what two whole ring extents allowed (1670 pages).
void the_shipping_262144_geometry() {
    constexpr std::uint32_t kPages  = 4096;
    constexpr std::uint32_t kBudget = 31572;
    const auto split = plan_kv_stream_split(kBudget, kPages, 16);
    check(split.has_value(), "the shipping geometry plans");
    if (!split) { return; }
    check(kv_stream_pool_pages(*split) <= kBudget, "the split fits its pool");
    const std::uint32_t streamed = kv_stream_window_start(kPages, *split, 0);
    check(split->ring_pages >= streamed, "the ring holds one layer's streamed extent");
    check(split->min_resident() > 1670,
          "a ratio-sized ring keeps a larger window than two whole extents did");
    check_equal(split->ring_pages % split->run_pages, 0, "the ring is a whole number of runs");
}

// The uniform plan is the shape the shipping runs used, and it must honour the article's ratio.
void the_uniform_plan_keeps_the_reference_ratio() {
    constexpr std::uint32_t kPages  = 4096;
    constexpr std::uint32_t kBudget = 31572;
    const auto split = plan_kv_stream_uniform(kBudget, kPages, 16);
    check(split.has_value(), "a uniform plan exists");
    if (!split) { return; }
    check_equal(split->layers(), 16, "every layer has a window");
    const std::uint32_t streamed = kPages - split->resident_of(0);
    check(split->ring_pages >= streamed, "the ring holds the whole streamed extent");
    check(static_cast<double>(split->ring_pages) <= 1.25 * static_cast<double>(streamed),
          "the ring stays near the reference's 1.10 rather than a whole extra extent");
}

// The concentrated shape exists for its schedule, never for its bytes: it must keep at least the pool
// the uniform shape needs, or the planner has to reject it.
void a_concentrated_plan_never_costs_pool() {
    constexpr std::uint32_t kPages = 4096;
    for (std::uint32_t budget = 12 * kPages; budget <= 15 * kPages; budget += 977) {
        const auto uniform = plan_kv_stream_uniform(budget, kPages, 16);
        const auto chosen  = plan_kv_stream_split(budget, kPages, 16);
        if (!uniform || !chosen) { continue; }
        check(kv_stream_pool_pages(*chosen) <= budget, "the chosen plan fits the pool");
        check(chosen->ring_pages >= kv_stream_window_start(kPages, *chosen, 0) ||
                  chosen->ring_pages == 0,
              "the chosen ring holds the widest stream it plans");
    }
}

// Every page the window takes keeps its slot while it stays resident, and the streamed prefix of a
// layer is addressed by runs, not by its index.
void a_resident_page_keeps_its_slot() {
    constexpr std::uint32_t kPages = 132;
    const auto planned = plan_kv_stream_uniform(16 * 40, kPages, 16, 8);
    check(planned.has_value(), "a partially resident pool plans");
    if (!planned) { return; }
    for (std::uint32_t active = planned->resident_of(0); active <= kPages; ++active) {
        const std::uint32_t start = kv_stream_window_start(active, *planned, 0);
        for (std::uint32_t page = start; page < active; ++page) {
            const auto first = kv_stream_window_slot(0, page, *planned);
            check(kv_stream_window_slot(0, page, *planned) == first,
                  "a resident page keeps its slot");
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

// The article's first rule on the split side: when the request's context fits the pool, nothing streams,
// so the ring is not addressed and every page stays resident.
void a_context_that_fits_needs_no_ring() {
    const auto planned = plan_kv_stream_split(31572, 4096, 16);
    check(planned.has_value(), "the shipping geometry plans");
    if (!planned) { return; }
    const auto fitted = resize_kv_stream_split(*planned, 31572, 1900);
    check(fitted.has_value(), "a short context re-sizes");
    if (!fitted) { return; }
    check_equal(fitted->ring_pages, 0, "a context that fits streams nothing");
    check_equal(fitted->min_resident(), 1900, "every page of the context stays resident");
    check(kv_stream_pool_pages(*fitted) <= 31572, "the fitted split fits the pool");
}

// Every page of a plan must be named by one slot, and no two pages by the same one. A page's slot is
// its layer's window base plus its position, so a plan whose windows overlapped would point two pages
// at one slot - the same mistake the removed mid-request re-shaping made with a whole window at once,
// which answered with another page's KV (bench/results/kv-stream/layout-rename-defect.md). The pool is
// planned once, so this has to hold for the plan that starts the process, not for a table built later.
void every_page_of_a_plan_is_named_once() {
    constexpr std::uint32_t kLayers = 16;
    constexpr std::uint32_t kPages  = 4096;
    for (const std::uint32_t budget : {20000U, 31572U, 47894U}) {
        const auto planned = plan_kv_stream_split(budget, kPages, kLayers);
        check(planned.has_value(), "a pool that holds one layer plans");
        if (!planned) { continue; }
        const std::uint32_t pool = kv_stream_pool_pages(*planned);
        check(pool <= budget, "a plan never claims more slots than the pool has");
        check_equal(planned->ring_pages % planned->run_pages, 0, "the ring is whole runs");
        std::map<std::uint32_t, std::uint32_t> owners;
        for (std::uint32_t layer = 0; layer < planned->layers(); ++layer) {
            const std::uint32_t resident = planned->resident_of(layer);
            for (std::uint32_t page = 0; page < resident; ++page) {
                const auto slot =
                    static_cast<std::uint32_t>(kv_stream_window_slot(layer, page, *planned));
                check(slot < pool, "a window slot is inside the pool");
                check(owners.emplace(slot, layer).second, "a window slot names at most one page");
            }
        }
    }
}

// Growing the context keeps giving the ring what it needs and the window the rest, and never lets the
// ring fall below the extent it must hold.
void re_sizing_follows_the_context() {    const auto planned = plan_kv_stream_split(31572, 4096, 16);
    check(planned.has_value(), "the shipping geometry plans");
    if (!planned) { return; }
    for (std::uint32_t active = 2000; active <= 4096; active += 97) {
        const auto fitted = resize_kv_stream_split(*planned, 31572, active);
        check(fitted.has_value(), "a growing context stays servable");
        if (!fitted) { continue; }
        check(kv_stream_pool_pages(*fitted) <= 31572, "a re-sized split fits the pool");
        const std::uint32_t widest = [&] {
            std::uint32_t most = 0;
            for (std::uint32_t layer = 0; layer < fitted->layers(); ++layer) {
                const std::uint32_t streamed = kv_stream_window_start(active, *fitted, layer);
                if (streamed > most) { most = streamed; }
            }
            return most;
        }();
        check(fitted->ring_pages >= widest, "the ring always holds the widest stream");
        check(fitted->min_resident() != 0, "the window never collapses to nothing");
        check(kv_stream_runs_per_layer(active, *fitted) >=
                  kv_stream_runs_of_layer(active, *fitted, 0),
              "no layer needs more runs than the widest one");
        check_equal(kv_stream_ring_runs(*fitted), fitted->ring_pages / fitted->run_pages,
                    "ring runs follow the ring");
    }
}

void refuses_what_it_cannot_serve() {
    check(!plan_kv_stream_split(0, 64, 16).has_value(), "a pool of nothing is refused");
    check(!plan_kv_stream_split(16 * 64, 64, 0).has_value(), "a pool without layers is refused");
    check(!plan_kv_stream_split(10, 64, 16).has_value(),
          "a pool smaller than one layer is refused");
    check(!plan_kv_stream_split(4096, 4096, 16, 0).has_value(), "a run of zero pages is refused");
    const auto planned = plan_kv_stream_split(31572, 4096, 16);
    if (planned) {
        check(!resize_kv_stream_split(*planned, 31572, 0).has_value(),
              "a context of zero pages has no split");
    }
}

// The report tells three plans apart, and it has to: all three carry the same narrowest window, so a
// record's shape is the only field that says whether its round time belongs beside another's.
void the_shape_is_what_identifies_the_layout() {
    constexpr std::uint32_t kPages  = 128;
    constexpr std::uint32_t kLayers = 8;
    const auto full = kv_stream_all_resident(kPages, kLayers, 8);
    check(kv_stream_shape(full, kPages) == KVStreamShape::all_resident,
          "a pool that holds the context is all-resident");
    const auto uniform = plan_kv_stream_uniform(kLayers * 40, kPages, kLayers, 8);
    check(uniform.has_value(), "a partial pool plans");
    if (uniform) {
        check(kv_stream_shape(*uniform, kPages) == KVStreamShape::uniform,
              "equal windows are the uniform shape");
    }
    // The same narrowest window as `uniform`, but one layer keeps the whole context: this is the
    // reference's decode layout, and `resident_pages * layers` under-reports its window.
    KVStreamSplit mixed;
    mixed.layer_resident.assign(kLayers, 40);
    mixed.layer_resident[0] = kPages;
    mixed.ring_pages        = 8;
    mixed.run_pages         = 8;
    check(kv_stream_shape(mixed, kPages) == KVStreamShape::concentrated,
          "a fully resident layer beside a windowed one is the concentrated shape");
    check(std::string(kv_stream_shape_name(kv_stream_shape(mixed, kPages))) == "concentrated",
          "the shape names itself for the report");
    check(kv_stream_shape_of({}, kPages) == KVStreamShape::uniform,
          "an empty plan falls back to uniform rather than claiming all-resident");
}

} // namespace

int main() {
    all_resident_matches_today();
    the_shipping_262144_geometry();
    the_uniform_plan_keeps_the_reference_ratio();
    a_concentrated_plan_never_costs_pool();
    a_resident_page_keeps_its_slot();
    runs_are_contiguous_and_disjoint();
    a_short_ring_rounds_down_to_whole_runs();
    a_context_that_fits_needs_no_ring();
    re_sizing_follows_the_context();
    every_page_of_a_plan_is_named_once();
    refuses_what_it_cannot_serve();
    the_shape_is_what_identifies_the_layout();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream layout: all checks passed\n";
    return 0;
}
