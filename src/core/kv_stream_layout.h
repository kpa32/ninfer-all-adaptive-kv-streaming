#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

namespace ninfer {

// The fixed split of a KV pool whose authority is pinned host memory
// (docs/maintainer/kv-streaming-fidelity.md).
//
// Two pools in one flat slot budget:
//
//   * a **window** of the newest pages of every streaming layer, which stays on the device;
//   * one **ring** of runs, shared by those layers, helding the older pages while a layer reads them.
//
// `layer_resident[i]` is layer `i`'s window, so the window is per layer and not one number: the
// reference concentrates a decode round's streaming into a few layers and leaves the rest fully
// resident (`kv_stream_resident_cache_layout`), which changes how many layers copy per round, not how
// many bytes travel - the pool identity fixes those. The layout below supports both shapes, and the
// planner picks the one whose ring is smallest for the context at hand.
//
//     sum(layer_resident) + ring_pages <= budget
//     ring_pages >= ring_ratio * max(active - layer_resident[i])        (the extent one call reads)
//
// A ring of one streamed extent can only hold one layer at a time, so a layer's copies cannot overlap
// the previous layer's attention; `ring_ratio` above one (the article's 1.10, the reference's
// `target_ring_working_set_ratio`) lets the next layer's copies start while the current one is still
// reading. Every slot the ring does not need is a page that stays resident and is not copied every
// round.
struct KVStreamSplit {
    // Window of each streaming layer, in layer order.
    std::vector<std::uint32_t> layer_resident;
    // Slots shared by every streaming layer, a whole number of runs.
    std::uint32_t ring_pages = 0;
    // Pages one copy (and one consumed span) covers: a batch the copy engine likes and the consumer
    // can dispatch without waiting for the whole layer. The reference's 32 pages of 256 tokens is 128
    // pages of this engine's 64-token pages by byte count.
    std::uint32_t run_pages = 1;

    [[nodiscard]] std::uint32_t layers() const noexcept {
        return static_cast<std::uint32_t>(layer_resident.size());
    }
    // The narrowest window any layer keeps: the floor the shared helpers use.
    [[nodiscard]] std::uint32_t min_resident() const noexcept {
        if (layer_resident.empty()) { return 0; }
        std::uint32_t least = layer_resident.front();
        for (const std::uint32_t pages : layer_resident) {
            if (pages < least) { least = pages; }
        }
        return least;
    }
    // The window a resident page is named in.
    [[nodiscard]] std::uint32_t resident_pages() const noexcept { return min_resident(); }
    // First slot of layer `layer`'s window.
    [[nodiscard]] std::uint32_t layer_offset(std::uint32_t layer) const noexcept {
        std::uint32_t offset = 0;
        for (std::uint32_t i = 0; i < layer && i < layers(); ++i) { offset += layer_resident[i]; }
        return offset;
    }
    [[nodiscard]] std::uint32_t resident_of(std::uint32_t layer) const noexcept {
        return layer < layers() ? layer_resident[layer] : 0;
    }
};

// Which shape a split has. Three plans read the same `resident_pages` (the narrowest window), so the
// report has to carry the shape as well: only it says whether a record ran the reference's decode
// layout (streams concentrated into a few layers, the rest fully resident) or the uniform one, and
// only it tells a reader whether `resident_pages * layers` is the window's size at all.
enum class KVStreamShape { all_resident, uniform, concentrated };

[[nodiscard]] inline KVStreamShape kv_stream_shape_of(const std::vector<std::uint32_t>& layer_resident,
                                                      std::uint32_t active_pages) noexcept {
    if (layer_resident.empty()) { return KVStreamShape::uniform; }
    const std::uint32_t first = layer_resident.front();
    bool same              = true;
    bool any_streaming     = false;
    for (const std::uint32_t pages : layer_resident) {
        if (pages != first) { same = false; }
        if (pages < active_pages) { any_streaming = true; }
    }
    if (!any_streaming) { return KVStreamShape::all_resident; }
    return same ? KVStreamShape::uniform : KVStreamShape::concentrated;
}

[[nodiscard]] inline KVStreamShape kv_stream_shape(const KVStreamSplit& split,
                                                   std::uint32_t active_pages) noexcept {
    return kv_stream_shape_of(split.layer_resident, active_pages);
}

[[nodiscard]] inline const char* kv_stream_shape_name(KVStreamShape shape) noexcept {
    switch (shape) {
        case KVStreamShape::all_resident: return "all-resident";
        case KVStreamShape::concentrated: return "concentrated";
        case KVStreamShape::uniform: break;
    }
    return "uniform";
}

// Ring slots relative to one layer's streamed pages. Above one leaves room to begin the next layer
// early; the reference's measured default is 1.10 (`llama-kv-stream-plan.h`). The default is a
// parameter everywhere because the ratio is a measured trade on the card in hand (bytes against
// overlap), not a property of the design: this fleet's numbers are in
// docs/maintainer/kv-streaming-measurements.md.
constexpr double kKVStreamRingRatio = 1.10;

// Pages a run covers when the ring is planned without an explicit choice.
constexpr std::uint32_t kKVStreamRunPages = 128;


// The run width a plan actually uses. `kKVStreamRunPages` is the default the design ships with; the
// environment variable exists so the width can be swept on a host without a rebuild, because the width
// is a *measured* trade - a run is the unit of copy and of look-ahead, so a narrower run overlaps
// sooner and costs more launches, a wider one does the opposite - and the answer depends on the card,
// the cache format and the ring. Every run record carries the value it ran with (`stream.run_pages`),
// so a measurement can never be read against the wrong geometry.
[[nodiscard]] inline std::uint32_t kv_stream_run_pages() noexcept {
    if (const char* env = std::getenv("NINFER_STREAM_RUN_PAGES")) {
        const long value = std::strtol(env, nullptr, 10);
        if (value >= 8 && value <= 4096) { return static_cast<std::uint32_t>(value); }
    }
    return kKVStreamRunPages;
}

// The ring ratio a plan actually uses. `kKVStreamRingRatio` is the measured default; the environment
// variable exists so the ring can be driven to its extremes on a host without a rebuild - a ratio at
// or above the pool's page count plans a zero window, which is the boundary the delivery must prove
// runs and stays bit-exact, and which no product flag reaches once the delivery no longer exposes one.
// Every record carries the split it ran with (`stream.layer_resident`, `stream.ring_runs`).
[[nodiscard]] inline double kv_stream_ring_ratio() noexcept {
    if (const char* env = std::getenv("NINFER_STREAM_RING_RATIO")) {
        const double value = std::strtod(env, nullptr);
        if (value >= 1.0 && value <= 1024.0) { return value; }
    }
    return kKVStreamRingRatio;
}

// Slots the pool holds for this split.
[[nodiscard]] inline std::uint32_t kv_stream_pool_pages(const KVStreamSplit& split) noexcept {
    std::uint32_t total = split.ring_pages;
    for (const std::uint32_t pages : split.layer_resident) { total += pages; }
    return total;
}

// The page layer `layer`'s window starts at: pages below it stream, pages from it up are resident.
[[nodiscard]] inline std::uint32_t kv_stream_window_start(std::uint32_t active_pages,
                                                          const KVStreamSplit& split,
                                                          std::uint32_t layer) noexcept {
    const std::uint32_t resident = split.resident_of(layer);
    return active_pages > resident ? active_pages - resident : 0U;
}

// The slot a resident page keeps for as long as it stays in its layer's window, so the table entry
// written when the page appeared needs no rewrite until it leaves.
[[nodiscard]] inline std::int32_t kv_stream_window_slot(std::uint32_t layer,
                                                        std::uint32_t logical_page,
                                                        const KVStreamSplit& split) noexcept {
    const std::uint32_t resident = split.resident_of(layer);
    const std::uint32_t offset   = split.layer_offset(layer);
    return static_cast<std::int32_t>(offset + (resident == 0 ? 0 : logical_page % resident));
}

// Runs layer `layer`'s streamed pages occupy, `ceil(window_start / run_pages)`.
[[nodiscard]] inline std::uint32_t kv_stream_runs_of_layer(std::uint32_t active_pages,
                                                           const KVStreamSplit& split,
                                                           std::uint32_t layer) noexcept {
    if (split.run_pages == 0) { return 0; }
    const std::uint32_t streamed = kv_stream_window_start(active_pages, split, layer);
    return (streamed + split.run_pages - 1) / split.run_pages;
}


// The most runs any layer's streamed pages take: what the ring has to hold at once.
[[nodiscard]] inline std::uint32_t kv_stream_runs_per_layer(std::uint32_t active_pages,
                                                            const KVStreamSplit& split) noexcept {
    std::uint32_t most = 0;
    for (std::uint32_t layer = 0; layer < split.layers(); ++layer) {
        const std::uint32_t runs = kv_stream_runs_of_layer(active_pages, split, layer);
        if (runs > most) { most = runs; }
    }
    return most;
}

// Runs the ring holds, `ring_pages / run_pages`.
[[nodiscard]] inline std::uint32_t kv_stream_ring_runs(const KVStreamSplit& split) noexcept {
    return split.run_pages == 0 ? 0 : split.ring_pages / split.run_pages;
}

// How much of a layer's streamed extent the ring can hold *on top of* the layer being read, as a
// ratio: `ring_runs / runs_per_layer`. One means the ring holds exactly one layer's extent, so a copy
// cannot start before the previous layer has been read and the round costs compute + copy instead of
// the larger of the two; 1.4 means two fifths of the next layer can be copied under the current one's
// compute. An integer "whole layers ahead" is too coarse to see that difference - at 251K, eleven runs
// against eight and nine runs against eight are both "zero layers ahead" yet measure 54.9 ms and
// 60.6 ms a round.
[[nodiscard]] inline double kv_stream_lookahead_ratio(std::uint32_t active_pages,
                                                     const KVStreamSplit& split) noexcept {
    const std::uint32_t runs = kv_stream_runs_per_layer(active_pages, split);
    const std::uint32_t ring = kv_stream_ring_runs(split);
    if (runs == 0 || ring == 0) { return 0.0; }
    return static_cast<double>(ring) / static_cast<double>(runs);
}


namespace detail {

[[nodiscard]] inline std::uint32_t kv_stream_ceil_to_run(std::uint64_t pages,
                                                         std::uint32_t run_pages) noexcept {
    if (run_pages == 0) { return static_cast<std::uint32_t>(pages); }
    const std::uint64_t rounded = pages + run_pages - 1;
    return static_cast<std::uint32_t>(rounded - rounded % run_pages);
}

} // namespace detail

// The all-resident split of a pool that holds every page of every layer: no ring, and layer `i`'s
// page `p` is slot `sum(layer_resident[0..i)) + p`, which is the layout the paged cache has today.
[[nodiscard]] inline KVStreamSplit kv_stream_all_resident(std::uint32_t active_pages,
                                                          std::uint32_t layers,
                                                          std::uint32_t run_pages) noexcept {
    KVStreamSplit split;
    split.layer_resident.assign(layers, active_pages);
    split.run_pages = run_pages;
    return split;
}

// The uniform split: every streaming layer keeps the same window and streams the rest. This is the
// shape the shipping runs used, and the fallback whenever concentrating the streams would not shrink
// the ring.
//
//     ring(R) = ceil_to_run(max(ring_ratio, 1) * (active - R))
//     feasible: layers * R + ring(R) <= budget
//
// The left side grows with R (the window term grows faster than the ring term shrinks, because the
// ring covers at most one layer's extent), so the largest feasible window is found by bisection
// rather than guessed by a fixed-point pass - which is what a pool that can only hold a ring and no
// window needs, the degenerate plan a capacity scan evaluates first.
//
// The run size is a request, not a promise: the ring must be a whole number of runs and hold a whole
// layer's streamed extent, so a pool too small to do that at `run_pages` is planned at the largest run
// it can actually hold. The split carries the run size it was planned with, so the copy and its
// consumer agree.
[[nodiscard]] inline std::optional<KVStreamSplit>
plan_kv_stream_uniform(std::uint32_t budget_pages, std::uint32_t active_pages,
                       std::uint32_t layers, std::uint32_t run_pages = kKVStreamRunPages,
                       double ring_ratio = kKVStreamRingRatio) noexcept {
    if (layers == 0 || active_pages == 0 || run_pages == 0) { return std::nullopt; }
    if (std::uint64_t(layers) * active_pages <= budget_pages) {
        return kv_stream_all_resident(active_pages, layers, run_pages);
    }
    if (!(ring_ratio >= 1.0)) { ring_ratio = 1.0; }
    // The ratio is a target: a pool that cannot afford a ring that deep even with no window plans at
    // 1.0 instead, which is the depth-one regime - a ring that holds exactly one layer's extent, so the
    // layers serialise. Refusing outright would make a capacity scan give up on a pool it can serve.
    const double ratios[2] = {ring_ratio, 1.0};
    for (const double ratio : ratios)
    for (std::uint32_t run = run_pages; run >= 1; run = run > 1 ? run / 2 : 0) {
        const auto ring_for = [&](std::uint32_t resident) {
            if (resident >= active_pages) { return std::uint32_t(0); }
            const std::uint64_t streamed = active_pages - resident;
            std::uint64_t ring = std::uint64_t(ratio * double(streamed));
            if (ring < streamed) { ring = streamed; }
            return detail::kv_stream_ceil_to_run(ring, run);
        };
        const auto fits = [&](std::uint32_t resident) {
            const std::uint64_t ring = ring_for(resident);
            return std::uint64_t(layers) * resident + ring <= budget_pages;
        };
        if (!fits(0)) {
            if (run == 1) { break; }
            continue;
        }
        // Bisect for the largest feasible window; the predicate is monotone over [0, active_pages].
        std::uint32_t low  = 0;
        std::uint32_t high = active_pages;
        while (low < high) {
            const std::uint32_t mid = low + (high - low + 1) / 2;
            if (fits(mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        // A window of zero pages is a legitimate plan: it is what a pool that holds only one layer's
        // streamed extent can serve.
        KVStreamSplit split;
        split.layer_resident.assign(layers, low);
        split.ring_pages = ring_for(low);
        split.run_pages  = run;
        return split;
    }
    return std::nullopt;
}

// The smallest ring that lets one layer's copy run under the previous layer's attention: two extents of
// the widest streamed layer, rounded up to a whole number of runs.
//
// Why this is a planning input and not a constant: a ring that holds exactly one extent cannot start a
// copy before its reader finishes, so the round costs copies *plus* attention instead of the larger of
// the two. At 251K on this card those two numbers are 46.1 ms of copies and ~8.3 ms of attention per
// round - 54.4 ms measured, i.e. no overlap at all - so a deep ring is worth up to 15% of a round. What
// it costs is the window it displaces: at a fixed pool a deeper ring means less residency, and at 251K
// `rk4v4` that is 44 pages per layer, about +4% of a round's bytes. Whether that trade pays is a
// measurement, not a property of the design, so the space proposes it as a trial and keeps it only if
// the round actually gets faster (kv_stream_space.cpp, evaluate_partition).
//
// Returns nothing when the pool cannot afford two extents even with a zero window, or when the model has
// too few streaming layers for the window term to dominate the ring's run rounding: then the depth-one
// regime is the only plan and the caller keeps the byte-minimal one.
[[nodiscard]] inline std::optional<KVStreamSplit>
plan_kv_stream_lookahead(std::uint32_t budget_pages, std::uint32_t active_pages,
                         std::uint32_t layers, std::uint32_t run_pages = kKVStreamRunPages) noexcept {
    if (layers < 4 || active_pages == 0 || run_pages == 0) { return std::nullopt; }
    if (std::uint64_t(layers) * active_pages <= budget_pages) { return std::nullopt; }
    const auto ring_for = [&](std::uint32_t resident) {
        if (resident >= active_pages) { return std::uint32_t(0); }
        const std::uint64_t streamed = active_pages - resident;
        return static_cast<std::uint32_t>(
            detail::kv_stream_ceil_to_run(detail::kv_stream_ceil_to_run(streamed, run_pages) * 2, run_pages));
    };
    const auto fits = [&](std::uint32_t resident) {
        return std::uint64_t(layers) * resident + ring_for(resident) <= budget_pages;
    };
    if (!fits(0)) { return std::nullopt; }
    std::uint32_t low  = 0;
    std::uint32_t high = active_pages - 1;
    while (low < high) {
        const std::uint32_t mid = low + (high - low + 1) / 2;
        if (fits(mid)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    KVStreamSplit split;
    split.layer_resident.assign(layers, low);
    split.ring_pages = ring_for(low);
    split.run_pages  = run_pages;
    if (kv_stream_ring_runs(split) < 2 * kv_stream_runs_per_layer(active_pages, split)) {
        return std::nullopt;
    }
    return split;
}

// The decode split the reference builds (`kv_stream_resident_cache_layout`): the same pool, but only
// `split_layers` layers stream and every other layer keeps its whole context resident. A layer that
// streams then reads a wider extent, so the ring must hold more per call and the window loses what the
// ring gains - which is why this is chosen only when it does not cost pool, and never for its bytes.
[[nodiscard]] inline std::optional<KVStreamSplit>
plan_kv_stream_concentrated(std::uint32_t budget_pages, std::uint32_t active_pages,
                            std::uint32_t layers, std::uint32_t run_pages, double ring_ratio,
                            std::uint32_t split_layers) noexcept {
    if (layers == 0 || active_pages == 0 || run_pages == 0 || split_layers == 0 ||
        split_layers > layers) {
        return std::nullopt;
    }
    if (!(ring_ratio >= 1.0)) { ring_ratio = 1.0; }
    // Each split layer streams `base` (or `base + 1`) of `streamed` pages, spread across model order
    // so the fully resident layers between them give the prefetch windows the reference relies on.
    for (std::uint64_t streamed = 1; streamed <= std::uint64_t(layers) * active_pages; ++streamed) {
        const std::uint64_t base      = streamed / split_layers;
        const std::uint64_t remainder = streamed % split_layers;
        const std::uint64_t widest    = base + (remainder != 0 ? 1 : 0);
        if (widest > active_pages) { break; }
        const std::uint32_t ring = detail::kv_stream_ceil_to_run(
            std::uint64_t(ring_ratio * double(widest) + 0.5), run_pages);
        const std::uint64_t pool = std::uint64_t(layers) * active_pages - streamed + ring;
        if (pool > budget_pages) { continue; }
        // Feasible: build the table.
        KVStreamSplit split;
        split.layer_resident.assign(layers, active_pages);
        std::uint64_t assigned = 0;
        for (std::uint32_t index = 0; index < split_layers; ++index) {
            const std::uint64_t take = base + (index < remainder ? 1 : 0);
            const std::uint32_t layer =
                std::uint32_t(std::uint64_t(index) * layers / split_layers);
            split.layer_resident[layer] = uint32_t(active_pages - take);
            assigned += take;
        }
        if (assigned != streamed) { continue; }
        split.ring_pages = ring;
        split.run_pages  = run_pages;
        if (kv_stream_pool_pages(split) > budget_pages) { continue; }
        return split;
    }
    return std::nullopt;
}

// The split a decode round should run: the concentrated shape when it needs no more pool than the
// uniform one, and the uniform shape otherwise. Concentrating changes how many layers copy per round
// and how long each copy run is; it can never make a round travel fewer bytes, because the pool
// identity fixes the total of window plus streamed pages.
[[nodiscard]] inline std::optional<KVStreamSplit>
plan_kv_stream_split(std::uint32_t budget_pages, std::uint32_t active_pages, std::uint32_t layers,
                     std::uint32_t run_pages = kKVStreamRunPages,
                     double ring_ratio = kKVStreamRingRatio) noexcept {
    const auto uniform = plan_kv_stream_uniform(budget_pages, active_pages, layers, run_pages,
                                                ring_ratio);
    if (!uniform) { return std::nullopt; }
    if (uniform->ring_pages == 0) { return uniform; }
    // The reference's bound: enough split layers that each one's extent fits the ring, and enough that
    // one layer's pages fit its own window.
    const std::uint32_t uniform_resident = uniform->resident_pages();
    if (uniform_resident >= active_pages) { return uniform; }
    const std::uint64_t streamed_total = std::uint64_t(layers) * (active_pages - uniform_resident);
    if (streamed_total == 0) { return uniform; }
    const std::uint64_t by_ring =
        (streamed_total + uniform->ring_pages - 1) / uniform->ring_pages;
    const std::uint64_t by_layer = (streamed_total + active_pages - 1) / active_pages;
    std::uint64_t split_layers = by_ring > by_layer ? by_ring : by_layer;
    if (split_layers < 2) { split_layers = 2; }
    if (split_layers > layers) { split_layers = layers; }
    const auto concentrated = plan_kv_stream_concentrated(
        budget_pages, active_pages, layers, run_pages, ring_ratio, std::uint32_t(split_layers));
    if (!concentrated) { return uniform; }
    // Accept it only if it keeps at least as much resident as the uniform shape: otherwise it bought a
    // schedule for pages that would have stayed resident.
    std::uint64_t concentrated_resident = 0;
    for (const std::uint32_t pages : concentrated->layer_resident) { concentrated_resident += pages; }
    std::uint64_t uniform_resident_total = std::uint64_t(uniform_resident) * layers;
    if (concentrated_resident < uniform_resident_total) { return uniform; }
    return concentrated;
}

// Re-sizes a split to the context that actually exists: a request far below the cache's capacity needs
// a ring only as large as its own streamed extent, and every slot the ring gives back is a page that
// stays resident (the article's rules 1 and 3: keep everything resident while it fits, grow the ring
// only when the context outgrows it).
[[nodiscard]] inline std::optional<KVStreamSplit>
resize_kv_stream_split(const KVStreamSplit& planned, std::uint32_t budget_pages,
                       std::uint32_t active_pages, double ring_ratio = kKVStreamRingRatio) noexcept {
    if (planned.layers() == 0 || planned.run_pages == 0 || active_pages == 0) { return std::nullopt; }
    const auto uniform = plan_kv_stream_split(budget_pages, active_pages, planned.layers(),
                                              planned.run_pages, ring_ratio);
    if (!uniform) { return std::nullopt; }
    return uniform;
}

} // namespace ninfer
