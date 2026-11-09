#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

// The fixed split of a KV pool whose authority is pinned host memory
// (docs/maintainer/kv-streaming-fidelity.md, the reference article's "ring buffer + adaptive
// repartition").
//
// `resident_pages` of every streaming layer stay on the device as a window over that layer's newest
// pages; the `ring_pages` slots shared by all of those layers hold the layer's older pages while it
// attends. The ring is a flat pool: a layer's streamed pages are copied into `run_pages`-sized runs
// that any layer may own, so the ring's size is set by the work it must cover - the article's
// "ring大得够隐藏预取、小得不多赶走驻留页" - rather than by whole layers.
//
//     layers * resident_pages + ring_pages <= budget
//     ring_pages >= the streamed extent one layer reads in one call
//
// A ring of one streamed extent can only hold one layer at a time, so a layer's copies cannot
// overlap the previous layer's attention; a ring of `ratio` extents (the article's 1.10, the
// reference implementation's `target_ring_working_set_ratio`) lets the next layer's copies start
// while the current one is still reading. That is the only knob the split has: every slot the ring
// does not need is a page that stays resident and is not copied every round.
struct KVStreamSplit {
    // Resident pages of one layer: a window over its newest pages.
    std::uint32_t resident_pages = 0;
    // Slots shared by every streaming layer, a whole number of runs.
    std::uint32_t ring_pages = 0;
    // Pages one copy (and one consumed span) covers. Sized so that a run is a batch the copy engine
    // likes and the consumer can dispatch without waiting for a whole layer: the reference uses 32
    // pages of 256 tokens, which is 128 pages of this engine's 64-token pages by byte count.
    std::uint32_t run_pages = 1;
};

// Ring slots relative to one layer's streamed pages. Above one leaves room to begin the next layer
// early; the reference's measured default is 1.10 (`llama-kv-stream-plan.h`).
constexpr double kKVStreamRingRatio = 1.10;

// Pages a run covers when the ring is planned without an explicit choice.
constexpr std::uint32_t kKVStreamRunPages = 128;

// Pages the pool holds for this split.
[[nodiscard]] constexpr std::uint32_t kv_stream_pool_pages(const KVStreamSplit& split,
                                                           std::uint32_t layers) noexcept {
    return layers * split.resident_pages + split.ring_pages;
}

// The page the window starts at: pages below it are streamed, pages from it up are resident.
[[nodiscard]] constexpr std::uint32_t kv_stream_window_start(std::uint32_t active_pages,
                                                             const KVStreamSplit& split) noexcept {
    return active_pages > split.resident_pages ? active_pages - split.resident_pages : 0U;
}

// The slot a resident page keeps for as long as it stays in its layer's window, so the table entry
// written when the page appeared needs no rewrite until it leaves.
[[nodiscard]] constexpr std::int32_t kv_stream_window_slot(std::uint32_t layer,
                                                           std::uint32_t logical_page,
                                                           const KVStreamSplit& split) noexcept {
    return static_cast<std::int32_t>(layer * split.resident_pages +
                                     (split.resident_pages == 0
                                          ? 0
                                          : logical_page % split.resident_pages));
}

// Runs one layer's streamed pages occupy, `ceil(window_start / run_pages)`.
[[nodiscard]] constexpr std::uint32_t kv_stream_runs_per_layer(std::uint32_t active_pages,
                                                               const KVStreamSplit& split) noexcept {
    if (split.run_pages == 0) { return 0; }
    const std::uint32_t streamed = kv_stream_window_start(active_pages, split);
    return (streamed + split.run_pages - 1) / split.run_pages;
}

// Runs the ring holds, `ring_pages / run_pages`.
[[nodiscard]] constexpr std::uint32_t kv_stream_ring_runs(const KVStreamSplit& split) noexcept {
    return split.run_pages == 0 ? 0 : split.ring_pages / split.run_pages;
}

// Chooses the split of `budget_pages` slots for `layers` streaming layers that each reach
// `pages_per_layer` logical pages, with a ring of `ring_ratio` times one layer's streamed extent.
//
// The ring must hold the extent it is sized for and the window takes what is left:
//
//     streamed = pages_per_layer - resident        (at the planned capacity)
//     ring     = ceil_to_run(ring_ratio * streamed)
//     resident = (budget - ring) / layers
//
// which is a fixed point rather than a formula - a larger window makes a smaller ring, which allows a
// larger window - so it is iterated to convergence (a few passes; the map is monotone). `nullopt`
// when no split serves the pool, so the caller refuses to start instead of quietly degrading. A
// budget that holds every layer in full returns the all-resident split, which takes no streaming
// path at all.
[[nodiscard]] constexpr std::optional<KVStreamSplit>
plan_kv_stream_split(std::uint32_t budget_pages, std::uint32_t pages_per_layer,
                     std::uint32_t layers, std::uint32_t run_pages = kKVStreamRunPages,
                     double ring_ratio = kKVStreamRingRatio) noexcept {
    if (layers == 0 || pages_per_layer == 0 || budget_pages < pages_per_layer || run_pages == 0) {
        return std::nullopt;
    }
    if (budget_pages >= static_cast<std::uint64_t>(layers) * pages_per_layer) {
        return KVStreamSplit{pages_per_layer, 0, run_pages};
    }
    if (!(ring_ratio >= 1.0)) { ring_ratio = 1.0; }
    const auto ceil_to_run = [run_pages](std::uint64_t pages) {
        return static_cast<std::uint32_t>(pages + run_pages - 1 - (pages + run_pages - 1) % run_pages);
    };
    std::uint32_t resident = budget_pages / layers;
    for (int pass = 0; pass < 8; ++pass) {
        if (resident > pages_per_layer) { resident = pages_per_layer; }
        const std::uint64_t streamed = pages_per_layer - resident;
        std::uint64_t ring = static_cast<std::uint64_t>(ring_ratio * static_cast<double>(streamed));
        // The ring must hold the whole extent regardless of the ratio, or the consumer would read
        // slots a copy has already reused.
        if (ring < streamed) { ring = streamed; }
        const std::uint32_t rounded = ceil_to_run(ring);
        if (rounded + run_pages > budget_pages) { return std::nullopt; }
        const std::uint32_t next =
            static_cast<std::uint32_t>((budget_pages - rounded) / layers);
        if (next == resident) {
            return KVStreamSplit{resident, rounded, run_pages};
        }
        resident = next;
    }
    // Eight passes of a monotone map either converged or oscillate between two neighbours; the
    // smaller window is the safe side, because it keeps the ring big enough for the extent.
    const std::uint64_t streamed = pages_per_layer > resident ? pages_per_layer - resident : 0;
    std::uint64_t ring = static_cast<std::uint64_t>(ring_ratio * static_cast<double>(streamed));
    if (ring < streamed) { ring = streamed; }
    const std::uint32_t rounded = ceil_to_run(ring);
    if (rounded + run_pages > budget_pages ||
        static_cast<std::uint64_t>(layers) * resident + rounded > budget_pages) {
        return std::nullopt;
    }
    return KVStreamSplit{resident, rounded, run_pages};
}

// The same split re-sized to the context that actually exists: a request far below the cache's
// capacity needs a ring only as large as its own streamed extent, and every slot the ring gives back
// is a page that stays resident (the article's rule 1, "只要固定的显存空间能够容纳整个键值缓存，环形
// 缓冲区就保持禁用状态", and rule 3, grow the ring only when the context outgrows it).
[[nodiscard]] constexpr std::optional<KVStreamSplit>
resize_kv_stream_split(const KVStreamSplit& planned, std::uint32_t budget_pages,
                       std::uint32_t active_pages, std::uint32_t layers,
                       double ring_ratio = kKVStreamRingRatio) noexcept {
    if (layers == 0 || planned.run_pages == 0 || active_pages == 0) { return std::nullopt; }
    if (static_cast<std::uint64_t>(layers) * active_pages <= budget_pages) {
        // The whole context fits, so nothing streams and the ring is not addressed.
        return KVStreamSplit{active_pages, 0, planned.run_pages};
    }
    const auto ceil_to_run = [run = planned.run_pages](std::uint64_t pages) {
        return static_cast<std::uint32_t>(pages + run - 1 - (pages + run - 1) % run);
    };
    std::uint32_t resident = active_pages > planned.resident_pages ? active_pages - planned.resident_pages
                                                                  : planned.resident_pages;
    if (resident > active_pages) { resident = active_pages; }
    for (int pass = 0; pass < 8; ++pass) {
        const std::uint64_t streamed = active_pages - resident;
        std::uint64_t ring = static_cast<std::uint64_t>(ring_ratio * static_cast<double>(streamed));
        if (ring < streamed) { ring = streamed; }
        const std::uint32_t rounded = ceil_to_run(ring);
        if (static_cast<std::uint64_t>(rounded) + static_cast<std::uint64_t>(layers) * planned.run_pages >
            budget_pages) {
            return std::nullopt;
        }
        const std::uint32_t next =
            static_cast<std::uint32_t>((budget_pages - rounded) / layers);
        if (next == resident) { return KVStreamSplit{resident, rounded, planned.run_pages}; }
        resident = next > active_pages ? active_pages : next;
    }
    return std::nullopt;
}

} // namespace ninfer
