#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

// The static split of a growing KV pool whose authority is pinned host memory.
//
// `resident_pages` of every streaming layer stay on the device as a window over that layer's
// newest pages; the remaining `ring_pages` slots are shared by all of those layers, in
// `ring_depth` regions of one layer's streamed extent each. One layer's streamed pages occupy one
// region while that layer attends, and the layer `ring_depth` steps away reuses the same region.
//
// A depth of one is the minimum and the shape the loop ran first: the ring holds exactly the layer
// that is attending, so its copy cannot begin until the previous layer's attention has finished
// reading the ring, and the compute stream waits for the whole copy before the next attention
// starts. Every layer then exposes its copy minus whatever compute the previous layer had left to
// enqueue (measured: 1.42 ms of 2.66 at 251K, 22.65 ms a round).
//
// A depth above one is what lets the next layers' copies overlap the current layer's attention:
// region `layer mod ring_depth` is free as soon as the layer `ring_depth` steps back has been read,
// which is one full layer of compute before it is needed. Two regions already cover a layer's
// compute (2.58 ms against a 2.90 ms copy at 251K), so a third buys nothing and costs window.
struct KVStreamSplit {
    // Resident pages of one layer, a window over its newest pages.
    std::uint32_t resident_pages = 0;
    // Slots shared by every streaming layer: `ring_depth` regions of one frame each.
    std::uint32_t ring_pages = 0;
    // The regions the ring is divided into, which is how many frames can be in flight at once.
    std::uint32_t ring_depth = 1;
    // Pages of one frame: the ring rotates through a layer's streamed pages a frame at a time, so
    // it holds the frames in flight rather than one layer's whole streamed extent.
    //
    // A frame is the smallest thing a copy can be *and* the smallest thing a launch can read, so
    // it is sized by the attention launch, not by the ring: a launch needs a device's worth of
    // CTAs, which is `sms / kv_heads` splits (design 5.13.1). Zero keeps the whole extent in the
    // ring, which is the layout every depth-2 measurement before it used.
    std::uint32_t frame_pages = 0;
    // Frames one layer has, `ceil(pages_per_layer / frame_pages)`. Planned rather than derived so
    // that a page's region stays a pure function of `(layer, page)`; zero with `frame_pages`.
    std::uint32_t frames_per_layer = 0;
};

// Layers' streamed extents the ring is planned to hold. The measured trade-off in
// docs/maintainer/kv-streaming-design.md 5.11: the second region costs one page of window per
// layer of every layer beyond the first and makes the copies overlap, which is worth several times
// that page.
constexpr std::uint32_t kKVStreamRingDepth = 2;

// Streamed pages of one layer that one region of the ring holds: the layer's whole streamed extent
// in the unframed layout, one frame in the framed one.
[[nodiscard]] constexpr std::uint32_t
kv_stream_ring_extent(const KVStreamSplit& split, std::uint32_t layers) noexcept {
    return split.ring_depth == 0 ? 0U : split.ring_pages / split.ring_depth;
}

// The region of the ring a page uses. The unframed layout gives every layer a region of its own;
// the framed one walks the ring once per frame, in the order the scheduler fills frames, so the
// region page `p` uses is free by the time `p` is copied: its previous occupant is the frame
// `ring_depth` frames back, which the consumer has already read.
[[nodiscard]] constexpr std::uint32_t
kv_stream_region(std::uint32_t layer, std::uint32_t logical_page,
                 const KVStreamSplit& split) noexcept {
    if (split.ring_depth == 0) { return 0; }
    if (split.frame_pages == 0 || split.frames_per_layer == 0) {
        return layer % split.ring_depth;
    }
    const std::uint64_t frame = static_cast<std::uint64_t>(layer) * split.frames_per_layer +
                                logical_page / split.frame_pages;
    return static_cast<std::uint32_t>(frame % split.ring_depth);
}

// Offset of a page inside its region: its own index in the unframed layout, its index inside the
// frame in the framed one.
[[nodiscard]] constexpr std::uint32_t
kv_stream_region_offset(std::uint32_t logical_page, const KVStreamSplit& split) noexcept {
    return split.frame_pages == 0 ? logical_page : logical_page % split.frame_pages;
}


// Pages the flat pool must hold for this split.
[[nodiscard]] constexpr std::uint32_t kv_stream_pool_pages(const KVStreamSplit& split,
                                                           std::uint32_t layers) noexcept {
    return layers * split.resident_pages + split.ring_pages;
}

// Chooses the split of a flat pool of `budget_pages` slots (each slot is one layer's page) for
// `layers` streaming layers that each reach `pages_per_layer` logical pages at max context.
//
// A dense attention reads a whole layer at once, so every streamed page of a layer must be in the
// ring at the same time: one region is exactly `pages_per_layer - resident_pages` deep, and the
// pool's resident region costs `layers * resident_pages`:
//
//     layers * R + depth * (P - R) <= B     =>     R <= (B - depth * P) / (layers - depth)
//
// The deepest ring that fits is chosen up to `max_ring_depth`, because a region the layer
// `depth` steps back has stopped reading is what lets a copy overlap attention. The largest
// feasible R for that depth is then taken, because every page that stays resident is a page not
// copied each round. `nullopt` means no split serves the context -- not even one layer's pages fit
// -- and the caller must refuse to start rather than quietly degrade. A budget that holds every
// layer in full returns `{P, 0, 1}`, which is the all-resident layout and takes no streaming path
// at all.
[[nodiscard]] constexpr std::optional<KVStreamSplit>
plan_kv_stream_split(std::uint32_t budget_pages, std::uint32_t pages_per_layer,
                     std::uint32_t layers,
                     std::uint32_t max_ring_depth = kKVStreamRingDepth,
                     std::uint32_t frame_pages = 0) noexcept {
    if (layers == 0 || pages_per_layer == 0 || budget_pages < pages_per_layer) { return std::nullopt; }
    const std::uint64_t full = static_cast<std::uint64_t>(layers) * pages_per_layer;
    if (budget_pages >= full) { return KVStreamSplit{pages_per_layer, 0, 1}; }
    // `layers == 1` leaves no sharing to exploit: its pages either fit or they do not, and the
    // division above would be by zero. A pool that does not hold the layer is refused.
    if (layers == 1) { return std::nullopt; }
    const std::uint32_t deepest = max_ring_depth < layers - 1 ? max_ring_depth : layers - 1;
    // A framed ring costs `depth * frame_pages` slots however far a layer reaches, so the window
    // takes whatever the ring leaves and no region has to hold the whole streamed extent:
    //
    //     layers * R + depth * frame_pages <= B     =>     R <= (B - depth * frame_pages) / layers
    //
    // Every page that a deeper frame does not need stays resident, so the deepest ring that fits is
    // taken: the two extra frames are what let a copy run while the launch before it reads.
    if (frame_pages != 0) {
        const std::uint32_t frames = (pages_per_layer - 1) / frame_pages + 1;
        for (std::uint32_t depth = deepest; depth >= 1; --depth) {
            const std::uint64_t ring = static_cast<std::uint64_t>(depth) * frame_pages;
            if (ring >= budget_pages) { continue; }
            const std::uint32_t resident =
                static_cast<std::uint32_t>((budget_pages - ring) / layers);
            if (resident >= pages_per_layer) { return KVStreamSplit{pages_per_layer, 0, 1}; }
            return KVStreamSplit{resident, static_cast<std::uint32_t>(ring), depth, frame_pages,
                                 frames};
        }
        return std::nullopt;
    }
    // A deeper ring costs the window one page per layer for every layer beyond the first, so it is
    // taken only from a pool that holds several contexts: six layers' pages of context is the
    // threshold below which a bigger window is worth more than the overlap (design 5.11).
    const std::uint32_t first_depth =
        static_cast<std::uint64_t>(budget_pages) >= 6ULL * pages_per_layer ? deepest : 1U;
    for (std::uint32_t depth = first_depth; depth >= 1; --depth) {
        const std::uint64_t ring = static_cast<std::uint64_t>(depth) * pages_per_layer;
        if (ring > budget_pages) { continue; }
        const std::uint32_t resident =
            static_cast<std::uint32_t>((budget_pages - ring) / (layers - depth));
        const std::uint32_t r = resident < pages_per_layer ? resident : pages_per_layer;
        // A budget of one layer's pages and nothing beside them plans as a window of zero pages:
        // the planner evaluates that anchor to model what a pool of one layer costs, and refuses
        // to build a cache from it (StreamedKVSpace wants a window).
        return KVStreamSplit{r, depth * (pages_per_layer - r), depth};
    }
    return std::nullopt;
}

// The slot a page takes once it leaves the window: region `layer mod ring_depth` of the ring, at
// its own page's offset inside that region. The region follows the layer, not the round, so the
// entry is written once and never rewritten.
[[nodiscard]] constexpr std::int32_t
kv_stream_ring_slot(std::uint32_t layer, std::uint32_t logical_page, const KVStreamSplit& split,
                    std::uint32_t layers) noexcept {
    return static_cast<std::int32_t>(
        layers * split.resident_pages +
        kv_stream_region(layer, logical_page, split) * kv_stream_ring_extent(split, layers) +
        kv_stream_region_offset(logical_page, split));
}

// Slot of logical page `logical_page` of `layer` in the flat pool.
//
// Both halves of the mapping depend on the page only, never on how far the context has grown:
// a streamed page is its own slot past the resident region (the frame it falls in, in the framed
// layout), and a resident page is the slot its page index leaves in its layer's region. That is
// what makes a table entry change exactly twice in a page's life - once when the page appears, once
// when it leaves the window - instead of on every page boundary. When a page leaves, the page
// taking its place is the one whose index differs by the window depth, which is the same slot.
//
// `logical_page` must be below `active_pages`, which must not exceed the layer's capacity; the
// result is always below `kv_stream_pool_pages(split, layers)`.
[[nodiscard]] constexpr std::int32_t kv_stream_slot(std::uint32_t layer, std::uint32_t logical_page,
                                                   std::uint32_t active_pages,
                                                   const KVStreamSplit& split,
                                                   std::uint32_t layers) noexcept {
    const std::uint32_t resident     = split.resident_pages;
    const std::uint32_t window_start = active_pages > resident ? active_pages - resident : 0;
    if (logical_page < window_start) {
        return kv_stream_ring_slot(layer, logical_page, split, layers);
    }
    // Every page is inside the window: no streamed page exists, and the ring is not addressed.
    return static_cast<std::int32_t>(layer * resident +
                                     (resident == 0 ? 0 : logical_page % resident));
}

// The slot a page keeps for as long as it stays in its layer's window, so that the table entry set
// when the page appeared needs no rewrite until the page leaves.
[[nodiscard]] constexpr std::int32_t kv_stream_window_slot(std::uint32_t layer,
                                                         std::uint32_t logical_page,
                                                         const KVStreamSplit& split) noexcept {
    return static_cast<std::int32_t>(layer * split.resident_pages +
                                     (split.resident_pages == 0
                                          ? 0
                                          : logical_page % split.resident_pages));
}

} // namespace ninfer
