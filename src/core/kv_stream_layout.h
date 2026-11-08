#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

// The static split of a growing KV pool whose authority is pinned host memory.
//
// `resident_pages` of every streaming layer stay on the device as a window over that layer's
// newest pages; the remaining `ring_pages` slots are shared by all of those layers -- one layer's
// streamed pages occupy them while that layer attends, and the next layer reuses the same slots.
// The ring is therefore one layer deep, not one pool deep, which is what makes a 16 GB card able
// to hold a 262 144-token context (docs/maintainer/kv-streaming-design.md, sections 4.1-4.5).
struct KVStreamSplit {
    // Resident pages of one layer, a window over its newest pages.
    std::uint32_t resident_pages = 0;
    // Slots shared by every streaming layer; one layer's streamed pages at a time.
    std::uint32_t ring_pages = 0;
};

// Pages the flat pool must hold for this split.
[[nodiscard]] constexpr std::uint32_t kv_stream_pool_pages(const KVStreamSplit& split,
                                                           std::uint32_t layers) noexcept {
    return layers * split.resident_pages + split.ring_pages;
}

// Chooses the split of a flat pool of `budget_pages` slots (each slot is one layer's page) for
// `layers` streaming layers that each reach `pages_per_layer` logical pages at max context.
//
// A dense attention reads a whole layer at once, so every streamed page of a layer must be in the
// ring at the same time: the ring is exactly `pages_per_layer - resident_pages` deep. With the
// pool's resident region costing `layers * resident_pages`:
//
//     layers * R + (P - R) <= B     =>     R <= (B - P) / (layers - 1)
//
// The largest feasible R is chosen, because every page that stays resident is a page not copied
// each round. `nullopt` means no split serves the context -- not even one layer's pages fit -- and
// the caller must refuse to start rather than quietly degrade. A budget that holds every layer in
// full returns `{P, 0}`, which is the all-resident layout and takes no streaming path at all.
[[nodiscard]] constexpr std::optional<KVStreamSplit>
plan_kv_stream_split(std::uint32_t budget_pages, std::uint32_t pages_per_layer,
                     std::uint32_t layers) noexcept {
    if (layers == 0 || pages_per_layer == 0 || budget_pages < pages_per_layer) { return std::nullopt; }
    const std::uint64_t full = static_cast<std::uint64_t>(layers) * pages_per_layer;
    if (budget_pages >= full) { return KVStreamSplit{pages_per_layer, 0}; }
    // `layers == 1` leaves no sharing to exploit: the single layer's pages either fit or they do
    // not, and `(budget - P) / 0` would not be the answer.
    const std::uint64_t resident =
        layers == 1 ? pages_per_layer
                    : (static_cast<std::uint64_t>(budget_pages) - pages_per_layer) / (layers - 1);
    const std::uint32_t r = static_cast<std::uint32_t>(
        resident < pages_per_layer ? resident : pages_per_layer);
    return KVStreamSplit{r, pages_per_layer - r};
}

// Slot of logical page `logical_page` of `layer` in the flat pool.
//
// Streamed pages map to the ring by identity, so that half of every layer's table is written once
// and never rewritten; the resident window rotates through its own region of the pool, one entry
// per page boundary (see the design, section 4.3). `logical_page` must be below `active_pages`,
// which must not exceed the layer's capacity; the result is always below
// `kv_stream_pool_pages(split, layers)`.
[[nodiscard]] constexpr std::int32_t kv_stream_slot(std::uint32_t layer, std::uint32_t logical_page,
                                                   std::uint32_t active_pages,
                                                   const KVStreamSplit& split,
                                                   std::uint32_t layers) noexcept {
    const std::uint32_t resident     = split.resident_pages;
    const std::uint32_t window_start = active_pages > resident ? active_pages - resident : 0;
    if (logical_page < window_start) {
        return static_cast<std::int32_t>(layers * resident + logical_page);
    }
    // Every page is inside the window: no streamed page exists, and the ring is not addressed.
    const std::uint32_t within = logical_page - window_start;
    return static_cast<std::int32_t>(layer * resident + (resident == 0 ? 0 : within % resident));
}

} // namespace ninfer
