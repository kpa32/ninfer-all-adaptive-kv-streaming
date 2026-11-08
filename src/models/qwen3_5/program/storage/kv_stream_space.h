#pragma once

#include "core/host_kv_arena.h"
#include "core/kv_stream_layout.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// The text cache's pages when they stream (docs/maintainer/kv-streaming-design.md).
//
// Pinned host memory holds every page of every full-attention layer and is the authority. The
// device pool holds those layers' pages without holding the whole context: a window over each
// layer's newest pages plus one ring of slots that every layer borrows in turn. One block table
// per layer names the slot a logical page occupies, so attention reaches a page through the table
// it already dereferences and a CUDA graph stays valid across rounds.
//
// The space owns three things the paged stores cannot express: all of the pool's slots (taken for
// the cache's lifetime, so the pool hands out nothing and lends nothing), the per-layer tables'
// contents, and the transfers that keep the host copy current. It owns no scheduling policy: when
// to move a page and which layer to fill next are the caller's.
class StreamedKVSpace {
public:
    // Takes every slot of `cache`'s pool and allocates `pages_per_layer` host pages from `arena`.
    // The cache must have been planned with this split (`PagedKVCache::stream_split`), and the
    // arena must support its geometry.
    StreamedKVSpace(PagedKVCache& cache, HostKVArena& arena);
    ~StreamedKVSpace();

    StreamedKVSpace(const StreamedKVSpace&)            = delete;
    StreamedKVSpace& operator=(const StreamedKVSpace&) = delete;
    StreamedKVSpace(StreamedKVSpace&&)                 = delete;
    StreamedKVSpace& operator=(StreamedKVSpace&&)      = delete;

    [[nodiscard]] const KVStreamSplit& split() const noexcept { return split_; }

    // Logical pages one layer reaches at the planned capacity.
    [[nodiscard]] std::uint32_t pages_per_layer() const noexcept { return pages_per_layer_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    // Pages the tables describe, and where the resident window currently starts.
    [[nodiscard]] std::uint32_t active_pages() const noexcept { return active_; }

    [[nodiscard]] std::uint32_t window_start() const noexcept {
        return active_ > split_.resident_pages ? active_ - split_.resident_pages : 0U;
    }

    // Host pages of one layer: page `p` is record `p` of this view, so a caller can copy a run of
    // them with a plain 2D transfer or point the append Ops at them.
    [[nodiscard]] HostKVAllocationView host_pages() noexcept;
    [[nodiscard]] HostKVAllocationConstView host_pages() const noexcept;

    // Device slot handles, indexed by slot number.
    [[nodiscard]] std::span<const DeviceKVPageHandle> slots() const noexcept {
        return slot_handles_;
    }

    // Brings the tables to `active_pages` logical pages of the context, creating the pages in
    // between: each new page's table entry points at its window slot, and each page the window
    // drops first moves to the host and then points at its ring slot. Both are ordered on
    // `streams`, so a page's slot is never reused before its bytes reached the host.
    //
    // Growing is the only transition a served request makes; `reset` is how a new request clears
    // the tables and the frontier.
    void advance(std::uint32_t active_pages, RankStreams streams);

    // Forgets the context: the frontier returns to zero and the next request republishes each
    // entry as its pages appear. Nothing is copied, because those pages are written again.
    void reset() noexcept { active_ = 0; }

    // Copies layer `layer`'s streamed pages (`[0, window_start)`) from their host pages into the
    // ring, on `copy`. The copies are asynchronous and land in the ring slots the tables name; the
    // caller records a ready event afterwards and has the consumer wait on it.
    //
    // The ring is shared, so a caller that issues this for consecutive layers must order each
    // against the previous layer's consumption (`cudaStreamWaitEvent(copy, consumed(layer - 1))`).
    void upload(std::uint32_t layer, cudaStream_t copy);

    // Device bytes one layer's streamed pages move in one upload, and what a round of every layer
    // moves, for the round's plan and its reporting.
    [[nodiscard]] std::uint64_t layer_upload_bytes() const noexcept {
        return static_cast<std::uint64_t>(window_start()) * page_bytes_;
    }
    [[nodiscard]] std::uint64_t uploaded_bytes() const noexcept {
        return layer_upload_bytes() * layers_;
    }

    // The per-layer events a round orders its copies and its consumers with. `ready(layer)` is
    // recorded by `upload`; `consumed(layer)` is recorded by the consumer once that layer's
    // attention has read the ring.
    [[nodiscard]] cudaEvent_t ready_event(std::uint32_t layer) const;
    [[nodiscard]] cudaEvent_t consumed_event(std::uint32_t layer) const;

private:
    void publish_slot(std::uint32_t layer, std::uint32_t page, std::int32_t slot,
                      RankStreams streams);

    PagedKVCache* cache_               = nullptr;
    HostKVArena* arena_                = nullptr;
    KVStreamSplit split_;
    std::uint32_t layers_              = 0;
    std::uint32_t pages_per_layer_     = 0;
    std::uint32_t ring_base_           = 0;
    std::size_t page_bytes_            = 0;
    HostKVPageLayout host_layout_;
    HostKVAllocation host_;
    // One lease per slot, held for the cache's lifetime, and the same handles by slot number.
    std::vector<DeviceKVPageLease> slot_leases_;
    std::vector<DeviceKVPageHandle> slot_handles_;
    std::vector<cudaEvent_t> ready_;
    std::vector<cudaEvent_t> consumed_;
    std::uint32_t active_ = 0;
};

} // namespace ninfer::models::qwen3_5
