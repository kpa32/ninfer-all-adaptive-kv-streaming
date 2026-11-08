#include "models/qwen3_5/program/storage/kv_stream_space.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

void create_events(std::vector<cudaEvent_t>& events, std::uint32_t count) {
    events.assign(count, nullptr);
    for (cudaEvent_t& event : events) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
}

void destroy_events(std::vector<cudaEvent_t>& events) noexcept {
    for (cudaEvent_t& event : events) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
        event = nullptr;
    }
}

[[nodiscard]] std::uint32_t page_count(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

} // namespace

StreamedKVSpace::StreamedKVSpace(PagedKVCache& cache)
    : cache_(&cache), split_(cache.stream_split()), layers_(cache.layers()) {
    if (!cache.flat_slots()) {
        throw std::invalid_argument("A streamed KV space needs a cache planned with a flat pool");
    }
    if (layers_ == 0 || split_.resident_pages == 0 || cache.max_context() == 0) {
        throw std::invalid_argument(
            "A streamed KV space needs layers, a window and a context: layers=" +
            std::to_string(layers_) + " resident=" + std::to_string(split_.resident_pages) +
            " ring=" + std::to_string(split_.ring_pages) + " slots=" +
            std::to_string(cache.page_pool().capacity_pages()) + " context=" +
            std::to_string(cache.max_context()));
    }
    pages_per_layer_ = page_count(cache.max_context());
    ring_base_       = layers_ * split_.resident_pages;
    // Every slot of the pool belongs to this space for the cache's lifetime: the mapping names
    // slots, so nothing else may allocate, lend or release one.
    const std::uint32_t slot_count = cache.page_pool().capacity_pages();
    slot_count_                    = slot_count;
    // The capacity resolution grants a slot count the split rounds down: `layers` slots at most
    // stay unaddressed at the end of the pool, and they cost one page each.
    if (slot_count < kv_stream_pool_pages(split_, layers_)) {
        throw std::invalid_argument(
            "A streamed KV pool is smaller than its split: slots=" + std::to_string(slot_count) +
            " needs=" + std::to_string(kv_stream_pool_pages(split_, layers_)));
    }
    if (ring_base_ + split_.ring_pages < pages_per_layer_) {
        throw std::invalid_argument("A streamed KV ring cannot hold one layer's streamed pages");
    }
    host_layout_ = plan_host_kv_page_layout(cache.page_pool().geometry());
    page_bytes_  = host_layout_.page_stride;
    // The authority: every page of every streaming layer, for the whole context. A streamed page
    // is read from here on every round, so this allocation is what lets the context exceed the
    // device pool.
    arena_ = std::make_unique<HostKVArena>(page_bytes_ * pages_per_layer_,
                                           std::span<const HostKVPageLayout>(&host_layout_, 1));
    auto host = arena_->allocate(host_layout_, pages_per_layer_);
    if (!host) {
        throw std::invalid_argument("The host KV arena is too small for this cache's context");
    }
    host_ = std::move(*host);

    // The slots are not leased. A lease would leave admission no page to reserve, and every
    // request would be turned away; what keeps the mapping's pages out of everyone else's hands is
    // that nothing else in a streaming configuration allocates from this pool - the retained-context
    // cache, the prefix cache and both loans are refused or absent - and the capacity resolution
    // already spent the budget on exactly these slots.
    create_events(ready_, layers_);
    create_events(consumed_, layers_);
}

StreamedKVSpace::~StreamedKVSpace() {
    destroy_events(ready_);
    destroy_events(consumed_);
}

HostKVAllocationView StreamedKVSpace::host_pages() noexcept { return arena_->writable_view(host_); }

HostKVAllocationConstView StreamedKVSpace::host_pages() const noexcept {
    return arena_->view(host_);
}
cudaEvent_t StreamedKVSpace::ready_event(std::uint32_t layer) const {
    if (layer >= ready_.size()) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return ready_[layer];
}

cudaEvent_t StreamedKVSpace::consumed_event(std::uint32_t layer) const {
    if (layer >= consumed_.size()) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return consumed_[layer];
}

void StreamedKVSpace::publish_slot(std::uint32_t layer, std::uint32_t page, std::int32_t slot,
                                   RankStreams streams) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= slot_handles_.size()) {
        throw std::logic_error("Streamed KV slot is outside its pool");
    }
    const std::span<const DeviceKVPageHandle> one(&slot_handles_[static_cast<std::size_t>(slot)], 1);
    cache_->execution_tables().publish_at(static_cast<std::int32_t>(layer), page, one, streams);
}

void StreamedKVSpace::advance(std::uint32_t active_pages, RankStreams streams) {
    if (active_pages < active_) {
        throw std::invalid_argument("A streamed KV frontier only grows; reset before rewinding");
    }
    if (active_pages > pages_per_layer_) {
        throw std::invalid_argument("A streamed KV frontier exceeds the cache's context");
    }
    HostKVAllocationView host = host_pages();
    for (std::uint32_t page = active_; page < active_pages; ++page) {
        // The page appears in its layer's window, where this round's append writes it.
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            publish_slot(layer, page, kv_stream_window_slot(layer, page, split_), streams);
            if (page < split_.resident_pages) { continue; }
            // The window is full, so the page it drops leaves the device: its bytes reach the host
            // first, and its table entry then points at the ring slot it will be read from. The
            // copy is ordered here, before this round's append overwrites the slot the arriving
            // page takes - the two pages' indices differ by the window depth, so it is the same.
            const std::uint32_t dropped      = page - split_.resident_pages;
            const std::int32_t dropped_slot  = kv_stream_window_slot(layer, dropped, split_);
            cache_->page_pool().copy_to_host_pages(dropped_slot, 1, host.subview(dropped, 1),
                                                   streams);
            publish_slot(layer, dropped, kv_stream_ring_slot(dropped, split_, layers_), streams);
        }
    }
    active_ = active_pages;
}

void StreamedKVSpace::upload(std::uint32_t layer, cudaStream_t copy) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (copy == nullptr) { throw std::invalid_argument("A streamed KV upload needs a copy stream"); }
    const std::uint32_t count = window_start();
    if (count != 0) {
        // Every streamed page of this layer, page 0 first, into the consecutive ring slots; the
        // pool turns the run into one strided transfer per plane.
        cache_->page_pool().copy_from_host_pages(host_pages().data(), host_layout_, ring_base_,
                                                 count, RankStreams(copy));
    }
    CUDA_CHECK(cudaEventRecord(ready_[layer], copy));
}

} // namespace ninfer::models::qwen3_5
