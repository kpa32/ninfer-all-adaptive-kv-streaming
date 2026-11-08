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

StreamedKVSpace::StreamedKVSpace(PagedKVCache& cache, bool report)
    : cache_(&cache), split_(cache.stream_split()), layers_(cache.layers()), report_(report) {
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
    if (kv_stream_ring_extent(split_, layers_) + split_.resident_pages < pages_per_layer_) {
        throw std::invalid_argument("A streamed KV ring region is shallower than a layer's stream");
    }
    host_layout_ = plan_host_kv_page_layout(cache.page_pool().geometry());
    page_bytes_  = host_layout_.page_stride;
    // The authority: every page of every streaming layer, for the whole context. A streamed page
    // is read from here on every round, so this allocation is what lets the context exceed the
    // device pool.
    arena_ = std::make_unique<HostKVArena>(page_bytes_ * pages_per_layer_ * layers_,
                                           std::span<const HostKVPageLayout>(&host_layout_, 1));
    auto host = arena_->allocate(host_layout_, pages_per_layer_ * layers_);
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
    create_events(dropped_, layers_);
    create_events(ready_frame_, layers_ * kFrameSlots);
    create_events(consumed_frame_, layers_ * kFrameSlots);
    region_layer_.assign(split_.ring_depth, kNoLayer);
    region_pages_.assign(split_.ring_depth, 0);
    round_.layers.resize(layers_);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) { round_.layers[layer].layer = layer; }
    // Timing events keep their times, so they are created without the disable-timing flag, and
    // only when this run asks to be timed.
    if (report_) {
        const auto create_timed = [&](std::vector<cudaEvent_t>& events) {
            events.assign(layers_, nullptr);
            for (cudaEvent_t& event : events) { CUDA_CHECK(cudaEventCreate(&event)); }
        };
        create_timed(copy_begin_);
        create_timed(copy_end_);
        create_timed(wait_begin_);
        create_timed(wait_end_);
        copy_pending_.assign(layers_, 0);
        wait_pending_.assign(layers_, 0);
        // One timing pair per (layer, frame slot) for the framed decode path, which has several
        // copies and several waits per layer.
        const auto create_frame_timed = [&](std::vector<cudaEvent_t>& events) {
            events.assign(layers_ * kFrameSlots, nullptr);
            for (cudaEvent_t& event : events) { CUDA_CHECK(cudaEventCreate(&event)); }
        };
        create_frame_timed(frame_copy_begin_);
        create_frame_timed(frame_copy_end_);
        create_frame_timed(frame_wait_begin_);
        create_frame_timed(frame_wait_end_);
        frame_copy_pending_.assign(layers_ * kFrameSlots, 0);
        frame_wait_pending_.assign(layers_ * kFrameSlots, 0);
    }
}

StreamedKVSpace::~StreamedKVSpace() {
    destroy_events(ready_);
    destroy_events(consumed_);
    destroy_events(dropped_);
    destroy_events(ready_frame_);
    destroy_events(consumed_frame_);
    destroy_events(copy_begin_);
    destroy_events(copy_end_);
    destroy_events(wait_begin_);
    destroy_events(wait_end_);
    destroy_events(frame_copy_begin_);
    destroy_events(frame_copy_end_);
    destroy_events(frame_wait_begin_);
    destroy_events(frame_wait_end_);
}

void StreamedKVSpace::add_copy_seconds(std::uint32_t layer, double seconds) {
    copy_seconds_ += seconds;
    if (rounds_ == 0) {
        prefill_copy_seconds_ += seconds;
        return;
    }
    round_.copy_seconds += seconds;
    round_.layers[layer].copy_seconds += seconds;
}

void StreamedKVSpace::add_exposed_seconds(std::uint32_t layer, double seconds) {
    exposed_seconds_ += seconds;
    if (rounds_ == 0) {
        prefill_exposed_seconds_ += seconds;
        return;
    }
    round_.exposed_seconds += seconds;
    round_.layers[layer].exposed_seconds += seconds;
}

std::optional<double> StreamedKVSpace::take_timing(cudaEvent_t begin, cudaEvent_t end,
                                                   std::uint8_t& pending) {
    if (!pending) { return std::nullopt; }
    if (cudaEventQuery(end) != cudaSuccess) {
        // Not finished yet; the query cleared any error it reported, and the pair stays pending.
        (void)cudaGetLastError();
        return std::nullopt;
    }
    float milliseconds = 0.0F;
    if (cudaEventElapsedTime(&milliseconds, begin, end) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    pending = 0;
    ++timing_samples_;
    return static_cast<double>(milliseconds) / 1000.0;
}

void StreamedKVSpace::close_timing(std::uint8_t& pending, cudaEvent_t begin, cudaEvent_t end,
                                   std::uint32_t layer, bool copy) {
    if (!pending) { return; }
    if (const std::optional<double> seconds = take_timing(begin, end, pending)) {
        if (copy) {
            add_copy_seconds(layer, *seconds);
        } else {
            add_exposed_seconds(layer, *seconds);
        }
        return;
    }
    // The pair is about to be recorded again, so its duration is gone: count it as lost rather than
    // silently reporting less copy time than the link carried.
    pending = 0;
    ++timing_lost_;
}

void StreamedKVSpace::collect_timings() {
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        if (const std::optional<double> seconds =
                take_timing(copy_begin_[layer], copy_end_[layer], copy_pending_[layer])) {
            add_copy_seconds(layer, *seconds);
        }
        if (const std::optional<double> seconds =
                take_timing(wait_begin_[layer], wait_end_[layer], wait_pending_[layer])) {
            add_exposed_seconds(layer, *seconds);
        }
    }
    for (std::size_t index = 0; index < frame_copy_begin_.size(); ++index) {
        const auto layer = static_cast<std::uint32_t>(index / kFrameSlots);
        if (const std::optional<double> seconds = take_timing(
                frame_copy_begin_[index], frame_copy_end_[index], frame_copy_pending_[index])) {
            add_copy_seconds(layer, *seconds);
        }
        if (const std::optional<double> seconds = take_timing(
                frame_wait_begin_[index], frame_wait_end_[index], frame_wait_pending_[index])) {
            add_exposed_seconds(layer, *seconds);
        }
    }
}

void StreamedKVSpace::reset_round() {
    round_.upload_bytes   = 0;
    round_.download_bytes = 0;
    round_.copy_seconds   = 0.0;
    round_.exposed_seconds = 0.0;
    for (StreamedKvLayerSample& sample : round_.layers) {
        sample.upload_bytes   = 0;
        sample.download_bytes = 0;
        sample.copy_seconds   = 0.0;
        sample.exposed_seconds = 0.0;
    }
}

StreamedKVSpace::RoundSample StreamedKVSpace::snapshot_round() const {
    RoundSample closed;
    closed.upload_bytes    = round_.upload_bytes;
    closed.download_bytes  = round_.download_bytes;
    closed.copy_seconds    = round_.copy_seconds;
    closed.exposed_seconds = round_.exposed_seconds;
    for (const StreamedKvLayerSample& sample : round_.layers) {
        if (sample.upload_bytes != 0 || sample.download_bytes != 0 || sample.copy_seconds != 0.0 ||
            sample.exposed_seconds != 0.0) {
            closed.layers.push_back(sample);
        }
    }
    return closed;
}

void StreamedKVSpace::close_round() {
    if (round_samples_.size() < kRoundSampleCap) { round_samples_.push_back(snapshot_round()); }
    reset_round();
}

void StreamedKVSpace::add_upload(std::uint32_t layer, std::uint64_t bytes) {
    upload_bytes_ += bytes;
    if (rounds_ == 0) {
        prefill_upload_bytes_ += bytes;
        return;
    }
    if (!report_) { return; }
    round_.upload_bytes += bytes;
    round_.layers[layer].upload_bytes += bytes;
}

void StreamedKVSpace::add_download(std::uint32_t layer, std::uint64_t bytes) {
    download_bytes_ += bytes;
    if (rounds_ == 0) {
        prefill_download_bytes_ += bytes;
        return;
    }
    if (!report_) { return; }
    round_.download_bytes += bytes;
    round_.layers[layer].download_bytes += bytes;
}

void StreamedKVSpace::begin_round() {
    // The per-round series is only built when the run asked to be timed: a normal run pays nothing
    // for it beyond the counters above.
    if (report_) {
        collect_timings();
        if (rounds_ != 0) { close_round(); }
    }
    ++rounds_;
    round_streamed_pages_ = window_start();
}

void StreamedKVSpace::wait_ready(std::uint32_t layer, cudaStream_t compute) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV wait needs a compute stream");
    }
    if (!report_) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
        return;
    }
    // The wait pair the device had not finished by this round is overwritten here; if it completed
    // in the meantime its time is taken now, otherwise it is counted as lost.
    close_timing(wait_pending_[layer], wait_begin_[layer], wait_end_[layer], layer, false);
    CUDA_CHECK(cudaEventRecord(wait_begin_[layer], compute));
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
    CUDA_CHECK(cudaEventRecord(wait_end_[layer], compute));
    wait_pending_[layer] = 1;
}

StreamedKVSpace::TransferStats StreamedKVSpace::transfer_stats() {
    if (report_) { collect_timings(); }
    TransferStats out;
    out.resident_pages  = split_.resident_pages;
    out.streamed_pages  = round_streamed_pages_;
    out.layers          = layers_;
    out.ring_depth      = split_.ring_depth;
    out.pages_per_layer = pages_per_layer_;
    out.page_bytes      = page_bytes_;
    out.host_bytes      = host_bytes();
    out.device_bytes    = device_bytes();
    out.rounds          = rounds_;
    out.upload_bytes    = upload_bytes_;
    out.download_bytes  = download_bytes_;
    out.copy_busy_seconds    = copy_seconds_;
    out.exposed_seconds      = exposed_seconds_;
    out.prefill_upload_bytes   = prefill_upload_bytes_;
    out.prefill_download_bytes = prefill_download_bytes_;
    out.prefill_copy_seconds     = prefill_copy_seconds_;
    out.prefill_exposed_seconds  = prefill_exposed_seconds_;
    out.timing_samples = timing_samples_;
    out.timing_lost    = timing_lost_;
    if (report_) {
        out.round_samples = round_samples_;
        // The round in flight is returned as a copy, not closed: publication may read this between
        // rounds, and the next `begin_round` is what pushes it into the series.
        if (rounds_ != 0 && round_samples_.size() == rounds_ - 1) {
            out.round_samples.push_back(snapshot_round());
        }
    }
    return out;
}

HostKVAllocationView StreamedKVSpace::host_pages(std::uint32_t layer) noexcept {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return arena_->writable_view(host_).subview(layer * pages_per_layer_, pages_per_layer_);
}

HostKVAllocationConstView StreamedKVSpace::host_pages(std::uint32_t layer) const noexcept {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return arena_->view(host_).subview(layer * pages_per_layer_, pages_per_layer_);
}
cudaEvent_t StreamedKVSpace::ready_event(std::uint32_t layer) const {
    if (layer >= ready_.size()) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return ready_[layer];
}

cudaEvent_t StreamedKVSpace::consumed_event(std::uint32_t layer) const {
    if (layer >= consumed_.size()) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return consumed_[layer];
}

cudaEvent_t StreamedKVSpace::dropped_event(std::uint32_t layer) const {
    if (layer >= dropped_.size()) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return dropped_[layer];
}

void StreamedKVSpace::publish_slot(std::uint32_t layer, std::uint32_t page, std::int32_t slot,
                                   RankStreams streams) {
    if (slot < 0 || static_cast<std::uint32_t>(slot) >= slot_count_) {
        throw std::logic_error("Streamed KV slot is outside its pool");
    }
    cache_->execution_tables().publish_slots_at(static_cast<std::int32_t>(layer), page,
                                                std::span<const std::int32_t>(&slot, 1), streams);
}

void StreamedKVSpace::advance(std::uint32_t active_pages, RankStreams streams) {
    if (active_pages < active_) {
        throw std::invalid_argument("A streamed KV frontier only grows; reset before rewinding");
    }
    if (active_pages > pages_per_layer_) {
        throw std::invalid_argument("A streamed KV frontier exceeds the cache's context");
    }
    for (std::uint32_t page = active_; page < active_pages; ++page) {
        // The page appears in its layer's window, where this round's append writes it.
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            publish_slot(layer, page, kv_stream_window_slot(layer, page, split_), streams);
            if (page < split_.resident_pages) { continue; }
            // The window is full, so the page it drops leaves the device: its bytes reach the host
            // first, and its table entry then points at the ring slot it will be read from. The
            // copy is ordered here, before this round's append overwrites the slot the arriving
            // page takes - the two pages' indices differ by the window depth, so it is the same.
            const std::uint32_t dropped     = page - split_.resident_pages;
            const std::int32_t dropped_slot = kv_stream_window_slot(layer, dropped, split_);
            cache_->page_pool().copy_to_host_pages(
                dropped_slot, 1, host_pages(layer).subview(dropped, 1), streams);
            add_download(layer, page_bytes_);
            publish_slot(layer, dropped, kv_stream_ring_slot(layer, dropped, split_, layers_), streams);
        }
    }
    // The host copies above are ordered on the compute streams, and the uploads that read those
    // pages run on the copy stream: without this mark the copy stream could read a page while the
    // compute stream is still writing it.
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        CUDA_CHECK(cudaEventRecord(dropped_[layer], streams[0]));
    }
    active_ = active_pages;
}

void StreamedKVSpace::upload(std::uint32_t layer, cudaStream_t copy) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (copy == nullptr) { throw std::invalid_argument("A streamed KV upload needs a copy stream"); }
    const std::uint32_t count   = window_start();
    const std::uint32_t region  = layer % split_.ring_depth;
    // The pages this region already holds for this layer are not copied twice: a round that
    // prefetched the next round's first layers only moves the pages appended since.
    const std::uint32_t already =
        region_layer_[region] == layer && region_pages_[region] <= count ? region_pages_[region]
                                                                        : 0U;
    if (report_) {
        // A prefetched copy was recorded for this layer and never collected: take its time before
        // the pair is recorded again, or the round would report less traffic than it moved.
        close_timing(copy_pending_[layer], copy_begin_[layer], copy_end_[layer], layer, true);
        CUDA_CHECK(cudaEventRecord(copy_begin_[layer], copy));
    }
    if (count > already) {
        add_upload(layer, static_cast<std::uint64_t>(count - already) * page_bytes_);
        // The pages `[already, count)` of this layer, into the ring slots they map to; the pool
        // turns the run into one strided transfer per plane. The host copies a round's window
        // dropped are ordered on the compute stream, so the uploads that read them wait first.
        CUDA_CHECK(cudaStreamWaitEvent(copy, dropped_[layer], 0));
        cache_->page_pool().copy_from_host_pages(
            host_pages(layer).subview(already, count - already).data(), host_layout_,
            static_cast<std::int32_t>(ring_region(layer) + already), count - already,
            RankStreams(copy));
    }
    region_layer_[region] = layer;
    region_pages_[region] = count;
    if (report_) {
        CUDA_CHECK(cudaEventRecord(copy_end_[layer], copy));
        copy_pending_[layer] = 1;
    }
    CUDA_CHECK(cudaEventRecord(ready_[layer], copy));
}

cudaEvent_t StreamedKVSpace::consumed_frame_event(std::uint32_t layer,
                                                  std::uint32_t frame) const {
    if (layer >= layers_ || frame >= kFrameSlots) {
        throw std::out_of_range("Streamed KV frame is out of range");
    }
    return consumed_frame_[layer * kFrameSlots + frame];
}

std::uint32_t StreamedKVSpace::upload_frame(std::uint32_t layer, std::uint32_t frame,
                                           std::int32_t first_page, std::int32_t page_count,
                                           cudaStream_t copy, bool seal) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (frame >= kFrameSlots) { throw std::out_of_range("Streamed KV frame is out of range"); }
    if (copy == nullptr) { throw std::invalid_argument("A streamed KV upload needs a copy stream"); }
    const std::size_t index = static_cast<std::size_t>(layer) * kFrameSlots + frame;
    // Only the pages below the window start are on the host; the rest are resident already.
    const std::int32_t window = static_cast<std::int32_t>(window_start());
    const std::int32_t first  = first_page < 0 ? 0 : (first_page > window ? window : first_page);
    const std::int32_t last   = first_page + page_count < window ? first_page + page_count : window;
    std::uint32_t moved       = 0;
    if (report_ && !frame_copy_pending_[index]) {
        CUDA_CHECK(cudaEventRecord(frame_copy_begin_[index], copy));
        frame_copy_pending_[index] = 1;
    }
    if (last > first) {
        // The ring slots these pages take are the ones the layer `ring_depth` back wrote, so the
        // copy is ordered after that layer's consumption of the same frame - and after the host
        // bytes of this round's dropped pages arrived.
        const std::uint32_t source = (layer + layers_ - split_.ring_depth) % layers_;
        CUDA_CHECK(cudaStreamWaitEvent(copy, consumed_frame_event(source, frame), 0));
        CUDA_CHECK(cudaStreamWaitEvent(copy, dropped_[layer], 0));
        const auto count = static_cast<std::uint32_t>(last - first);
        cache_->page_pool().copy_from_host_pages(
            host_pages(layer).subview(static_cast<std::uint32_t>(first), count).data(), host_layout_,
            static_cast<std::int32_t>(ring_region(layer) + static_cast<std::uint32_t>(first)),
            count, RankStreams(copy));
        add_upload(layer, static_cast<std::uint64_t>(count) * page_bytes_);
        moved = count;
    }
    if (seal) { seal_frame(layer, frame, copy); }
    return moved;
}

void StreamedKVSpace::seal_frame(std::uint32_t layer, std::uint32_t frame, cudaStream_t copy) {
    const std::size_t index = static_cast<std::size_t>(layer) * kFrameSlots + frame;
    if (report_) {
        // A frame that moved nothing has no begin to pair with, and an event that was never
        // recorded has no time to read: only a frame whose copy began is timed.
        if (frame_copy_pending_[index]) {
            CUDA_CHECK(cudaEventRecord(frame_copy_end_[index], copy));
        }
    }
    CUDA_CHECK(cudaEventRecord(ready_frame_[index], copy));
}

void StreamedKVSpace::wait_ready_frame(std::uint32_t layer, std::uint32_t frame,
                                       cudaStream_t compute) {
    if (layer >= layers_ || frame >= kFrameSlots) {
        throw std::out_of_range("Streamed KV frame is out of range");
    }
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV wait needs a compute stream");
    }
    const std::size_t index = static_cast<std::size_t>(layer) * kFrameSlots + frame;
    if (!report_) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready_frame_[index], 0));
        return;
    }
    // The wait is measured: whatever it costs here is copy time this round could not hide.
    CUDA_CHECK(cudaEventRecord(frame_wait_begin_[index], compute));
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_frame_[index], 0));
    CUDA_CHECK(cudaEventRecord(frame_wait_end_[index], compute));
    frame_wait_pending_[index] = 1;
}

} // namespace ninfer::models::qwen3_5
