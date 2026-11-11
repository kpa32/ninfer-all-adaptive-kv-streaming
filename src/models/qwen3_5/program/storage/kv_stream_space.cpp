#include "models/qwen3_5/program/storage/kv_stream_space.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

void create_events(std::vector<cudaEvent_t>& events, std::uint32_t count) {
    events.assign(count, nullptr);
    for (cudaEvent_t& event : events) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
}

void grow_events(std::vector<cudaEvent_t>& events, std::uint32_t count) {
    for (std::uint32_t index = static_cast<std::uint32_t>(events.size()); index < count; ++index) {
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events.push_back(event);
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
    // A layer may keep no window at all (every page of it streams, which is the degenerate plan a pool
    // that holds one layer's extent can serve), but the space as a whole needs the ring, the run size and
    // a context to describe.
    if (layers_ == 0 || split_.layers() != layers_ || split_.run_pages == 0 ||
        cache.max_context() == 0) {
        throw std::invalid_argument(
            "A streamed KV space needs layers, runs and a context: layers=" +
            std::to_string(layers_) + " planned=" + std::to_string(split_.layers()) +
            " resident=" + std::to_string(split_.resident_pages()) +
            " ring=" + std::to_string(split_.ring_pages) + " run=" +
            std::to_string(split_.run_pages) + " slots=" +
            std::to_string(cache.page_pool().capacity_pages()) + " context=" +
            std::to_string(cache.max_context()));
    }
    pages_per_layer_ = page_count(cache.max_context());
    ring_base_       = split_.layer_offset(layers_);
    // Every slot of the pool belongs to this space for the cache's lifetime: the mapping names
    // slots, so nothing else may allocate, lend or release one.
    const std::uint32_t slot_count = cache.page_pool().capacity_pages();
    slot_count_                    = slot_count;
    if (slot_count < kv_stream_pool_pages(split_)) {
        throw std::invalid_argument(
            "A streamed KV pool is smaller than its split: slots=" + std::to_string(slot_count) +
            " needs=" + std::to_string(kv_stream_pool_pages(split_)));
    }
    // A layer reads its whole streamed extent in one call, so the ring must hold that extent -
    // otherwise the copy would overwrite slots the consumer has not read yet.
    if (split_.ring_pages < pages_per_layer_ - split_.min_resident()) {
        throw std::invalid_argument(
            "A streamed KV ring cannot hold one layer's streamed pages: ring=" +
            std::to_string(split_.ring_pages) + " extent=" +
            std::to_string(pages_per_layer_ - split_.min_resident()));
    }
    host_layout_ = plan_host_kv_page_layout(cache.page_pool().geometry());
    page_bytes_  = host_layout_.page_stride;
    arena_ = std::make_unique<HostKVArena>(page_bytes_ * pages_per_layer_ * layers_,
                                           std::span<const HostKVPageLayout>(&host_layout_, 1));
    auto host = arena_->allocate(host_layout_, pages_per_layer_ * layers_);
    if (!host) {
        throw std::invalid_argument("The host KV arena is too small for this cache's context");
    }
    host_ = std::move(*host);

    create_events(ready_, layers_);
    create_events(consumed_, layers_);
    create_events(dropped_, layers_);
    // The ring is a flat pool of runs: any layer may own any run, and a run's previous owner is what
    // its copy waits for (kv_stream_slot_map.h, the reference's per-slot ready/consumed pair).
    ring_runs_ = kv_stream_ring_runs(split_);
    create_events(ready_run_, ring_runs_);
    create_events(consumed_run_, ring_runs_);
    run_used_.assign(ring_runs_, 0);
    run_owner_.assign(ring_runs_, kNoLayer);
    run_layer_.assign(ring_runs_, kNoLayer);
    layer_first_run_.assign(layers_, 0);
    layer_runs_.assign(layers_, 0);
    stream_slots_.assign(pages_per_layer_, 0);
    // One deadline flag per layer: a layer's streamed pages are copied as one run sequence, so a
    // layer is the unit that can be late.
    signal_ = std::make_unique<KVStreamSignal>(layers_);
    round_.layers.resize(layers_);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) { round_.layers[layer].layer = layer; }
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
    }
}
StreamedKVSpace::~StreamedKVSpace() {
    destroy_events(ready_);
    destroy_events(consumed_);
    destroy_events(dropped_);
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
}

void StreamedKVSpace::reset_round() {
    round_.upload_bytes   = 0;
    round_.download_bytes = 0;
    round_.copy_seconds   = 0.0;
    round_.exposed_seconds = 0.0;
    round_.deadline_samples = 0;
    round_.deadline_misses  = 0;
    round_.copy_busy_ratio  = 0.0;
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
    closed.deadline_samples = round_.deadline_samples;
    closed.deadline_misses  = round_.deadline_misses;
    closed.copy_busy_ratio  = round_.copy_busy_ratio;
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
    // Per-round bytes are measurement the campaigns read, and they cost an addition: only the timing
    // events that price a copy are optional (the `report_` pairs in the copy path).
    round_.upload_bytes += bytes;
    round_.layers[layer].upload_bytes += bytes;
}

void StreamedKVSpace::add_download(std::uint32_t layer, std::uint64_t bytes) {
    download_bytes_ += bytes;
    if (rounds_ == 0) {
        prefill_download_bytes_ += bytes;
        return;
    }
    round_.download_bytes += bytes;
    round_.layers[layer].download_bytes += bytes;
}

void StreamedKVSpace::begin_round(cudaStream_t compute) {
    // The feedback window covers one round of the decode loop: it opens on the compute stream before
    // the round's first layer and closes at the next round's boundary, when everything the round
    // enqueued has run. A window the device has not finished stays open instead of being replaced -
    // a replaced window is a measurement lost, not one guessed (kv_stream_signal.h).
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV round needs a compute stream");
    }
    if (signal_window_open_ && !signal_end_recorded_) {
        signal_->end_window(compute);
        signal_end_recorded_ = true;
    }
    if (signal_window_open_ && close_signal_window()) {
        signal_window_open_ = false;
        signal_end_recorded_ = false;
    }
    if (!signal_window_open_) {
        signal_->begin_window(compute);
        signal_base_         = signal_->deadline_counters();
        signal_uploads_      = 0;
        signal_window_open_  = true;
        signal_end_recorded_ = false;
    }
    // The per-round series is built on every run: it is what the benchmark campaigns read to prove a
    // round really moved pages. Only the timing pairs that price a copy are optional (`report_`).
    if (report_) { collect_timings(); }
    if (rounds_ != 0) { close_round(); }
    ++rounds_;
    round_streamed_pages_ = max_streamed_pages();
}

bool StreamedKVSpace::close_signal_window() noexcept {
    if (!signal_window_open_) { return true; }
    const KVStreamSignal::WindowTime window = signal_->window_time();
    if (!window.recorded) { return false; }
    const KVStreamSignal::CopyTime copy = signal_->copy_time();
    const KVStreamDeadlineCounters current = signal_->deadline_counters();
    const KVStreamFeedbackDelta delta = kv_stream_feedback_delta(current, signal_base_);
    if (delta.valid) {
        deadline_samples_ += delta.samples;
        deadline_misses_  += delta.misses;
        round_.deadline_samples = delta.samples;
        round_.deadline_misses  = delta.misses;
    }
    KVStreamBusySample sample;
    sample.copy_seconds    = copy.seconds;
    sample.sampled_uploads = copy.uploads == 0 ? 1 : copy.uploads;
    sample.window_uploads  = signal_uploads_;
    sample.window_seconds  = window.seconds;
    sample.recorded        = copy.recorded;
    round_.copy_busy_ratio = kv_stream_copy_busy_ratio(sample);
    window_seconds_        = window.seconds;
    if (copy.recorded) {
        copy_busy_ratio_ = round_.copy_busy_ratio;
        ++copy_busy_windows_;
    }
    return true;
}
std::uint32_t StreamedKVSpace::upload(std::uint32_t layer, cudaStream_t copy) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (copy == nullptr) { throw std::invalid_argument("A streamed KV upload needs a copy stream"); }
    const std::uint32_t streamed = window_start(layer);
    const std::uint32_t runs     = kv_stream_runs_of_layer(active_, split_, layer);
    layer_first_run_[layer]      = run_cursor_;
    layer_runs_[layer]           = runs;
    if (runs == 0) {
        // Nothing streams: every page is resident, and the layer has no run to wait for.
        CUDA_CHECK(cudaEventRecord(ready_[layer], copy));
        return 0;
    }
    if (report_) {
        close_timing(copy_pending_[layer], copy_begin_[layer], copy_end_[layer], layer, true);
        CUDA_CHECK(cudaEventRecord(copy_begin_[layer], copy));
    }
    // The slots this layer's pages are named by: run `k` of the layer takes ring run
    // `(first + k) mod ring_runs`, and a page inside a run keeps its offset. Built once for the layer
    // and published in one copy.
    for (std::uint32_t page = 0; page < streamed; ++page) {
        stream_slots_[page] = static_cast<std::int32_t>(slot_of_page(layer, page));
    }
    CUDA_CHECK(cudaStreamWaitEvent(copy, dropped_[layer], 0));
    signal_->begin_copy_sample(copy);
    signal_->clear_run(layer, copy);
    std::uint32_t moved = 0;
    for (std::uint32_t run = 0; run < runs; ++run) {
        const std::uint32_t ring_run = (layer_first_run_[layer] + run) % ring_runs_;
        // The run's previous owner consumed it before this copy may overwrite it. A ring that holds
        // only one extent therefore serialises the layers - the copy waits for the reader - while a
        // ring that holds one extent plus a margin lets the next layer start as this one reads, which
        // is what the split's ratio buys (kv_stream_layout.h).
        if (run_used_[ring_run]) {
            CUDA_CHECK(cudaStreamWaitEvent(copy, consumed_run_[ring_run], 0));
        }
        const std::uint32_t first = run*split_.run_pages;
        const std::uint32_t count =
            streamed - first < split_.run_pages ? streamed - first : split_.run_pages;
        cache_->page_pool().copy_from_host_pages(
            host_pages(layer).subview(first, count).data(), host_layout_,
            static_cast<std::int32_t>(ring_base_ + ring_run*split_.run_pages), count,
            RankStreams(copy));
        CUDA_CHECK(cudaEventRecord(ready_run_[ring_run], copy));
        run_used_[ring_run]  = 1;
        run_owner_[ring_run] = layer;
        run_layer_[ring_run] = layer;
        moved += count;
    }
    ++signal_uploads_;
    signal_->mark_run(layer, copy);
    signal_->end_copy_sample(copy);
    // One publication for the whole layer: the entries above are its streamed prefix, and a single
    // strided H2D per layer is what keeps a per-round re-addressing affordable.
    cache_->execution_tables().publish_slots_at(
        static_cast<std::int32_t>(layer), 0,
        std::span<const std::int32_t>(stream_slots_.data(), streamed), RankStreams(copy));
    add_upload(layer, static_cast<std::uint64_t>(moved) * page_bytes_);
    if (report_) {
        CUDA_CHECK(cudaEventRecord(copy_end_[layer], copy));
        copy_pending_[layer] = 1;
    }
    run_cursor_ = (layer_first_run_[layer] + runs) % ring_runs_;
    CUDA_CHECK(cudaEventRecord(ready_[layer], copy));
    verify_streamed_page(layer, streamed);
    return runs;
}

void StreamedKVSpace::verify_streamed_page(std::uint32_t layer, std::uint32_t streamed) {
    // A diagnostic that answers one question exactly: do the bytes the ring now holds for a streamed
    // page match the host record the table points the consumer at? It synchronises, so it only runs in
    // an eager (`--no-cuda-graph`) run with `NINFER_STREAM_VERIFY` set.
    if (std::getenv("NINFER_STREAM_VERIFY") == nullptr || streamed == 0 || verify_budget_ <= 0) {
        return;
    }
    --verify_budget_;
    if (!verify_arena_) {
        verify_arena_ = std::make_unique<HostKVArena>(
            page_bytes_, std::span<const HostKVPageLayout>(&host_layout_, 1));
        verify_host_ = verify_arena_->allocate(host_layout_, 1);
        if (!verify_host_) { throw std::logic_error("the verify staging arena is too small"); }
    }
    for (const std::uint32_t page : {0U, streamed - 1U}) {
        const std::uint32_t slot = slot_of_page(layer, page);
        // Wait for every copy this layer issued - `ready_` is recorded after the last one - so the probe
        // compares the bytes the copies left, not the bytes the previous owner left.
        CUDA_CHECK(cudaEventSynchronize(ready_[layer]));
        cache_->page_pool().copy_to_host_pages(slot, 1, verify_arena_->writable_view(*verify_host_),
                                               RankStreams());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto* got  = verify_arena_->view(*verify_host_).data();
        const auto* want = host_pages(layer).subview(page, 1).data();
        ++verify_checks_;
        if (std::memcmp(got, want, page_bytes_) != 0) {
            ++verify_mismatches_;
            std::size_t first_diff = 0;
            while (first_diff < page_bytes_ && got[first_diff] == want[first_diff]) { ++first_diff; }
            // Which record does the slot actually hold? The fingerprint names the writer, which is what
            // decides whether the table, the copy or the reuse gate is wrong.
            char holder[64] = "none";
            bool named      = false;
            for (std::uint32_t other_layer = 0; other_layer < layers_ && !named; ++other_layer) {
                for (std::uint32_t other_page = 0; other_page < active_; ++other_page) {
                    if (std::memcmp(got, host_pages(other_layer).subview(other_page, 1).data(),
                                    page_bytes_) != 0) {
                        continue;
                    }
                    std::snprintf(holder, sizeof(holder), "layer=%u page=%u", other_layer,
                                  other_page);
                    named = true;
                    break;
                }
            }
            std::fprintf(stderr,
                         "[verify] MISMATCH layer=%u page=%u slot=%u active=%u streamed=%u "
                         "first_byte=%zu holds=%s\n",
                         layer, page, slot, active_, streamed, first_diff, holder);
        } else {
            std::fprintf(stderr, "[verify] ok layer=%u page=%u slot=%u streamed=%u\n", layer, page,
                         slot, streamed);
        }
    }
}

void StreamedKVSpace::wait_ready(std::uint32_t layer, cudaStream_t compute) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV wait needs a compute stream");
    }
    const std::uint32_t runs = layer_runs_[layer];
    if (runs == 0) { return; }
    // The runs were recorded in order on one copy stream, so the last one landing implies the rest.
    // `ready_` is recorded after the block-table publication, so waiting on it as well is what orders
    // the consumer's *addressing* with the bytes: without it the attention can still resolve a streamed
    // page to the slot it held last round, and the append of this round's tokens can write into a ring
    // slot the table names - which measured as a 0.1-nat shift with the greedy ids unchanged.
    const std::uint32_t last = (layer_first_run_[layer] + runs - 1) % ring_runs_;
    if (report_) {
        signal_->sample_deadline(layer, compute);
        close_timing(wait_pending_[layer], wait_begin_[layer], wait_end_[layer], layer, false);
        CUDA_CHECK(cudaEventRecord(wait_begin_[layer], compute));
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready_run_[last], 0));
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
        CUDA_CHECK(cudaEventRecord(wait_end_[layer], compute));
        wait_pending_[layer] = 1;
        return;
    }
    signal_->sample_deadline(layer, compute);
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_run_[last], 0));
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
}

void StreamedKVSpace::wait_run(std::uint32_t layer, std::uint32_t run, cudaStream_t compute) {
    if (layer >= layers_ || run >= layer_runs_[layer]) {
        throw std::out_of_range("Streamed KV run is out of range");
    }
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV wait needs a compute stream");
    }
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_run_[(layer_first_run_[layer] + run) % ring_runs_],
                                  0));
    CUDA_CHECK(cudaStreamWaitEvent(compute, ready_[layer], 0));
}

void StreamedKVSpace::wait_run_range(std::uint32_t layer, std::uint32_t first_run,
                                     std::uint32_t last_run, cudaStream_t compute) {
    for (std::uint32_t run = first_run; run <= last_run; ++run) { wait_run(layer, run, compute); }
}

void StreamedKVSpace::consume(std::uint32_t layer, cudaStream_t compute) {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    // The post-attention half of the verify probe: the same slots the upload wrote, read again after the
    // attention that consumed them. A mismatch here means something other than the copy wrote those
    // slots while the layer read them.
    verify_after_attention(layer);
    for (std::uint32_t run = 0; run < layer_runs_[layer]; ++run) { consume_run(layer, run, compute); }
    CUDA_CHECK(cudaEventRecord(consumed_[layer], compute));
}

void StreamedKVSpace::verify_after_attention(std::uint32_t layer) {
    if (std::getenv("NINFER_STREAM_VERIFY") == nullptr || verify_after_budget_ <= 0) { return; }
    const std::uint32_t streamed = window_start(layer);
    if (streamed == 0 || !verify_arena_ || !verify_host_) { return; }
    --verify_after_budget_;
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::uint32_t page = 0;
    const std::uint32_t slot = slot_of_page(layer, page);
    cache_->page_pool().copy_to_host_pages(slot, 1, verify_arena_->writable_view(*verify_host_),
                                           RankStreams());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto* got  = verify_arena_->view(*verify_host_).data();
    const auto* want = host_pages(layer).subview(page, 1).data();
    ++verify_checks_;
    if (std::memcmp(got, want, page_bytes_) != 0) {
        ++verify_mismatches_;
        std::size_t first_diff = 0;
        while (first_diff < page_bytes_ && got[first_diff] == want[first_diff]) { ++first_diff; }
        std::size_t last_diff = page_bytes_;
        while (last_diff > 0 && got[last_diff - 1] == want[last_diff - 1]) { --last_diff; }
        std::size_t differs = 0;
        for (std::size_t i = 0; i < page_bytes_; ++i) { differs += got[i] != want[i]; }
        // Whose bytes are these now? A full-page match names the copy that overwrote the slot; no match
        // at all means the writer is not a copy of a host record but the append itself.
        int found_layer = -1;
        unsigned found_page = 0;
        for (std::uint32_t other = 0; other < layers_ && found_layer < 0; ++other) {
            const std::uint32_t limit = window_start(other);
            for (std::uint32_t candidate = 0; candidate < limit; ++candidate) {
                if (std::memcmp(got, host_pages(other).subview(candidate, 1).data(), page_bytes_) ==
                    0) {
                    found_layer = static_cast<int>(other);
                    found_page  = candidate;
                    break;
                }
            }
        }
        std::fprintf(stderr, "[verify-after] MISMATCH layer=%u page=%u slot=%u active=%u streamed=%u "
                     "first_byte=%zu last_byte=%zu differs=%zu/%zu first_run=%u runs=%u ring=%u "
                     "holds=%s%u\n",
                     layer, page, slot, active_, streamed, first_diff, last_diff, differs, page_bytes_,
                     layer_first_run_[layer], layer_runs_[layer], ring_runs_,
                     found_layer < 0 ? "nothing" : "host(layer,page)=",
                     found_layer < 0 ? 0U : (static_cast<unsigned>(found_layer) << 16) | found_page);
    } else {
        std::fprintf(stderr, "[verify-after] ok layer=%u page=%u slot=%u streamed=%u\n", layer, page,
                     slot, streamed);
    }
}

void StreamedKVSpace::consume_run(std::uint32_t layer, std::uint32_t run, cudaStream_t compute) {
    if (layer >= layers_ || run >= layer_runs_[layer]) {
        throw std::out_of_range("Streamed KV run is out of range");
    }
    if (compute == nullptr) {
        throw std::invalid_argument("A streamed KV consume needs a compute stream");
    }
    const std::uint32_t ring_run = (layer_first_run_[layer] + run) % ring_runs_;
    CUDA_CHECK(cudaEventRecord(consumed_run_[ring_run], compute));
}

void StreamedKVSpace::consume_run_range(std::uint32_t layer, std::uint32_t first_run,
                                        std::uint32_t last_run, cudaStream_t compute) {
    for (std::uint32_t run = first_run; run <= last_run; ++run) { consume_run(layer, run, compute); }
}

std::uint32_t StreamedKVSpace::first_run(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    return layer_first_run_[layer];
}

std::uint32_t StreamedKVSpace::slot_of_page(std::uint32_t layer, std::uint32_t page) const {
    if (layer >= layers_) { throw std::out_of_range("Streamed KV layer is out of range"); }
    if (page >= window_start(layer)) {
        return static_cast<std::uint32_t>(kv_stream_window_slot(layer, page, split_));
    }
    if (ring_runs_ == 0) {
        // Nothing can hold a page the window dropped: an all-resident split's window is its whole
        // frontier (`advance` keeps them equal). Reaching here is a split that contradicts its own
        // frontier, which is a bug to report rather than a modulo by zero to die on.
        throw std::logic_error(
            "An all-resident streamed KV split has no ring to hold a page outside its window");
    }
    const std::uint32_t run = (layer_first_run_[layer] + page / split_.run_pages) % ring_runs_;
    return ring_base_ + run*split_.run_pages + page % split_.run_pages;
}

void StreamedKVSpace::enter_request(std::uint32_t tokens) {
    const std::uint32_t pages = tokens == 0 ? 0U : page_count(tokens);
    if (pages > active_) {
        throw std::logic_error("streamed KV frontier is beyond the space's context");
    }
    if (pages < active_) {
        // Nothing continues: every page the prompt reaches is republished as it is written, and any
        // page past its frontier is never read again.
        active_ = 0;
    }
    retained_          = false;
    retained_frontier_ = 0;
    clear_round_bookkeeping();
}

void StreamedKVSpace::advance(std::uint32_t active_pages, RankStreams streams) {
    if (active_pages < active_) {
        throw std::invalid_argument("A streamed KV frontier only grows; reset before rewinding");
    }
    if (active_pages > pages_per_layer_) {
        throw std::invalid_argument("A streamed KV frontier exceeds the cache's context");
    }
    if (ring_runs_ == 0 && active_pages > split_.resident_of(0)) {
        // An all-resident split is planned for the whole context, so its window is every page a frontier
        // can name and this cannot happen. Widening it would be a layout change, and a page's slot is its
        // layer's base plus its position in the window, so a new shape renames every resident page: a
        // rename that does not move every byte answers with another page's KV
        // (bench/results/kv-stream/layout-rename-defect.md). Report it rather than re-shape it.
        throw std::logic_error(
            "An all-resident streamed KV split cannot grow past the window it was planned with");
    }
    for (std::uint32_t page = active_; page < active_pages; ++page) {
        // The page appears in its layer's window, where this round's append writes it.
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            publish_slot(layer, page, kv_stream_window_slot(layer, page, split_), streams);
            const std::uint32_t resident = split_.resident_of(layer);
            if (page < resident) { continue; }
            // The window is full, so the page it drops leaves the device: its bytes reach the host
            // first, and this round's copy names its streamed slot when it brings it back.
            const std::uint32_t dropped = page - resident;
            cache_->page_pool().copy_to_host_pages(
                kv_stream_window_slot(layer, dropped, split_), 1, host_pages(layer).subview(dropped, 1),
                streams);
            add_download(layer, page_bytes_);
        }
    }
    // The host copies above are ordered on the compute streams, and the uploads that read those pages
    // run on the copy stream: without this mark the copy stream could read a page while the compute
    // stream is still writing it.
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        CUDA_CHECK(cudaEventRecord(dropped_[layer], streams[0]));
    }
    active_ = active_pages;
}
StreamedKVSpace::TransferStats StreamedKVSpace::transfer_stats() {
    if (report_) { collect_timings(); }
    TransferStats out;
    out.resident_pages  = split_.min_resident();
    out.streamed_pages  = round_streamed_pages_;
    out.layers          = layers_;
    out.ring_runs       = ring_runs_;
    out.run_pages       = split_.run_pages;
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
    out.deadline_samples  = deadline_samples_;
    out.deadline_misses   = deadline_misses_;
    out.copy_busy_windows = copy_busy_windows_;
    out.copy_busy_ratio   = copy_busy_ratio_;
    // The shape a round ran in: every measurement is only comparable inside one, and the report is
    // the only place a campaign can learn which one produced it.
    out.layer_resident      = split_.layer_resident;
    out.active_pages        = active_;
    // Whether a finished request's context still lives here, and at which token frontier: the next
    // request either continues it or starts from an empty space.
    out.retained          = retained_;
    out.retained_frontier = retained_frontier_;
    out.timing_samples = timing_samples_;
    out.timing_lost    = timing_lost_;
    out.verify_checks     = verify_checks_;
    out.verify_mismatches = verify_mismatches_;
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

} // namespace ninfer::models::qwen3_5
